"""
端到端碱基识别流水线 (Basecalling Pipeline)

整合从原始 Fast5 电流信号到 DNA 序列的完整流程：
1. 读取 Fast5 原始电流信号
2. MAD 归一化
3. 信号去噪（尖峰修复、接头切除）
4. 神经网络前向推理
5. CTC 贪心解码
6. 输出 FASTA/FASTQ 格式结果

支持单条序列处理和批量处理两种模式。
"""

import os
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Generator
from pathlib import Path

from ..io.fast5_reader import StreamingFast5Reader, Fast5Reader
from ..signal.normalizer import MADNormalizer
from ..signal.denoiser import SignalDenoiser
from ..model.basecaller_net import BasecallerNet
from ..decoding.ctc_decoder import CTCGreedyDecoder


@dataclass
class BasecallResult:
    """碱基识别结果数据类"""

    read_id: str
    sequence: str
    quality_scores: Optional[np.ndarray] = None
    mean_confidence: float = 0.0
    signal_length: int = 0
    sequence_length: int = 0
    num_bases: int = 0
    process_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_fasta(self) -> str:
        """转换为 FASTA 格式字符串"""
        header = f">{self.read_id}"
        if self.mean_confidence > 0:
            header += f" mean_q={self.mean_confidence:.3f}"
        return f"{header}\n{self.sequence}\n"

    def to_fastq(self) -> str:
        """转换为 FASTQ 格式字符串"""
        if self.quality_scores is None:
            q_str = "I" * len(self.sequence)
        else:
            q_scores = np.clip(self.quality_scores, 0, 40)
            q_chars = [chr(int(q) + 33) for q in q_scores]
            q_str = "".join(q_chars)

        header = f"@{self.read_id}"
        return f"{header}\n{self.sequence}\n+\n{q_str}\n"


class BasecallPipeline:
    """端到端碱基识别流水线
    
    整合所有处理模块，提供简洁的调用接口。
    """

    def __init__(
        self,
        model: Optional[BasecallerNet] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        normalizer: Optional[MADNormalizer] = None,
        denoiser: Optional[SignalDenoiser] = None,
        decoder: Optional[CTCGreedyDecoder] = None,
        normalize_mode: str = "global",
        trim_adapter: bool = True,
        denoise_signal: bool = True,
        batch_size: int = 16,
    ):
        """
        Args:
            model: 预加载的模型
            model_path: 模型权重文件路径
            device: 计算设备 ('cpu', 'cuda', None=自动选择)
            normalizer: 自定义归一化器
            denoiser: 自定义去噪器
            decoder: 自定义解码器
            normalize_mode: 归一化模式 ('global', 'windowed', 'chunks')
            trim_adapter: 是否切除接头序列
            denoise_signal: 是否进行信号去噪
            batch_size: 批处理大小
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if model is not None:
            self.model = model
        else:
            self.model = BasecallerNet()

        if model_path is not None and os.path.exists(model_path):
            self.load_model(model_path)

        self.model.to(self.device)
        self.model.eval()

        self.normalizer = normalizer or MADNormalizer()
        self.denoiser = denoiser or SignalDenoiser()
        self.decoder = decoder or CTCGreedyDecoder()

        self.normalize_mode = normalize_mode
        self.trim_adapter = trim_adapter
        self.denoise_signal = denoise_signal
        self.batch_size = batch_size

    def load_model(self, model_path: str) -> None:
        """加载模型权重
        
        Args:
            model_path: 模型权重文件路径
        """
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """信号预处理：归一化 + 去噪 + 接头切除
        
        Args:
            raw_signal: 原始 pA 电流信号
            
        Returns:
            预处理后的归一化信号
        """
        signal = raw_signal.astype(np.float32)

        if self.normalize_mode == "global":
            signal = self.normalizer.normalize(signal)
        elif self.normalize_mode == "windowed":
            signal = self.normalizer.normalize_windowed(signal)
        elif self.normalize_mode == "chunks":
            signal = self.normalizer.normalize_chunks(signal)
        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize_mode}")

        if self.trim_adapter:
            signal, _ = self.denoiser.trim_adapter(signal)

        if self.denoise_signal:
            signal, _ = self.denoiser.denoise(signal)

        return signal

    def basecall_single(
        self,
        signal: np.ndarray,
        read_id: str = "unknown",
    ) -> BasecallResult:
        """对单条信号进行碱基识别
        
        Args:
            signal: 原始电流信号 (pA)
            read_id: 读取 ID
            
        Returns:
            BasecallResult 对象
        """
        start_time = time.time()

        processed_signal = self.preprocess_signal(signal)

        if len(processed_signal) == 0:
            return BasecallResult(
                read_id=read_id,
                sequence="",
                signal_length=len(signal),
                process_time=time.time() - start_time,
            )

        signal_tensor = torch.from_numpy(processed_signal).float()
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
        signal_tensor = signal_tensor.to(self.device)

        length_tensor = torch.tensor([len(processed_signal)], dtype=torch.int32)

        with torch.no_grad():
            log_probs, out_lengths = self.model(signal_tensor, length_tensor)

        log_probs_np = log_probs.cpu().numpy()
        out_len = int(out_lengths[0].item()) if out_lengths is not None else log_probs.shape[0]

        sequence, confidences, q_scores = self.decoder.decode_with_quality(
            log_probs_np[:, 0, :], length=out_len
        )

        mean_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0

        return BasecallResult(
            read_id=read_id,
            sequence=sequence,
            quality_scores=q_scores,
            mean_confidence=mean_conf,
            signal_length=len(signal),
            sequence_length=len(sequence),
            num_bases=len(sequence),
            process_time=time.time() - start_time,
        )

    def basecall_batch(
        self,
        signals: List[np.ndarray],
        read_ids: List[str] = None,
    ) -> List[BasecallResult]:
        """批量碱基识别
        
        Args:
            signals: 原始信号列表
            read_ids: 读取 ID 列表
            
        Returns:
            BasecallResult 列表
        """
        if read_ids is None:
            read_ids = [f"read_{i}" for i in range(len(signals))]

        assert len(signals) == len(read_ids)

        results = []
        for i in range(0, len(signals), self.batch_size):
            batch_signals = signals[i : i + self.batch_size]
            batch_ids = read_ids[i : i + self.batch_size]

            for sig, rid in zip(batch_signals, batch_ids):
                result = self.basecall_single(sig, rid)
                results.append(result)

        return results

    def basecall_fast5(
        self,
        fast5_path: str,
        max_reads: Optional[int] = None,
        output_fastq: Optional[str] = None,
        output_fasta: Optional[str] = None,
    ) -> List[BasecallResult]:
        """对 Fast5 文件进行碱基识别
        
        Args:
            fast5_path: Fast5 文件路径
            max_reads: 最大处理读取数
            output_fastq: 输出 FASTQ 文件路径
            output_fasta: 输出 FASTA 文件路径
            
        Returns:
            BasecallResult 列表
        """
        reader = StreamingFast5Reader(fast5_path)
        results = []

        with reader:
            for read_id, signal in reader.iter_reads(
                convert_to_pa=True, max_reads=max_reads
            ):
                result = self.basecall_single(signal, read_id)
                results.append(result)

        if output_fastq:
            self._write_fastq(results, output_fastq)

        if output_fasta:
            self._write_fasta(results, output_fasta)

        return results

    def basecall_fast5_streaming(
        self,
        fast5_path: str,
        max_reads: Optional[int] = None,
    ) -> Generator[BasecallResult, None, None]:
        """流式处理 Fast5 文件，逐条产出结果
        
        对于超大文件，流式处理可以显著降低内存占用。
        
        Args:
            fast5_path: Fast5 文件路径
            max_reads: 最大处理读取数
            
        Yields:
            BasecallResult 对象
        """
        reader = StreamingFast5Reader(fast5_path)

        with reader:
            for read_id, signal in reader.iter_reads(
                convert_to_pa=True, max_reads=max_reads
            ):
                result = self.basecall_single(signal, read_id)
                yield result

    def _write_fasta(
        self, results: List[BasecallResult], output_path: str
    ) -> None:
        """写入 FASTA 文件"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            for result in results:
                f.write(result.to_fasta())

    def _write_fastq(
        self, results: List[BasecallResult], output_path: str
    ) -> None:
        """写入 FASTQ 文件"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            for result in results:
                f.write(result.to_fastq())

    def summary(self, results: List[BasecallResult]) -> Dict[str, Any]:
        """生成结果统计摘要
        
        Args:
            results: 碱基识别结果列表
            
        Returns:
            统计摘要字典
        """
        total_bases = sum(r.num_bases for r in results)
        total_signal = sum(r.signal_length for r in results)
        total_time = sum(r.process_time for r in results)
        confidences = [r.mean_confidence for r in results if r.num_bases > 0]

        return {
            "num_reads": len(results),
            "total_bases": total_bases,
            "total_signal_samples": total_signal,
            "avg_read_length": total_bases / len(results) if results else 0,
            "avg_confidence": float(np.mean(confidences)) if confidences else 0,
            "bases_per_second": total_bases / total_time if total_time > 0 else 0,
            "total_process_time": total_time,
        }

    @classmethod
    def default(cls, device: Optional[str] = None) -> "BasecallPipeline":
        """创建默认配置的流水线"""
        return cls(
            normalize_mode="global",
            trim_adapter=True,
            denoise_signal=True,
            batch_size=16,
            device=device,
        )
