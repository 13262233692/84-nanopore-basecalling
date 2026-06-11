"""
Fast5 (HDF5) 测序数据读取器
支持流式读取数十 GB 级别的纳米孔测序原始电流数据
"""

import os
from typing import Generator, List, Optional, Tuple, Dict, Any
import numpy as np
import h5py


class Fast5Reader:
    """Fast5 文件读取器，用于解析纳米孔测序的 HDF5 格式数据
    
    Fast5 文件结构（多读取版本）：
    /read_<read_id>/Raw/Signal - 原始电流信号（int16 或 float32）
    /read_<read_id>/channel_id - 通道信息
    /read_<read_id>/tracking_id - 追踪信息
    """

    BASECALL_GROUP_PATH = "Analyses/Basecall_1D_000"
    RAW_SIGNAL_PATH = "Raw/Signal"
    CHANNEL_ID_PATH = "channel_id"
    TRACKING_ID_PATH = "tracking_id"

    def __init__(self, filepath: str):
        """
        Args:
            filepath: Fast5 文件路径
        """
        self.filepath = filepath
        self._file = None
        self._read_ids = None

    def open(self) -> None:
        """打开 HDF5 文件"""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Fast5 file not found: {self.filepath}")
        self._file = h5py.File(self.filepath, "r")

    def close(self) -> None:
        """关闭 HDF5 文件"""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def _ensure_open(self) -> None:
        if not self.is_open:
            raise RuntimeError("Fast5 file is not open. Call open() first.")

    def get_read_ids(self) -> List[str]:
        """获取所有读取 ID
        
        Returns:
            读取 ID 列表
        """
        self._ensure_open()
        if self._read_ids is None:
            self._read_ids = [
                key for key in self._file.keys() if key.startswith("read_")
            ]
        return self._read_ids.copy()

    def get_read_count(self) -> int:
        """获取读取数量"""
        return len(self.get_read_ids())

    def get_raw_signal(self, read_id: str) -> np.ndarray:
        """获取指定读取的原始电流信号
        
        Args:
            read_id: 读取 ID
            
        Returns:
            原始电流信号数组（单位：pA 皮安级微电流）
        """
        self._ensure_open()
        signal_path = f"/{read_id}/{self.RAW_SIGNAL_PATH}"
        if signal_path not in self._file:
            raise KeyError(f"Signal not found for read {read_id}")
        return np.array(self._file[signal_path], dtype=np.float32)

    def get_signal_length(self, read_id: str) -> int:
        """获取信号长度（不加载全部数据）"""
        self._ensure_open()
        signal_path = f"/{read_id}/{self.RAW_SIGNAL_PATH}"
        if signal_path not in self._file:
            raise KeyError(f"Signal not found for read {read_id}")
        return self._file[signal_path].shape[0]

    def get_channel_info(self, read_id: str) -> Dict[str, Any]:
        """获取通道信息"""
        self._ensure_open()
        channel_path = f"/{read_id}/{self.CHANNEL_ID_PATH}"
        info = {}
        if channel_path in self._file:
            for key, val in self._file[channel_path].attrs.items():
                info[key] = val
        return info

    def get_digitisation_params(
        self, read_id: str
    ) -> Tuple[float, float, float]:
        """获取数字化参数，用于将原始 int16 转换为 pA
        
        Returns:
            (digitisation, offset, range) 元组
        """
        info = self.get_channel_info(read_id)
        digitisation = float(info.get("digitisation", 8192.0))
        offset = float(info.get("offset", 0.0))
        signal_range = float(info.get("range", 1200.0))
        return digitisation, offset, signal_range

    def get_pa_signal(self, read_id: str) -> np.ndarray:
        """获取转换为皮安（pA）的电流信号
        
        原始信号通常是 int16 数字化值，需要转换为实际电流值：
        current_pA = (raw_signal + offset) * range / digitisation
        
        Args:
            read_id: 读取 ID
            
        Returns:
            pA 单位的电流信号数组
        """
        raw_signal = self.get_raw_signal(read_id)
        digitisation, offset, signal_range = self.get_digitisation_params(read_id)
        pa_signal = (raw_signal + offset) * (signal_range / digitisation)
        return pa_signal.astype(np.float32)

    def get_read_metadata(self, read_id: str) -> Dict[str, Any]:
        """获取读取的完整元数据"""
        self._ensure_open()
        meta = {"read_id": read_id}
        meta["channel_info"] = self.get_channel_info(read_id)
        meta["signal_length"] = self.get_signal_length(read_id)
        
        tracking_path = f"/{read_id}/{self.TRACKING_ID_PATH}"
        if tracking_path in self._file:
            meta["tracking"] = {}
            for key, val in self._file[tracking_path].attrs.items():
                meta["tracking"][key] = val
        
        return meta


class StreamingFast5Reader(Fast5Reader):
    """流式 Fast5 读取器，支持大文件分块读取
    
    对于数十 GB 的 Fast5 文件，一次性加载全部数据会导致内存不足。
    流式读取器按读取逐条处理，或按信号块分块读取。
    """

    def iter_reads(
        self, convert_to_pa: bool = True, max_reads: Optional[int] = None
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        """迭代读取所有 read 的信号
        
        Args:
            convert_to_pa: 是否转换为 pA 单位
            max_reads: 最大读取数量，None 表示全部
            
        Yields:
            (read_id, signal_array) 元组
        """
        self._ensure_open()
        read_ids = self.get_read_ids()
        if max_reads is not None:
            read_ids = read_ids[:max_reads]

        for read_id in read_ids:
            if convert_to_pa:
                signal = self.get_pa_signal(read_id)
            else:
                signal = self.get_raw_signal(read_id)
            yield read_id, signal

    def iter_signal_chunks(
        self,
        read_id: str,
        chunk_size: int = 100000,
        convert_to_pa: bool = True,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """对单个 read 的信号进行分块迭代读取
        
        Args:
            read_id: 读取 ID
            chunk_size: 每块的样本数
            convert_to_pa: 是否转换为 pA
            
        Yields:
            (start_index, signal_chunk) 元组
        """
        self._ensure_open()
        signal_path = f"/{read_id}/{self.RAW_SIGNAL_PATH}"
        if signal_path not in self._file:
            raise KeyError(f"Signal not found for read {read_id}")

        dset = self._file[signal_path]
        total_length = dset.shape[0]
        digitisation, offset, signal_range = self.get_digitisation_params(read_id)

        for start in range(0, total_length, chunk_size):
            end = min(start + chunk_size, total_length)
            chunk = np.array(dset[start:end], dtype=np.float32)

            if convert_to_pa:
                chunk = (chunk + offset) * (signal_range / digitisation)

            yield start, chunk.astype(np.float32)

    def batch_iter_reads(
        self,
        batch_size: int = 32,
        convert_to_pa: bool = True,
        pad: bool = True,
        max_length: Optional[int] = None,
    ) -> Generator[Tuple[List[str], np.ndarray, np.ndarray], None, None]:
        """批量迭代读取
        
        Args:
            batch_size: 批次大小
            convert_to_pa: 是否转换为 pA
            pad: 是否对不等长信号进行填充
            max_length: 最大信号长度，超出则截断
            
        Yields:
            (read_ids, signals, lengths) 元组
            - read_ids: 本批次的 read_id 列表
            - signals: (batch, max_len) 形状的信号张量
            - lengths: 各信号的实际长度
        """
        self._ensure_open()
        read_ids = self.get_read_ids()
        n_reads = len(read_ids)

        for batch_start in range(0, n_reads, batch_size):
            batch_end = min(batch_start + batch_size, n_reads)
            batch_ids = read_ids[batch_start:batch_end]
            batch_signals = []
            lengths = []

            for rid in batch_ids:
                if convert_to_pa:
                    sig = self.get_pa_signal(rid)
                else:
                    sig = self.get_raw_signal(rid)

                if max_length is not None and len(sig) > max_length:
                    sig = sig[:max_length]

                batch_signals.append(sig)
                lengths.append(len(sig))

            if pad and batch_signals:
                max_len = max(lengths)
                padded = np.zeros((len(batch_signals), max_len), dtype=np.float32)
                for i, sig in enumerate(batch_signals):
                    padded[i, : len(sig)] = sig
                batch_signals = padded
            else:
                batch_signals = np.array(
                    [s for s in batch_signals], dtype=object
                )

            yield batch_ids, batch_signals, np.array(lengths, dtype=np.int32)
