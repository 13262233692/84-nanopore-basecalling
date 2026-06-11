"""
CTC (Connectionist Temporal Classification) 贪心解码模块

CTC 是处理序列标注问题的经典算法，解决了输入序列与输出标签
长度不一致且未知对齐方式的问题。在纳米孔碱基识别中：
- 输入：电流信号时间序列（数万个时间步）
- 输出：DNA 碱基序列（数千个碱基）
- 问题：同一碱基对应多少个时间步的电流信号是未知的

贪心解码 (Greedy Search) 是 CTC 解码中最简单高效的方式：
1. 每个时间步取概率最大的标签
2. 折叠连续重复的标签
3. 去除空白符 (Blank)
4. 得到最终的碱基序列

本模块还提供了：
- 批量解码
- 置信度计算
- 序列质量评估
- 基于 numba 的高速实现
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from numba import jit, int32, float32


BASES = ["A", "T", "C", "G"]
BLANK_INDEX = 4
VOCAB_SIZE = 5


@jit(nopython=True, cache=True)
def _greedy_decode_single(
    log_probs: np.ndarray,
    blank: int,
) -> Tuple[np.ndarray, float]:
    """单条序列的 CTC 贪心解码核心算法（numba 加速）
    
    Args:
        log_probs: (T, C) 对数概率矩阵
        blank: 空白符索引
        
    Returns:
        (labels, avg_confidence) 元组
        - labels: 解码后的标签序列（不含空白符、已折叠重复）
        - avg_confidence: 平均置信度
    """
    T = log_probs.shape[0]

    argmax_labels = np.zeros(T, dtype=np.int32)
    max_probs = np.zeros(T, dtype=np.float32)

    for t in range(T):
        max_idx = 0
        max_val = log_probs[t, 0]
        for c in range(1, log_probs.shape[1]):
            if log_probs[t, c] > max_val:
                max_val = log_probs[t, c]
                max_idx = c
        argmax_labels[t] = max_idx
        max_probs[t] = max_val

    decoded = np.zeros(T, dtype=np.int32)
    confidences = np.zeros(T, dtype=np.float32)
    n_decoded = 0
    prev_label = -1

    for t in range(T):
        label = argmax_labels[t]
        if label != blank and label != prev_label:
            decoded[n_decoded] = label
            confidences[n_decoded] = max_probs[t]
            n_decoded += 1
        prev_label = label

    result_labels = decoded[:n_decoded]
    if n_decoded > 0:
        avg_conf = float(np.sum(np.exp(confidences[:n_decoded])) / n_decoded)
    else:
        avg_conf = 0.0

    return result_labels, avg_conf


@jit(nopython=True, cache=True)
def _greedy_decode_batch(
    log_probs: np.ndarray,
    lengths: np.ndarray,
    blank: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """批量 CTC 贪心解码
    
    Args:
        log_probs: (T, B, C) 对数概率矩阵
        lengths: (B,) 各序列的有效长度
        blank: 空白符索引
        
    Returns:
        (decoded_list, confidences) 元组
    """
    batch_size = log_probs.shape[1]
    confidences = np.zeros(batch_size, dtype=np.float32)
    decoded_list = []

    for b in range(batch_size):
        length = int(lengths[b]) if lengths is not None else log_probs.shape[0]
        seq_log_probs = log_probs[:length, b, :]
        labels, conf = _greedy_decode_single(seq_log_probs, blank)
        decoded_list.append(labels)
        confidences[b] = conf

    return decoded_list, confidences


def labels_to_bases(labels: np.ndarray) -> str:
    """将整数标签序列转换为碱基字符串
    
    Args:
        labels: 整数标签数组
        
    Returns:
        DNA 碱基字符串
    """
    bases = []
    for label in labels:
        if label == BLANK_INDEX:
            continue
        if label < len(BASES):
            bases.append(BASES[label])
    return "".join(bases)


def bases_to_labels(sequence: str) -> np.ndarray:
    """将碱基字符串转换为整数标签序列
    
    Args:
        sequence: DNA 碱基字符串
        
    Returns:
        整数标签数组
    """
    base_to_idx = {base: i for i, base in enumerate(BASES)}
    labels = []
    for base in sequence.upper():
        if base in base_to_idx:
            labels.append(base_to_idx[base])
    return np.array(labels, dtype=np.int32)


class CTCGreedyDecoder:
    """CTC 贪心解码器
    
    将神经网络输出的对数概率矩阵解码为 DNA 碱基序列。
    
    算法流程：
    1. 逐时间步取概率最大的类别
    2. 折叠连续重复的类别
    3. 去除空白符 (Blank)
    4. 得到最终的碱基序列
    """

    def __init__(
        self,
        blank_index: int = BLANK_INDEX,
        vocab_size: int = VOCAB_SIZE,
        bases: List[str] = None,
    ):
        """
        Args:
            blank_index: 空白符索引
            vocab_size: 词表大小
            bases: 碱基列表（不含空白符）
        """
        self.blank_index = blank_index
        self.vocab_size = vocab_size
        self.bases = bases if bases is not None else BASES
        self._base_to_idx = {b: i for i, b in enumerate(self.bases)}

    def decode(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> Tuple[str, float]:
        """解码单条序列
        
        Args:
            log_probs: (T, C) 对数概率矩阵
            length: 有效长度，None 表示使用全部
            
        Returns:
            (sequence, confidence) 元组
            - sequence: 解码后的 DNA 碱基序列
            - confidence: 平均置信度 (0~1)
        """
        if length is not None:
            log_probs = log_probs[:length]

        labels, confidence = _greedy_decode_single(log_probs, self.blank_index)
        sequence = labels_to_bases(labels)
        return sequence, float(confidence)

    def decode_batch(
        self,
        log_probs: np.ndarray,
        lengths: Optional[np.ndarray] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """批量解码
        
        Args:
            log_probs: (T, B, C) 对数概率矩阵（时间优先）
            lengths: (B,) 各序列的有效长度
            
        Returns:
            (sequences, confidences) 元组
            - sequences: 解码后的 DNA 序列列表
            - confidences: 各序列的平均置信度
        """
        T, B, C = log_probs.shape

        if lengths is None:
            lengths = np.full(B, T, dtype=np.int32)

        labels_list, confidences = _greedy_decode_batch(
            log_probs, lengths, self.blank_index
        )

        sequences = []
        for labels in labels_list:
            sequences.append(labels_to_bases(labels))

        return sequences, confidences

    def decode_from_torch(
        self,
        log_probs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """从 PyTorch 张量直接解码
        
        Args:
            log_probs: (T, B, C) 对数概率张量
            lengths: (B,) 有效长度张量
            
        Returns:
            (sequences, confidences) 元组
        """
        log_probs_np = log_probs.detach().cpu().numpy()
        if lengths is not None:
            lengths_np = lengths.detach().cpu().numpy().astype(np.int32)
        else:
            lengths_np = None
        return self.decode_batch(log_probs_np, lengths_np)

    def decode_with_timesteps(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """解码并返回碱基对应的时间步信息
        
        Args:
            log_probs: (T, C) 对数概率矩阵
            length: 有效长度
            
        Returns:
            (sequence, start_steps, end_steps) 元组
            - sequence: 碱基序列
            - start_steps: 各碱基对应的起始时间步
            - end_steps: 各碱基对应的结束时间步
        """
        if length is not None:
            log_probs = log_probs[:length]

        T = log_probs.shape[0]
        argmax_labels = np.argmax(log_probs, axis=1)
        max_probs = np.max(log_probs, axis=1)

        bases = []
        starts = []
        ends = []
        current_base = -1
        current_start = -1

        for t in range(T):
            label = argmax_labels[t]
            if label == self.blank_index:
                if current_base != -1:
                    bases.append(self.bases[current_base])
                    starts.append(current_start)
                    ends.append(t - 1)
                    current_base = -1
                continue

            if label != current_base:
                if current_base != -1:
                    bases.append(self.bases[current_base])
                    starts.append(current_start)
                    ends.append(t - 1)
                current_base = label
                current_start = t

        if current_base != -1:
            bases.append(self.bases[current_base])
            starts.append(current_start)
            ends.append(T - 1)

        return "".join(bases), np.array(starts), np.array(ends)

    def compute_sequence_confidence(
        self,
        log_probs: np.ndarray,
        sequence: str,
    ) -> np.ndarray:
        """计算已解码序列中每个碱基的置信度
        
        这比平均置信度更精细，用于标识低质量碱基。
        
        Args:
            log_probs: (T, C) 对数概率矩阵
            sequence: 已解码的碱基序列
            
        Returns:
            各碱基的置信度数组
        """
        _, starts, ends = self.decode_with_timesteps(log_probs)

        if len(starts) != len(sequence):
            raise ValueError(
                f"Sequence length ({len(sequence)}) doesn't match "
                f"detected segments ({len(starts)})"
            )

        confidences = np.zeros(len(sequence), dtype=np.float32)
        for i, (s, e) in enumerate(zip(starts, ends)):
            base = sequence[i]
            base_idx = self._base_to_idx.get(base, -1)
            if base_idx >= 0:
                confidences[i] = np.mean(np.exp(log_probs[s : e + 1, base_idx]))
            else:
                confidences[i] = 0.0

        return confidences

    def quality_scores(
        self, confidences: np.ndarray, scale: float = 10.0
    ) -> np.ndarray:
        """将置信度转换为 Phred 质量分
        
        Q = -10 * log10(1 - P)
        
        Args:
            confidences: 置信度数组 (0~1)
            scale: 缩放因子
            
        Returns:
            Phred 质量分数组
        """
        eps = 1e-10
        confidences = np.clip(confidences, eps, 1.0 - eps)
        error_probs = 1.0 - confidences
        error_probs = np.clip(error_probs, eps, 1.0)
        q_scores = -scale * np.log10(error_probs)
        return q_scores

    def decode_with_quality(
        self,
        log_probs: np.ndarray,
        length: Optional[int] = None,
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """解码并返回每个碱基的质量分
        
        Args:
            log_probs: (T, C) 对数概率矩阵
            length: 有效长度
            
        Returns:
            (sequence, confidences, q_scores) 元组
        """
        sequence, starts, ends = self.decode_with_timesteps(log_probs, length)
        confidences = self.compute_sequence_confidence(log_probs, sequence)
        q_scores = self.quality_scores(confidences)
        return sequence, confidences, q_scores
