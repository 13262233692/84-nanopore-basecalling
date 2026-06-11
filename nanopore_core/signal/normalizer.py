"""
MAD (Median Absolute Deviation) 中值绝对偏差归一化模块

对于纳米孔测序的微电流信号，由于孔蛋白卡顿、接头序列等原因会产生
大量尖峰伪影，常规的均值-标准差归一化会被这些异常值严重干扰。
MAD 归一化基于中值，具有极强的鲁棒性，是 Nanopore 碱基识别的
标准预处理步骤。

MAD = median(|x - median(x)|)
归一化后: x_norm = (x - median) / (MAD * 1.4826)
其中 1.4826 是使 MAD 在正态分布下等于标准差的校正因子。
"""

import numpy as np
from typing import Tuple, Optional
from numba import jit, float32, int64


@jit(nopython=True, cache=True)
def _median_odd(arr: np.ndarray) -> float:
    """快速计算奇数长度数组的中位数（numba 加速）"""
    n = len(arr)
    mid = n // 2
    sorted_arr = np.sort(arr)
    return float(sorted_arr[mid])


@jit(nopython=True, cache=True)
def _median_even(arr: np.ndarray) -> float:
    """快速计算偶数长度数组的中位数（numba 加速）"""
    n = len(arr)
    mid = n // 2
    sorted_arr = np.sort(arr)
    return float((sorted_arr[mid - 1] + sorted_arr[mid]) / 2.0)


@jit(nopython=True, cache=True)
def fast_median(arr: np.ndarray) -> float:
    """快速中位数计算"""
    n = len(arr)
    if n % 2 == 1:
        return _median_odd(arr)
    return _median_even(arr)


@jit(nopython=True, cache=True)
def mad_normalize(signal: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    MAD 归一化核心算法（numba JIT 编译加速）
    
    Args:
        signal: 原始电流信号数组
        
    Returns:
        (normalized_signal, median, mad) 元组
    """
    median_val = fast_median(signal)
    abs_dev = np.abs(signal - median_val)
    mad_val = fast_median(abs_dev)
    
    if mad_val == 0:
        mad_val = 1e-8
    
    scale_factor = 1.4826 * mad_val
    normalized = (signal - median_val) / scale_factor
    
    return normalized.astype(np.float32), float(median_val), float(mad_val)


class MADNormalizer:
    """MAD 中值绝对偏差归一化器
    
    提供全局归一化、滑动窗口归一化、分块归一化三种模式，
    适配不同长度和特性的纳米孔电流信号。
    """

    def __init__(
        self,
        mad_scale: float = 1.4826,
        clip_outliers: bool = True,
        outlier_threshold: float = 5.0,
    ):
        """
        Args:
            mad_scale: MAD 缩放因子，1.4826 使 MAD 等价于正态分布的标准差
            clip_outliers: 是否截断极端异常值
            outlier_threshold: 异常值截断阈值（单位：MAD）
        """
        self.mad_scale = mad_scale
        self.clip_outliers = clip_outliers
        self.outlier_threshold = outlier_threshold

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """全局 MAD 归一化
        
        对整段信号计算单一的 median 和 MAD，适用于相对平稳的信号段。
        
        Args:
            signal: 原始电流信号 (N,)
            
        Returns:
            归一化后的信号
        """
        normalized, _, _ = mad_normalize(signal)
        
        if self.clip_outliers:
            normalized = np.clip(
                normalized,
                -self.outlier_threshold,
                self.outlier_threshold,
            )
        
        return normalized

    def normalize_windowed(
        self,
        signal: np.ndarray,
        window_size: int = 5000,
        step_size: int = 1000,
    ) -> np.ndarray:
        """滑动窗口 MAD 归一化
        
        对于长信号，使用滑动窗口计算局部 median 和 MAD，
        可以更好地处理信号的缓慢漂移（如孔蛋白老化导致的电流漂移）。
        
        Args:
            signal: 原始电流信号 (N,)
            window_size: 窗口大小（样本数）
            step_size: 滑动步长
            
        Returns:
            归一化后的信号
        """
        n = len(signal)
        if n <= window_size:
            return self.normalize(signal)

        normalized_full = np.zeros_like(signal, dtype=np.float32)
        weight_sum = np.zeros_like(signal, dtype=np.float32)

        starts = range(0, n - window_size + 1, step_size)
        if (n - window_size) % step_size != 0:
            starts = list(starts) + [n - window_size]

        for start in starts:
            end = start + window_size
            window = signal[start:end]
            norm_window, med, mad_val = mad_normalize(window)

            if self.clip_outliers:
                norm_window = np.clip(
                    norm_window,
                    -self.outlier_threshold,
                    self.outlier_threshold,
                )

            normalized_full[start:end] += norm_window
            weight_sum[start:end] += 1.0

        weight_sum = np.maximum(weight_sum, 1.0)
        normalized_full /= weight_sum

        return normalized_full.astype(np.float32)

    def normalize_chunks(
        self,
        signal: np.ndarray,
        chunk_size: int = 10000,
    ) -> np.ndarray:
        """分块 MAD 归一化
        
        将信号分成不重叠的块，每块独立归一化。
        适用于超长信号的快速预处理。
        
        Args:
            signal: 原始电流信号 (N,)
            chunk_size: 每块大小
            
        Returns:
            归一化后的信号
        """
        n = len(signal)
        if n <= chunk_size:
            return self.normalize(signal)

        normalized = np.zeros_like(signal, dtype=np.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = signal[start:end]
            normalized[start:end] = self.normalize(chunk)

        return normalized

    def denormalize(
        self,
        normalized_signal: np.ndarray,
        median_val: float,
        mad_val: float,
    ) -> np.ndarray:
        """逆归一化，将归一化信号还原为原始电流值
        
        Args:
            normalized_signal: 归一化信号
            median_val: 原始信号的中值
            mad_val: 原始信号的 MAD
            
        Returns:
            原始尺度的电流信号
        """
        scale_factor = self.mad_scale * mad_val
        return normalized_signal * scale_factor + median_val

    def get_stats(self, signal: np.ndarray) -> dict:
        """计算信号的统计特征
        
        Args:
            signal: 电流信号
            
        Returns:
            包含 median, mad, mean, std, min, max 的字典
        """
        _, median_val, mad_val = mad_normalize(signal)
        return {
            "median": median_val,
            "mad": mad_val,
            "std_equivalent": mad_val * self.mad_scale,
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
            "length": len(signal),
        }
