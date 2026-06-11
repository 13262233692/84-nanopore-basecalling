"""
信号去噪模块 - 去除纳米孔测序中的尖峰伪影与接头序列

纳米孔测序信号中常见的噪声源：
1. 孔蛋白卡顿 (Pore blocking) - 导致电流突然下降或升高的尖峰
2. 接头序列 (Adapter) - 测序开始和结束时的人工序列信号
3. 热噪声 - 高频随机噪声
4. 基线漂移 - 缓慢的电流变化

本模块提供多种去噪算法，包括中值滤波、尖峰检测与修复、
以及接头序列检测与切除。
"""

import numpy as np
from typing import Tuple, Optional, List
from numba import jit


@jit(nopython=True, cache=True)
def _median_filter_1d(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    """1D 中值滤波（numba 加速）
    
    Args:
        signal: 输入信号
        kernel_size: 核大小（奇数）
        
    Returns:
        滤波后的信号
    """
    n = len(signal)
    pad = kernel_size // 2
    result = np.zeros(n, dtype=np.float32)
    window = np.zeros(kernel_size, dtype=np.float32)

    for i in range(n):
        start = max(0, i - pad)
        end = min(n, i + pad + 1)
        k = end - start
        window[:k] = signal[start:end]
        window_k = window[:k]
        result[i] = np.sort(window_k)[k // 2]

    return result


@jit(nopython=True, cache=True)
def _detect_spikes(
    signal: np.ndarray,
    threshold: float,
    mad_val: float,
) -> np.ndarray:
    """尖峰检测
    
    基于 MAD 的尖峰检测，标记偏离中值超过阈值的样本点。
    
    Args:
        signal: 归一化后的信号
        threshold: 尖峰阈值（单位：MAD）
        mad_val: MAD 值
        
    Returns:
        布尔数组，True 表示尖峰位置
    """
    abs_signal = np.abs(signal)
    return abs_signal > threshold


@jit(nopython=True, cache=True)
def _interpolate_spikes(
    signal: np.ndarray,
    spike_mask: np.ndarray,
    max_spike_length: int,
) -> np.ndarray:
    """用线性插值修复尖峰
    
    对于较短的尖峰（< max_spike_length），用两侧正常信号的线性插值替代。
    对于过长的尖峰（可能是真实的生物学信号），保留不变。
    
    Args:
        signal: 原始信号
        spike_mask: 尖峰掩码
        max_spike_length: 最大可修复尖峰长度
        
    Returns:
        修复后的信号
    """
    n = len(signal)
    result = signal.copy()

    i = 0
    while i < n:
        if not spike_mask[i]:
            i += 1
            continue

        start = i
        while i < n and spike_mask[i]:
            i += 1
        end = i
        spike_len = end - start

        if spike_len <= max_spike_length:
            left_val = signal[start - 1] if start > 0 else signal[end]
            right_val = signal[end] if end < n else signal[start - 1]

            for j in range(spike_len):
                t = (j + 1) / (spike_len + 1)
                result[start + j] = left_val * (1 - t) + right_val * t

    return result


class SignalDenoiser:
    """纳米孔信号去噪器
    
    整合中值滤波、尖峰检测与修复、接头切除等多种去噪策略。
    """

    def __init__(
        self,
        median_filter_size: int = 5,
        spike_threshold: float = 5.0,
        max_spike_length: int = 50,
        apply_median_filter: bool = True,
        remove_spikes: bool = True,
    ):
        """
        Args:
            median_filter_size: 中值滤波核大小
            spike_threshold: 尖峰检测阈值（单位：MAD）
            max_spike_length: 最大可修复尖峰长度（样本数）
            apply_median_filter: 是否应用中值滤波
            remove_spikes: 是否检测并修复尖峰
        """
        self.median_filter_size = median_filter_size
        self.spike_threshold = spike_threshold
        self.max_spike_length = max_spike_length
        self.apply_median_filter = apply_median_filter
        self.remove_spikes = remove_spikes

    def denoise(
        self,
        signal: np.ndarray,
        mad_val: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """对信号进行去噪处理
        
        Args:
            signal: 归一化后的电流信号
            mad_val: MAD 值，用于尖峰阈值计算。None 则自动计算
            
        Returns:
            (denoised_signal, stats) 元组
        """
        stats = {}

        if len(signal) == 0:
            stats["median_filter_applied"] = False
            stats["spikes_removed"] = False
            stats["n_spike_samples"] = 0
            stats["spike_fraction"] = 0.0
            return signal.copy(), stats

        result = signal.copy()

        if self.apply_median_filter and self.median_filter_size > 1:
            result = _median_filter_1d(result, self.median_filter_size)
            stats["median_filter_applied"] = True
        else:
            stats["median_filter_applied"] = False

        if self.remove_spikes:
            if mad_val is None:
                from .normalizer import mad_normalize
                _, _, mad_val = mad_normalize(signal)

            spike_mask = _detect_spikes(result, self.spike_threshold, mad_val)
            n_spikes = int(np.sum(spike_mask))
            stats["n_spike_samples"] = n_spikes
            stats["spike_fraction"] = n_spikes / len(signal)

            if n_spikes > 0:
                result = _interpolate_spikes(
                    result, spike_mask, self.max_spike_length
                )
                stats["spikes_removed"] = True
            else:
                stats["spikes_removed"] = False
        else:
            stats["spikes_removed"] = False

        return result.astype(np.float32), stats

    def trim_adapter(
        self,
        signal: np.ndarray,
        window_size: int = 200,
        threshold: float = 2.0,
        min_trim: int = 100,
    ) -> Tuple[np.ndarray, int]:
        """切除接头序列区域
        
        纳米孔测序开始时，DNA 尚未进入孔道，信号处于开放孔道电平。
        本方法通过检测信号突变点来定位接头-样本转换位置。
        
        Args:
            signal: 归一化信号
            window_size: 滑动窗口大小
            threshold: 突变检测阈值
            min_trim: 最小切除长度
            
        Returns:
            (trimmed_signal, trim_start) 元组
        """
        n = len(signal)
        if n < window_size * 2:
            return signal, 0

        means = np.zeros(n - window_size + 1, dtype=np.float32)
        cumsum = np.cumsum(signal)
        means[0] = cumsum[window_size - 1] / window_size
        for i in range(1, len(means)):
            means[i] = means[i - 1] + (
                signal[i + window_size - 1] - signal[i - 1]
            ) / window_size

        diffs = np.abs(np.diff(means))

        if len(diffs) == 0:
            return signal, 0

        max_idx = int(np.argmax(diffs))

        if diffs[max_idx] < threshold:
            trim_pos = min_trim
        else:
            trim_pos = max_idx + window_size // 2
            trim_pos = max(trim_pos, min_trim)

        if trim_pos >= n:
            trim_pos = min_trim

        return signal[trim_pos:].astype(np.float32), trim_pos

    def detect_open_pore(
        self,
        signal: np.ndarray,
        sample_rate: int = 4000,
        open_pore_current: float = 0.0,
        tolerance: float = 0.5,
    ) -> List[Tuple[int, int]]:
        """检测开放孔道段（无 DNA 时的信号）
        
        Args:
            signal: 归一化信号
            sample_rate: 采样率 (Hz)
            open_pore_current: 开放孔道电流（归一化后）
            tolerance: 容差
            
        Returns:
            开放孔道段列表 [(start, end), ...]
        """
        n = len(signal)
        min_pore_len = int(sample_rate * 0.01)

        mask = np.abs(signal - open_pore_current) < tolerance
        pore_regions = []

        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue

            start = i
            while i < n and mask[i]:
                i += 1

            if i - start >= min_pore_len:
                pore_regions.append((start, i))

        return pore_regions

    def running_std(
        self, signal: np.ndarray, window_size: int = 100
    ) -> np.ndarray:
        """计算滑动窗口标准差，用于信号质量评估
        
        Args:
            signal: 输入信号
            window_size: 窗口大小
            
        Returns:
            滑动标准差序列
        """
        n = len(signal)
        result = np.zeros(n, dtype=np.float32)
        pad = window_size // 2

        for i in range(n):
            start = max(0, i - pad)
            end = min(n, i + pad + 1)
            result[i] = np.std(signal[start:end])

        return result
