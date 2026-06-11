"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture
def sample_signal():
    """生成模拟的纳米孔电流信号（用于测试）"""
    np.random.seed(42)
    n = 10000
    base_current = 100.0
    noise = np.random.normal(0, 5, n)
    signal = base_current + noise

    spike_positions = np.random.choice(n, 50, replace=False)
    signal[spike_positions] += np.random.uniform(50, 100, 50)

    drift = np.linspace(0, 10, n)
    signal += drift

    return signal.astype(np.float32)


@pytest.fixture
def long_signal():
    """生成长度为 50000 的模拟信号"""
    np.random.seed(123)
    n = 50000
    base = 80.0
    noise = np.random.normal(0, 3, n)
    signal = base + noise

    pattern_len = n // 50
    pattern = np.sin(np.linspace(0, 8 * np.pi, pattern_len)) * 20
    for i in range(0, n - pattern_len, pattern_len * 2):
        signal[i:i + pattern_len] += pattern

    return signal.astype(np.float32)
