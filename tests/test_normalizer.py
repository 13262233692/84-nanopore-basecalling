"""
单元测试 - MAD 归一化模块
"""

import numpy as np
import pytest
from nanopore_core.signal.normalizer import MADNormalizer, mad_normalize, fast_median


class TestFastMedian:
    def test_median_odd(self):
        arr = np.array([3, 1, 4, 1, 5], dtype=np.float32)
        assert abs(fast_median(arr) - 3.0) < 1e-6

    def test_median_even(self):
        arr = np.array([1, 2, 3, 4], dtype=np.float32)
        assert abs(fast_median(arr) - 2.5) < 1e-6

    def test_median_single_element(self):
        arr = np.array([42.0], dtype=np.float32)
        assert abs(fast_median(arr) - 42.0) < 1e-6

    def test_median_negative(self):
        arr = np.array([-5, -2, -8, -1, -3], dtype=np.float32)
        assert abs(fast_median(arr) - (-3.0)) < 1e-6


class TestMADNormalize:
    def test_normalize_output_shape(self, sample_signal):
        normalized, median_val, mad_val = mad_normalize(sample_signal)
        assert normalized.shape == sample_signal.shape
        assert normalized.dtype == np.float32
        assert isinstance(median_val, float)
        assert isinstance(mad_val, float)

    def test_normalize_zero_median(self, sample_signal):
        normalized, _, _ = mad_normalize(sample_signal)
        med = np.median(normalized)
        assert abs(med) < 0.5

    def test_normalize_robust_to_spikes(self):
        np.random.seed(42)
        clean = np.random.normal(100, 5, 1000).astype(np.float32)
        with_spikes = clean.copy()
        with_spikes[::10] += 100.0

        _, med_clean, mad_clean = mad_normalize(clean)
        _, med_spike, mad_spike = mad_normalize(with_spikes)

        assert abs(med_clean - med_spike) < 2.0
        assert mad_spike < mad_clean * 3.0


class TestMADNormalizer:
    def test_global_normalize(self, sample_signal):
        normalizer = MADNormalizer()
        result = normalizer.normalize(sample_signal)
        assert result.shape == sample_signal.shape
        assert result.dtype == np.float32

    def test_clip_outliers(self, sample_signal):
        normalizer = MADNormalizer(clip_outliers=True, outlier_threshold=3.0)
        result = normalizer.normalize(sample_signal)
        assert np.max(result) <= 3.0 + 1e-6
        assert np.min(result) >= -3.0 - 1e-6

    def test_no_clip(self, sample_signal):
        normalizer = MADNormalizer(clip_outliers=False)
        result = normalizer.normalize(sample_signal)
        assert np.max(np.abs(result)) > 3.0

    def test_windowed_normalize(self, long_signal):
        normalizer = MADNormalizer()
        result = normalizer.normalize_windowed(long_signal, window_size=1000, step_size=200)
        assert result.shape == long_signal.shape
        assert result.dtype == np.float32

    def test_windowed_short_signal(self, sample_signal):
        normalizer = MADNormalizer()
        result = normalizer.normalize_windowed(sample_signal, window_size=20000, step_size=5000)
        assert result.shape == sample_signal.shape

    def test_chunks_normalize(self, long_signal):
        normalizer = MADNormalizer()
        result = normalizer.normalize_chunks(long_signal, chunk_size=10000)
        assert result.shape == long_signal.shape
        assert result.dtype == np.float32

    def test_denormalize(self, sample_signal):
        normalizer = MADNormalizer(clip_outliers=False)
        _, med, mad_val = mad_normalize(sample_signal)
        normalized = normalizer.normalize(sample_signal)
        denormalized = normalizer.denormalize(normalized, med, mad_val)
        np.testing.assert_allclose(denormalized, sample_signal, rtol=1e-5)

    def test_get_stats(self, sample_signal):
        normalizer = MADNormalizer()
        stats = normalizer.get_stats(sample_signal)
        assert "median" in stats
        assert "mad" in stats
        assert "std_equivalent" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "length" in stats
        assert stats["length"] == len(sample_signal)
