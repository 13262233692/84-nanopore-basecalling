"""
单元测试 - 信号去噪模块
"""

import numpy as np
import pytest
from nanopore_core.signal.denoiser import SignalDenoiser


class TestSignalDenoiser:
    def test_denoise_basic(self, sample_signal):
        denoiser = SignalDenoiser()
        result, stats = denoiser.denoise(sample_signal)
        assert result.shape == sample_signal.shape
        assert result.dtype == np.float32
        assert "n_spike_samples" in stats
        assert "spike_fraction" in stats

    def test_median_filter_smooths(self):
        np.random.seed(42)
        n = 1000
        clean = np.full(n, 100.0, dtype=np.float32)
        noise = np.random.normal(0, 10, n).astype(np.float32)
        signal = clean + noise

        denoiser = SignalDenoiser(
            median_filter_size=11,
            remove_spikes=False,
        )
        result, _ = denoiser.denoise(signal)
        assert np.std(result) < np.std(signal)

    def test_spike_removal(self):
        np.random.seed(42)
        n = 1000
        signal = np.full(n, 0.0, dtype=np.float32)
        spike_pos = [100, 200, 300, 400, 500]
        for pos in spike_pos:
            signal[pos] = 10.0

        denoiser = SignalDenoiser(
            spike_threshold=5.0,
            max_spike_length=2,
            apply_median_filter=False,
        )
        result, stats = denoiser.denoise(signal, mad_val=1.0)

        assert stats["n_spike_samples"] == len(spike_pos)
        assert stats["spikes_removed"] is True

        for pos in spike_pos:
            assert abs(result[pos]) < 5.0

    def test_no_spikes_found(self):
        np.random.seed(42)
        signal = np.random.normal(0, 0.5, 1000).astype(np.float32)

        denoiser = SignalDenoiser(
            spike_threshold=10.0,
            apply_median_filter=False,
        )
        result, stats = denoiser.denoise(signal, mad_val=1.0)
        assert stats["n_spike_samples"] == 0
        assert stats["spikes_removed"] is False

    def test_long_spikes_preserved(self):
        np.random.seed(42)
        n = 1000
        signal = np.zeros(n, dtype=np.float32)
        signal[100:200] = 10.0

        denoiser = SignalDenoiser(
            spike_threshold=5.0,
            max_spike_length=10,
            apply_median_filter=False,
        )
        result, stats = denoiser.denoise(signal, mad_val=1.0)

        assert stats["n_spike_samples"] == 100
        assert abs(np.mean(result[100:200]) - 10.0) < 0.1

    def test_disable_all(self):
        denoiser = SignalDenoiser(
            apply_median_filter=False,
            remove_spikes=False,
        )
        signal = np.random.randn(100).astype(np.float32)
        result, stats = denoiser.denoise(signal)
        np.testing.assert_array_equal(result, signal)
        assert stats["median_filter_applied"] is False
        assert stats["spikes_removed"] is False


class TestAdapterTrimming:
    def test_trim_adapter_basic(self):
        np.random.seed(42)
        adapter_len = 500
        adapter = np.random.normal(5.0, 0.5, adapter_len).astype(np.float32)
        sample = np.random.normal(0.0, 1.0, 2000).astype(np.float32)
        signal = np.concatenate([adapter, sample])

        denoiser = SignalDenoiser()
        trimmed, trim_pos = denoiser.trim_adapter(
            signal, window_size=100, threshold=0.5, min_trim=10
        )

        assert trim_pos > 0
        assert len(trimmed) == len(signal) - trim_pos
        assert trim_pos < len(signal) // 2

    def test_trim_min_trim(self):
        signal = np.random.normal(0, 0.1, 1000).astype(np.float32)

        denoiser = SignalDenoiser()
        trimmed, trim_pos = denoiser.trim_adapter(
            signal, window_size=50, threshold=10.0, min_trim=50
        )

        assert trim_pos == 50
        assert len(trimmed) == 950

    def test_trim_short_signal(self):
        signal = np.random.randn(100).astype(np.float32)

        denoiser = SignalDenoiser()
        trimmed, trim_pos = denoiser.trim_adapter(signal)

        assert trim_pos == 0
        assert len(trimmed) == 100


class TestOpenPoreDetection:
    def test_detect_open_pore(self):
        np.random.seed(42)
        n = 5000
        signal = np.random.normal(2.0, 0.2, n).astype(np.float32)
        signal[1000:2000] = np.random.normal(0.0, 0.2, 1000).astype(np.float32)
        signal[3000:4000] = np.random.normal(0.0, 0.2, 1000).astype(np.float32)

        denoiser = SignalDenoiser()
        regions = denoiser.detect_open_pore(
            signal, sample_rate=1000, open_pore_current=0.0, tolerance=0.5
        )

        assert len(regions) >= 2

    def test_no_open_pore(self):
        np.random.seed(42)
        signal = np.random.normal(5.0, 0.2, 1000).astype(np.float32)

        denoiser = SignalDenoiser()
        regions = denoiser.detect_open_pore(
            signal, sample_rate=1000, open_pore_current=0.0, tolerance=0.5
        )

        assert len(regions) == 0


class TestRunningStd:
    def test_running_std_shape(self, sample_signal):
        denoiser = SignalDenoiser()
        result = denoiser.running_std(sample_signal, window_size=50)
        assert result.shape == sample_signal.shape

    def test_running_std_constant(self):
        signal = np.ones(1000, dtype=np.float32) * 10.0
        denoiser = SignalDenoiser()
        result = denoiser.running_std(signal, window_size=50)
        assert np.allclose(result, 0.0, atol=1e-5)
