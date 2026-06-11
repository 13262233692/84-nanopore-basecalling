"""
单元测试 - Fast5/HDF5 读取器
使用模拟的 HDF5 文件进行测试
"""

import os
import tempfile
import numpy as np
import h5py
import pytest
from nanopore_core.io.fast5_reader import Fast5Reader, StreamingFast5Reader


def create_mock_fast5(filepath: str, n_reads: int = 5, signal_len: int = 1000):
    """创建模拟的 Fast5 文件用于测试"""
    with h5py.File(filepath, "w") as f:
        for i in range(n_reads):
            read_id = f"read_test_{i:08d}-{i:04d}-{i:04d}-{i:04d}-{i:012d}"
            read_group = f.create_group(f"read_{read_id}")

            raw_group = read_group.create_group("Raw")
            signal = np.random.normal(100, 10, signal_len).astype(np.int16)
            raw_group.create_dataset("Signal", data=signal)

            channel_group = read_group.create_group("channel_id")
            channel_group.attrs["digitisation"] = 8192.0
            channel_group.attrs["offset"] = 0.0
            channel_group.attrs["range"] = 1200.0
            channel_group.attrs["sampling_rate"] = 4000.0
            channel_group.attrs["channel_number"] = str(i + 1)

            tracking_group = read_group.create_group("tracking_id")
            tracking_group.attrs["device_id"] = "test_device"
            tracking_group.attrs["run_id"] = "test_run"
            tracking_group.attrs["sample_id"] = "test_sample"

    return filepath


@pytest.fixture
def mock_fast5_file():
    """创建临时模拟 Fast5 文件"""
    with tempfile.NamedTemporaryFile(suffix=".fast5", delete=False) as tmp:
        filepath = tmp.name
    create_mock_fast5(filepath, n_reads=5, signal_len=1000)
    yield filepath
    os.unlink(filepath)


class TestFast5Reader:
    def test_open_close(self, mock_fast5_file):
        reader = Fast5Reader(mock_fast5_file)
        assert not reader.is_open
        reader.open()
        assert reader.is_open
        reader.close()
        assert not reader.is_open

    def test_context_manager(self, mock_fast5_file):
        with Fast5Reader(mock_fast5_file) as reader:
            assert reader.is_open
            read_ids = reader.get_read_ids()
            assert len(read_ids) == 5
        assert not reader.is_open

    def test_get_read_ids(self, mock_fast5_file):
        with Fast5Reader(mock_fast5_file) as reader:
            read_ids = reader.get_read_ids()
            assert len(read_ids) == 5
            for rid in read_ids:
                assert rid.startswith("read_read_test_")

    def test_get_read_count(self, mock_fast5_file):
        with Fast5Reader(mock_fast5_file) as reader:
            assert reader.get_read_count() == 5

    def test_get_raw_signal(self, mock_fast5_file):
        with Fast5Reader(mock_fast5_file) as reader:
            read_ids = reader.get_read_ids()
            signal = reader.get_raw_signal(read_ids[0])
            assert len(signal) == 1000
            assert signal.dtype == np.float32

    def test_get_signal_length(self, mock_fast5_file):
        with Fast5Reader(mock_fast5_file) as reader:
            read_ids = reader.get_read_ids()
            length = reader.get_signal_length(read_ids[0])
            assert length == 1000

    def test_get_channel_info(self, mock_fast5_file):
        with Fast5Reader(mock_fast5_file) as reader:
            read_ids = reader.get_read_ids()
            info = reader.get_channel_info(read_ids[0])
            assert "digitisation" in info
            assert "range" in info
            assert "offset" in info
            assert info["digitisation"] == 8192.0

    def test_get_digitisation_params(self, mock_fast5_file):
        with Fast5Reader(mock_fast5_file) as reader:
            read_ids = reader.get_read_ids()
            dig, offset, rng = reader.get_digitisation_params(read_ids[0])
            assert dig == 8192.0
            assert offset == 0.0
            assert rng == 1200.0

    def test_get_pa_signal(self, mock_fast5_file):
        with Fast5Reader(mock_fast5_file) as reader:
            read_ids = reader.get_read_ids()
            pa_signal = reader.get_pa_signal(read_ids[0])
            assert len(pa_signal) == 1000
            assert pa_signal.dtype == np.float32

    def test_get_read_metadata(self, mock_fast5_file):
        with Fast5Reader(mock_fast5_file) as reader:
            read_ids = reader.get_read_ids()
            meta = reader.get_read_metadata(read_ids[0])
            assert "read_id" in meta
            assert "channel_info" in meta
            assert "signal_length" in meta
            assert "tracking" in meta
            assert meta["signal_length"] == 1000

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            reader = Fast5Reader("/nonexistent/file.fast5")
            reader.open()

    def test_not_open_error(self):
        reader = Fast5Reader("/fake/path.fast5")
        with pytest.raises(RuntimeError):
            reader.get_read_ids()


class TestStreamingFast5Reader:
    def test_iter_reads(self, mock_fast5_file):
        reader = StreamingFast5Reader(mock_fast5_file)
        count = 0
        with reader:
            for read_id, signal in reader.iter_reads(convert_to_pa=True):
                assert len(signal) == 1000
                assert signal.dtype == np.float32
                count += 1
        assert count == 5

    def test_iter_reads_max_reads(self, mock_fast5_file):
        reader = StreamingFast5Reader(mock_fast5_file)
        count = 0
        with reader:
            for _ in reader.iter_reads(max_reads=3):
                count += 1
        assert count == 3

    def test_iter_signal_chunks(self, mock_fast5_file):
        reader = StreamingFast5Reader(mock_fast5_file)
        with reader:
            read_ids = reader.get_read_ids()
            chunks = list(reader.iter_signal_chunks(
                read_ids[0], chunk_size=300, convert_to_pa=True
            ))
            assert len(chunks) == 4
            total_len = sum(len(c) for _, c in chunks)
            assert total_len == 1000

    def test_batch_iter_reads(self, mock_fast5_file):
        reader = StreamingFast5Reader(mock_fast5_file)
        with reader:
            batches = list(reader.batch_iter_reads(
                batch_size=2, convert_to_pa=True, pad=True
            ))
            assert len(batches) == 3
            for read_ids, signals, lengths in batches:
                assert len(read_ids) <= 2
                assert signals.shape[0] == len(read_ids)
                assert len(lengths) == len(read_ids)

    def test_batch_iter_max_length(self, mock_fast5_file):
        reader = StreamingFast5Reader(mock_fast5_file)
        with reader:
            batches = list(reader.batch_iter_reads(
                batch_size=3, convert_to_pa=True, pad=True, max_length=500
            ))
            for read_ids, signals, lengths in batches:
                assert signals.shape[1] <= 500
