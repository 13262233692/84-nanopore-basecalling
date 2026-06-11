"""
单元测试 - 端到端碱基识别流水线
"""

import os
import tempfile
import numpy as np
import pytest
from nanopore_core.pipeline.basecall_pipeline import BasecallPipeline, BasecallResult


class TestBasecallResult:
    def test_to_fasta(self):
        result = BasecallResult(
            read_id="test_read",
            sequence="ATCG",
            mean_confidence=0.95,
        )
        fasta = result.to_fasta()
        assert fasta.startswith(">test_read")
        assert "ATCG" in fasta
        assert "mean_q=0.950" in fasta

    def test_to_fastq(self):
        result = BasecallResult(
            read_id="test_read",
            sequence="ATCG",
            quality_scores=np.array([30, 35, 28, 32], dtype=np.float32),
        )
        fastq = result.to_fastq()
        lines = fastq.strip().split("\n")
        assert len(lines) == 4
        assert lines[0].startswith("@test_read")
        assert lines[1] == "ATCG"
        assert lines[2] == "+"
        assert len(lines[3]) == 4

    def test_to_fastq_no_quality(self):
        result = BasecallResult(
            read_id="test_read",
            sequence="ATCG",
        )
        fastq = result.to_fastq()
        lines = fastq.strip().split("\n")
        assert lines[3] == "IIII"


class TestBasecallPipeline:
    def test_default_creation(self):
        pipeline = BasecallPipeline.default(device="cpu")
        assert pipeline is not None
        assert pipeline.device.type == "cpu"

    def test_preprocess_signal(self, sample_signal):
        pipeline = BasecallPipeline.default(device="cpu")
        processed = pipeline.preprocess_signal(sample_signal)
        assert processed.dtype == np.float32
        assert len(processed) <= len(sample_signal)

    def test_preprocess_signal_modes(self, sample_signal):
        pipeline = BasecallPipeline.default(device="cpu")

        pipeline.normalize_mode = "global"
        result_global = pipeline.preprocess_signal(sample_signal)

        pipeline.normalize_mode = "chunks"
        result_chunks = pipeline.preprocess_signal(sample_signal)

        assert result_global.shape == sample_signal.shape or len(result_global) <= len(sample_signal)
        assert result_chunks.shape == sample_signal.shape or len(result_chunks) <= len(sample_signal)

    def test_preprocess_invalid_mode(self, sample_signal):
        pipeline = BasecallPipeline.default(device="cpu")
        pipeline.normalize_mode = "invalid"
        with pytest.raises(ValueError):
            pipeline.preprocess_signal(sample_signal)

    def test_basecall_single(self, sample_signal):
        pipeline = BasecallPipeline.default(device="cpu")
        result = pipeline.basecall_single(sample_signal, read_id="test_read")

        assert isinstance(result, BasecallResult)
        assert result.read_id == "test_read"
        assert isinstance(result.sequence, str)
        assert result.signal_length == len(sample_signal)
        assert result.sequence_length == len(result.sequence)
        assert result.num_bases == len(result.sequence)
        assert 0.0 <= result.mean_confidence <= 1.0
        assert result.process_time > 0

    def test_basecall_single_empty(self):
        pipeline = BasecallPipeline.default(device="cpu")
        empty_signal = np.array([], dtype=np.float32)
        result = pipeline.basecall_single(empty_signal, read_id="empty")
        assert result.sequence == ""
        assert result.num_bases == 0

    def test_basecall_batch(self, sample_signal, long_signal):
        pipeline = BasecallPipeline.default(device="cpu")
        signals = [
            sample_signal,
            long_signal[:10000].copy(),
            long_signal[20000:30000].copy(),
        ]
        read_ids = ["read_1", "read_2", "read_3"]

        results = pipeline.basecall_batch(signals, read_ids)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.read_id == read_ids[i]

    def test_basecall_batch_default_ids(self, sample_signal):
        pipeline = BasecallPipeline.default(device="cpu")
        signals = [sample_signal, sample_signal]

        results = pipeline.basecall_batch(signals)

        assert len(results) == 2
        assert results[0].read_id == "read_0"
        assert results[1].read_id == "read_1"

    def test_summary(self, sample_signal):
        pipeline = BasecallPipeline.default(device="cpu")
        results = []
        for i in range(3):
            results.append(pipeline.basecall_single(sample_signal, f"read_{i}"))

        summary = pipeline.summary(results)

        assert summary["num_reads"] == 3
        assert summary["total_bases"] == sum(r.num_bases for r in results)
        assert "avg_read_length" in summary
        assert "avg_confidence" in summary
        assert "bases_per_second" in summary
        assert "total_process_time" in summary

    def test_trim_adapter_option(self, sample_signal):
        pipeline = BasecallPipeline.default(device="cpu")
        pipeline.trim_adapter = True
        result_trimmed = pipeline.basecall_single(sample_signal)

        pipeline.trim_adapter = False
        result_not_trimmed = pipeline.basecall_single(sample_signal)

        assert result_trimmed.signal_length == len(sample_signal)
        assert result_not_trimmed.signal_length == len(sample_signal)

    def test_denoise_option(self, sample_signal):
        pipeline = BasecallPipeline.default(device="cpu")
        pipeline.denoise_signal = True
        result_denoised = pipeline.basecall_single(sample_signal)

        pipeline.denoise_signal = False
        result_raw = pipeline.basecall_single(sample_signal)

        assert isinstance(result_denoised.sequence, str)
        assert isinstance(result_raw.sequence, str)

    def test_write_fasta(self, sample_signal, tmp_path):
        pipeline = BasecallPipeline.default(device="cpu")
        results = []
        for i in range(3):
            results.append(pipeline.basecall_single(sample_signal, f"read_{i}"))

        output_file = str(tmp_path / "output.fasta")
        pipeline._write_fasta(results, output_file)

        assert os.path.exists(output_file)
        with open(output_file) as f:
            content = f.read()
        assert content.count(">") == 3

    def test_write_fastq(self, sample_signal, tmp_path):
        pipeline = BasecallPipeline.default(device="cpu")
        results = []
        for i in range(2):
            results.append(pipeline.basecall_single(sample_signal, f"read_{i}"))

        output_file = str(tmp_path / "output.fastq")
        pipeline._write_fastq(results, output_file)

        assert os.path.exists(output_file)
        with open(output_file) as f:
            content = f.read()
        assert content.count("@read_") == 2
