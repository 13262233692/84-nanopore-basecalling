"""
单元测试 - CTC 贪心解码模块
"""

import numpy as np
import torch
import pytest
from nanopore_core.decoding.ctc_decoder import (
    CTCGreedyDecoder,
    labels_to_bases,
    bases_to_labels,
    _greedy_decode_single,
    BASES,
    BLANK_INDEX,
)


class TestLabelConversion:
    def test_labels_to_bases(self):
        labels = np.array([0, 1, 2, 3], dtype=np.int32)
        result = labels_to_bases(labels)
        assert result == "ATCG"

    def test_labels_to_bases_with_blank(self):
        labels = np.array([0, 4, 1, 4, 2], dtype=np.int32)
        result = labels_to_bases(labels)
        assert result == "ATC"

    def test_labels_to_bases_empty(self):
        labels = np.array([], dtype=np.int32)
        result = labels_to_bases(labels)
        assert result == ""

    def test_bases_to_labels(self):
        seq = "ATCG"
        result = bases_to_labels(seq)
        assert list(result) == [0, 1, 2, 3]

    def test_bases_to_labels_lowercase(self):
        seq = "atcg"
        result = bases_to_labels(seq)
        assert list(result) == [0, 1, 2, 3]

    def test_bases_to_labels_invalid(self):
        seq = "ATXN"
        result = bases_to_labels(seq)
        assert list(result) == [0, 1]

    def test_roundtrip(self):
        original = "ATCGATCG"
        labels = bases_to_labels(original)
        recovered = labels_to_bases(labels)
        assert original == recovered


class TestGreedyDecodeSingle:
    def test_decode_simple(self):
        T, C = 20, 5
        log_probs = np.full((T, C), -10.0, dtype=np.float32)
        sequence = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
        for i, label in enumerate(sequence):
            log_probs[i, label] = 0.0
        for i in range(len(sequence), T):
            log_probs[i, BLANK_INDEX] = 0.0

        labels, conf = _greedy_decode_single(log_probs, BLANK_INDEX)
        decoded = labels_to_bases(labels)

        assert decoded == "ATCG"
        assert conf > 0.0

    def test_decode_all_blanks(self):
        T = 10
        log_probs = np.full((T, 5), -10.0, dtype=np.float32)
        log_probs[:, BLANK_INDEX] = 0.0

        labels, conf = _greedy_decode_single(log_probs, BLANK_INDEX)
        assert len(labels) == 0
        assert conf == 0.0

    def test_decode_collapse_repeats(self):
        T = 15
        log_probs = np.full((T, 5), -10.0, dtype=np.float32)
        for i in range(5):
            log_probs[i, 0] = 0.0
        for i in range(5, 10):
            log_probs[i, BLANK_INDEX] = 0.0
        for i in range(10, 15):
            log_probs[i, 0] = 0.0

        labels, _ = _greedy_decode_single(log_probs, BLANK_INDEX)
        decoded = labels_to_bases(labels)

        assert decoded == "AA"

    def test_decode_single_base(self):
        T = 5
        log_probs = np.full((T, 5), -10.0, dtype=np.float32)
        log_probs[:, 2] = 0.0

        labels, _ = _greedy_decode_single(log_probs, BLANK_INDEX)
        decoded = labels_to_bases(labels)
        assert decoded == "C"


class TestCTCGreedyDecoder:
    def test_decode_single(self):
        decoder = CTCGreedyDecoder()
        T, C = 30, 5
        log_probs = np.full((T, C), -10.0, dtype=np.float32)
        seq_labels = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
        for i, lbl in enumerate(seq_labels):
            log_probs[i, lbl] = 0.0
        for i in range(len(seq_labels), T):
            log_probs[i, BLANK_INDEX] = 0.0

        sequence, confidence = decoder.decode(log_probs)

        assert sequence == "ATCG"
        assert 0.0 < confidence <= 1.0

    def test_decode_with_length(self):
        decoder = CTCGreedyDecoder()
        T = 100
        log_probs = np.full((T, 5), -10.0, dtype=np.float32)
        log_probs[:10, 0] = 0.0
        log_probs[10:, BLANK_INDEX] = 0.0

        seq_full, _ = decoder.decode(log_probs)
        seq_short, _ = decoder.decode(log_probs, length=15)

        assert seq_full == "A"
        assert seq_short == "A"

    def test_decode_batch(self):
        decoder = CTCGreedyDecoder()
        T, B, C = 20, 3, 5
        log_probs = np.full((T, B, C), -10.0, dtype=np.float32)

        for b in range(B):
            for t in range(5):
                log_probs[t, b, b] = 0.0
            for t in range(5, T):
                log_probs[t, b, BLANK_INDEX] = 0.0

        lengths = np.array([20, 20, 20], dtype=np.int32)
        sequences, confs = decoder.decode_batch(log_probs, lengths)

        assert len(sequences) == 3
        assert len(confs) == 3
        assert sequences[0] == "A"
        assert sequences[1] == "T"
        assert sequences[2] == "C"

    def test_decode_from_torch(self):
        decoder = CTCGreedyDecoder()
        T, B, C = 20, 2, 5
        log_probs = torch.full((T, B, C), -10.0)
        for t in range(5):
            log_probs[t, 0, 0] = 0.0
        for t in range(5, T):
            log_probs[t, :, BLANK_INDEX] = 0.0
        for t in range(5):
            log_probs[t, 1, 3] = 0.0

        lengths = torch.tensor([20, 20], dtype=torch.int32)
        sequences, confs = decoder.decode_from_torch(log_probs, lengths)

        assert len(sequences) == 2
        assert sequences[0] == "A"
        assert sequences[1] == "G"

    def test_decode_with_timesteps(self):
        decoder = CTCGreedyDecoder()
        T = 30
        log_probs = np.full((T, 5), -10.0, dtype=np.float32)

        for t in range(0, 5):
            log_probs[t, 0] = 0.0
        for t in range(5, 7):
            log_probs[t, BLANK_INDEX] = 0.0
        for t in range(7, 12):
            log_probs[t, 1] = 0.0
        for t in range(12, T):
            log_probs[t, BLANK_INDEX] = 0.0

        sequence, starts, ends = decoder.decode_with_timesteps(log_probs)

        assert sequence == "AT"
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0] == 0
        assert ends[0] == 4
        assert starts[1] == 7
        assert ends[1] == 11

    def test_quality_scores(self):
        decoder = CTCGreedyDecoder()
        confidences = np.array([0.9, 0.5, 0.99], dtype=np.float32)
        q_scores = decoder.quality_scores(confidences)

        assert len(q_scores) == 3
        assert q_scores[2] > q_scores[0] > q_scores[1]
        assert q_scores[0] > 0

    def test_decode_with_quality(self):
        decoder = CTCGreedyDecoder()
        T = 20
        log_probs = np.full((T, 5), -10.0, dtype=np.float32)
        for t in range(5):
            log_probs[t, 0] = 0.0
        for t in range(5, 10):
            log_probs[t, 1] = 0.0
        for t in range(10, T):
            log_probs[t, BLANK_INDEX] = 0.0

        sequence, confs, qs = decoder.decode_with_quality(log_probs)

        assert sequence == "AT"
        assert len(confs) == 2
        assert len(qs) == 2
