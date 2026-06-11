"""
单元测试 - K-mer 工具库
"""

import numpy as np
import pytest
from nanopore_core.assembly.kmer_util import KmerUtil, BASE_TO_BIT, BIT_TO_BASE


class TestKmerUtil:
    def test_create(self):
        kmer = KmerUtil(k=5)
        assert kmer.k == 5
        assert kmer.k_minus_1 == 4

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            KmerUtil(k=1)
        with pytest.raises(ValueError):
            KmerUtil(k=50)

    def test_encode_decode_roundtrip(self):
        kmer = KmerUtil(k=10)
        seq = "ATCGATCGAT"
        code = kmer.encode(seq)
        decoded = kmer.decode(code)
        assert decoded == seq

    def test_encode_atcg(self):
        kmer = KmerUtil(k=4)
        code = kmer.encode("ATCG")
        decoded = kmer.decode(code)
        assert decoded == "ATCG"

    def test_get_prefix(self):
        kmer = KmerUtil(k=5)
        code = kmer.encode("ATCGA")
        prefix = kmer.get_prefix(code)
        assert kmer.decode_node(prefix) == "ATCG"

    def test_get_suffix(self):
        kmer = KmerUtil(k=5)
        code = kmer.encode("ATCGA")
        suffix = kmer.get_suffix(code)
        assert kmer.decode_node(suffix) == "TCGA"

    def test_extend_right(self):
        kmer = KmerUtil(k=5)
        node = kmer.encode_node("ATCG")
        new_node = kmer.extend_right(node, BASE_TO_BIT["A"])
        assert kmer.decode_node(new_node) == "TCGA"

    def test_iter_kmers(self):
        kmer = KmerUtil(k=3)
        seq = "ATCGAT"
        kmers = kmer.iter_kmers(seq)

        assert len(kmers) == len(seq) - 2
        assert kmer.decode(kmers[0]) == "ATC"
        assert kmer.decode(kmers[1]) == "TCG"
        assert kmer.decode(kmers[2]) == "CGA"
        assert kmer.decode(kmers[3]) == "GAT"

    def test_iter_kmers_short(self):
        kmer = KmerUtil(k=10)
        seq = "ATCG"
        kmers = kmer.iter_kmers(seq)
        assert kmers == []

    def test_kmer_count_dict(self):
        kmer = KmerUtil(k=3)
        seqs = ["ATCG", "TCGA", "CGAT"]
        counts = kmer.kmer_count_dict(seqs)

        assert len(counts) == 4
        atc = kmer.encode("ATC")
        tcg = kmer.encode("TCG")
        assert counts.get(atc, 0) == 1
        assert counts.get(tcg, 0) == 2

    def test_kmer_count_min_count(self):
        kmer = KmerUtil(k=3)
        seqs = ["ATCG", "ATCG", "CGAT"]
        counts = kmer.kmer_count_dict(seqs, min_count=2)

        atc = kmer.encode("ATC")
        tcg = kmer.encode("TCG")
        assert counts.get(atc, 0) >= 2
        assert counts.get(tcg, 0) >= 2

    def test_reverse_complement(self):
        kmer = KmerUtil(k=4)
        rc = kmer.reverse_complement("ATCG")
        assert rc == "CGAT"

    def test_reverse_complement_code(self):
        kmer = KmerUtil(k=4)
        code = kmer.encode("ATCG")
        rc_code = kmer.reverse_complement_code(code)
        rc_str = kmer.decode(rc_code)
        assert rc_str == "CGAT"

    def test_canonical_kmer(self):
        kmer = KmerUtil(k=4)
        code = kmer.encode("ATCG")
        canon = kmer.canonical_kmer(code)
        rc = kmer.reverse_complement_code(code)
        assert canon == min(code, rc)

    def test_kmer_to_edges(self):
        kmer = KmerUtil(k=5)
        code = kmer.encode("ATCGA")
        from_node, to_node = kmer.kmer_to_edges(code)

        assert kmer.decode_node(from_node) == "ATCG"
        assert kmer.decode_node(to_node) == "TCGA"

    def test_node_to_kmer(self):
        kmer = KmerUtil(k=5)
        from_code = kmer.encode_node("ATCG")
        to_code = kmer.encode_node("TCGA")
        kmer_code = kmer.node_to_kmer(from_code, to_code)
        assert kmer.decode(kmer_code) == "ATCGA"

    def test_edge_roundtrip(self):
        kmer = KmerUtil(k=10)
        original = kmer.encode("ATCGATCGAT")
        f, t = kmer.kmer_to_edges(original)
        recovered = kmer.node_to_kmer(f, t)
        assert recovered == original

    def test_repr(self):
        kmer = KmerUtil(k=31)
        assert "KmerUtil" in repr(kmer)
        assert "k=31" in repr(kmer)
