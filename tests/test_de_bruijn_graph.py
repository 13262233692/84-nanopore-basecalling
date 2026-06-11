"""
单元测试 - De Bruijn 图
"""

import numpy as np
import pytest
from nanopore_core.assembly.kmer_util import KmerUtil
from nanopore_core.assembly.de_bruijn_graph import DeBruijnGraph


class TestDeBruijnGraph:
    def test_create(self):
        graph = DeBruijnGraph(k=5)
        assert graph.k == 5
        assert graph.num_nodes == 0
        assert graph.num_edges == 0

    def test_add_kmer(self):
        graph = DeBruijnGraph(k=5)
        kmer_util = KmerUtil(5)
        kmer = kmer_util.encode("ATCGA")

        graph.add_kmer(kmer)

        assert graph.num_edges == 1
        assert graph.num_nodes == 2

    def test_add_sequence(self):
        graph = DeBruijnGraph(k=5)
        seq = "ATCGATCGAT"

        graph.add_sequence(seq)

        assert graph.num_edges == 4
        assert graph.num_nodes == 4

    def test_add_sequences(self):
        graph = DeBruijnGraph(k=5)
        seqs = ["ATCGAT", "CGATCG", "GATCGA"]

        graph.add_sequences(seqs)

        assert graph.num_edges > 0
        assert graph.num_nodes > 0

    def test_out_degree(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("ATCG")

        at_node = graph.kmer_util.encode_node("AT")
        assert graph.out_degree(at_node) == 1

    def test_in_degree(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("ATCG")

        cg_node = graph.kmer_util.encode_node("CG")
        assert graph.in_degree(cg_node) == 1

    def test_out_neighbors(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("ATCG")

        at_node = graph.kmer_util.encode_node("AT")
        neighbors = graph.out_neighbors(at_node)

        assert len(neighbors) == 1
        tc_node = graph.kmer_util.encode_node("TC")
        assert tc_node in neighbors

    def test_in_neighbors(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("ATCG")

        cg_node = graph.kmer_util.encode_node("CG")
        neighbors = graph.in_neighbors(cg_node)

        assert len(neighbors) == 1
        tc_node = graph.kmer_util.encode_node("TC")
        assert tc_node in neighbors

    def test_remove_edge(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("ATCG")

        from_n = graph.kmer_util.encode_node("AT")
        to_n = graph.kmer_util.encode_node("TC")

        assert graph.num_edges == 2
        result = graph.remove_edge(from_n, to_n)
        assert result is True
        assert graph.num_edges == 1

    def test_remove_edge_nonexistent(self):
        graph = DeBruijnGraph(k=3)
        result = graph.remove_edge(123, 456)
        assert result is False

    def test_remove_node(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("ATCG")

        tc_node = graph.kmer_util.encode_node("TC")

        result = graph.remove_node(tc_node)
        assert result is True
        assert graph.num_nodes == 2
        assert graph.num_edges == 0

    def test_get_path_forward(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("ATCGAT")

        start = graph.kmer_util.encode_node("AT")

        nodes, weights = graph.get_path_forward(start, max_length=10)
        assert len(nodes) > 1
        assert len(weights) == len(nodes) - 1

    def test_get_path_backward(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("ATCGAT")

        end = graph.kmer_util.encode_node("AT")

        nodes, weights = graph.get_path_backward(end, max_length=10)
        assert len(nodes) > 1

    def test_is_tip_start(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("AAAAA")
        graph.add_sequence("AAAT")

        at_node = graph.kmer_util.encode_node("AT")
        assert graph.is_tip_start(at_node, max_depth=5)

    def test_is_tip_end(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("AAAAA")
        graph.add_sequence("TAAA")

        ta_node = graph.kmer_util.encode_node("TA")
        assert graph.is_tip_end(ta_node, max_depth=5)

    def test_is_not_tip(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("AAAAA")

        aa_node = graph.kmer_util.encode_node("AA")
        assert not graph.is_tip_start(aa_node, max_depth=5)

    def test_is_tip_dead_end(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("AAT")

        at_node = graph.kmer_util.encode_node("AT")
        assert graph.out_degree(at_node) == 0
        assert graph.is_tip_start(at_node, max_depth=0)

    def test_path_to_sequence(self):
        graph = DeBruijnGraph(k=5)
        seq = "ATCGATCGAT"
        graph.add_sequence(seq)

        start_node = graph.kmer_util.encode_node("ATCG")
        nodes, _ = graph.get_path_forward(start_node, max_length=10)

        recovered = graph.path_to_sequence(nodes)
        assert len(recovered) == len(nodes) + graph.k - 2

    def test_path_to_sequence_single(self):
        graph = DeBruijnGraph(k=5)
        node = graph.kmer_util.encode_node("ATCG")
        seq = graph.path_to_sequence([node])
        assert seq == "ATCG"

    def test_get_stats(self):
        graph = DeBruijnGraph(k=5)
        graph.add_sequence("ATCGATCGAT")

        stats = graph.get_stats()
        assert stats.k == 5
        assert stats.num_nodes > 0
        assert stats.num_edges > 0
        assert stats.total_kmer_count > 0

    def test_repr(self):
        graph = DeBruijnGraph(k=31)
        assert "DeBruijnGraph" in repr(graph)
        assert "k=31" in repr(graph)
