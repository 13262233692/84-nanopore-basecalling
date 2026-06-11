"""
单元测试 - 图简化算法
"""

import numpy as np
import pytest
from nanopore_core.assembly.kmer_util import KmerUtil
from nanopore_core.assembly.de_bruijn_graph import DeBruijnGraph
from nanopore_core.assembly.graph_simplifier import GraphSimplifier


class TestGraphSimplifier:
    def test_create(self):
        simp = GraphSimplifier(
            tip_max_length=10,
            tip_min_depth=2.0,
            bubble_max_length=100,
        )
        assert simp.tip_max_length == 10
        assert simp.tip_min_depth == 2.0

    def test_trim_tips(self):
        graph = DeBruijnGraph(k=3)

        for _ in range(5):
            graph.add_sequence("AAAAATTTTG")
        graph.add_sequence("AAAAACCCG")

        nodes_before = graph.num_nodes
        edges_before = graph.num_edges

        simp = GraphSimplifier(tip_max_length=10, tip_min_depth=2.0)
        stats = simp.trim_tips(graph)

        assert stats > 0
        assert graph.num_nodes < nodes_before
        assert graph.num_edges < edges_before

    def test_compress_bubbles(self):
        graph = DeBruijnGraph(k=3)

        seq1 = "AAAACAAAA"
        seq2 = "AAAATAAAA"

        graph.add_sequence(seq1)
        graph.add_sequence(seq2)

        simp = GraphSimplifier(bubble_max_length=10)
        bubbles = simp.detect_bubbles(graph)

        assert len(bubbles) >= 1

    def test_simplify(self):
        graph = DeBruijnGraph(k=5)

        main_seq = "ATCGATCGATCGATCGATCG"
        graph.add_sequence(main_seq)

        tip_seq = "ATCGATCGATCGGG"
        graph.add_sequence(tip_seq)

        simp = GraphSimplifier(
            tip_max_length=10,
            tip_min_depth=0.5,
            bubble_max_length=20,
        )

        stats = simp.simplify(graph, max_iterations=5)

        assert stats.tips_removed >= 0
        assert stats.bubbles_removed >= 0
        assert stats.tips_iterations > 0

    def test_remove_low_coverage_edges(self):
        graph = DeBruijnGraph(k=3)

        for _ in range(5):
            graph.add_sequence("ATCGATCG")
        graph.add_sequence("ATCGGG")

        simp = GraphSimplifier()
        removed = simp.remove_low_coverage_edges(graph, min_count=3)

        assert removed > 0

    def test_remove_low_coverage_nodes(self):
        graph = DeBruijnGraph(k=5)

        for _ in range(5):
            graph.add_sequence("ATCGATCGATCG")

        simp = GraphSimplifier()
        removed = simp.remove_low_coverage_nodes(graph, min_average_depth=3.0)

        assert removed >= 0

    def test_bubble_detection_simple(self):
        graph = DeBruijnGraph(k=3)

        prefix = "AAA"
        suffix = "TTT"
        path1 = prefix + "C" + suffix
        path2 = prefix + "G" + suffix

        graph.add_sequence(path1)
        graph.add_sequence(path2)

        simp = GraphSimplifier(bubble_max_length=10)
        bubbles = simp.detect_bubbles(graph)

        assert len(bubbles) >= 1
        assert len(bubbles[0].paths) >= 2

    def test_no_bubbles(self):
        graph = DeBruijnGraph(k=5)
        graph.add_sequence("ATCGATCGATCGATCG")

        simp = GraphSimplifier()
        bubbles = simp.detect_bubbles(graph)

        assert len(bubbles) == 0

    def test_tip_removal_iterative(self):
        graph = DeBruijnGraph(k=3)

        main = "AAAAAAAAAAG"
        tip = "AAAAAAAAAATT"

        for _ in range(5):
            graph.add_sequence(main)
        graph.add_sequence(tip)

        edges_before = graph.num_edges

        simp = GraphSimplifier(tip_max_length=5, tip_min_depth=2.0)
        stats = simp.simplify(graph, max_iterations=10)

        assert graph.num_edges < edges_before

    def test_bubble_paths_count(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("AAACCCAAA")
        graph.add_sequence("AAAGGGAAA")
        graph.add_sequence("AAATTTAAA")

        simp = GraphSimplifier(bubble_max_length=10)
        bubbles = simp.detect_bubbles(graph)

        assert len(bubbles) >= 1
        assert len(bubbles[0].paths) >= 2

    def test_compress_bubbles_removes_edges(self):
        graph = DeBruijnGraph(k=3)
        graph.add_sequence("AAACCCAAA")
        graph.add_sequence("AAAGGGAAA")

        edges_before = graph.num_edges

        simp = GraphSimplifier(bubble_max_length=10)
        count = simp.compress_bubbles(graph)

        assert count >= 1
        assert graph.num_edges < edges_before
