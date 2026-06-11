"""
单元测试 - 欧拉路径拼接与端到端组装
"""

import os
import tempfile
import numpy as np
import pytest
from nanopore_core.assembly.kmer_util import KmerUtil
from nanopore_core.assembly.de_bruijn_graph import DeBruijnGraph
from nanopore_core.assembly.graph_simplifier import GraphSimplifier
from nanopore_core.assembly.eulerian_path import (
    EulerianPathAssembler,
    Contig,
    AssemblyStats,
)
from nanopore_core.assembly.assembler import DeNovoAssembler, AssemblyResult


class TestContig:
    def test_create(self):
        contig = Contig(
            name="contig_0001",
            sequence="ATCGATCG",
            length=8,
            num_kmers=4,
            avg_depth=10.0,
            path_nodes=[],
        )
        assert contig.name == "contig_0001"
        assert contig.length == 8

    def test_to_fasta(self):
        contig = Contig(
            name="contig_0001",
            sequence="ATCGATCG",
            length=8,
            num_kmers=4,
            avg_depth=5.5,
            path_nodes=[],
        )
        fasta = contig.to_fasta()
        assert fasta.startswith(">contig_0001")
        assert "ATCGATCG" in fasta
        assert "depth=5.50" in fasta


class TestEulerianPathAssembler:
    def test_create(self):
        assembler = EulerianPathAssembler(min_contig_length=10)
        assert assembler.min_contig_length == 10

    def test_assemble_simple(self):
        np.random.seed(42)
        k = 5
        seq_len = 100
        bases = ["A", "T", "C", "G"]
        seq = "".join([bases[i] for i in np.random.randint(0, 4, seq_len)])

        graph = DeBruijnGraph(k=k)
        graph.add_sequence(seq)

        assembler = EulerianPathAssembler(min_contig_length=20)
        contigs = assembler.assemble(graph)

        assert len(contigs) >= 1
        total_len = sum(c.length for c in contigs)
        assert total_len >= 30

    def test_assemble_with_branches(self):
        graph = DeBruijnGraph(k=5)

        prefix = "A" * 30
        branch1 = prefix + "T" * 30
        branch2 = prefix + "G" * 30

        graph.add_sequence(branch1)
        graph.add_sequence(branch2)

        assembler = EulerianPathAssembler(min_contig_length=5)
        contigs = assembler.assemble(graph)

        assert len(contigs) >= 2

    def test_compute_assembly_stats(self):
        assembler = EulerianPathAssembler()

        contigs = [
            Contig("c1", "A" * 1000, 1000, 996, 10.0, []),
            Contig("c2", "T" * 500, 500, 496, 5.0, []),
            Contig("c3", "G" * 2000, 2000, 1996, 8.0, []),
        ]

        stats = assembler.compute_assembly_stats(contigs)

        assert stats.num_contigs == 3
        assert stats.total_bases == 3500
        assert stats.max_contig == 2000
        assert stats.n50 == 2000

    def test_assembly_stats_empty(self):
        assembler = EulerianPathAssembler()
        stats = assembler.compute_assembly_stats([])
        assert stats.num_contigs == 0
        assert stats.total_bases == 0

    def test_write_fasta(self, tmp_path):
        assembler = EulerianPathAssembler()

        contigs = [
            Contig("c1", "ATCG" * 20, 80, 76, 5.0, []),
            Contig("c2", "GCAT" * 10, 40, 36, 3.0, []),
        ]

        output_file = str(tmp_path / "test_contigs.fasta")
        assembler.write_fasta(contigs, output_file)

        assert os.path.exists(output_file)
        with open(output_file) as f:
            content = f.read()
        assert content.count(">") == 2


class TestDeNovoAssembler:
    def test_create(self):
        assembler = DeNovoAssembler(k=21, min_kmer_count=2)
        assert assembler.k == 21
        assert assembler.min_kmer_count == 2

    def test_default(self):
        assembler = DeNovoAssembler.default()
        assert assembler is not None
        assert assembler.k == 31

    def test_assemble(self):
        np.random.seed(42)
        genome_len = 500
        bases = ["A", "T", "C", "G"]
        genome = "".join([bases[i] for i in np.random.randint(0, 4, genome_len)])

        reads = []
        for i in range(0, len(genome) - 50, 10):
            reads.append(genome[i:i + 50])

        assembler = DeNovoAssembler(k=15, min_kmer_count=1, min_contig_length=30)
        result = assembler.assemble(reads)

        assert isinstance(result, AssemblyResult)
        assert len(result.contigs) > 0
        assert result.assembly_stats.total_bases > 0

    def test_assembly_result(self):
        reads = [
            "ATCGATCGATCGATCG",
            "TCGATCGATCGATCGA",
            "CGATCGATCGATCGAT",
        ]

        assembler = DeNovoAssembler(k=8, min_kmer_count=1, min_contig_length=10)
        result = assembler.assemble(reads)

        assert result.num_input_reads == 3
        assert result.k == 8
        assert result.total_time >= 0

    def test_assemble_with_output(self, tmp_path):
        reads = ["ATCGATCGATCG", "TCGATCGATCGATCG", "CGATCGATCGATCGA"]

        output_file = str(tmp_path / "output.fasta")
        assembler = DeNovoAssembler(k=8, min_kmer_count=1, min_contig_length=10)
        result = assembler.assemble(reads, output_fasta=output_file)

        assert os.path.exists(output_file)
        assert len(result.contigs) > 0

    def test_get_graph(self):
        reads = ["ATCGATCGATCG", "TCGATCGATCGATCG"]

        assembler = DeNovoAssembler(k=8, min_kmer_count=1, min_contig_length=10)
        assembler.assemble(reads)

        graph = assembler.get_graph()
        assert graph is not None
        assert graph.num_nodes > 0

    def test_assembly_stats_increase_k(self):
        np.random.seed(42)
        genome_len = 500
        bases = ["A", "T", "C", "G"]
        genome = "".join([bases[i] for i in np.random.randint(0, 4, genome_len)])

        reads = []
        for _ in range(10):
            start = np.random.randint(0, genome_len - 50)
            reads.append(genome[start:start + 50])

        assembler = DeNovoAssembler(k=20, min_kmer_count=2, min_contig_length=30)
        result = assembler.assemble(reads)

        assert result.assembly_stats.n50 >= 0
        assert result.graph_stats_before["num_nodes"] > 0
        assert result.graph_stats_after["num_nodes"] > 0
