"""
端到端从头组装流水线 (De Novo Assembly Pipeline)

整合 De Bruijn 图构建、图简化、欧拉路径拼接等
所有步骤，提供简洁的调用接口。

完整流程：
1. 输入：百万条 Reads（DNA 序列）
2. K-mer 统计与过滤
3. De Bruijn 图构建
4. 图简化（Tips 修剪、Bubbles 压缩）
5. 欧拉路径寻找与 Contig 拼接
6. 输出：FASTA 格式的 Contigs
"""

import time
import os
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .kmer_util import KmerUtil
from .de_bruijn_graph import DeBruijnGraph
from .graph_simplifier import GraphSimplifier, SimplificationStats
from .eulerian_path import EulerianPathAssembler, Contig, AssemblyStats


@dataclass
class AssemblyResult:
    """从头组装结果"""
    contigs: List[Contig]
    assembly_stats: AssemblyStats
    graph_stats_before: Dict
    graph_stats_after: Dict
    simplification_stats: SimplificationStats
    total_time: float = 0.0
    k: int = 31
    num_input_reads: int = 0
    total_input_bases: int = 0


class DeNovoAssembler:
    """从头基因组组装器
    
    基于 De Bruijn 图的完整从头组装流水线。
    """

    def __init__(
        self,
        k: int = 31,
        min_kmer_count: int = 2,
        min_contig_length: int = 200,
        tip_max_length: int = 10,
        tip_min_depth: float = 2.0,
        bubble_max_length: int = 100,
        simplify_iterations: int = 5,
    ):
        """
        Args:
            k: K-mer 长度
            min_kmer_count: 最小 K-mer 出现次数（过滤错误 K-mer）
            min_contig_length: 最小 Contig 长度（碱基）
            tip_max_length: Tip 最大长度
            tip_min_depth: Tip 最小深度阈值
            bubble_max_length: Bubble 最大长度
            simplify_iterations: 图简化最大迭代次数
        """
        self.k = k
        self.min_kmer_count = min_kmer_count
        self.min_contig_length = min_contig_length
        self.tip_max_length = tip_max_length
        self.tip_min_depth = tip_min_depth
        self.bubble_max_length = bubble_max_length
        self.simplify_iterations = simplify_iterations

        self.kmer_util = KmerUtil(k)
        self.graph: Optional[DeBruijnGraph] = None
        self.simplifier = GraphSimplifier(
            tip_max_length=tip_max_length,
            tip_min_depth=tip_min_depth,
            bubble_max_length=bubble_max_length,
        )
        self.assembler = EulerianPathAssembler(
            min_contig_length=min_contig_length,
            prefer_high_depth=True,
        )

    def assemble(
        self,
        reads: List[str],
        output_fasta: Optional[str] = None,
    ) -> AssemblyResult:
        """执行完整的从头组装
        
        Args:
            reads: 输入的 DNA 序列列表（reads）
            output_fasta: 输出 FASTA 文件路径（可选）
            
        Returns:
            AssemblyResult 组装结果
        """
        start_time = time.time()

        total_bases = sum(len(r) for r in reads)
        num_reads = len(reads)

        print(f"[Assembler] 输入 Reads: {num_reads:,} 条")
        print(f"[Assembler] 输入碱基: {total_bases:,} bp")
        print(f"[Assembler] K-mer 长度: {self.k}")

        print(f"[Assembler] 步骤 1: K-mer 统计...")
        kmer_counts = self._count_kmers(reads)
        print(f"[Assembler]   原始 K-mer 种类: {len(kmer_counts):,}")

        if self.min_kmer_count > 1:
            kmer_counts = {
                k: v for k, v in kmer_counts.items()
                if v >= self.min_kmer_count
            }
            print(f"[Assembler]   过滤后 K-mer 种类: {len(kmer_counts):,}")

        print(f"[Assembler] 步骤 2: 构建 De Bruijn 图...")
        graph = self._build_graph(kmer_counts)
        self.graph = graph
        stats_before = graph.get_stats().__dict__
        print(f"[Assembler]   节点数: {stats_before['num_nodes']:,}")
        print(f"[Assembler]   边数: {stats_before['num_edges']:,}")
        print(f"[Assembler]   平均覆盖度: {stats_before['avg_kmer_depth']:.2f}")

        print(f"[Assembler] 步骤 3: 图简化...")
        simp_stats = self.simplifier.simplify(
            graph, max_iterations=self.simplify_iterations
        )
        stats_after = graph.get_stats().__dict__
        print(f"[Assembler]   移除 Tips: {simp_stats.tips_removed}")
        print(f"[Assembler]   压缩 Bubbles: {simp_stats.bubbles_removed}")
        print(f"[Assembler]   移除边: {simp_stats.edges_removed:,}")
        print(f"[Assembler]   移除节点: {simp_stats.nodes_removed:,}")
        print(f"[Assembler]   剩余节点: {stats_after['num_nodes']:,}")
        print(f"[Assembler]   剩余边: {stats_after['num_edges']:,}")

        print(f"[Assembler] 步骤 4: 欧拉路径拼接 Contigs...")
        contigs = self.assembler.assemble(graph)
        assembly_stats = self.assembler.compute_assembly_stats(contigs)
        print(f"[Assembler]   Contig 数量: {assembly_stats.num_contigs}")
        print(f"[Assembler]   总碱基数: {assembly_stats.total_bases:,} bp")
        print(f"[Assembler]   最大 Contig: {assembly_stats.max_contig:,} bp")
        print(f"[Assembler]   N50: {assembly_stats.n50:,} bp")
        print(f"[Assembler]   N90: {assembly_stats.n90:,} bp")
        print(f"[Assembler]   GC 含量: {assembly_stats.gc_content*100:.2f}%")

        if output_fasta:
            print(f"[Assembler] 步骤 5: 输出 FASTA 文件...")
            self.assembler.write_fasta(contigs, output_fasta)
            print(f"[Assembler]   输出: {output_fasta}")

        total_time = time.time() - start_time
        print(f"[Assembler] 总耗时: {total_time:.2f} 秒")

        return AssemblyResult(
            contigs=contigs,
            assembly_stats=assembly_stats,
            graph_stats_before=stats_before,
            graph_stats_after=stats_after,
            simplification_stats=simp_stats,
            total_time=total_time,
            k=self.k,
            num_input_reads=num_reads,
            total_input_bases=total_bases,
        )

    def _count_kmers(self, reads: List[str]) -> Dict[int, int]:
        """统计所有 reads 中的 K-mer 频次"""
        counts: Dict[int, int] = {}
        for read in reads:
            self.kmer_util.iter_kmers_with_count(read, counts)
        return counts

    def _build_graph(self, kmer_counts: Dict[int, int]) -> DeBruijnGraph:
        """从 K-mer 频次构建 De Bruijn 图"""
        graph = DeBruijnGraph(self.k)
        for kmer_code, count in kmer_counts.items():
            graph.add_kmer(kmer_code, count)
        return graph

    def get_graph(self) -> Optional[DeBruijnGraph]:
        """获取构建的 De Bruijn 图"""
        return self.graph

    @classmethod
    def default(cls) -> "DeNovoAssembler":
        """创建默认配置的组装器"""
        return cls(
            k=31,
            min_kmer_count=2,
            min_contig_length=200,
            tip_max_length=10,
            tip_min_depth=2.0,
            bubble_max_length=100,
            simplify_iterations=5,
        )
