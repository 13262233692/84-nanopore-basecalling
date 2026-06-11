"""
欧拉路径寻找与 Contig 拼接算法

在简化后的 De Bruijn 图中寻找欧拉路径，将分散的边序列
拼接成长的连续重叠群 (Contigs)。

De Bruijn 图的欧拉路径性质：
- 每条边代表一个 K-mer
- 欧拉路径经过每条边恰好一次
- 路径长度 = 边数 + K - 1 个碱基

实际数据中的图结构复杂，策略：
1. 线性路径直接拼接（出度=入度=1 的节点链）
2. 分支处作为 contig 边界
3. 环状 contig 单独处理
4. 按长度排序输出 contigs
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass

from .de_bruijn_graph import DeBruijnGraph


@dataclass
class Contig:
    """重叠群 (Contig) 数据类"""
    name: str
    sequence: str
    length: int
    num_kmers: int
    avg_depth: float
    path_nodes: List[int]
    is_circular: bool = False

    def to_fasta(self, wrap: int = 80) -> str:
        """转换为 FASTA 格式"""
        header = f">{self.name} length={self.length} depth={self.avg_depth:.2f}"
        if self.is_circular:
            header += " circular=yes"

        seq_lines = []
        for i in range(0, len(self.sequence), wrap):
            seq_lines.append(self.sequence[i:i + wrap])

        return header + "\n" + "\n".join(seq_lines) + "\n"


@dataclass
class AssemblyStats:
    """组装结果统计"""
    num_contigs: int = 0
    total_bases: int = 0
    n50: int = 0
    n90: int = 0
    max_contig: int = 0
    min_contig: int = 0
    avg_contig_length: float = 0.0
    gc_content: float = 0.0


class EulerianPathAssembler:
    """基于欧拉路径的 Contig 拼接器
    
    在简化后的 De Bruijn 图上构建重叠群。
    """

    def __init__(
        self,
        min_contig_length: int = 100,
        prefer_high_depth: bool = True,
    ):
        """
        Args:
            min_contig_length: 最小 contig 长度（碱基）
            prefer_high_depth: 分支处优先选择高覆盖度路径
        """
        self.min_contig_length = min_contig_length
        self.prefer_high_depth = prefer_high_depth

    def assemble(
        self,
        graph: DeBruijnGraph,
    ) -> List[Contig]:
        """执行组装，生成 Contigs
        
        Args:
            graph: 简化后的 De Bruijn 图
            
        Returns:
            Contig 列表（按长度降序排列）
        """
        contigs = []
        visited_edges: Set[Tuple[int, int]] = set()

        start_nodes = self._find_start_nodes(graph)

        contig_idx = 0

        for start_node in start_nodes:
            new_contigs = self._assemble_from_node(
                graph, start_node, visited_edges, contig_idx
            )
            contigs.extend(new_contigs)
            contig_idx += len(new_contigs)

        remaining_nodes = set()
        for from_n, to_n in graph._edge_counts:
            if (from_n, to_n) not in visited_edges:
                remaining_nodes.add(from_n)
                remaining_nodes.add(to_n)

        for node in remaining_nodes:
            new_contigs = self._assemble_from_node(
                graph, node, visited_edges, contig_idx
            )
            contigs.extend(new_contigs)
            contig_idx += len(new_contigs)

        contigs = [c for c in contigs if c.length >= self.min_contig_length]
        contigs.sort(key=lambda c: c.length, reverse=True)

        for i, contig in enumerate(contigs):
            contig.name = f"contig_{i:04d}"

        return contigs

    def _find_start_nodes(self, graph: DeBruijnGraph) -> List[int]:
        """寻找欧拉路径的起点节点
        
        起点的定义：
        1. 入度为 0 且出度 > 0 的节点（路径起点）
        2. 出度 > 入度的节点（分支起点）
        
        对于欧拉图（所有节点入度=出度），任选一个节点即可。
        """
        start_nodes = []

        for node in graph.nodes:
            out_deg = graph.out_degree(node)
            in_deg = graph.in_degree(node)

            if in_deg == 0 and out_deg > 0:
                start_nodes.append(node)
            elif out_deg > in_deg:
                start_nodes.append(node)

        if not start_nodes and graph.num_nodes > 0:
            for node in graph.nodes:
                if graph.out_degree(node) > 0:
                    start_nodes.append(node)
                    break

        start_nodes.sort(
            key=lambda n: graph.out_degree(n) - graph.in_degree(n),
            reverse=True,
        )

        return start_nodes

    def _assemble_from_node(
        self,
        graph: DeBruijnGraph,
        start_node: int,
        visited_edges: Set[Tuple[int, int]],
        start_idx: int,
    ) -> List[Contig]:
        """从指定节点出发组装 Contigs
        
        贪心策略：每次选择覆盖度最高的未访问边。
        """
        contigs = []

        out_neighbors = graph.out_neighbors(start_node)
        for to_node in sorted(
            out_neighbors.keys(),
            key=lambda n: out_neighbors[n],
            reverse=True,
        ):
            edge = (start_node, to_node)
            if edge in visited_edges:
                continue

            path_nodes, path_edges = self._extend_path(
                graph, start_node, to_node, visited_edges
            )

            if len(path_nodes) < 2:
                continue

            seq = graph.path_to_sequence(path_nodes)
            if len(seq) < self.min_contig_length:
                continue

            avg_depth = np.mean(path_edges) if path_edges else 0.0

            contig = Contig(
                name=f"contig_{start_idx + len(contigs):04d}",
                sequence=seq,
                length=len(seq),
                num_kmers=len(path_edges),
                avg_depth=float(avg_depth),
                path_nodes=path_nodes,
                is_circular=(path_nodes[0] == path_nodes[-1]),
            )
            contigs.append(contig)

        return contigs

    def _extend_path(
        self,
        graph: DeBruijnGraph,
        from_node: int,
        to_node: int,
        visited_edges: Set[Tuple[int, int]],
        max_steps: int = 1000000,
    ) -> Tuple[List[int], List[int]]:
        """从一条边出发，贪心扩展路径
        
        规则：
        - 如果当前节点只有一条未访问出边，继续扩展
        - 如果有多条未访问出边，选覆盖度最高的
        - 如果没有未访问出边，停止
        """
        path_nodes = [from_node, to_node]
        path_edges = [graph.get_edge_count(from_node, to_node)]
        visited_edges.add((from_node, to_node))

        current = to_node

        for _ in range(max_steps):
            out_neighbors = graph.out_neighbors(current)
            unvisited = [
                nb for nb in out_neighbors
                if (current, nb) not in visited_edges
            ]

            if not unvisited:
                break

            if len(unvisited) == 1:
                next_node = unvisited[0]
            elif self.prefer_high_depth:
                next_node = max(
                    unvisited,
                    key=lambda n: out_neighbors[n],
                )
            else:
                next_node = unvisited[0]

            edge = (current, next_node)
            if edge in visited_edges:
                break

            visited_edges.add(edge)
            path_edges.append(out_neighbors[next_node])
            path_nodes.append(next_node)
            current = next_node

        return path_nodes, path_edges

    def compute_assembly_stats(
        self, contigs: List[Contig]
    ) -> AssemblyStats:
        """计算组装统计指标
        
        包括 N50、N90、GC 含量等经典组装评估指标。
        """
        stats = AssemblyStats()
        stats.num_contigs = len(contigs)

        if not contigs:
            return stats

        lengths = [c.length for c in contigs]
        stats.total_bases = sum(lengths)
        stats.max_contig = max(lengths)
        stats.min_contig = min(lengths)
        stats.avg_contig_length = stats.total_bases / stats.num_contigs

        sorted_lengths = sorted(lengths, reverse=True)
        cumulative = 0
        half_total = stats.total_bases / 2
        ninety_percent = stats.total_bases * 0.9

        for length in sorted_lengths:
            cumulative += length
            if stats.n50 == 0 and cumulative >= half_total:
                stats.n50 = length
            if stats.n90 == 0 and cumulative >= ninety_percent:
                stats.n90 = length

        total_gc = 0
        total_at = 0
        for contig in contigs:
            seq = contig.sequence.upper()
            total_gc += seq.count('G') + seq.count('C')
            total_at += seq.count('A') + seq.count('T')

        total_bases_gc = total_gc + total_at
        if total_bases_gc > 0:
            stats.gc_content = total_gc / total_bases_gc

        return stats

    def write_fasta(
        self,
        contigs: List[Contig],
        output_path: str,
        wrap: int = 80,
    ) -> None:
        """将 Contigs 写入 FASTA 文件"""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "w") as f:
            for contig in contigs:
                f.write(contig.to_fasta(wrap=wrap))
