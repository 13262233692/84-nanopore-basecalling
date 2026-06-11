"""
De Bruijn 有向图数据结构

节点 = (K-1)-mer
边  = K-mer（连接前缀节点到后缀节点）
边权重 = K-mer 出现频次（测序覆盖度）

图的性质：
- 每个节点最多 4 个出边（A/T/C/G 四种碱基扩展）
- 每个节点最多 4 个入边
- 欧拉路径：经过每条边恰好一次
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .kmer_util import KmerUtil, BIT_TO_BASE


@dataclass
class GraphStats:
    """De Bruijn 图统计信息"""
    k: int
    num_nodes: int = 0
    num_edges: int = 0
    num_tips: int = 0
    num_bubbles: int = 0
    total_kmer_count: int = 0
    avg_kmer_depth: float = 0.0
    max_node_degree: int = 0


class DeBruijnGraph:
    """De Bruijn 有向图
    
    使用邻接表表示法存储图，支持高效的邻居查询和图遍历。
    """

    def __init__(self, k: int):
        """
        Args:
            k: K-mer 长度
        """
        self.kmer_util = KmerUtil(k)
        self.k = k

        self._out_edges: Dict[int, Dict[int, int]] = defaultdict(dict)
        self._in_edges: Dict[int, Dict[int, int]] = defaultdict(dict)
        self._edge_counts: Dict[Tuple[int, int], int] = {}

        self._nodes: Set[int] = set()

    @property
    def kmer_length(self) -> int:
        return self.k

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edge_counts)

    @property
    def nodes(self) -> Set[int]:
        return self._nodes.copy()

    def add_node(self, node_code: int) -> None:
        """添加节点"""
        self._nodes.add(node_code)

    def add_edge(
        self,
        from_node: int,
        to_node: int,
        count: int = 1,
    ) -> None:
        """添加边（from_node -> to_node）
        
        Args:
            from_node: 源节点 (K-1 mer)
            to_node: 目标节点 (K-1 mer)
            count: 边的权重（K-mer 出现次数）
        """
        self._nodes.add(from_node)
        self._nodes.add(to_node)

        key = (from_node, to_node)
        current = self._edge_counts.get(key, 0)
        self._edge_counts[key] = current + count

        self._out_edges[from_node][to_node] = self._edge_counts[key]
        self._in_edges[to_node][from_node] = self._edge_counts[key]

    def add_kmer(self, kmer_code: int, count: int = 1) -> None:
        """添加 K-mer 作为一条边
        
        Args:
            kmer_code: K-mer 整数编码
            count: 出现次数
        """
        from_node = self.kmer_util.get_prefix(kmer_code)
        to_node = self.kmer_util.get_suffix(kmer_code)
        self.add_edge(from_node, to_node, count)

    def add_sequence(self, sequence: str) -> None:
        """从 DNA 序列构建图
        
        将序列切分为 K-mers，每条 K-mer 作为一条边加入图中。
        """
        kmers = self.kmer_util.iter_kmers(sequence)
        for kmer in kmers:
            self.add_kmer(kmer, 1)

    def add_sequences(self, sequences: List[str]) -> None:
        """批量添加序列"""
        for seq in sequences:
            self.add_sequence(seq)

    def remove_edge(self, from_node: int, to_node: int) -> bool:
        """移除一条边
        
        Returns:
            是否成功移除
        """
        key = (from_node, to_node)
        if key not in self._edge_counts:
            return False

        del self._edge_counts[key]

        if from_node in self._out_edges:
            self._out_edges[from_node].pop(to_node, None)
            if not self._out_edges[from_node]:
                del self._out_edges[from_node]

        if to_node in self._in_edges:
            self._in_edges[to_node].pop(from_node, None)
            if not self._in_edges[to_node]:
                del self._in_edges[to_node]

        return True

    def remove_node(self, node_code: int) -> bool:
        """移除一个节点及其所有边"""
        if node_code not in self._nodes:
            return False

        out_neighbors = list(self._out_edges.get(node_code, {}).keys())
        for to_node in out_neighbors:
            self.remove_edge(node_code, to_node)

        in_neighbors = list(self._in_edges.get(node_code, {}).keys())
        for from_node in in_neighbors:
            self.remove_edge(from_node, node_code)

        self._nodes.discard(node_code)
        return True

    def out_degree(self, node_code: int) -> int:
        """获取节点的出度"""
        return len(self._out_edges.get(node_code, {}))

    def in_degree(self, node_code: int) -> int:
        """获取节点的入度"""
        return len(self._in_edges.get(node_code, {}))

    def out_neighbors(self, node_code: int) -> Dict[int, int]:
        """获取出邻居节点及边权重
        
        Returns:
            {node_code: edge_count} 字典
        """
        return dict(self._out_edges.get(node_code, {}))

    def in_neighbors(self, node_code: int) -> Dict[int, int]:
        """获取入邻居节点及边权重"""
        return dict(self._in_edges.get(node_code, {}))

    def get_edge_count(self, from_node: int, to_node: int) -> int:
        """获取边的权重（K-mer 出现次数）"""
        return self._edge_counts.get((from_node, to_node), 0)

    def is_tip_start(self, node_code: int, max_depth: int = 5) -> bool:
        """判断节点是否在悬挂分支（Tip）上（从起点正向判断）
        
        Tip 定义：从该节点出发，沿唯一出边向前走，
        在有限步数内到达死胡同（出度为 0）。
        """
        depth = 0
        current = node_code
        while depth <= max_depth:
            if self.out_degree(current) == 0:
                return True
            if self.out_degree(current) > 1:
                return False
            current = list(self._out_edges[current].keys())[0]
            depth += 1
        return False

    def is_tip_end(self, node_code: int, max_depth: int = 5) -> bool:
        """判断节点是否在悬挂分支（Tip）上（从终点反向判断）
        
        Tip 定义：从该节点出发，沿唯一入边向后走，
        在有限步数内到达源头（入度为 0）。
        """
        depth = 0
        current = node_code
        while depth <= max_depth:
            if self.in_degree(current) == 0:
                return True
            if self.in_degree(current) > 1:
                return False
            current = list(self._in_edges[current].keys())[0]
            depth += 1
        return False

    def get_path_forward(
        self, start_node: int, max_length: int = 1000
    ) -> Tuple[List[int], List[int]]:
        """沿唯一路径向前遍历
        
        在每个节点只有一个出边的情况下，沿路径向前遍历。
        
        Args:
            start_node: 起始节点
            max_length: 最大路径长度
            
        Returns:
            (nodes, edges) 节点列表和边权重列表
        """
        nodes = [start_node]
        edge_weights = []
        current = start_node

        for _ in range(max_length):
            if self.out_degree(current) != 1:
                break
            next_node = list(self._out_edges[current].keys())[0]
            edge_weights.append(self._out_edges[current][next_node])
            nodes.append(next_node)
            current = next_node

            if current == start_node:
                break

        return nodes, edge_weights

    def get_path_backward(
        self, end_node: int, max_length: int = 1000
    ) -> Tuple[List[int], List[int]]:
        """沿唯一路径向后遍历"""
        nodes = [end_node]
        edge_weights = []
        current = end_node

        for _ in range(max_length):
            if self.in_degree(current) != 1:
                break
            prev_node = list(self._in_edges[current].keys())[0]
            edge_weights.append(self._in_edges[current][prev_node])
            nodes.insert(0, prev_node)
            current = prev_node

            if current == end_node:
                break

        return nodes, edge_weights

    def get_stats(self) -> GraphStats:
        """获取图的统计信息"""
        total_count = sum(self._edge_counts.values())
        avg_depth = total_count / len(self._edge_counts) if self._edge_counts else 0

        max_deg = 0
        for node in self._nodes:
            deg = self.out_degree(node) + self.in_degree(node)
            if deg > max_deg:
                max_deg = deg

        return GraphStats(
            k=self.k,
            num_nodes=len(self._nodes),
            num_edges=len(self._edge_counts),
            total_kmer_count=total_count,
            avg_kmer_depth=avg_depth,
            max_node_degree=max_deg,
        )

    def node_to_sequence(self, node_code: int) -> str:
        """将节点编码转换为 (K-1)-mer 字符串"""
        return self.kmer_util.decode_node(node_code)

    def path_to_sequence(self, node_path: List[int]) -> str:
        """将节点路径转换为 DNA 序列
        
        路径中相邻节点重叠 K-2 个碱基，所以序列长度 = K-1 + len(path)-1 = len(path) + K - 2
        """
        if not node_path:
            return ""

        first_node = self.kmer_util.decode_node(node_path[0])
        if len(node_path) == 1:
            return first_node

        suffixes = []
        for node in node_path[1:]:
            last_base = node & 3
            suffixes.append(BIT_TO_BASE[last_base])

        return first_node + "".join(suffixes)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DeBruijnGraph(k={self.k}, nodes={stats.num_nodes}, "
            f"edges={stats.num_edges}, avg_depth={stats.avg_kmer_depth:.2f})"
        )
