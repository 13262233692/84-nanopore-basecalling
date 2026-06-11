"""
图简化算法 - 悬挂分支 (Tips) 修剪与气泡 (Bubbles) 压缩

纳米孔测序存在较高的错误率，导致 De Bruijn 图中存在大量
由测序错误引入的噪声结构：

1. Tips (悬挂分支) - 从主路径分出后很快终止的短分支
   - 通常由序列末端的测序错误产生
   - 长度较短（< K*2）、覆盖度低
   
2. Bubbles (气泡) - 两个节点之间存在两条或多条平行路径
   - 通常由单核苷酸多态性 (SNP) 或局部测序错误产生
   - 两侧共享同一个起点和终点节点

简化策略：
- 迭代修剪 Tips，直到图中不再有新的 Tip 产生
- 检测并压缩 Bubbles，保留覆盖度最高的路径
"""

import numpy as np
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass

from .de_bruijn_graph import DeBruijnGraph


@dataclass
class SimplificationStats:
    """图简化统计信息"""
    tips_removed: int = 0
    tips_iterations: int = 0
    bubbles_removed: int = 0
    edges_removed: int = 0
    nodes_removed: int = 0


@dataclass
class Bubble:
    """气泡结构"""
    start_node: int
    end_node: int
    paths: List[List[int]]
    path_counts: List[int]
    best_path_idx: int


class GraphSimplifier:
    """De Bruijn 图简化器
    
    实现 Tip 修剪和 Bubble 压缩两种核心简化算法。
    """

    def __init__(
        self,
        tip_max_length: int = 10,
        tip_min_depth: float = 2.0,
        bubble_max_length: int = 100,
        bubble_max_paths: int = 4,
    ):
        """
        Args:
            tip_max_length: 悬挂分支的最大长度（节点数）
            tip_min_depth: 低于此深度的分支才会被修剪
            bubble_max_length: 气泡的最大路径长度
            bubble_max_paths: 气泡中最多的路径数
        """
        self.tip_max_length = tip_max_length
        self.tip_min_depth = tip_min_depth
        self.bubble_max_length = bubble_max_length
        self.bubble_max_paths = bubble_max_paths

    def simplify(
        self,
        graph: DeBruijnGraph,
        max_iterations: int = 10,
    ) -> SimplificationStats:
        """执行完整的图简化
        
        先修剪 Tips，再压缩 Bubbles，反复迭代直到图稳定。
        
        Args:
            graph: De Bruijn 图（原地修改）
            max_iterations: 最大迭代次数
            
        Returns:
            简化统计信息
        """
        stats = SimplificationStats()

        for iteration in range(max_iterations):
            tips_before = stats.tips_removed
            self.trim_tips(graph, stats)

            bubbles_before = stats.bubbles_removed
            self.compress_bubbles(graph, stats)

            if stats.tips_removed == tips_before and stats.bubbles_removed == bubbles_before:
                stats.tips_iterations = iteration + 1
                break

            stats.tips_iterations = iteration + 1

        return stats

    def trim_tips(
        self,
        graph: DeBruijnGraph,
        stats: Optional[SimplificationStats] = None,
    ) -> int:
        """修剪悬挂分支 (Tips)

        策略：
        1. 找出所有死胡同（出度为 0）和源头（入度为 0）
        2. 沿唯一路径追溯，忽略自环，找到分支点或汇合点
        3. 如果分支短且覆盖度低，则删除整个分支

        Args:
            graph: De Bruijn 图
            stats: 统计对象

        Returns:
            移除的 Tip 数量
        """
        if stats is None:
            stats = SimplificationStats()

        tips_removed = 0
        tip_nodes_to_remove = set()
        processed = set()

        for node in graph.nodes:
            if node in processed:
                continue

            is_dead_end = graph.out_degree(node) == 0 and self._in_degree_no_self(graph, node) > 0
            is_source = graph.in_degree(node) == 0 and self._out_degree_no_self(graph, node) > 0

            if not is_dead_end and not is_source:
                continue

            path_nodes = []
            path_edge_counts = []

            if is_dead_end:
                self._walk_backward(graph, node, path_nodes, path_edge_counts, self.tip_max_length + 2)
            else:
                self._walk_forward(graph, node, path_nodes, path_edge_counts, self.tip_max_length + 2)

            if len(path_nodes) <= 1:
                continue

            tip_len = len(path_nodes) - 1
            if tip_len > self.tip_max_length:
                continue

            if not path_edge_counts:
                continue

            avg_depth = float(np.mean(path_edge_counts[:tip_len]))
            if avg_depth > self.tip_min_depth:
                continue

            if is_dead_end:
                root = path_nodes[0]
                root_branching = self._out_degree_no_self(graph, root) > 1
                root_isolated = self._in_degree_no_self(graph, root) == 0
            else:
                root = path_nodes[-1]
                root_branching = self._in_degree_no_self(graph, root) > 1
                root_isolated = self._out_degree_no_self(graph, root) == 0

            if not root_branching and not root_isolated:
                continue

            if root_isolated:
                nodes_to_del = list(path_nodes)
            else:
                if is_dead_end:
                    nodes_to_del = path_nodes[1:]
                else:
                    nodes_to_del = path_nodes[:-1]

            for n in nodes_to_del:
                if n in graph.nodes:
                    tip_nodes_to_remove.add(n)
                    processed.add(n)

            stats.edges_removed += tip_len
            tips_removed += 1
            stats.tips_removed += 1

        for node in tip_nodes_to_remove:
            if node in graph.nodes:
                graph.remove_node(node)
                stats.nodes_removed += 1

        return tips_removed

    def _in_degree_no_self(self, graph: DeBruijnGraph, node: int) -> int:
        """计算入度（忽略自环）"""
        count = 0
        for src in graph._in_edges.get(node, {}).keys():
            if src != node:
                count += 1
        return count

    def _out_degree_no_self(self, graph: DeBruijnGraph, node: int) -> int:
        """计算出度（忽略自环）"""
        count = 0
        for dst in graph._out_edges.get(node, {}).keys():
            if dst != node:
                count += 1
        return count

    def _walk_backward(
        self,
        graph: DeBruijnGraph,
        start_node: int,
        path_nodes: List[int],
        path_edge_counts: List[int],
        max_steps: int,
    ) -> None:
        """沿入边向后走，忽略自环，得到路径 [root, ..., start_node]"""
        current = start_node
        path_nodes.append(current)

        for _ in range(max_steps):
            in_neighbors = [n for n in graph._in_edges.get(current, {}).keys() if n != current]
            if len(in_neighbors) != 1:
                break

            prev_node = in_neighbors[0]
            if prev_node in path_nodes:
                break

            path_edge_counts.append(graph._in_edges[current][prev_node])
            path_nodes.insert(0, prev_node)
            current = prev_node

    def _walk_forward(
        self,
        graph: DeBruijnGraph,
        start_node: int,
        path_nodes: List[int],
        path_edge_counts: List[int],
        max_steps: int,
    ) -> None:
        """沿出边向前走，忽略自环，得到路径 [start_node, ..., end]"""
        current = start_node
        path_nodes.append(current)

        for _ in range(max_steps):
            out_neighbors = [n for n in graph._out_edges.get(current, {}).keys() if n != current]
            if len(out_neighbors) != 1:
                break

            next_node = out_neighbors[0]
            if next_node in path_nodes:
                break

            path_edge_counts.append(graph._out_edges[current][next_node])
            path_nodes.append(next_node)
            current = next_node

    def detect_bubbles(
        self,
        graph: DeBruijnGraph,
    ) -> List[Bubble]:
        """检测图中的气泡结构
        
        气泡定义：
        - 存在一个分支节点（出度 >= 2）
        - 多条路径在汇合节点重新汇合
        - 所有路径长度不超过 bubble_max_length
        
        Returns:
            气泡列表
        """
        bubbles = []
        visited_starts = set()

        for node in graph.nodes:
            if graph.out_degree(node) < 2:
                continue
            if node in visited_starts:
                continue

            bubble = self._find_bubble_from(graph, node)
            if bubble is not None:
                bubbles.append(bubble)
                visited_starts.add(node)

        return bubbles

    def _find_bubble_from(
        self,
        graph: DeBruijnGraph,
        start_node: int,
    ) -> Optional[Bubble]:
        """从分支节点出发寻找气泡
        
        使用 BFS 搜索从分支节点出发的所有路径，
        找到第一个汇合节点即为气泡终点。
        """
        out_neighbors = list(graph.out_neighbors(start_node).keys())
        if len(out_neighbors) < 2:
            return None

        visited: Dict[int, Tuple[int, List[int]]] = {}

        queue = []
        for nb in out_neighbors:
            queue.append((nb, [start_node, nb]))

        end_node = None
        paths = []
        max_depth = self.bubble_max_length

        for depth in range(max_depth):
            next_queue = []
            current_level: Dict[int, List[List[int]]] = {}

            for node, path in queue:
                if node not in current_level:
                    current_level[node] = []
                current_level[node].append(path)

                if len(current_level[node]) >= 2:
                    end_node = node
                    paths = current_level[node]
                    break

            if end_node is not None:
                break

            for node, paths_here in current_level.items():
                if node == start_node:
                    continue

                for path in paths_here:
                    for nb in graph.out_neighbors(node).keys():
                        if nb == start_node:
                            new_path = path + [nb]
                            next_queue.append((nb, new_path))
                        elif nb not in path:
                            new_path = path + [nb]
                            next_queue.append((nb, new_path))

            queue = next_queue
            if not queue:
                break

        if end_node is None or len(paths) < 2:
            return None

        path_counts = []
        for path in paths:
            total = 0
            for i in range(len(path) - 1):
                total += graph.get_edge_count(path[i], path[i + 1])
            path_counts.append(total)

        best_idx = int(np.argmax(path_counts))

        return Bubble(
            start_node=start_node,
            end_node=end_node,
            paths=paths,
            path_counts=path_counts,
            best_path_idx=best_idx,
        )

    def compress_bubbles(
        self,
        graph: DeBruijnGraph,
        stats: Optional[SimplificationStats] = None,
    ) -> int:
        """压缩气泡结构
        
        对于每个气泡，保留覆盖度最高的路径，删除其他路径。
        
        Args:
            graph: De Bruijn 图
            stats: 统计对象
            
        Returns:
            压缩的气泡数量
        """
        if stats is None:
            stats = SimplificationStats()

        bubbles_removed = 0
        bubbles = self.detect_bubbles(graph)

        for bubble in bubbles:
            if len(bubble.paths) < 2:
                continue

            best_idx = bubble.best_path_idx
            best_path = bubble.paths[best_idx]

            for i, path in enumerate(bubble.paths):
                if i == best_idx:
                    continue

                for j in range(len(path) - 1):
                    from_n = path[j]
                    to_n = path[j + 1]
                    if graph.get_edge_count(from_n, to_n) > 0:
                        graph.remove_edge(from_n, to_n)
                        stats.edges_removed += 1

            for node in graph.nodes.copy():
                if graph.out_degree(node) == 0 and graph.in_degree(node) == 0:
                    if node not in best_path:
                        graph.remove_node(node)
                        stats.nodes_removed += 1

            bubbles_removed += 1
            stats.bubbles_removed += 1

        return bubbles_removed

    def remove_low_coverage_edges(
        self,
        graph: DeBruijnGraph,
        min_count: int = 2,
        stats: Optional[SimplificationStats] = None,
    ) -> int:
        """移除低覆盖度的边
        
        Args:
            graph: De Bruijn 图
            min_count: 最小覆盖度阈值
            stats: 统计对象
            
        Returns:
            移除的边数
        """
        if stats is None:
            stats = SimplificationStats()

        edges_to_remove = []
        for (from_n, to_n), count in graph._edge_counts.items():
            if count < min_count:
                edges_to_remove.append((from_n, to_n))

        for from_n, to_n in edges_to_remove:
            graph.remove_edge(from_n, to_n)
            stats.edges_removed += 1

        return len(edges_to_remove)

    def remove_low_coverage_nodes(
        self,
        graph: DeBruijnGraph,
        min_average_depth: float = 1.0,
        stats: Optional[SimplificationStats] = None,
    ) -> int:
        """移除低覆盖度的孤立节点
        
        Args:
            graph: De Bruijn 图
            min_average_depth: 最小平均深度
            stats: 统计对象
            
        Returns:
            移除的节点数
        """
        if stats is None:
            stats = SimplificationStats()

        nodes_to_remove = []
        for node in graph.nodes:
            in_counts = list(graph.in_neighbors(node).values())
            out_counts = list(graph.out_neighbors(node).values())
            all_counts = in_counts + out_counts

            if not all_counts:
                nodes_to_remove.append(node)
                continue

            avg_depth = np.mean(all_counts)
            if avg_depth < min_average_depth:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            graph.remove_node(node)
            stats.nodes_removed += 1

        return len(nodes_to_remove)
