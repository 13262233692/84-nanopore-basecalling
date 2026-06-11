"""
K-mer 工具库

提供 K-mer 的切分、整数编码、反向互补等核心操作。
为了构建 De Bruijn 图，我们需要高效地：
1. 将长序列切分为 K 长度的子串
2. 将 K-mer 编码为整数以节省内存
3. 计算 K-1 mer 作为图的节点
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from numba import jit


BASE_TO_BIT = {"A": 0, "T": 1, "C": 2, "G": 3}
BIT_TO_BASE = ["A", "T", "C", "G"]
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _encode_kmer(kmer_str: str, k: int) -> int:
    """将 K-mer 字符串编码为整数（2 bit 编码）
    
    A=00, T=01, C=10, G=11
    每个碱基 2 bits，K-mer 编码为 2*K bits 的整数。
    """
    code = 0
    for c in kmer_str.upper():
        bit = BASE_TO_BIT.get(c, 0)
        code = (code << 2) | bit
    return code


def _decode_kmer(code: int, k: int) -> str:
    """将整数解码为 K-mer 字符串"""
    chars = []
    mask = 3
    for i in range(k - 1, -1, -1):
        bit = (code >> (2 * i)) & mask
        chars.append(BIT_TO_BASE[bit])
    return "".join(chars)


def _reverse_complement_code(code: int, k: int) -> int:
    """计算 K-mer 编码的反向互补编码"""
    rc = 0
    mask = 3
    for i in range(k):
        bit = (code >> (2 * i)) & mask
        comp = bit ^ 1
        rc = (rc << 2) | comp
    return rc


class KmerUtil:
    """K-mer 工具类
    
    提供 K-mer 相关的切分、编码、解码等操作。
    使用 2-bit 整数编码，大幅节省内存。
    """

    def __init__(self, k: int):
        """
        Args:
            k: K-mer 长度
        """
        if k < 3:
            raise ValueError("K-mer length must be at least 3")
        if k > 32:
            raise ValueError("K-mer length must be at most 32 (64-bit integer limit)")
        self.k = k
        self.k_minus_1 = k - 1
        self._mask = (1 << (2 * k)) - 1
        self._k1_mask = (1 << (2 * (k - 1))) - 1

    @property
    def kmer_length(self) -> int:
        return self.k

    def encode(self, kmer: str) -> int:
        """将 K-mer 字符串编码为整数
        
        Args:
            kmer: K-mer 字符串
            
        Returns:
            整数编码
        """
        if len(kmer) != self.k:
            raise ValueError(f"K-mer length must be {self.k}")
        return _encode_kmer(kmer, self.k)

    def decode(self, code: int) -> str:
        """将整数编码解码为 K-mer 字符串"""
        return _decode_kmer(code, self.k)

    def encode_node(self, node_str: str) -> int:
        """将 (K-1)-mer 节点字符串编码为整数"""
        if len(node_str) != self.k_minus_1:
            raise ValueError(f"Node length must be {self.k_minus_1}")
        return _encode_kmer(node_str, self.k_minus_1)

    def decode_node(self, code: int) -> str:
        """将整数编码解码为 (K-1)-mer 节点字符串"""
        return _decode_kmer(code, self.k_minus_1)

    def reverse_complement(self, kmer: str) -> str:
        """计算 K-mer 的反向互补序列"""
        return "".join(COMPLEMENT.get(c, "N") for c in reversed(kmer))

    def reverse_complement_code(self, code: int) -> int:
        """计算 K-mer 编码的反向互补编码"""
        return _reverse_complement_code(code, self.k)

    def get_prefix(self, kmer_code: int) -> int:
        """获取 K-mer 的 K-1 前缀（作为 De Bruijn 图的源节点）
        
        即去掉最后一个碱基的 K-1 mer
        """
        return (kmer_code >> 2) & self._k1_mask

    def get_suffix(self, kmer_code: int) -> int:
        """获取 K-mer 的 K-1 后缀（作为 De Bruijn 图的目标节点）
        
        即去掉第一个碱基的 K-1 mer
        """
        return kmer_code & self._k1_mask

    def extend_right(self, node_code: int, base_bit: int) -> int:
        """从 (K-1)-mer 节点向右扩展一个碱基，得到下一个 (K-1)-mer
        
        Args:
            node_code: (K-1)-mer 编码
            base_bit: 新加入的碱基编码 (0-3)
            
        Returns:
            新的 (K-1)-mer 编码
        """
        return ((node_code << 2) | base_bit) & self._k1_mask

    def iter_kmers(self, sequence: str) -> List[int]:
        """将序列切分为 K-mer 编码列表
        
        Args:
            sequence: DNA 序列
            
        Returns:
            K-mer 整数编码列表
        """
        n = len(sequence)
        if n < self.k:
            return []

        kmers = []
        code = 0

        for i in range(self.k - 1):
            base = sequence[i]
            bit = BASE_TO_BIT.get(base.upper(), 0)
            code = (code << 2) | bit

        for i in range(self.k - 1, n):
            base = sequence[i]
            bit = BASE_TO_BIT.get(base.upper(), 0)
            code = ((code << 2) | bit) & self._mask
            kmers.append(code)

        return kmers

    def iter_kmers_with_count(
        self,
        sequence: str,
        count_dict: Optional[Dict[int, int]] = None,
    ) -> Dict[int, int]:
        """统计序列中各 K-mer 的出现次数
        
        Args:
            sequence: DNA 序列
            count_dict: 已有的计数字典，用于累加
            
        Returns:
            K-mer 出现次数字典
        """
        if count_dict is None:
            count_dict = {}

        n = len(sequence)
        if n < self.k:
            return count_dict

        code = 0

        for i in range(self.k - 1):
            base = sequence[i]
            bit = BASE_TO_BIT.get(base.upper(), 0)
            code = (code << 2) | bit

        for i in range(self.k - 1, n):
            base = sequence[i]
            bit = BASE_TO_BIT.get(base.upper(), 0)
            code = ((code << 2) | bit) & self._mask
            count_dict[code] = count_dict.get(code, 0) + 1

        return count_dict

    def kmer_count_dict(
        self,
        sequences: List[str],
        min_count: int = 1,
    ) -> Dict[int, int]:
        """批量统计多条序列中的 K-mer 频次
        
        Args:
            sequences: DNA 序列列表
            min_count: 最小出现次数，低于此值的 K-mer 会被过滤
            
        Returns:
            过滤后的 K-mer 频次
        """
        counts: Dict[int, int] = {}
        for seq in sequences:
            self.iter_kmers_with_count(seq, counts)

        if min_count > 1:
            filtered = {k: v for k, v in counts.items() if v >= min_count}
            return filtered

        return counts

    def canonical_kmer(self, kmer_code: int) -> int:
        """获取正则 K-mer（K-mer 和反向互补中较小的那个）
        
        用于构建有向图时，正向和反向互补的 K-mer 被视为同一节点的两个方向。
        """
        rc = self.reverse_complement_code(kmer_code)
        return min(kmer_code, rc)

    def is_canonical(self, kmer_code: int) -> bool:
        """判断是否为正则 K-mer"""
        return kmer_code <= self.reverse_complement_code(kmer_code)

    def kmer_to_edges(self, kmer_code: int) -> Tuple[int, int]:
        """将 K-mer 转换为 De Bruijn 图的一条边 (from_node, to_node)
        
        from_node 是 K-1 前缀，to_node 是 K-1 后缀。
        """
        from_node = self.get_prefix(kmer_code)
        to_node = self.get_suffix(kmer_code)
        return from_node, to_node

    def node_to_kmer(self, from_node: int, to_node: int) -> int:
        """根据相邻节点重建 K-mer
        
        Args:
            from_node: 源 (K-1)-mer
            to_node: 目标 (K-1)-mer
            
        Returns:
            K-mer 编码
        """
        last_base = to_node & 3
        return (from_node << 2) | last_base

    def __repr__(self) -> str:
        return f"KmerUtil(k={self.k})"
