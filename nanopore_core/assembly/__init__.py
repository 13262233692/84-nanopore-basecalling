"""
De novo Assembly Module - 从头基因组组装模块

基于 De Bruijn 图的从头组装算法，将百万条测序 Reads
拼接成连续的重叠群 (Contigs)。

核心组件：
- kmer_util: K-mer 切分与编码工具
- de_bruijn_graph: De Bruijn 有向图数据结构
- graph_simplifier: 图简化（Tips 修剪、Bubbles 压缩）
- eulerian_path: 欧拉路径寻找与 Contig 拼接
- assembler: 端到端组装流水线
"""

from .kmer_util import KmerUtil
from .de_bruijn_graph import DeBruijnGraph
from .graph_simplifier import GraphSimplifier
from .eulerian_path import EulerianPathAssembler
from .assembler import DeNovoAssembler, AssemblyResult

__all__ = [
    "KmerUtil",
    "DeBruijnGraph",
    "GraphSimplifier",
    "EulerianPathAssembler",
    "DeNovoAssembler",
    "AssemblyResult",
]
