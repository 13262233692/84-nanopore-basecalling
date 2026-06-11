"""
碱基识别神经网络模型 - 基于 1D 卷积 + 双向 GRU 的声学模型

网络架构参考 Nanopore 社区主流设计（类似 Guppy/Bonito 风格）：
1. 多层 1D 卷积提取局部电流特征并降采样
2. 多层双向 GRU 建模长程序列依赖
3. 线性投影层输出碱基 + 空白符的对数概率矩阵

由于 DNA 过孔速度不均匀，同一碱基可能对应多个时间步的电流信号，
因此模型输出与标签序列之间不存在一一对应关系，需配合 CTC 训练和解码。

输出标签定义：
    0: 'A' (腺嘌呤)
    1: 'T' (胸腺嘧啶)
    2: 'C' (胞嘧啶)
    3: 'G' (鸟嘌呤)
    4: '-' (Blank 空白符)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """1D 卷积块 - Conv1d + BatchNorm + GELU
    
    带可选的下采样（stride > 1），用于压缩序列长度。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
            
        Returns:
            (batch, out_channels, seq_len / stride)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualConvBlock(nn.Module):
    """残差 1D 卷积块
    
    带残差连接，便于训练深层网络。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = ConvBlock(
            channels, channels, kernel_size, stride=1,
            dilation=dilation, dropout=dropout
        )
        self.conv2 = ConvBlock(
            channels, channels, kernel_size, stride=1,
            dilation=dilation, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class BidirectionalGRU(nn.Module):
    """双向 GRU 层
    
    封装单层双向 GRU，带层归一化和 dropout。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
            lengths: 各序列的实际长度，用于 PackedSequence
            
        Returns:
            (batch, seq_len, hidden_size * 2)
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True
            )
        else:
            x, _ = self.gru(x)

        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class BasecallerNet(nn.Module):
    """碱基识别神经网络
    
    将归一化后的电流信号转换为碱基概率的对数矩阵。
    
    架构：
        输入层 -> [Conv下采样] x N -> [双向GRU] x M -> 线性投影 -> LogSoftmax
    
    输入形状:
        (batch, 1, signal_len) - 单通道电流信号
    
    输出形状:
        (output_len, batch, 5) - 各时间步的碱基对数概率
        5 个类别: A, T, C, G, Blank
    """

    BASES = ["A", "T", "C", "G"]
    BLANK = "-"
    VOCAB = BASES + [BLANK]
    NUM_CLASSES = len(VOCAB)

    def __init__(
        self,
        conv_channels: list = None,
        conv_kernels: list = None,
        conv_strides: list = None,
        gru_hidden: int = 256,
        gru_layers: int = 3,
        dropout: float = 0.2,
        input_channels: int = 1,
    ):
        """
        Args:
            conv_channels: 各卷积层的输出通道数
            conv_kernels: 各卷积层的核大小
            conv_strides: 各卷积层的步长（决定下采样率）
            gru_hidden: GRU 隐藏层大小（单向）
            gru_layers: 双向 GRU 层数
            dropout: Dropout 概率
            input_channels: 输入通道数（默认单通道电流）
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]
        if conv_kernels is None:
            conv_kernels = [5, 5, 5, 5]
        if conv_strides is None:
            conv_strides = [2, 2, 2, 1]

        assert len(conv_channels) == len(conv_kernels) == len(conv_strides)

        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self._total_stride = 1
        for s in conv_strides:
            self._total_stride *= s

        conv_layers = []
        in_ch = input_channels
        for out_ch, kernel, stride in zip(conv_channels, conv_kernels, conv_strides):
            conv_layers.append(
                ConvBlock(in_ch, out_ch, kernel, stride=stride, dropout=dropout)
            )
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*conv_layers)

        gru_layers_list = []
        gru_input_size = conv_channels[-1]
        for _ in range(gru_layers):
            gru_layers_list.append(
                BidirectionalGRU(gru_input_size, gru_hidden, dropout=dropout)
            )
            gru_input_size = gru_hidden * 2
        self.gru_stack = nn.ModuleList(gru_layers_list)

        self.output_proj = nn.Linear(gru_hidden * 2, self.NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    @property
    def total_stride(self) -> int:
        """网络的总下采样倍率"""
        return self._total_stride

    def get_output_length(self, input_length: int) -> int:
        """根据输入长度计算输出序列长度
        
        Conv1d 输出长度公式:
        output_len = floor((input_len + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
        其中 padding = (kernel-1)*dilation // 2
        """
        length = input_length
        for kernel, stride in zip(self.conv_kernels, self.conv_strides):
            padding = (kernel - 1) // 2
            length = (length + 2 * padding - (kernel - 1) - 1) // stride + 1
        return length

    def get_output_lengths(
        self, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        """批量计算输出长度"""
        lengths = input_lengths.clone().float()
        for kernel, stride in zip(self.conv_kernels, self.conv_strides):
            padding = (kernel - 1) // 2
            lengths = torch.floor(
                (lengths + 2 * padding - (kernel - 1) - 1) / stride
            ) + 1
        return lengths.long()

    def forward(
        self,
        signal: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播
        
        Args:
            signal: (batch, 1, signal_len) 归一化后的电流信号
            lengths: (batch,) 各信号的实际长度（可选）
            
        Returns:
            (log_probs, output_lengths)
            - log_probs: (output_len, batch, num_classes) 对数概率矩阵
              形状为时间优先，符合 PyTorch CTCLoss 的要求
            - output_lengths: (batch,) 各输出序列的实际长度
        """
        batch_size = signal.shape[0]

        if lengths is not None:
            output_lengths = self.get_output_lengths(lengths)
        else:
            output_lengths = None

        x = self.conv_stack(signal)

        x = x.transpose(1, 2)

        for gru_layer in self.gru_stack:
            x = gru_layer(x, output_lengths)

        logits = self.output_proj(x)
        log_probs = self.log_softmax(logits)

        log_probs = log_probs.transpose(0, 1).contiguous()

        return log_probs, output_lengths

    def inference(
        self,
        signal: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """推理模式前向传播
        
        与 forward 相同，但确保 eval 模式。
        
        Args:
            signal: (batch, 1, signal_len)
            lengths: (batch,)
            
        Returns:
            log_probs: (output_len, batch, num_classes)
        """
        self.eval()
        with torch.no_grad():
            log_probs, _ = self.forward(signal, lengths)
        return log_probs

    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict) -> "BasecallerNet":
        """从配置字典创建模型"""
        return cls(
            conv_channels=config.get("conv_channels"),
            conv_kernels=config.get("conv_kernels"),
            conv_strides=config.get("conv_strides"),
            gru_hidden=config.get("gru_hidden", 256),
            gru_layers=config.get("gru_layers", 3),
            dropout=config.get("dropout", 0.2),
        )
