"""
单元测试 - 碱基识别神经网络模型
"""

import torch
import pytest
from nanopore_core.model.basecaller_net import BasecallerNet, ConvBlock, BidirectionalGRU


class TestConvBlock:
    def test_forward_shape(self):
        block = ConvBlock(1, 32, kernel_size=5, stride=1)
        x = torch.randn(2, 1, 100)
        out = block(x)
        assert out.shape == (2, 32, 100)

    def test_forward_stride(self):
        block = ConvBlock(16, 32, kernel_size=5, stride=2)
        x = torch.randn(2, 16, 100)
        out = block(x)
        assert out.shape == (2, 32, 50)

    def test_forward_dilation(self):
        block = ConvBlock(8, 16, kernel_size=3, dilation=2)
        x = torch.randn(2, 8, 100)
        out = block(x)
        assert out.shape == (2, 16, 100)


class TestBidirectionalGRU:
    def test_forward_shape(self):
        gru = BidirectionalGRU(64, 128)
        x = torch.randn(2, 50, 64)
        out = gru(x)
        assert out.shape == (2, 50, 256)

    def test_forward_with_lengths(self):
        gru = BidirectionalGRU(64, 128)
        x = torch.randn(2, 50, 64)
        lengths = torch.tensor([50, 30], dtype=torch.int32)
        out = gru(x, lengths)
        assert out.shape == (2, 50, 256)


class TestBasecallerNet:
    def test_default_creation(self):
        model = BasecallerNet()
        assert model is not None
        assert model.NUM_CLASSES == 5

    def test_forward_shape(self):
        model = BasecallerNet()
        batch_size = 2
        signal_len = 2000
        signal = torch.randn(batch_size, 1, signal_len)
        lengths = torch.tensor([signal_len, signal_len], dtype=torch.int32)

        log_probs, out_lengths = model(signal, lengths)

        assert log_probs.shape[0] == model.get_output_length(signal_len)
        assert log_probs.shape[1] == batch_size
        assert log_probs.shape[2] == 5
        assert out_lengths.shape[0] == batch_size

    def test_log_probabilities(self):
        model = BasecallerNet()
        signal = torch.randn(1, 1, 1000)
        log_probs, _ = model(signal)

        probs = torch.exp(log_probs)
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    def test_get_output_length(self):
        model = BasecallerNet()
        assert model.get_output_length(1000) == 1000 // model.total_stride

    def test_get_output_lengths(self):
        model = BasecallerNet()
        lengths = torch.tensor([1000, 500, 2000], dtype=torch.int32)
        out_lengths = model.get_output_lengths(lengths)
        expected = [
            model.get_output_length(1000),
            model.get_output_length(500),
            model.get_output_length(2000),
        ]
        assert list(out_lengths.numpy()) == expected

    def test_total_stride(self):
        model = BasecallerNet(
            conv_channels=[32, 64],
            conv_kernels=[5, 5],
            conv_strides=[2, 2],
        )
        assert model.total_stride == 4

    def test_count_parameters(self):
        model = BasecallerNet()
        n_params = model.count_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_inference_mode(self):
        model = BasecallerNet()
        signal = torch.randn(1, 1, 500)

        log_probs_train = model.inference(signal)
        assert not model.training
        assert log_probs_train.shape[0] == model.get_output_length(500)

    def test_from_config(self):
        config = {
            "conv_channels": [16, 32],
            "conv_kernels": [3, 3],
            "conv_strides": [2, 2],
            "gru_hidden": 64,
            "gru_layers": 2,
            "dropout": 0.1,
        }
        model = BasecallerNet.from_config(config)
        assert model.gru_hidden == 64
        assert model.gru_layers == 2
        assert len(model.conv_stack) == 2

    def test_vocab_constants(self):
        assert BasecallerNet.BASES == ["A", "T", "C", "G"]
        assert BasecallerNet.BLANK == "-"
        assert len(BasecallerNet.VOCAB) == 5

    def test_batches_various_lengths(self):
        model = BasecallerNet()
        signal = torch.randn(4, 1, 2000)
        lengths = torch.tensor([2000, 1500, 1000, 500], dtype=torch.int32)

        log_probs, out_lengths = model(signal, lengths)

        assert out_lengths[0] > out_lengths[1] > out_lengths[2] > out_lengths[3]

    def test_gradients_flow(self):
        model = BasecallerNet()
        signal = torch.randn(2, 1, 500)

        log_probs, _ = model(signal)
        loss = log_probs.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
