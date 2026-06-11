"""
示例 1：快速开始 - 基本的单条信号碱基识别

演示如何使用 BasecallPipeline 对模拟的纳米孔电流信号进行碱基识别。
"""

import numpy as np
from nanopore_core import BasecallPipeline


def generate_simulated_signal(length: int = 20000):
    """生成模拟的纳米孔电流信号"""
    np.random.seed(42)

    base_current = 100.0
    noise = np.random.normal(0, 5, length)
    signal = base_current + noise

    pattern_len = 40
    pattern = np.sin(np.linspace(0, 4 * np.pi, pattern_len)) * 15
    for i in range(0, length - pattern_len, pattern_len * 3):
        signal[i:i + pattern_len] += pattern

    spike_pos = np.random.choice(length, 100, replace=False)
    signal[spike_pos] += np.random.uniform(30, 60, 100)

    return signal.astype(np.float32)


def main():
    print("=" * 60)
    print("纳米孔碱基识别引擎 - 快速开始示例")
    print("=" * 60)

    print("\n1. 生成模拟电流信号...")
    signal = generate_simulated_signal(20000)
    print(f"   信号长度: {len(signal)} 样本")
    print(f"   电流范围: {signal.min():.2f} ~ {signal.max():.2f} pA")
    print(f"   平均电流: {signal.mean():.2f} pA")

    print("\n2. 创建碱基识别流水线...")
    pipeline = BasecallPipeline.default(device="cpu")
    print(f"   设备: {pipeline.device}")
    print(f"   模型参数量: {pipeline.model.count_parameters():,}")
    print(f"   归一化模式: {pipeline.normalize_mode}")
    print(f"   信号去噪: {'开启' if pipeline.denoise_signal else '关闭'}")
    print(f"   接头切除: {'开启' if pipeline.trim_adapter else '关闭'}")

    print("\n3. 执行碱基识别...")
    result = pipeline.basecall_single(signal, read_id="sim_read_001")

    print("\n4. 识别结果:")
    print(f"   Read ID: {result.read_id}")
    print(f"   原始信号长度: {result.signal_length:,} 样本")
    print(f"   识别碱基数目: {result.num_bases:,} bp")
    print(f"   平均置信度: {result.mean_confidence:.4f}")
    print(f"   处理时间: {result.process_time:.3f} 秒")
    print(f"   处理速度: {result.num_bases / result.process_time:.1f} bp/s")

    seq_preview = result.sequence[:80]
    if len(result.sequence) > 80:
        seq_preview += "..."
    print(f"\n   序列预览 (前80bp):")
    print(f"   {seq_preview}")

    print("\n5. FASTA 格式输出:")
    print(result.to_fasta()[:200])

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
