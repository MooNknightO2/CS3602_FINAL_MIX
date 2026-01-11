"""
测试 mixSampling 模块 - KV Cache + Speculative Decoding
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from mixSampling import (
    MixSamplingConfig,
    mixSampling,
    mixSampling_adaptive,
    mixSampling_simple,
    specSampling_press,
)
import gc


def compare_with_baseline():
    """与基准方法对比"""
    from specSampling import specSampling
    from regrSampling import regrSampling
    
    print("=" * 70)
    print("性能对比测试: KV Press + Speculative Decoding")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    q_model = AutoModelForCausalLM.from_pretrained(
        "models/pythia-70m", device_map="auto", dtype=torch.float16
    )
    p_model = AutoModelForCausalLM.from_pretrained(
        "models/pythia-2.8b", device_map="auto", dtype=torch.float16
    )
    q_model.eval()
    p_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("models/pythia-2.8b")
    
    prompt = "He redid the main theme"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    max_new_tokens = 1000
    
    results = {}
    outputs = {}
    
    print(f"\n输入: {prompt}")
    print(f"生成 {max_new_tokens} 个新 tokens\n")

    # 1. 自适应混合采样
    print("\n" + "-" * 50)
    print("1. mixSampling_adaptive - 自适应 gamma")
    print("-" * 50)
    config_adaptive = MixSamplingConfig(
        gamma=4,
        compression_ratio=0.7,
        press_type="streaming",
        apply_to_target=True,
        apply_to_draft=True
    )
    gc.disable()
    start = time.time()
    output6 = mixSampling_adaptive(input_ids.clone(), q_model, p_model, max_new_tokens, config_adaptive)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.time() - start
    gc.enable()
    tpot = total / max_new_tokens * 1000
    throughput = max_new_tokens / total
    results["mixSampling_adaptive"] = (tpot, throughput)
    outputs["mixSampling_adaptive"] = output6
    print(f"   TPOT: {tpot:.2f} ms/token")
    print(f"   吞吐量: {throughput:.2f} tokens/s")

    # 2. KV Press + Speculative Decoding（仅 KV 压缩，无增量 cache）
    print("\n" + "-" * 50)
    print("2. specSampling_press - 仅 KV Press")
    print("-" * 50)
    config_press_only = MixSamplingConfig(
        gamma=4,
        compression_ratio=0.7,
        press_type="streaming",
        apply_to_target=True,
        apply_to_draft=False,
        num_sink_tokens=4
    )
    gc.disable()
    start = time.time()
    output3 = specSampling_press(input_ids.clone(), q_model, p_model, max_new_tokens, config_press_only)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.time() - start
    gc.enable()
    tpot = total / max_new_tokens * 1000
    throughput = max_new_tokens / total
    results["specSampling_press"] = (tpot, throughput)
    outputs["specSampling_press"] = output3
    print(f"   TPOT: {tpot:.2f} ms/token")
    print(f"   吞吐量: {throughput:.2f} tokens/s")
    


    # 3. 原始 Speculative Decoding（无优化）
    print("\n" + "-" * 50)
    print("3. Speculative Decoding (specSampling) - 原始实现")
    print("-" * 50)
    gc.disable()
    start = time.time()
    output2 = specSampling(input_ids.clone(), q_model, p_model, max_new_tokens, gamma=4)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.time() - start
    gc.enable()
    tpot = total / max_new_tokens * 1000
    throughput = max_new_tokens / total
    results["specSampling"] = (tpot, throughput)
    outputs["specSampling"] = output2
    print(f"   TPOT: {tpot:.2f} ms/token")
    print(f"   吞吐量: {throughput:.2f} tokens/s")

    # 打印对比结果
    print("\n" + "=" * 70)
    print("性能对比总结")
    print("=" * 70)
    
    # 使用 specSampling 作为基准
    baseline_throughput = results["specSampling"][1]
    
    print(f"\n{'方法':<30} {'TPOT (ms)':<12} {'吞吐量':<15} {'vs specSampling':<15}")
    print("-" * 75)
    
    for name, (tpot, throughput) in results.items():
        speedup = throughput / baseline_throughput if baseline_throughput > 0 else 0
        print(f"{name:<30} {tpot:<12.2f} {throughput:<15.2f} {speedup:<15.2f}x")
    
    # 显示生成的文本示例
    print("\n" + "=" * 70)
    print("生成文本示例 (specSampling_press):")
    print("=" * 70)
    print(tokenizer.decode(outputs["specSampling_press"][0], skip_special_tokens=True)[:500])
    
    print("\n" + "=" * 70)
    print("生成文本示例 (mixSampling_adaptive):")
    print("=" * 70)
    print(tokenizer.decode(outputs["mixSampling_adaptive"][0], skip_special_tokens=True)[:500])


def test_correctness():
    """测试正确性：验证不同方法生成的分布是否一致"""
    print("=" * 70)
    print("正确性测试")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    q_model = AutoModelForCausalLM.from_pretrained("models/pythia-70m").to(device)
    p_model = AutoModelForCausalLM.from_pretrained("models/pythia-410m").to(device)
    q_model.eval()
    p_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("models/pythia-410m")
    
    prompt = "Hello, world!"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 设置固定的随机种子以获得可重复的结果
    torch.manual_seed(42)
    
    config = MixSamplingConfig(
        gamma=4,
        temperature=1.0,
        press_type="none",
        apply_to_target=False
    )
    
    output = mixSampling_simple(input_ids.clone(), q_model, p_model, 50, config)
    print(f"\n生成结果: {tokenizer.decode(output[0], skip_special_tokens=True)}")
    print("\n✓ 测试通过!")


if __name__ == "__main__":
    compare_with_baseline()
