# CS3602_FINAL_MIX

## Fast Inference from Transformers

基于 **Speculative Decoding** 和 **KV Cache Compression (KVPress)** 加速 Pythia-2.8B 模型的推理

---

## 📖 项目简介

Transformer 架构的大语言模型（LLM）在部署时面临着高推理延迟和内存带宽限制的挑战。本项目研究了通过内存优化和算法效率提升来加速 LLM 推理的实用策略。

### 核心工作

1. **复现与评估**：全面复现和评估了多种 KV Cache 压缩技术（KVPress）和标准推测解码框架
2. **创新方法**：
   - **Dynamic Gamma 机制**：自适应调整推测深度，实现更高效的推测解码
   - **Multi-Level Verification 策略**：探索多层级验证的可能性
3. **方法融合**：将 Speculative Decoding 与 KVPress 结合，实现最高 **6.48x** 的加速

---

## 🔬 方法介绍

### 1. Speculative Decoding（推测解码）

核心原理：使用小型草稿模型（Draft Model）$M_q$ 生成 $\gamma$ 个候选 token，再由目标模型（Target Model）$M_p$ 并行验证。

**验证机制**：
- 对于草稿 token $x_i$ 及其概率分布 $q(x_i)$ 和 $p(x_i)$
- 当随机变量 $r \sim U(0,1)$ 满足 $r < \min(1, p(x_i)/q(x_i))$ 时接受该 token
- 被拒绝的 token 从修正分布中重采样，保证输出分布与目标模型完全一致

### 2. Dynamic Gamma（动态 Gamma）

固定的推测深度 $\gamma$ 并非最优：
- 模型分歧时，激进的推测会浪费计算资源
- 模型一致时，保守的推测会低估带宽利用率

**自适应调整策略**（灵感来自计算机网络的 AIMD 拥塞控制算法）：
- **Exploitation（开发）**：若所有推测 token 都被接受，增加 $\gamma$（$\gamma \leftarrow \gamma + 1$）
- **Correction（修正）**：若发生拒绝，几何级减少 $\gamma$（$\gamma \leftarrow \max(4, \lfloor\gamma / 2\rfloor)$）

### 3. Multi-Layer Speculative Decoding（多层推测解码）

将范式扩展为三个模型的级联：$M_q$（Draft）、$M_r$（Intermediate）、$M_p$（Target）

**两阶段过程**：
1. $M_q$ 生成 token，由 $M_r$ 进行初步验证
2. 被 $M_r$ 接受的序列传递给 $M_p$ 进行最终验证

> 💡 灵感来自计算机组成原理中的多级缓存设计，但实验表明对于当前模型尺寸比例效果不佳

### 4. KV Cache Compression (KVPress)

自回归解码需要存储每层的 Key/Value 状态，KV Cache 随序列长度线性增长。

**压缩策略**：
- **Streaming**：保留前 $N_{sink}$ 个 token（attention sinks）和最近的 $N_{recent}$ 个 token，丢弃中间上下文
- **Hybrid Conservative**：结合滑动窗口与重要性采样（基于 Key L2 范数），保留语义重要的中间 token

**关键发现**：
- 短上下文（WikiText-2）：Hybrid 策略更优（PPL 427 vs 508）
- 长上下文（PG19）：简单的 Streaming 策略反而更好（PPL 108 vs 136）

### 5. Integration（方法融合）

将 Speculative Decoding 与 KVPress 结合：
- 前者通过提前生成草稿 token 减少计算延迟
- 后者通过压缩 KV Cache 减少内存延迟

---

## 📊 实验结果

### Speculative Decoding 加速效果

| 方法 | TTFT (s) | TPOT (ms) | 吞吐量 (tok/s) |
|------|----------|-----------|----------------|
| Baseline (2.8B) | 1.55 | 268.51 | 3.72 |
| Speculative Average | 0.36 | 148.75 | 6.72 |
| Speculative Best | 0.48 | 107.19 | 9.33 |

### Dynamic Gamma 与多层方法对比

| 方法 | TTFT (s) | TPOT (ms) | 吞吐量 (tok/s) |
|------|----------|-----------|----------------|
| Pythia-2.8B + 70m | 1.18 | 378.81 | 2.64 |
| + Dynamic Gamma (Best) | 3.23 | 82.41 | 12.13 |
| + Dynamic Gamma (Avg) | 1.55 | 363.41 | 2.75 |
| Multi-Layer (70M+410M+2.8B) | - | 400.52 | 2.50 |

### 融合方法加速效果

| 方法 | TPOT (ms) | 吞吐量 (tok/s) | 相对加速 |
|------|-----------|----------------|----------|
| Speculative Decoding | 387.54 | 2.58 | 1.00x |
| + KVpress | 185.97 | 5.38 | 2.08x |
| + KVpress + Dynamic Gamma | 59.78 | 16.73 | **6.48x** |

---

## 🛠️ 环境配置

### 依赖安装

```bash
pip install torch transformers datasets huggingface_hub tqdm
```

### 硬件要求

- GPU：建议使用 CUDA 兼容的 NVIDIA GPU（至少 8GB 显存）
- 内存：建议 16GB 以上

---

## 📥 数据与模型准备

### 下载数据集

```bash
python downloadData.py
```

将下载：
- **WikiText-2**：短文本评测数据集
- **PG-19**：长文本评测数据集

### 下载模型

```bash
python downloadModel.py
```

将下载 Pythia 系列模型：
- `pythia-70m`：Draft Model（草稿模型）
- `pythia-410m`：Intermediate Model（中间模型）
- `pythia-2.8b`：Target Model（目标模型）

---

## 🚀 使用方法

### 1. 标准 Speculative Decoding

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from specSampling import specSampling
import torch

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("./models/pythia-2.8b")
p_model = AutoModelForCausalLM.from_pretrained("./models/pythia-2.8b", device_map="auto", torch_dtype=torch.float16)
q_model = AutoModelForCausalLM.from_pretrained("./models/pythia-70m", device_map="auto", torch_dtype=torch.float16)

# 推理
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(p_model.device)
output = specSampling(
    prefix=inputs["input_ids"],
    q_model=q_model,
    p_model=p_model,
    maxLen=100,
    gamma=4
)
print(tokenizer.decode(output[0]))
```

### 2. Dynamic Gamma Speculative Decoding

```python
from specSampling import specSampling_new

output = specSampling_new(
    prefix=inputs["input_ids"],
    q_model=q_model,
    p_model=p_model,
    maxLen=100,
    gamma=4  # 初始 gamma，会自动调整
)
```

### 3. 混合采样（Speculative Decoding + KVPress）

```python
from mixSampling import mixSampling_adaptive, MixSamplingConfig

config = MixSamplingConfig(
    gamma=4,
    compression_ratio=0.5,
    press_type="streaming",  # 或 "hybrid", "snapkv" 等
    apply_to_target=True,
    apply_to_draft=False
)

output = mixSampling_adaptive(
    prefix=inputs["input_ids"],
    q_model=q_model,
    p_model=p_model,
    maxLen=100,
    config=config
)
```

### 4. 运行完整测试

```bash
python test.py
```

可在 `test.py` 中修改 `SpecConfig` 来选择测试内容：
- `speed`: 标准速度测试
- `ppl`: PPL 计算
- `speedNew`: Dynamic Gamma 速度测试
- `speedMulti`: 多层推测解码测试

---

## 📁 项目结构

```
CS3602_FINAL_MIX/
├── specSampling.py      # Speculative Decoding 实现
│                        # - specSampling: 标准推测解码
│                        # - specSampling_new: Dynamic Gamma 版本
│                        # - specSampling_new_multi: 多层推测解码
├── mixSampling.py       # 混合采样实现（SpecDec + KVPress）
│                        # - mixSampling: KV Cache 增量推理版本
│                        # - mixSampling_adaptive: 自适应 gamma 版本
│                        # - mixSampling_simple: 简化版本
├── regrSampling.py      # 标准自回归采样（Baseline）
├── gptneox_press.py     # GPTNeoX 的 KVPress 适配
├── kvpress/             # KVPress 压缩策略库
│   ├── presses/         # 各种压缩策略实现
│   │   ├── streaming_llm_press.py
│   │   ├── snapkv_press.py
│   │   ├── knorm_press.py
│   │   └── ...
│   └── ...
├── PPL.py               # Perplexity 计算
├── utils.py             # 工具函数（采样、归一化等）
├── test.py              # 测试脚本
├── test_mixSampling.py  # 混合采样测试脚本
├── downloadData.py      # 数据集下载脚本
├── downloadModel.py     # 模型下载脚本
├── main.tex             # 项目论文（LaTeX）
└── README.md            # 项目说明
```

---

## 📈 关键结论

1. **Speculative Decoding** 是加速 LLM 推理的有效方法，可实现最高 **2.5x** 加速
2. **Dynamic Gamma** 机制通过动态调整推测深度成功提升效率约 **5%**
3. **Multi-Layer** 扩展对当前模型尺寸比例效果不佳，需要更精确的中间模型
4. **KVPress** 在长上下文场景下简单的 Streaming 策略反而最稳健
5. **方法融合**（SpecDec + KVPress + Dynamic Gamma）可实现 **6.48x** 加速，但以 PPL 上升为代价

---

## 🔗 相关链接

- **Wang Siyuan's Repo**: [CS3602_FINAL_SpeDec](https://github.com/MooNknightO2/CS3602_FINAL_SpeDec)
- **Lin Ruikang's Repo**: [CS3604_FINAL](https://github.com/ephuon/CS3604_FINAL)
