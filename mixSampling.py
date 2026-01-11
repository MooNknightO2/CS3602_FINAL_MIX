"""
混合采样方法：KV Press + Speculative Decoding
将 KV 缓存压缩与推测解码结合，实现更高效的推理

核心优化：
1. 使用 KV Cache 进行增量推理，避免重复计算
2. Draft model 使用 cache 逐 token 生成
3. Target model 一次性验证 gamma 个 tokens，利用 cache 加速
4. KV Press 压缩长序列的 cache，减少内存和计算
"""

import torch
import copy
from tqdm import tqdm
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from transformers import DynamicCache
from utils import norm_logits, sample, max_fn
from gptneox_press import (
    GPTNeoXKnormPress,
    GPTNeoXStreamingPress,
    GPTNeoXSnapKVPress,
    GPTNeoXAdaptivePress,
    GPTNeoXOptimizedHybridPress,
    GPTNeoXHybridPress,
)


@dataclass
class MixSamplingConfig:
    """混合采样的配置类"""
    # Speculative Decoding 参数
    gamma: int = 4  # draft model 每次猜测的 token 数量
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    
    # KV Press 参数
    compression_ratio: float = 0.5  # 压缩比例 (删除的比例)
    press_type: str = "streaming"  # 压缩类型
    
    # 应用策略
    apply_to_target: bool = True  # 是否对 target model 应用压缩
    apply_to_draft: bool = False  # 是否对 draft model 应用压缩
    
    # StreamingPress 特有参数
    num_sink_tokens: int = 4
    
    # SnapKV 特有参数
    window_size: int = 32
    kernel_size: int = 5


def get_press(config: MixSamplingConfig):
    """根据配置获取对应的 Press 实例"""
    press_type = config.press_type.lower()
    
    if press_type == "knorm":
        return GPTNeoXKnormPress(compression_ratio=config.compression_ratio)
    elif press_type == "streaming":
        return GPTNeoXStreamingPress(
            compression_ratio=config.compression_ratio,
            num_sink_tokens=config.num_sink_tokens
        )
    elif press_type == "snapkv":
        return GPTNeoXSnapKVPress(
            compression_ratio=config.compression_ratio,
            window_size=config.window_size,
            kernel_size=config.kernel_size
        )
    elif press_type == "adaptive":
        return GPTNeoXAdaptivePress(
            compression_ratio=config.compression_ratio,
            num_sink_tokens=config.num_sink_tokens
        )
    elif press_type == "hybrid":
        return GPTNeoXHybridPress(
            compression_ratio=config.compression_ratio,
            num_sink_tokens=config.num_sink_tokens,
            window_size=config.window_size
        )
    elif press_type == "optimized_hybrid":
        return GPTNeoXOptimizedHybridPress(
            compression_ratio=config.compression_ratio,
            num_sink_tokens=config.num_sink_tokens,
            window_size=config.window_size
        )
    elif press_type == "none":
        return None
    else:
        raise ValueError(f"未知的 press_type: {press_type}")


def get_cache_length(past_key_values) -> int:
    """获取 cache 的层数（兼容不同版本的 transformers）"""
    if past_key_values is None:
        return 0
    
    # 新版本 transformers: DynamicCache 有 key_cache 属性
    if hasattr(past_key_values, 'key_cache'):
        return len(past_key_values.key_cache)
    # 老版本或元组格式
    elif isinstance(past_key_values, (list, tuple)):
        return len(past_key_values)
    else:
        return 0


def get_cache_kv(past_key_values, layer_idx: int):
    """获取指定层的 key 和 value（兼容不同版本）"""
    if hasattr(past_key_values, 'key_cache'):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    else:
        return past_key_values[layer_idx]


def truncate_kv_cache(past_key_values, target_len: int):
    """截断 KV Cache 到指定长度"""
    if past_key_values is None:
        return None
    
    num_layers = get_cache_length(past_key_values)
    if num_layers == 0:
        return None
    
    # 检查是否是 DynamicCache 类型
    if hasattr(past_key_values, 'key_cache'):
        new_cache = DynamicCache()
        for layer_idx in range(num_layers):
            key = past_key_values.key_cache[layer_idx][:, :, :target_len, :]
            value = past_key_values.value_cache[layer_idx][:, :, :target_len, :]
            new_cache.update(key, value, layer_idx)
        return new_cache
    else:
        # 元组格式
        new_cache = []
        for layer_idx in range(num_layers):
            key, value = past_key_values[layer_idx]
            new_cache.append((
                key[:, :, :target_len, :],
                value[:, :, :target_len, :]
            ))
        return tuple(new_cache)


def clone_kv_cache(past_key_values):
    """深拷贝 KV Cache"""
    if past_key_values is None:
        return None
    
    num_layers = get_cache_length(past_key_values)
    if num_layers == 0:
        return None
    
    # 检查是否是 DynamicCache 类型
    if hasattr(past_key_values, 'key_cache'):
        new_cache = DynamicCache()
        for layer_idx in range(num_layers):
            key = past_key_values.key_cache[layer_idx].clone()
            value = past_key_values.value_cache[layer_idx].clone()
            new_cache.update(key, value, layer_idx)
        return new_cache
    else:
        # 元组格式
        new_cache = []
        for layer_idx in range(num_layers):
            key, value = past_key_values[layer_idx]
            new_cache.append((key.clone(), value.clone()))
        return tuple(new_cache)


@torch.no_grad()
def mixSampling(
    prefix: torch.Tensor,
    q_model: torch.nn.Module,
    p_model: torch.nn.Module,
    maxLen: int,
    config: Optional[MixSamplingConfig] = None
) -> torch.Tensor:
    """
    KV Press + Speculative Decoding 混合采样（使用 KV Cache 增量推理）
    
    工作流程：
    1. Draft model 使用 KV Cache 逐 token 生成 gamma 个候选
    2. Target model 一次性验证所有候选 tokens（利用并行计算）
    3. 根据验证结果接受/拒绝 tokens，并回滚 cache 到正确位置
    4. KV Press 在 cache 过长时进行压缩
    
    Args:
        prefix: 输入序列, shape=(batch=1, prefix_seqLen)
        q_model: draft model (小模型)
        p_model: target model (大模型)
        maxLen: 最大生成 token 数量
        config: 混合采样配置
        
    Returns:
        torch.Tensor: 生成的 tokens (batch, target_seqLen)
    """
    if config is None:
        config = MixSamplingConfig()
    
    gamma = config.gamma
    temperature = config.temperature
    top_k = config.top_k
    top_p = config.top_p
    
    seqLen = prefix.shape[1]
    T = seqLen + maxLen
    device = prefix.device
    assert prefix.shape[0] == 1, "仅支持 batch_size=1"
    
    # 获取 press 实例
    press = get_press(config)
    
    # ==================== 初始化：Prefill 阶段 ====================
    # Draft model prefill
    q_outputs = q_model(prefix, use_cache=True)
    q_cache = q_outputs.past_key_values
    
    # Target model prefill（使用 KV Press 压缩）
    if press is not None and config.apply_to_target:
        with press(p_model):
            p_outputs = p_model(prefix, use_cache=True)
    else:
        p_outputs = p_model(prefix, use_cache=True)
    p_cache = p_outputs.past_key_values
    
    # 当前生成的序列
    generated = prefix.clone()
    
    with tqdm(total=T, desc="mixSampling (KVCache + SpecDec)") as pbar:
        pbar.update(seqLen)
        
        while generated.shape[1] < T:
            preLen = generated.shape[1]
            
            # 保存 cache 状态（用于可能的回滚）
            q_cache_backup = clone_kv_cache(q_cache)
            p_cache_len = p_cache.get_seq_length() if p_cache is not None else 0
            
            # ==================== 阶段 1: Draft Model 增量生成 gamma 个候选 ====================
            draft_tokens = []
            draft_probs = []
            
            current_token = generated[:, -1:]  # 最后一个 token
            
            for _ in range(gamma):
                # 增量推理：只传入最后一个 token，使用 cache
                if press is not None and config.apply_to_draft:
                    with press(q_model):
                        q_out = q_model(current_token, past_key_values=q_cache, use_cache=True)
                else:
                    q_out = q_model(current_token, past_key_values=q_cache, use_cache=True)
                
                q_cache = q_out.past_key_values
                logits = q_out.logits[:, -1, :]
                
                # 计算概率分布
                probs = norm_logits(logits, temperature, top_k, top_p)
                draft_probs.append(probs)
                
                # 采样下一个 token
                next_tok = sample(probs)
                draft_tokens.append(next_tok)
                current_token = next_tok
            
            # 拼接所有 draft tokens
            draft_sequence = torch.cat(draft_tokens, dim=1)  # [1, gamma]
            
            # ==================== 阶段 2: Target Model 并行验证 ====================
            # Target model 一次性处理所有 gamma 个新 tokens
            # 这里利用了 transformer 的并行性：一次前向传播计算所有位置的 logits
            
            if press is not None and config.apply_to_target:
                with press(p_model):
                    p_out = p_model(draft_sequence, past_key_values=p_cache, use_cache=True)
            else:
                p_out = p_model(draft_sequence, past_key_values=p_cache, use_cache=True)
            
            p_cache = p_out.past_key_values
            p_logits = p_out.logits  # [1, gamma, vocab_size]
            
            # ==================== 阶段 3: 验证与采样 ====================
            n_accepted = 0
            rejected_at = -1
            
            for i in range(gamma):
                # Target model 在位置 i 的概率（预测位置 i+1）
                p_probs = norm_logits(p_logits[:, i, :], temperature, top_k, top_p)
                
                # Draft model 在位置 i 的概率
                q_probs = draft_probs[i]
                
                # 被 draft 采样的 token
                draft_tok = draft_tokens[i]
                tok_idx = draft_tok.item()
                
                # 计算接受概率
                p_prob = p_probs[0, tok_idx]
                q_prob = q_probs[0, tok_idx]
                
                accept_prob = min(1.0, p_prob.item() / (q_prob.item() + 1e-10))
                
                r = torch.rand(1, device=device).item()
                
                if r < accept_prob:
                    n_accepted += 1
                else:
                    rejected_at = i
                    # 计算修正分布并采样
                    diff = p_probs - q_probs
                    diff = torch.clamp(diff, min=0)
                    diff_sum = diff.sum()
                    if diff_sum > 0:
                        diff = diff / diff_sum
                    else:
                        diff = p_probs  # fallback
                    correction_token = sample(diff)
                    break
            
            # ==================== 阶段 4: 更新序列和 Cache ====================
            if rejected_at == -1:
                # 全部接受
                accepted_tokens = draft_sequence
                # 从 target model 的最后一个位置采样额外的 token
                final_probs = norm_logits(p_logits[:, -1, :], temperature, top_k, top_p)
                bonus_token = sample(final_probs)
                new_tokens = torch.cat([accepted_tokens, bonus_token], dim=1)
            else:
                # 部分接受
                if n_accepted > 0:
                    accepted_tokens = draft_sequence[:, :n_accepted]
                    new_tokens = torch.cat([accepted_tokens, correction_token], dim=1)
                else:
                    new_tokens = correction_token
                
                # 回滚 target model 的 cache
                # cache 应该保留到 preLen + n_accepted 的位置
                target_cache_len = preLen + n_accepted
                p_cache = truncate_kv_cache(p_cache, target_cache_len)
                
                # 需要用 correction_token 更新 target cache
                if press is not None and config.apply_to_target:
                    with press(p_model):
                        p_out = p_model(correction_token, past_key_values=p_cache, use_cache=True)
                else:
                    p_out = p_model(correction_token, past_key_values=p_cache, use_cache=True)
                p_cache = p_out.past_key_values
            
            # 更新生成序列
            generated = torch.cat([generated, new_tokens], dim=1)
            
            # 回滚并更新 draft model 的 cache
            # Draft cache 需要同步到当前生成位置
            q_cache = truncate_kv_cache(q_cache_backup, preLen - 1)
            # 用新接受的 tokens 更新 draft cache
            if press is not None and config.apply_to_draft:
                with press(q_model):
                    q_out = q_model(new_tokens, past_key_values=q_cache, use_cache=True)
            else:
                q_out = q_model(new_tokens, past_key_values=q_cache, use_cache=True)
            q_cache = q_out.past_key_values
            
            pbar.update(generated.shape[1] - pbar.n)
    
    return generated


@torch.no_grad()
def mixSampling_adaptive(
    prefix: torch.Tensor,
    q_model: torch.nn.Module,
    p_model: torch.nn.Module,
    maxLen: int,
    config: Optional[MixSamplingConfig] = None
) -> torch.Tensor:
    """
    高效版本：KV Press + Speculative Decoding
    
    优化点：
    1. 不克隆 cache（避免巨大的内存拷贝开销）
    2. Draft model 不使用增量 cache（每轮重新计算，但小模型很快）
    3. Target model 使用增量 cache + KV Press 压缩
    4. 自适应调整 gamma
    """
    if config is None:
        config = MixSamplingConfig()
    
    gamma = config.gamma
    temperature = config.temperature
    top_k = config.top_k
    top_p = config.top_p
    
    seqLen = prefix.shape[1]
    T = seqLen + maxLen
    device = prefix.device
    assert prefix.shape[0] == 1, "仅支持 batch_size=1"
    
    press = get_press(config)
    generated = prefix.clone()
    
    # 统计
    total_accepted = 0
    total_proposed = 0
    
    with tqdm(total=T, desc="mixSampling_adaptive") as pbar:
        pbar.update(seqLen)
        
        while generated.shape[1] < T:
            preLen = generated.shape[1]
            x = generated
            
            actual_gamma = min(gamma, T - generated.shape[1])
            
            # ==================== Draft Model 生成（不使用 cache，小模型重算很快）====================
            draft_tokens = []
            draft_probs = []
            
            for _ in range(actual_gamma):
                # Draft model 每次重算整个序列（小模型开销可接受）
                q_out = q_model(x)
                logits = q_out.logits[:, -1, :]
                probs = norm_logits(logits, temperature, top_k, top_p)
                draft_probs.append(probs)
                next_tok = sample(probs)
                draft_tokens.append(next_tok)
                x = torch.cat([x, next_tok], dim=1)
            
            # ==================== Target Model 验证（使用 KV Press）====================
            if press is not None and config.apply_to_target:
                with press(p_model):
                    p_out = p_model(x)
            else:
                p_out = p_model(x)
            
            p_logits = p_out.logits
            
            # ==================== 验证逻辑 ====================
            n_accepted = 0
            rejected_at = -1
            
            for i in range(actual_gamma):
                # p_logits[:, preLen + i - 1, :] 是预测位置 preLen + i 的分布
                p_probs = norm_logits(p_logits[:, preLen + i - 1, :], temperature, top_k, top_p)
                q_probs = draft_probs[i]
                tok_idx = draft_tokens[i].item()
                
                p_prob = p_probs[0, tok_idx]
                q_prob = q_probs[0, tok_idx]
                accept_prob = min(1.0, p_prob.item() / (q_prob.item() + 1e-10))
                
                r = torch.rand(1, device=device).item()
                
                if r < accept_prob:
                    n_accepted += 1
                else:
                    rejected_at = i
                    # 计算修正分布
                    diff = torch.clamp(p_probs - q_probs, min=0)
                    diff_sum = diff.sum()
                    if diff_sum > 0:
                        diff = diff / diff_sum
                    else:
                        diff = p_probs
                    correction_token = sample(diff)
                    break
            
            # 更新统计
            total_accepted += n_accepted
            total_proposed += actual_gamma
            
            # ==================== 构建新序列 ====================
            if rejected_at == -1:
                # 全部接受
                gamma = min(gamma + 1, 16)
                final_probs = norm_logits(p_logits[:, -1, :], temperature, top_k, top_p)
                bonus_token = sample(final_probs)
                new_tokens = torch.cat(draft_tokens + [bonus_token], dim=1)
            else:
                # 部分接受
                gamma = max(2, gamma - 1)
                if n_accepted > 0:
                    new_tokens = torch.cat(draft_tokens[:n_accepted] + [correction_token], dim=1)
                else:
                    new_tokens = correction_token
            
            generated = torch.cat([generated, new_tokens], dim=1)
            pbar.update(generated.shape[1] - pbar.n)
    
    if total_proposed > 0:
        accept_rate = total_accepted / total_proposed
        print(f"\n接受率: {accept_rate:.2%} ({total_accepted}/{total_proposed}), 最终 gamma: {gamma}")
    
    return generated


@torch.no_grad()
def mixSampling_simple(
    prefix: torch.Tensor,
    q_model: torch.nn.Module,
    p_model: torch.nn.Module,
    maxLen: int,
    config: Optional[MixSamplingConfig] = None
) -> torch.Tensor:
    """
    简化版本：不使用复杂的 cache 管理，但正确使用 KV Cache
    
    适用于不需要 KV Press 的场景，或者作为性能对比基准
    """
    if config is None:
        config = MixSamplingConfig()
    
    gamma = config.gamma
    temperature = config.temperature
    top_k = config.top_k
    top_p = config.top_p
    
    seqLen = prefix.shape[1]
    T = seqLen + maxLen
    device = prefix.device
    assert prefix.shape[0] == 1
    
    generated = prefix.clone()
    
    with tqdm(total=T, desc="mixSampling_simple") as pbar:
        pbar.update(seqLen)
        
        while generated.shape[1] < T:
            preLen = generated.shape[1]
            x = generated
            
            # Draft model 生成（使用 cache）
            q_cache = None
            draft_tokens = []
            draft_probs = []
            
            for _ in range(gamma):
                if q_cache is None:
                    q_out = q_model(x, use_cache=True)
                else:
                    q_out = q_model(x[:, -1:], past_key_values=q_cache, use_cache=True)
                
                q_cache = q_out.past_key_values
                logits = q_out.logits[:, -1, :]
                probs = norm_logits(logits, temperature, top_k, top_p)
                draft_probs.append(probs)
                next_tok = sample(probs)
                draft_tokens.append(next_tok)
                x = torch.cat([x, next_tok], dim=1)
            
            # Target model 一次性验证
            p_out = p_model(x)
            p_logits = p_out.logits
            
            # 验证
            n_accepted = 0
            final_token = None
            
            for i in range(gamma):
                # p_logits[:, preLen + i - 1, :] 是预测第 preLen + i 个位置的分布
                p_probs = norm_logits(p_logits[:, preLen + i - 1, :], temperature, top_k, top_p)
                q_probs = draft_probs[i]
                
                tok_idx = draft_tokens[i].item()
                p_prob = p_probs[0, tok_idx]
                q_prob = q_probs[0, tok_idx]
                
                accept_prob = min(1.0, p_prob.item() / (q_prob.item() + 1e-10))
                r = torch.rand(1, device=device).item()
                
                if r < accept_prob:
                    n_accepted += 1
                else:
                    # 拒绝，采样修正 token
                    diff = torch.clamp(p_probs - q_probs, min=0)
                    diff_sum = diff.sum()
                    if diff_sum > 0:
                        diff = diff / diff_sum
                    else:
                        diff = p_probs
                    final_token = sample(diff)
                    break
            
            # 构建新序列
            if final_token is None:
                # 全部接受，从最后位置采样 bonus token
                final_probs = norm_logits(p_logits[:, -1, :], temperature, top_k, top_p)
                final_token = sample(final_probs)
                new_tokens = torch.cat(draft_tokens + [final_token], dim=1)
            else:
                if n_accepted > 0:
                    new_tokens = torch.cat(draft_tokens[:n_accepted] + [final_token], dim=1)
                else:
                    new_tokens = final_token
            
            generated = torch.cat([generated, new_tokens], dim=1)
            pbar.update(generated.shape[1] - pbar.n)
    
    return generated


# ===================== 纯 KV Press + Speculative Decoding =====================

@torch.no_grad()
def specSampling_press(
    prefix: torch.Tensor,
    q_model: torch.nn.Module,
    p_model: torch.nn.Module,
    maxLen: int,
    config: Optional[MixSamplingConfig] = None
) -> torch.Tensor:
    """
    纯 KV Press + Speculative Decoding（不使用增量 KV Cache）
    
    这个版本类似原始的 specSampling，每次重新计算整个序列，
    但在 target model 推理时使用 KV Press 压缩 cache。
    
    适用场景：
    - 对比 KV Press 单独带来的加速效果
    - 长序列生成时减少 attention 计算量
    
    Args:
        prefix: 输入序列, shape=(batch=1, prefix_seqLen)
        q_model: draft model (小模型)
        p_model: target model (大模型)
        maxLen: 最大生成 token 数量
        config: 配置
        
    Returns:
        torch.Tensor: 生成的 tokens (batch, target_seqLen)
    """
    if config is None:
        config = MixSamplingConfig()
    
    gamma = config.gamma
    temperature = config.temperature
    top_k = config.top_k
    top_p = config.top_p
    
    seqLen = prefix.shape[1]
    T = seqLen + maxLen
    device = prefix.device
    assert prefix.shape[0] == 1, "仅支持 batch_size=1"
    
    # 获取 press 实例
    press = get_press(config)
    
    with tqdm(total=T, desc="specSampling_press (KVPress only)") as pbar:
        pbar.update(seqLen)
        
        while prefix.shape[1] < T:
            x = prefix
            preLen = prefix.shape[1]
            
            # ==================== 阶段 1: Draft Model 生成候选 ====================
            # 这里不使用增量 cache，每次重算（与原始 specSampling 一致）
            for _ in range(gamma):
                if press is not None and config.apply_to_draft:
                    with press(q_model):
                        q = q_model(x).logits
                else:
                    q = q_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            
            # 归一化所有位置的 logits（用于验证）
            for i in range(q.shape[1]):
                q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)
            
            # ==================== 阶段 2: Target Model 验证（使用 KV Press）====================
            if press is not None and config.apply_to_target:
                with press(p_model):
                    p = p_model(x).logits
            else:
                p = p_model(x).logits
            
            # 归一化
            for i in range(p.shape[1]):
                p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)
            
            # ==================== 阶段 3: 验证与采样 ====================
            flag = True
            n = preLen - 1
            
            for i in range(gamma):
                r = torch.rand(1, device=device)
                j = x[:, preLen + i]
                
                # 接受概率
                accept_prob = torch.min(
                    torch.tensor([1.0], device=device),
                    p[:, preLen + i - 1, j] / (q[:, preLen + i - 1, j] + 1e-10)
                )
                
                if r < accept_prob:
                    n += 1
                else:
                    # 拒绝，使用修正分布采样
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    flag = False
                    break
            
            prefix = x[:, :(n + 1)]
            
            if flag:
                # 全部接受
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(prefix.shape[1] - pbar.n)
    
    return prefix


# ===================== 便捷接口函数 =====================

@torch.no_grad()
def mixSampling_streaming(
    prefix: torch.Tensor,
    q_model: torch.nn.Module,
    p_model: torch.nn.Module,
    maxLen: int,
    gamma: int = 4,
    compression_ratio: float = 0.5,
    num_sink_tokens: int = 4,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9
) -> torch.Tensor:
    """使用 StreamingPress 的混合采样"""
    config = MixSamplingConfig(
        gamma=gamma,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        compression_ratio=compression_ratio,
        press_type="streaming",
        apply_to_target=True,
        apply_to_draft=False,
        num_sink_tokens=num_sink_tokens
    )
    return mixSampling(prefix, q_model, p_model, maxLen, config)


@torch.no_grad()
def mixSampling_snapkv(
    prefix: torch.Tensor,
    q_model: torch.nn.Module,
    p_model: torch.nn.Module,
    maxLen: int,
    gamma: int = 4,
    compression_ratio: float = 0.5,
    window_size: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9
) -> torch.Tensor:
    """使用 SnapKVPress 的混合采样"""
    config = MixSamplingConfig(
        gamma=gamma,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        compression_ratio=compression_ratio,
        press_type="snapkv",
        apply_to_target=True,
        apply_to_draft=False,
        window_size=window_size
    )
    return mixSampling(prefix, q_model, p_model, maxLen, config)


# ===================== 测试代码 =====================

if __name__ == "__main__":
    print("混合采样模块 v2 - 使用 KV Cache 增量推理")
    print("\n核心优化:")
    print("  1. Draft model 使用 KV Cache 逐 token 增量生成")
    print("  2. Target model 一次性并行验证 gamma 个 tokens")
    print("  3. 验证失败时正确回滚 Cache")
    print("  4. KV Press 压缩长序列的 Cache")
    
    print("\n可用函数:")
    print("  - mixSampling: 完整版（KV Cache + KV Press）")
    print("  - mixSampling_adaptive: 自适应 gamma")
    print("  - mixSampling_simple: 简化版（仅 KV Cache）")
    print("  - mixSampling_streaming: StreamingPress 便捷接口")
    print("  - mixSampling_snapkv: SnapKVPress 便捷接口")
