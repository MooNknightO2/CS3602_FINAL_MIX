"""
支持 GPTNeoX (Pythia) 模型的 kvpress 适配器
使用 DynamicCache monkey-patching 实现压缩
"""

import torch
import torch.nn.functional as F
from typing import Optional, Any
from dataclasses import dataclass
from kvpress.presses.base_press import BasePress


@dataclass
class GPTNeoXKnormPress(BasePress):
    """
    为 GPTNeoX 架构定制的 Knorm Press
    基于 key 的范数进行压缩
    
    Parameters
    ----------
    compression_ratio : float, default=0.5
        保留的 KV 缓存比例（0-1之间）
    """
    
    compression_ratio: float = 0.5
    
    def __post_init__(self):
        # 与 kvpress 语义对齐：compression_ratio 表示“删除比例”
        assert 0 <= self.compression_ratio < 1, "compression_ratio 必须在 [0, 1) 之间"

    def compress_cache(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """压缩 key 和 value states（保留较小 key 范数的 tokens，与 kvpress 一致）"""
        seq_len = key_states.shape[2]
        keep_size = max(int(seq_len * (1 - self.compression_ratio)), 1)

        if keep_size >= seq_len or keep_size <= 0:
            return key_states, value_states

        # 计算 key 的范数（kvpress 使用 -norm 得分，等价于选择最小范数）
        scores = torch.norm(key_states, dim=-1, keepdim=False)  # [batch, num_heads, seq_len]

        # 选择最小范数的 top-k（使用负号与 kvpress 行为一致）
        _, top_indices = torch.topk(-scores, k=keep_size, dim=-1, sorted=False)
        top_indices = top_indices.sort(dim=-1)[0]  # 保持原始顺序

        # 压缩 key 和 value
        batch_size, num_heads, _, head_dim = key_states.shape
        compressed_key = torch.gather(
            key_states,
            dim=2,
            index=top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )
        compressed_value = torch.gather(
            value_states,
            dim=2,
            index=top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )

        return compressed_key, compressed_value
    
    def __call__(self, model):
        """使用 monkey-patching 拦截 DynamicCache.update"""
        from contextlib import contextmanager
        from transformers.cache_utils import DynamicCache
        
        @contextmanager
        def _context():
            # 保存原始的 update 方法
            original_update = DynamicCache.update
            
            # 创建包装函数
            def wrapped_update(cache_self, key_states: torch.Tensor, value_states: torch.Tensor, 
                             layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None):
                # 在存入 cache 之前压缩
                key_states, value_states = self.compress_cache(key_states, value_states)
                # 调用原始的 update 方法
                return original_update(cache_self, key_states, value_states, layer_idx, cache_kwargs)
            
            try:
                # Monkey-patch
                DynamicCache.update = wrapped_update
                yield
            finally:
                # 恢复原始方法
                DynamicCache.update = original_update
        
        return _context()


@dataclass
class GPTNeoXStreamingPress(BasePress):
    """
    为 GPTNeoX 架构定制的 Streaming Press
    保留开始和最近的 tokens
    
    Parameters
    ----------
    compression_ratio : float, default=0.5
        保留的 KV 缓存比例
    num_sink_tokens : int, default=4
        保留的开头 tokens 数量（重要的初始 tokens）
    """
    
    compression_ratio: float = 0.5
    num_sink_tokens: int = 4

    def __post_init__(self):
        # 与 kvpress 语义对齐：compression_ratio 表示“删除比例”
        assert 0 <= self.compression_ratio < 1, "compression_ratio 必须在 [0, 1) 之间"
        assert self.num_sink_tokens >= 0, "num_sink_tokens 必须 >= 0"

    def compress_cache(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """压缩 key 和 value states - 保留 sink tokens + recent tokens"""
        seq_len = key_states.shape[2]
        keep_size = max(int(seq_len * (1 - self.compression_ratio)), self.num_sink_tokens + 1)

        if keep_size >= seq_len or keep_size <= 0:
            return key_states, value_states

        # 先保留 sink tokens，再分配预算给最近 tokens
        num_sink = min(self.num_sink_tokens, seq_len, keep_size)
        remaining_budget = max(keep_size - num_sink, 0)
        num_recent = min(remaining_budget, seq_len - num_sink)

        # 创建索引：开始的 sink_tokens + 最后的 recent_tokens
        device = key_states.device
        sink_indices = torch.arange(num_sink, device=device)
        recent_indices = torch.arange(seq_len - num_recent, seq_len, device=device) if num_recent > 0 else torch.tensor([], device=device, dtype=torch.long)
        keep_indices = torch.cat([sink_indices, recent_indices])

        # 压缩
        batch_size, num_heads, _, head_dim = key_states.shape
        keep_indices = keep_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        keep_indices = keep_indices.expand(batch_size, num_heads, -1, head_dim)

        compressed_key = torch.gather(key_states, dim=2, index=keep_indices)
        compressed_value = torch.gather(value_states, dim=2, index=keep_indices)

        return compressed_key, compressed_value
    
    def __call__(self, model):
        """使用 monkey-patching 拦截 DynamicCache.update"""
        from contextlib import contextmanager
        from transformers.cache_utils import DynamicCache
        
        @contextmanager
        def _context():
            # 保存原始的 update 方法
            original_update = DynamicCache.update
            
            # 创建包装函数
            def wrapped_update(cache_self, key_states: torch.Tensor, value_states: torch.Tensor, 
                             layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None):
                # 在存入 cache 之前压缩
                key_states, value_states = self.compress_cache(key_states, value_states)
                # 调用原始的 update 方法
                return original_update(cache_self, key_states, value_states, layer_idx, cache_kwargs)
            
            try:
                # Monkey-patch
                DynamicCache.update = wrapped_update
                yield
            finally:
                # 恢复原始方法
                DynamicCache.update = original_update
        
        return _context()


@dataclass
class GPTNeoXSnapKVPress(BasePress):
    """
    为 GPTNeoX 架构定制的 SnapKV Press（与 kvpress 语义更一致）
    使用最近窗口的 attention 模式估计历史 tokens 重要性
    
    Parameters
    ----------
    compression_ratio : float, default=0.5
        删除的 KV 缓存比例（0 表示不压缩）
    window_size : int, default=32
        计算 attention 的窗口大小
    kernel_size : int, default=5
        对分数做平滑的卷积核大小
    """
    
    compression_ratio: float = 0.5
    window_size: int = 32
    kernel_size: int = 5
    
    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio 必须在 [0, 1) 之间"
        assert self.window_size > 0, "window_size 必须 > 0"
        assert self.kernel_size > 0, "kernel_size 必须 > 0"
    
    def compress_cache(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """基于最近窗口 attention 的压缩，保持与 kvpress 思路一致"""
        seq_len = key_states.shape[2]
        keep_size = max(int(seq_len * (1 - self.compression_ratio)), 1)

        if keep_size >= seq_len or seq_len <= self.window_size:
            return key_states, value_states

        # 使用最近 window_size 个位置作为查询，计算对全序列的注意力
        window_start = max(0, seq_len - self.window_size)
        query_window = key_states[:, :, window_start:, :]  # [B, H, W, D]

        scale = key_states.shape[-1] ** -0.5
        attn_scores = torch.matmul(query_window, key_states.transpose(-2, -1)) * scale  # [B, H, W, S]
        attn_scores = torch.softmax(attn_scores, dim=-1)

        # 对查询窗口求平均得到每个 key 位置的重要性
        scores = attn_scores.mean(dim=2)  # [B, H, S]

        # 平滑处理（与 kvpress 中的 avg_pool1d 行为一致）
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        scores = scores[..., :seq_len]  # 形状对齐

        # 确保最近窗口不被裁剪：给窗口内分数赋最大值
        max_scores = scores.max(dim=-1, keepdim=True).values
        scores[..., -self.window_size:] = max_scores

        # 选择 top-k
        _, top_indices = torch.topk(scores, k=keep_size, dim=-1, sorted=False)
        top_indices = top_indices.sort(dim=-1)[0]

        # 压缩
        batch_size, num_heads, _, head_dim = key_states.shape
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        compressed_key = torch.gather(key_states, dim=2, index=gather_index)
        compressed_value = torch.gather(value_states, dim=2, index=gather_index)

        return compressed_key, compressed_value
    
    def __call__(self, model):
        """使用 monkey-patching 拦截 DynamicCache.update"""
        from contextlib import contextmanager
        from transformers.cache_utils import DynamicCache
        
        @contextmanager
        def _context():
            original_update = DynamicCache.update
            
            def wrapped_update(cache_self, key_states: torch.Tensor, value_states: torch.Tensor, 
                             layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None):
                key_states, value_states = self.compress_cache(key_states, value_states)
                return original_update(cache_self, key_states, value_states, layer_idx, cache_kwargs)
            
            try:
                DynamicCache.update = wrapped_update
                yield
            finally:
                DynamicCache.update = original_update
        
        return _context()


@dataclass
class GPTNeoXAdaptivePress(BasePress):
    """
    自适应轻量级压缩 - StreamingPress的智能增强版
    
    核心思想：
    1. 保持StreamingPress的简单高效
    2. 在中间区域添加轻量级的重要性筛选
    3. 避免复杂的多维度评分
    
    策略：
    - 保留开头的sink tokens（prompt和初始上下文）
    - 保留最近的tokens（当前生成依赖）
    - 在中间区域用key norm快速选择少量重要tokens
    
    Parameters
    ----------
    compression_ratio : float, default=0.5
        保留的 KV 缓存比例
    num_sink_tokens : int, default=4
        保留的开头 tokens 数量
    recent_ratio : float, default=0.4
        分配给最近tokens的比例（相对于总保留量）
    middle_ratio : float, default=0.3
        分配给中间区域的比例（相对于总保留量）
    """
    
    compression_ratio: float = 0.5
    num_sink_tokens: int = 4
    recent_ratio: float = 0.4
    middle_ratio: float = 0.3
    
    def __post_init__(self):
        assert 0 < self.compression_ratio <= 1, "compression_ratio 必须在 (0, 1] 之间"
        assert self.num_sink_tokens >= 0, "num_sink_tokens 必须 >= 0"
        assert 0 < self.recent_ratio < 1, "recent_ratio 必须在 (0, 1) 之间"
        assert 0 < self.middle_ratio < 1, "middle_ratio 必须在 (0, 1) 之间"
        assert self.recent_ratio + self.middle_ratio <= 1, "recent_ratio + middle_ratio 不能超过 1"
    
    def compress_cache(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """自适应轻量级压缩"""
        seq_len = key_states.shape[2]
        target_size = max(int(seq_len * (1 - self.compression_ratio)), self.num_sink_tokens + 1)
        
        if target_size >= seq_len:
            return key_states, value_states
        
        device = key_states.device
        
        # ==================== 阶段1: 计算各部分的token数量 ====================
        # 1. Sink tokens (固定)
        num_sink = min(self.num_sink_tokens, seq_len, target_size)
        
        # 2. Recent tokens (按比例分配)
        remaining_budget = target_size - num_sink
        num_recent = max(int(remaining_budget * self.recent_ratio), 1)
        num_recent = min(num_recent, seq_len - num_sink)
        
        # 3. Middle tokens (剩余的预算)
        num_middle = max(target_size - num_sink - num_recent, 0)
        
        # ==================== 阶段2: 选择tokens ====================
        selected_indices = []
        
        # 2.1 选择sink tokens
        if num_sink > 0:
            sink_indices = torch.arange(num_sink, device=device)
            selected_indices.append(sink_indices)
        
        # 2.2 选择recent tokens
        if num_recent > 0:
            recent_start = seq_len - num_recent
            recent_indices = torch.arange(recent_start, seq_len, device=device)
            selected_indices.append(recent_indices)
        
        # 2.3 在中间区域选择重要tokens（基于key norm）
        if num_middle > 0 and seq_len > num_sink + num_recent:
            # 中间区域的范围
            middle_start = num_sink
            middle_end = seq_len - num_recent
            
            if middle_end > middle_start:
                # 计算中间区域的key norm
                middle_keys = key_states[:, :, middle_start:middle_end, :]
                key_norms = torch.norm(middle_keys, dim=-1)  # [B, H, M]
                
                # 对batch和heads取平均，得到每个position的综合重要性
                avg_norms = key_norms.mean(dim=(0, 1))  # [M]
                
                # 选择top-k
                num_to_select = min(num_middle, middle_end - middle_start)
                _, top_k = torch.topk(avg_norms, k=num_to_select, sorted=True)
                
                # 转换为全局索引
                middle_indices = top_k + middle_start
                selected_indices.append(middle_indices)
        
        # ==================== 阶段3: 合并并排序 ====================
        if len(selected_indices) == 0:
            # 降级为只保留开头
            final_indices = torch.arange(min(target_size, seq_len), device=device)
        else:
            final_indices = torch.cat(selected_indices)
            final_indices = final_indices.sort()[0]  # 保持原始顺序
        
        # ==================== 阶段4: 收集结果 ====================
        return self._gather_by_indices(key_states, value_states, final_indices)
    
    def _gather_by_indices(self, key_states: torch.Tensor, value_states: torch.Tensor, 
                          indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据索引收集 key 和 value"""
        batch_size, num_heads, _, head_dim = key_states.shape
        indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        indices = indices.expand(batch_size, num_heads, -1, head_dim)
        
        compressed_key = torch.gather(key_states, dim=2, index=indices)
        compressed_value = torch.gather(value_states, dim=2, index=indices)
        
        return compressed_key, compressed_value
    
    def __call__(self, model):
        """使用 monkey-patching 拦截 DynamicCache.update"""
        from contextlib import contextmanager
        from transformers.cache_utils import DynamicCache
        
        @contextmanager
        def _context():
            original_update = DynamicCache.update
            
            def wrapped_update(cache_self, key_states: torch.Tensor, value_states: torch.Tensor, 
                             layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None):
                key_states, value_states = self.compress_cache(key_states, value_states)
                return original_update(cache_self, key_states, value_states, layer_idx, cache_kwargs)
            
            try:
                DynamicCache.update = wrapped_update
                yield
            finally:
                DynamicCache.update = original_update
        
        return _context()


@dataclass  
class GPTNeoXOptimizedHybridPress(BasePress):
    """
    优化的复合压缩方法 - 简化版HybridPress
    
    改进点：
    1. 移除position scores（贡献小，计算开销大）
    2. 降低knorm_threshold或移除
    3. 简化attention计算
    4. 优化权重分配
    
    Parameters
    ----------
    compression_ratio : float, default=0.5
        保留的 KV 缓存比例
    num_sink_tokens : int, default=4
        保留的开头 tokens 数量
    window_size : int, default=16
        用于计算 attention score 的窗口大小（减小以降低开销）
    use_attention : bool, default=True
        是否使用attention scores（可以关闭以加速）
    """
    
    compression_ratio: float = 0.5
    num_sink_tokens: int = 4
    window_size: int = 16  # 从32减少到16
    use_attention: bool = True
    
    def __post_init__(self):
        assert 0 < self.compression_ratio <= 1, "compression_ratio 必须在 (0, 1] 之间"
        assert self.num_sink_tokens >= 0, "num_sink_tokens 必须 >= 0"
        assert self.window_size > 0, "window_size 必须 > 0"
    
    def compress_cache(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """优化的复合压缩"""
        seq_len = key_states.shape[2]
        target_size = max(int(seq_len * (1 - self.compression_ratio)), self.num_sink_tokens + 1)
        
        if target_size >= seq_len:
            return key_states, value_states
        
        device = key_states.device
        
        # ==================== 步骤1: 保留必需的tokens ====================
        num_sink = min(self.num_sink_tokens, seq_len, target_size)
        num_recent = max(int(target_size * 0.3), 1)  # 至少30%给最近的
        num_recent = min(num_recent, seq_len - num_sink)
        
        sink_indices = torch.arange(num_sink, device=device)
        recent_start = max(num_sink, seq_len - num_recent)
        recent_indices = torch.arange(recent_start, seq_len, device=device)
        
        must_keep = torch.cat([sink_indices, recent_indices]).unique()
        
        if len(must_keep) >= target_size:
            return self._gather_by_indices(key_states, value_states, must_keep[:target_size])
        
        # ==================== 步骤2: 在中间区域选择 ====================
        all_indices = torch.arange(seq_len, device=device)
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask[must_keep] = False
        candidate_indices = all_indices[mask]
        
        num_to_select = target_size - len(must_keep)
        
        if len(candidate_indices) == 0 or num_to_select <= 0:
            return self._gather_by_indices(key_states, value_states, must_keep)
        
        # 简化评分：只用 knorm 和 attention（如果启用）
        candidate_keys = key_states[:, :, candidate_indices, :]
        
        # Key norm分数
        key_norms = torch.norm(candidate_keys, dim=-1)  # [B, H, C]
        knorm_scores = key_norms.mean(dim=1)  # [B, C]
        
        # Attention分数（可选）
        if self.use_attention and seq_len > self.window_size:
            window_start = max(0, seq_len - self.window_size)
            query_window = key_states[:, :, window_start:, :]
            
            attn_scores = torch.matmul(
                query_window,
                candidate_keys.transpose(-2, -1)
            ).mean(dim=2).mean(dim=1)  # [B, C]
            
            # 归一化
            attn_scores = attn_scores / (attn_scores.max(dim=-1, keepdim=True)[0] + 1e-8)
            knorm_scores = knorm_scores / (knorm_scores.max(dim=-1, keepdim=True)[0] + 1e-8)
            
            # 简单平均组合
            combined_scores = 0.4 * knorm_scores + 0.6 * attn_scores
        else:
            combined_scores = knorm_scores
        
        # 对batch取平均
        final_scores = combined_scores.mean(dim=0)
        
        # 选择top-k
        _, top_k = torch.topk(final_scores, k=min(num_to_select, len(candidate_indices)), sorted=False)
        selected = candidate_indices[top_k]
        
        # ==================== 步骤3: 合并并排序 ====================
        final_indices = torch.cat([must_keep, selected]).sort()[0]
        
        return self._gather_by_indices(key_states, value_states, final_indices)
    
    def _gather_by_indices(self, key_states: torch.Tensor, value_states: torch.Tensor, 
                          indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据索引收集 key 和 value"""
        batch_size, num_heads, _, head_dim = key_states.shape
        indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        indices = indices.expand(batch_size, num_heads, -1, head_dim)
        
        compressed_key = torch.gather(key_states, dim=2, index=indices)
        compressed_value = torch.gather(value_states, dim=2, index=indices)
        
        return compressed_key, compressed_value
    
    def __call__(self, model):
        """使用 monkey-patching 拦截 DynamicCache.update"""
        from contextlib import contextmanager
        from transformers.cache_utils import DynamicCache
        
        @contextmanager
        def _context():
            original_update = DynamicCache.update
            
            def wrapped_update(cache_self, key_states: torch.Tensor, value_states: torch.Tensor, 
                             layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None):
                key_states, value_states = self.compress_cache(key_states, value_states)
                return original_update(cache_self, key_states, value_states, layer_idx, cache_kwargs)
            
            try:
                DynamicCache.update = wrapped_update
                yield
            finally:
                DynamicCache.update = original_update
        
        return _context()


@dataclass
class GPTNeoXHybridPress(BasePress):
    """
    复合压缩方法 - 结合多种策略的优势
    
    策略组合：
    1. 保留 sink tokens（开头的重要tokens）- 来自 StreamingPress
    2. 基于 attention scores 选择重要 tokens - 来自 SnapKVPress  
    3. 基于 key norm 过滤低重要性 tokens - 来自 KnormPress
    4. 保留最近的 tokens（局部性）- 来自 StreamingPress
    
    Parameters
    ----------
    compression_ratio : float, default=0.5
        保留的 KV 缓存比例
    num_sink_tokens : int, default=8
        保留的开头 tokens 数量（增加到8以保留更多初始context）
    window_size : int, default=64
        用于计算 attention score 的窗口大小（增加窗口以获得更好的全局视图）
    knorm_threshold : float, default=0.0
        key norm 的阈值百分比（设为0禁用过滤，避免误删重要tokens）
    recent_ratio : float, default=0.4
        分配给最近tokens的比例（增加到0.4，因为局部性很重要）
    """
    
    compression_ratio: float = 0.5
    num_sink_tokens: int = 8
    window_size: int = 64
    knorm_threshold: float = 0.0
    recent_ratio: float = 0.4
    
    def __post_init__(self):
        assert 0 < self.compression_ratio <= 1, "compression_ratio 必须在 (0, 1] 之间"
        assert self.num_sink_tokens >= 0, "num_sink_tokens 必须 >= 0"
        assert self.window_size > 0, "window_size 必须 > 0"
        assert 0 <= self.knorm_threshold < 1, "knorm_threshold 必须在 [0, 1) 之间"
    
    def compress_cache(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """复合压缩策略"""
        seq_len = key_states.shape[2]
        target_size = max(int(seq_len * (1 - self.compression_ratio)), self.num_sink_tokens + 1)
        
        if target_size >= seq_len:
            return key_states, value_states
        
        batch_size, num_heads, _, head_dim = key_states.shape
        device = key_states.device
        
        # ==================== 步骤 1: 确定必须保留的 tokens ====================
        # 1.1 保留开头的 sink tokens
        sink_indices = torch.arange(min(self.num_sink_tokens, seq_len), device=device)
        
        # 1.2 保留最近的一些 tokens（至少保留一部分，确保局部性）
        num_recent = max(int(target_size * 0.2), 1)  # 至少20%给最近的tokens
        recent_start = max(self.num_sink_tokens, seq_len - num_recent)
        recent_indices = torch.arange(recent_start, seq_len, device=device)
        
        # 必须保留的 indices
        must_keep_indices = torch.cat([sink_indices, recent_indices]).unique()
        
        # ==================== 步骤 2: 对剩余 tokens 进行评分 ====================
        # 可选择的 token 范围（排除必须保留的）
        all_indices = torch.arange(seq_len, device=device)
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask[must_keep_indices] = False
        candidate_indices = all_indices[mask]
        
        if len(candidate_indices) == 0 or len(must_keep_indices) >= target_size:
            # 已经达到目标大小
            return self._gather_by_indices(key_states, value_states, must_keep_indices[:target_size])
        
        # 计算剩余需要选择的数量
        num_to_select = target_size - len(must_keep_indices)
        
        if num_to_select <= 0:
            return self._gather_by_indices(key_states, value_states, must_keep_indices[:target_size])
        
        # ==================== 步骤 3: 多维度评分 ====================
        # 3.1 基于 key norm 的分数（KnormPress）
        key_norms = torch.norm(key_states[:, :, candidate_indices, :], dim=-1)  # [B, H, C]
        knorm_scores = key_norms / (key_norms.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        # 3.2 基于 attention 的分数（SnapKVPress）
        if seq_len > self.window_size:
            window_start = max(0, seq_len - self.window_size)
            query_window = key_states[:, :, window_start:, :]  # [B, H, W, D]
            
            # 只计算候选位置的 attention scores
            candidate_keys = key_states[:, :, candidate_indices, :]  # [B, H, C, D]
            attn_scores = torch.matmul(
                query_window,  # [B, H, W, D]
                candidate_keys.transpose(-2, -1)  # [B, H, D, C]
            )  # [B, H, W, C]
            attn_scores = attn_scores.mean(dim=2)  # [B, H, C]
            attn_scores = attn_scores / (attn_scores.max(dim=-1, keepdim=True)[0] + 1e-8)
        else:
            attn_scores = knorm_scores  # fallback
        
        # 3.3 位置分数（中间的 tokens 可能更重要）
        position_scores = torch.ones_like(knorm_scores)
        if len(candidate_indices) > 0:
            positions = candidate_indices.float() / seq_len
            # 使用高斯函数，中间位置得分更高
            position_scores = position_scores * torch.exp(-((positions - 0.5) ** 2) / 0.3).unsqueeze(0).unsqueeze(0)
        
        # ==================== 步骤 4: 综合评分 ====================
        # 权重: knorm(30%) + attention(50%) + position(20%)
        combined_scores = (
            0.3 * knorm_scores + 
            0.5 * attn_scores + 
            0.2 * position_scores
        )  # [B, H, C]
        
        # 对所有 heads 取平均
        final_scores = combined_scores.mean(dim=1)  # [B, C]
        
        # 过滤掉 norm 太小的 tokens
        if self.knorm_threshold > 0:
            knorm_threshold_val = knorm_scores.mean(dim=1) * self.knorm_threshold  # [B, C]
            valid_mask = (knorm_scores.mean(dim=1) >= knorm_threshold_val)
            final_scores = final_scores * valid_mask.float()
        
        # ==================== 步骤 5: 选择 top-k ====================
        # 对batch取平均（简化处理）
        final_scores = final_scores.mean(dim=0)  # [C]
        
        _, top_k_indices = torch.topk(
            final_scores, 
            k=min(num_to_select, len(candidate_indices)), 
            sorted=False
        )
        
        selected_indices = candidate_indices[top_k_indices]
        
        # ==================== 步骤 6: 合并所有保留的 indices ====================
        final_indices = torch.cat([must_keep_indices, selected_indices])
        final_indices = final_indices.sort()[0]  # 保持原始顺序
        
        return self._gather_by_indices(key_states, value_states, final_indices)
    
    def _gather_by_indices(self, key_states: torch.Tensor, value_states: torch.Tensor, 
                          indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据索引收集 key 和 value"""
        batch_size, num_heads, _, head_dim = key_states.shape
        indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        indices = indices.expand(batch_size, num_heads, -1, head_dim)
        
        compressed_key = torch.gather(key_states, dim=2, index=indices)
        compressed_value = torch.gather(value_states, dim=2, index=indices)
        
        return compressed_key, compressed_value
    
    def __call__(self, model):
        """使用 monkey-patching 拦截 DynamicCache.update"""
        from contextlib import contextmanager
        from transformers.cache_utils import DynamicCache
        
        @contextmanager
        def _context():
            original_update = DynamicCache.update
            
            def wrapped_update(cache_self, key_states: torch.Tensor, value_states: torch.Tensor, 
                             layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None):
                key_states, value_states = self.compress_cache(key_states, value_states)
                return original_update(cache_self, key_states, value_states, layer_idx, cache_kwargs)
            
            try:
                DynamicCache.update = wrapped_update
                yield
            finally:
                DynamicCache.update = original_update
        
        return _context()
