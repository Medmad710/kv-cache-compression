"""KV-cache eviction policies: FullCache, SlidingWindow, H2O, and SnapKV.

All attention importance scoring is strictly O(L) in memory (never materializes
an L x L attention matrix) and computes softmax in float32 to prevent FP16
overflow/NaNs on long sequences.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DynamicCache cross-version adapter (supports key_cache/value_cache & layers)
# ---------------------------------------------------------------------------


def get_num_layers(cache: Any) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return len(cache)


def get_layer_kv(cache: Any, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if hasattr(cache, "layers"):
        if layer_idx >= len(cache.layers):
            return None, None
        layer = cache.layers[layer_idx]
        return getattr(layer, "keys", None), getattr(layer, "values", None)
    if hasattr(cache, "key_cache"):
        if layer_idx >= len(cache.key_cache):
            return None, None
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    return cache[layer_idx][0], cache[layer_idx][1]


def set_layer_kv(
    cache: Any,
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        layer.keys = key_states
        layer.values = value_states
    elif hasattr(cache, "key_cache"):
        cache.key_cache[layer_idx] = key_states
        cache.value_cache[layer_idx] = value_states
    else:
        raise TypeError(f"Unsupported cache object type: {type(cache)}")


def get_cache_seq_len(cache: Any, layer_idx: int = 0) -> int:
    if cache is None:
        return 0
    k, _ = get_layer_kv(cache, layer_idx)
    if k is not None and k.numel() > 0:
        return int(k.shape[-2])
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length(layer_idx))
    return 0


# ---------------------------------------------------------------------------
# O(L) Memory-Safe Attention Importance Scoring (FP32 Softmax + GQA Grouping)
# ---------------------------------------------------------------------------


def compute_windowed_importance_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    window_size: int,
    apply_causal_mask: bool = True,
) -> torch.Tensor:
    """Compute O(W * L) FP32 attention importance scores aggregated per KV-head.

    Args:
        query_states: ``[B, num_q_heads, q_len, head_dim]``
        key_states:   ``[B, num_kv_heads, k_len, head_dim]``
        window_size:  Number of trailing query tokens to observe (W <= q_len).
        apply_causal_mask: Whether to mask future keys inside the trailing W x W block.

    Returns:
        Importance scores of shape ``[B, num_kv_heads, k_len]`` in ``torch.float32``.
    """
    bsz, num_q_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, k_len, _ = key_states.shape
    num_groups = num_q_heads // num_kv_heads

    obs_len = min(window_size, q_len)
    # Slice ONLY the trailing observation queries -> O(obs_len * k_len) memory
    q_obs = query_states[:, :, -obs_len:, :]  # [B, H_q, W, D]
    # Reshape into KV-head groups so we never duplicate key_states
    q_grouped = q_obs.reshape(bsz, num_kv_heads, num_groups * obs_len, head_dim)

    scale = 1.0 / math.sqrt(head_dim)
    # Compute logits in float32 to avoid FP16 overflow on long sequences
    attn_logits = torch.matmul(
        q_grouped.float(), key_states.float().transpose(-2, -1)
    ) * scale  # [B, H_kv, G * W, L]
    attn_logits = attn_logits.view(bsz, num_kv_heads, num_groups, obs_len, k_len)

    if apply_causal_mask and obs_len > 1 and k_len >= obs_len:
        # Query i (0..obs_len-1) sits at position (k_len - obs_len + i).
        # Keys at j > k_len - obs_len + i are strictly future tokens.
        causal_tri = torch.triu(
            torch.ones((obs_len, obs_len), dtype=torch.bool, device=attn_logits.device),
            diagonal=1,
        )
        attn_logits[:, :, :, :, -obs_len:] = attn_logits[:, :, :, :, -obs_len:].masked_fill(
            causal_tri.view(1, 1, 1, obs_len, obs_len), float("-inf")
        )

    # Strictly FP32 softmax
    attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32)
    # Aggregate across GQA query-head groups (mean) and observation queries (sum)
    # Shape: [B, num_kv_heads, k_len]
    kv_scores = attn_probs.mean(dim=2).sum(dim=2)
    return kv_scores


def _gather_kv_by_indices(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather K and V along seq_len dim using per-KV-head indices [B, H_kv, K]."""
    head_dim = key_states.shape[-1]
    expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_kept = torch.gather(key_states, dim=2, index=expanded)
    v_kept = torch.gather(value_states, dim=2, index=expanded)
    return k_kept, v_kept


# ---------------------------------------------------------------------------
# Eviction Policy Hierarchy
# ---------------------------------------------------------------------------


class BaseEvictionPolicy(ABC):
    """Base interface for per-layer KV-cache eviction policies."""

    name: str = "base"

    def __init__(self, budget: Optional[int] = None):
        if budget is not None and budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        self.budget = budget

    def reset(self) -> None:
        """Reset any per-sequence internal state before a new generation."""

    @abstractmethod
    def compress_layer(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        is_prefill: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optionally compress a layer's KV-cache after attention computation."""


class FullCachePolicy(BaseEvictionPolicy):
    """Uncompressed Full KV-Cache baseline."""

    name = "full"

    def __init__(self, budget: Optional[int] = None):
        super().__init__(budget=None)

    def compress_layer(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        is_prefill: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return key_states, value_states


class SlidingWindowPolicy(BaseEvictionPolicy):
    """StreamingLLM-style Sliding Window + Attention Sinks baseline.

    Retains the first ``sink_size`` tokens plus the most recent
    ``budget - sink_size`` tokens.
    """

    name = "sliding_window"

    def __init__(self, budget: int, sink_size: int = 4):
        super().__init__(budget=budget)
        self.sink_size = sink_size

    def compress_layer(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        is_prefill: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = key_states.shape[-2]
        if self.budget is None or seq_len <= self.budget:
            return key_states, value_states

        eff_sink = min(self.sink_size, self.budget)
        recent = self.budget - eff_sink

        if eff_sink == 0:
            return (
                key_states[:, :, -recent:, :].contiguous(),
                value_states[:, :, -recent:, :].contiguous(),
            )
        if recent == 0:
            return (
                key_states[:, :, :eff_sink, :].contiguous(),
                value_states[:, :, :eff_sink, :].contiguous(),
            )

        k_out = torch.cat([key_states[:, :, :eff_sink, :], key_states[:, :, -recent:, :]], dim=2)
        v_out = torch.cat([value_states[:, :, :eff_sink, :], value_states[:, :, -recent:, :]], dim=2)
        return k_out, v_out


class SnapKVPolicy(BaseEvictionPolicy):
    """SnapKV (NeurIPS 2024): Observation-window pooled KV-cache compression.

    During prefill, computes O(W * L) FP32 attention weights for the last
    ``window_size`` queries against all keys, aggregates across GQA query-head
    groups, applies 1D average pooling (kernel_size=5, padding=2) over the
    prefix scores, and selects the top ``budget - window_size`` prefix tokens
    plus the final ``window_size`` recent tokens.
    """

    name = "snapkv"

    def __init__(self, budget: int, window_size: int = 32, kernel_size: int = 5):
        super().__init__(budget=budget)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding.")
        self.window_size = window_size
        self.kernel_size = kernel_size

    def compress_layer(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        is_prefill: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = key_states.shape[-2]
        if self.budget is None or seq_len <= self.budget:
            return key_states, value_states

        eff_window = min(self.window_size, self.budget, seq_len)
        prefix_budget = self.budget - eff_window

        if prefix_budget <= 0:
            return (
                key_states[:, :, -self.budget :, :].contiguous(),
                value_states[:, :, -self.budget :, :].contiguous(),
            )

        if not is_prefill:
            # During decode, preserve the SnapKV-selected prefix and keep a
            # sliding observation window of size eff_window so cache_len <= budget.
            k_out = torch.cat(
                [key_states[:, :, :prefix_budget, :], key_states[:, :, -eff_window:, :]], dim=2
            )
            v_out = torch.cat(
                [value_states[:, :, :prefix_budget, :], value_states[:, :, -eff_window:, :]], dim=2
            )
            return k_out, v_out

        # Prefill compression: O(W * L) attention scoring in FP32
        kv_scores = compute_windowed_importance_scores(
            query_states=query_states,
            key_states=key_states,
            window_size=eff_window,
            apply_causal_mask=True,
        )  # [B, H_kv, L]

        prefix_len = seq_len - eff_window
        prefix_scores = kv_scores[:, :, :prefix_len]  # [B, H_kv, L - W]

        # 1D average pooling over prefix scores (SnapKV clustering)
        if prefix_len >= self.kernel_size:
            pooled_scores = F.avg_pool1d(
                prefix_scores,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )
        else:
            pooled_scores = prefix_scores

        # Select top prefix_budget tokens per KV-head and sort to preserve causal order
        topk_idx = torch.topk(pooled_scores, k=prefix_budget, dim=-1, largest=True).indices
        topk_idx, _ = torch.sort(topk_idx, dim=-1)

        k_prefix, v_prefix = _gather_kv_by_indices(
            key_states[:, :, :prefix_len, :],
            value_states[:, :, :prefix_len, :],
            topk_idx,
        )
        k_recent = key_states[:, :, -eff_window:, :]
        v_recent = value_states[:, :, -eff_window:, :]

        return torch.cat([k_prefix, k_recent], dim=2), torch.cat([v_prefix, v_recent], dim=2)


class H2OPolicy(BaseEvictionPolicy):
    """H2O — Heavy-Hitter Oracle (NeurIPS 2023).

    Maintains per-layer accumulated Heavy-Hitter importance scores per KV-head.
    - Prefill: initializes scores via a memory-safe windowed query pass
      (last ``init_window`` queries, default 128) in FP32 without O(L^2) memory.
    - Decode: accumulates each step's 1 x cache_len FP32 attention scores.
    - Eviction: retains ``sink_size`` (first 4 tokens) + top Heavy-Hitters +
      ``recent_size`` tokens up to ``budget``.
    """

    name = "h2o"

    def __init__(
        self,
        budget: int,
        sink_size: int = 4,
        recent_ratio: float = 0.5,
        recent_size: Optional[int] = None,
        init_window: int = 128,
    ):
        super().__init__(budget=budget)
        self.sink_size = sink_size
        self.recent_ratio = recent_ratio
        self.recent_size = recent_size
        self.init_window = init_window
        self._hh_scores: Dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self._hh_scores.clear()

    def _split_budget(self) -> Tuple[int, int, int]:
        assert self.budget is not None
        eff_sink = min(self.sink_size, self.budget)
        rem = self.budget - eff_sink
        if self.recent_size is not None:
            eff_recent = min(self.recent_size, rem)
        else:
            eff_recent = min(max(1, int(rem * self.recent_ratio)), rem) if rem > 0 else 0
        hh_budget = max(0, rem - eff_recent)
        return eff_sink, hh_budget, eff_recent

    def compress_layer(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        is_prefill: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_kv_heads, seq_len, _ = key_states.shape
        device = key_states.device

        if is_prefill or layer_idx not in self._hh_scores:
            step_scores = compute_windowed_importance_scores(
                query_states=query_states,
                key_states=key_states,
                window_size=self.init_window,
                apply_causal_mask=True,
            )  # [B, H_kv, L]
            self._hh_scores[layer_idx] = step_scores
        else:
            # Decode step: query_states has q_len=1, key_states has k_len=seq_len
            step_scores = compute_windowed_importance_scores(
                query_states=query_states,
                key_states=key_states,
                window_size=query_states.shape[-2],
                apply_causal_mask=False,
            )  # [B, H_kv, seq_len]
            prev = self._hh_scores[layer_idx]
            added = seq_len - prev.shape[-1]
            if added > 0:
                pad = torch.zeros(
                    (bsz, num_kv_heads, added), dtype=torch.float32, device=device
                )
                prev = torch.cat([prev, pad], dim=-1)
            elif added < 0:
                prev = prev[:, :, -seq_len:]
            self._hh_scores[layer_idx] = prev + step_scores

        if self.budget is None or seq_len <= self.budget:
            return key_states, value_states

        eff_sink, hh_budget, eff_recent = self._split_budget()
        scores = self._hh_scores[layer_idx]  # [B, H_kv, seq_len]

        parts_idx = []
        if eff_sink > 0:
            sink_idx = (
                torch.arange(0, eff_sink, device=device, dtype=torch.long)
                .view(1, 1, eff_sink)
                .expand(bsz, num_kv_heads, eff_sink)
            )
            parts_idx.append(sink_idx)

        mid_start = eff_sink
        mid_end = seq_len - eff_recent
        if hh_budget > 0 and mid_end > mid_start:
            mid_scores = scores[:, :, mid_start:mid_end]
            k_select = min(hh_budget, mid_scores.shape[-1])
            hh_rel = torch.topk(mid_scores, k=k_select, dim=-1, largest=True).indices
            hh_idx, _ = torch.sort(hh_rel + mid_start, dim=-1)
            parts_idx.append(hh_idx)

        if eff_recent > 0:
            recent_idx = (
                torch.arange(seq_len - eff_recent, seq_len, device=device, dtype=torch.long)
                .view(1, 1, eff_recent)
                .expand(bsz, num_kv_heads, eff_recent)
            )
            parts_idx.append(recent_idx)

        keep_idx = torch.cat(parts_idx, dim=-1)  # [B, H_kv, <= budget]
        k_kept, v_kept = _gather_kv_by_indices(key_states, value_states, keep_idx)
        self._hh_scores[layer_idx] = torch.gather(scores, dim=-1, index=keep_idx)
        return k_kept, v_kept


def build_policy(name: str, budget: Optional[int] = None, **kwargs: Any) -> BaseEvictionPolicy:
    """Factory helper to instantiate an eviction policy by name."""
    key = name.strip().lower()
    if key in ("full", "full_cache", "fullcache"):
        return FullCachePolicy()
    if budget is None:
        raise ValueError(f"Policy '{name}' requires an explicit integer budget.")
    if key in ("sliding", "sliding_window", "sink", "sinks"):
        return SlidingWindowPolicy(budget=budget, **kwargs)
    if key in ("snapkv", "snap_kv"):
        return SnapKVPolicy(budget=budget, **kwargs)
    if key in ("h2o", "heavy_hitter"):
        return H2OPolicy(budget=budget, **kwargs)
    raise ValueError(f"Unknown policy name: '{name}'")
