"""Clean, reversible monkey-patch for Qwen2Attention enabling O(L) KV-cache compression.

Executes attention via memory-efficient ``F.scaled_dot_product_attention`` (SDPA)
and delegates KV-cache eviction to an attached ``BaseEvictionPolicy`` without ever
materializing an L x L attention matrix.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .policies import BaseEvictionPolicy, FullCachePolicy, set_layer_kv


def _check_sdpa_enable_gqa() -> bool:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (2, 5)


_SDPA_SUPPORTS_ENABLE_GQA = _check_sdpa_enable_gqa()


def _make_patched_forward(policy: BaseEvictionPolicy):
    """Build a Qwen2Attention.forward replacement bound to ``policy``."""

    def patched_qwen2_attention_forward(
        self: Qwen2Attention,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Support both `past_key_value` (transformers <=4.55) and `past_key_values` (>=4.56)
        if past_key_value is None and "past_key_values" in kwargs:
            past_key_value = kwargs["past_key_values"]

        input_shape = hidden_states.shape[:-1]
        q_len = input_shape[-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            # Fallback for older transformers versions that compute RoPE inside attention
            position_ids = kwargs.get("position_ids")
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        k_len = key_states.shape[-2]
        is_prefill = q_len > 1

        # Prepare SDPA causal mask safely (never materialize L x L when avoidable)
        sdpa_mask = None
        if attention_mask is not None:
            if q_len > 1:
                if attention_mask.ndim == 4 and attention_mask.shape[-1] >= k_len:
                    sdpa_mask = attention_mask[:, :, :, :k_len]
            else:
                # Single-token decode: for batch=1 (no padding), every retained KV entry is
                # a valid past token, so drop any mask that could carry stale pre-eviction width.
                if (
                    attention_mask.ndim == 4
                    and input_shape[0] > 1
                    and attention_mask.shape[-1] == k_len
                ):
                    sdpa_mask = attention_mask

        is_causal = is_prefill and sdpa_mask is None and q_len == k_len

        # Memory-efficient SDPA (Flash / MemEfficient kernel on CUDA, native SDPA on CPU)
        use_gqa = _SDPA_SUPPORTS_ENABLE_GQA and (query_states.shape[1] != key_states.shape[1])
        if use_gqa:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                enable_gqa=True,
            )
        else:
            key_for_sdpa = repeat_kv(key_states, self.num_key_value_groups)
            value_for_sdpa = repeat_kv(value_states, self.num_key_value_groups)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_for_sdpa,
                value_for_sdpa,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )

        # Compress KV-cache in-place after computing exact attention for current step
        if past_key_value is not None and not isinstance(policy, FullCachePolicy):
            compressed_k, compressed_v = policy.compress_layer(
                layer_idx=self.layer_idx,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                is_prefill=is_prefill,
            )
            set_layer_kv(past_key_value, self.layer_idx, compressed_k, compressed_v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return patched_qwen2_attention_forward


@contextmanager
def patch_qwen2_attention(
    model: nn.Module,
    policy: BaseEvictionPolicy,
) -> Iterator[None]:
    """Context manager that patches all ``Qwen2Attention`` modules on ``model``."""
    patched_fn = _make_patched_forward(policy)
    originals = []
    policy.reset()

    try:
        for module in model.modules():
            if isinstance(module, Qwen2Attention):
                originals.append((module, module.forward))
                module.forward = patched_fn.__get__(module, Qwen2Attention)
        yield
    finally:
        for module, orig_forward in originals:
            module.forward = orig_forward
