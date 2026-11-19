"""Manual prefill + autoregressive decode loop with strict logical RoPE position tracking."""

from __future__ import annotations

import inspect
import time
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from .attention_patch import patch_qwen2_attention
from .metrics import (
    GenerationMetrics,
    PeakVRAMTracker,
    get_kv_cache_memory_mb,
    theoretical_kv_cache_mb,
)
from .policies import BaseEvictionPolicy, get_cache_seq_len


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_with_policy(
    model: nn.Module,
    input_ids: torch.Tensor,
    policy: BaseEvictionPolicy,
    max_new_tokens: int = 20,
    eos_token_id: Optional[int] = None,
    stop_fn: Optional[Callable[[torch.Tensor], bool]] = None,
    return_cache: bool = False,
) -> Tuple[torch.Tensor, GenerationMetrics] | Tuple[torch.Tensor, GenerationMetrics, DynamicCache]:
    """Run greedy prefill + autoregressive decoding under a KV-cache eviction policy.

    Crucially, after KV-cache compression shrinks ``cache.get_seq_length()`` from
    ``L`` down to ``budget``, this loop explicitly tracks ``logical_pos = L, L+1, ...``
    and passes both ``position_ids`` and ``cache_position`` using ``logical_pos`` so
    RoPE rotary phases never corrupt after eviction.

    Args:
        model: ``Qwen2ForCausalLM`` instance in eval mode.
        input_ids: Prompt token IDs of shape ``[1, L]``.
        policy: Eviction policy (FullCache, SlidingWindow, H2O, SnapKV).
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_id: Optional EOS token ID to terminate generation early.
        stop_fn: Optional callback receiving generated token IDs ``[1, t]``; returns
            ``True`` to stop early (e.g., once a NIAH passkey is emitted).
        return_cache: If ``True``, also return the final ``DynamicCache``.

    Returns:
        ``(generated_ids, metrics)`` or ``(generated_ids, metrics, past_key_values)``.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids of shape [1, L], got {tuple(input_ids.shape)}")

    device = input_ids.device
    prompt_len = int(input_ids.shape[1])
    past_key_values = DynamicCache()

    generated_tokens = []

    # Avoid projecting lm_head over all L prompt tokens during prefill (~2.5-5 GB VRAM saved at 8K)
    fwd_params = inspect.signature(model.forward).parameters
    prefill_extra = {}
    if "logits_to_keep" in fwd_params:
        prefill_extra["logits_to_keep"] = 1
    elif "num_logits_to_keep" in fwd_params:
        prefill_extra["num_logits_to_keep"] = 1

    with PeakVRAMTracker(device) as vram_tracker, patch_qwen2_attention(model, policy):
        # ------------------------------------------------------------------
        # 1. Prefill Phase: process prompt tokens [0, L)
        # ------------------------------------------------------------------
        prefill_pos = torch.arange(0, prompt_len, device=device, dtype=torch.long)
        position_ids = prefill_pos.unsqueeze(0)  # [1, L]

        _sync_cuda(device)
        t0 = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=prefill_pos,
            past_key_values=past_key_values,
            use_cache=True,
            **prefill_extra,
        )
        _sync_cuda(device)
        prefill_time_s = time.perf_counter() - t0

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        del outputs
        generated_tokens.append(next_token)

        # ------------------------------------------------------------------
        # 2. Autoregressive Decode Phase with strict logical RoPE positions
        # ------------------------------------------------------------------
        logical_pos = prompt_len
        _sync_cuda(device)
        t_decode_start = time.perf_counter()

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
            if stop_fn is not None and stop_fn(torch.cat(generated_tokens, dim=-1)):
                break

            # RoPE uses strict logical_pos (L, L+1, ...), while cache_position tracks
            # physical cache length so transformers >= 4.53 create_causal_mask never mismatches.
            physical_cache_len = get_cache_seq_len(past_key_values, layer_idx=0)
            step_position_ids = torch.tensor([[logical_pos]], device=device, dtype=torch.long)
            step_cache_position = torch.tensor([physical_cache_len], device=device, dtype=torch.long)
            outputs = model(
                input_ids=next_token,
                position_ids=step_position_ids,      # [1, 1] -> logical RoPE position
                cache_position=step_cache_position,  # [1]    -> physical cache slot
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            del outputs
            generated_tokens.append(next_token)
            logical_pos += 1

        _sync_cuda(device)
        decode_time_s = time.perf_counter() - t_decode_start

    generated_ids = torch.cat(generated_tokens, dim=-1) if generated_tokens else input_ids[:, :0]
    num_gen = int(generated_ids.shape[-1])
    kv_len = get_cache_seq_len(past_key_values, layer_idx=0)
    kv_mb = get_kv_cache_memory_mb(past_key_values)

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    param_dtype = next(model.parameters()).dtype
    total_logical_len = prompt_len + max(0, num_gen - 1)
    full_kv_mb = theoretical_kv_cache_mb(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        seq_len=total_logical_len,
        dtype=param_dtype,
        batch_size=1,
    )
    compression_ratio = (kv_mb / full_kv_mb) if full_kv_mb > 0 else 1.0
    decode_steps = max(0, num_gen - 1)
    decode_tps = (decode_steps / decode_time_s) if decode_time_s > 0 and decode_steps > 0 else 0.0

    metrics = GenerationMetrics(
        policy_name=policy.name,
        budget=policy.budget,
        prompt_tokens=prompt_len,
        generated_tokens=num_gen,
        kv_cache_len=kv_len,
        kv_cache_mb=kv_mb,
        full_kv_cache_mb=full_kv_mb,
        compression_ratio=compression_ratio,
        prefill_time_s=prefill_time_s,
        decode_time_s=decode_time_s,
        decode_tokens_per_s=decode_tps,
        peak_vram_mb=vram_tracker.peak_mb,
    )

    if return_cache:
        return generated_ids, metrics, past_key_values
    del past_key_values
    return generated_ids, metrics
