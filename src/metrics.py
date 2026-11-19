"""Memory, precision, VRAM, and latency accounting utilities for KV-cache compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


def get_optimal_dtype(device: Optional[torch.device | str] = None) -> torch.dtype:
    """Select hardware-optimal floating-point precision.

    - CUDA with compute capability major < 8 (e.g., Tesla T4 / Turing, Volta):
      strictly ``torch.float16`` for native FP16 Tensor Cores.
    - CUDA with compute capability major >= 8 (Ampere, Ada, Hopper):
      ``torch.bfloat16``.
    - CPU / non-CUDA: ``torch.float32``.
    """
    if device is not None:
        dev = torch.device(device)
        if dev.type != "cuda":
            return torch.float32
        device_idx = dev.index if dev.index is not None else 0
    else:
        if not torch.cuda.is_available():
            return torch.float32
        device_idx = torch.cuda.current_device()

    major, _ = torch.cuda.get_device_capability(device_idx)
    if major < 8:
        return torch.float16
    return torch.bfloat16


def iter_cache_kv(past_key_values: Any):
    """Yield (key_tensor, value_tensor) pairs across DynamicCache API versions."""
    if past_key_values is None:
        return
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is not None and v is not None:
                yield k, v
    elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
            if k is not None and v is not None:
                yield k, v
    elif isinstance(past_key_values, (tuple, list)):
        for item in past_key_values:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                k, v = item[0], item[1]
                if k is not None and v is not None:
                    yield k, v


def get_kv_cache_memory_mb(past_key_values: Any) -> float:
    """Compute exact memory footprint (in MiB) of all tensors stored in KV-cache."""
    total_bytes = 0
    for k, v in iter_cache_kv(past_key_values):
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024.0 * 1024.0)


def theoretical_kv_cache_mb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype,
    batch_size: int = 1,
) -> float:
    """Compute theoretical KV-cache footprint (in MiB) for given dimensions."""
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    total_bytes = 2 * num_layers * batch_size * num_kv_heads * seq_len * head_dim * elem_bytes
    return total_bytes / (1024.0 * 1024.0)


class PeakVRAMTracker:
    """Context manager tracking peak CUDA memory allocated in MiB."""

    def __init__(self, device: Optional[torch.device | str] = None):
        self.device = device
        self.peak_mb: float = 0.0
        self._enabled = False

    def __enter__(self) -> "PeakVRAMTracker":
        dev = torch.device(self.device) if self.device is not None else None
        if torch.cuda.is_available() and (dev is None or dev.type == "cuda"):
            self._enabled = True
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._enabled:
            dev = torch.device(self.device) if self.device is not None else None
            torch.cuda.synchronize(dev)
            self.peak_mb = torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)


@dataclass
class GenerationMetrics:
    """Performance and memory telemetry for a single prefill + decode run."""

    policy_name: str
    budget: Optional[int]
    prompt_tokens: int
    generated_tokens: int
    kv_cache_len: int
    kv_cache_mb: float
    full_kv_cache_mb: float
    compression_ratio: float
    prefill_time_s: float
    decode_time_s: float
    decode_tokens_per_s: float
    peak_vram_mb: float
