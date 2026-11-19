"""KV-Cache Compression from scratch in PyTorch (SnapKV, H2O, Sliding Window)."""

from .metrics import (
    GenerationMetrics,
    PeakVRAMTracker,
    get_kv_cache_memory_mb,
    get_optimal_dtype,
    theoretical_kv_cache_mb,
)
from .policies import (
    BaseEvictionPolicy,
    FullCachePolicy,
    H2OPolicy,
    SlidingWindowPolicy,
    SnapKVPolicy,
    build_policy,
    get_cache_seq_len,
    get_layer_kv,
    get_num_layers,
    set_layer_kv,
)
from .attention_patch import patch_qwen2_attention
from .generator import generate_with_policy

__all__ = [
    "BaseEvictionPolicy",
    "FullCachePolicy",
    "SlidingWindowPolicy",
    "H2OPolicy",
    "SnapKVPolicy",
    "build_policy",
    "get_cache_seq_len",
    "get_layer_kv",
    "get_num_layers",
    "set_layer_kv",
    "patch_qwen2_attention",
    "generate_with_policy",
    "GenerationMetrics",
    "PeakVRAMTracker",
    "get_kv_cache_memory_mb",
    "get_optimal_dtype",
    "theoretical_kv_cache_mb",
]
