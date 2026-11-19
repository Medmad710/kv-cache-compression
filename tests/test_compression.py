"""Fast local unit tests using a tiny in-memory Qwen2Config (no downloads)."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from src.attention_patch import patch_qwen2_attention
from src.generator import generate_with_policy
from src.metrics import get_kv_cache_memory_mb, theoretical_kv_cache_mb
from src.policies import (
    FullCachePolicy,
    H2OPolicy,
    SlidingWindowPolicy,
    SnapKVPolicy,
    compute_windowed_importance_scores,
    get_cache_seq_len,
    get_layer_kv,
)


@pytest.fixture(scope="module")
def tiny_qwen2_model() -> Qwen2ForCausalLM:
    """Create a deterministic tiny in-memory Qwen2 model with GQA (4 Q-heads, 2 KV-heads)."""
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def test_full_cache_matches_hf_greedy_exact(tiny_qwen2_model: Qwen2ForCausalLM) -> None:
    """a) FullCache matches standard HuggingFace greedy output 100% token-for-token."""
    torch.manual_seed(0)
    prompt = torch.randint(0, 256, (1, 48), dtype=torch.long)
    max_new_tokens = 12

    with torch.inference_mode():
        hf_out = tiny_qwen2_model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=-1,
        )
    hf_generated = hf_out[:, prompt.shape[1] :]

    our_generated, metrics = generate_with_policy(
        model=tiny_qwen2_model,
        input_ids=prompt,
        policy=FullCachePolicy(),
        max_new_tokens=max_new_tokens,
        eos_token_id=None,
    )

    assert our_generated.shape == hf_generated.shape
    assert torch.equal(our_generated, hf_generated), (
        f"FullCache mismatch!\nOurs: {our_generated.tolist()}\nHF:   {hf_generated.tolist()}"
    )
    assert metrics.kv_cache_len == prompt.shape[1] + max_new_tokens - 1


@pytest.mark.parametrize(
    "policy_factory",
    [
        lambda b: SnapKVPolicy(budget=b, window_size=8, kernel_size=5),
        lambda b: H2OPolicy(budget=b, sink_size=4, recent_ratio=0.5, init_window=16),
        lambda b: SlidingWindowPolicy(budget=b, sink_size=4),
    ],
    ids=["snapkv", "h2o", "sliding_window"],
)
def test_policies_strictly_enforce_budget_prefill_and_decode(
    tiny_qwen2_model: Qwen2ForCausalLM,
    policy_factory,
) -> None:
    """b) SnapKV, H2O, and SlidingWindow strictly enforce cache_len <= budget
    after prefill and at every single autoregressive decode step."""
    torch.manual_seed(123)
    prompt_len = 64
    budget = 24
    decode_steps = 10
    prompt = torch.randint(0, 256, (1, prompt_len), dtype=torch.long)

    policy = policy_factory(budget)
    past_key_values = DynamicCache()

    with torch.inference_mode(), patch_qwen2_attention(tiny_qwen2_model, policy):
        # Prefill
        pos = torch.arange(0, prompt_len, dtype=torch.long)
        out = tiny_qwen2_model(
            input_ids=prompt,
            position_ids=pos.unsqueeze(0),
            cache_position=pos,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        for layer_idx in range(tiny_qwen2_model.config.num_hidden_layers):
            assert get_cache_seq_len(past_key_values, layer_idx) <= budget

        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        logical_pos = prompt_len

        # Autoregressive decode with explicit logical RoPE positions & physical cache_position
        for _ in range(decode_steps):
            physical_len = get_cache_seq_len(past_key_values, layer_idx=0)
            out = tiny_qwen2_model(
                input_ids=next_tok,
                position_ids=torch.tensor([[logical_pos]], dtype=torch.long),
                cache_position=torch.tensor([physical_len], dtype=torch.long),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            assert torch.isfinite(out.logits).all(), "Logits contain NaN/Inf after eviction!"
            for layer_idx in range(tiny_qwen2_model.config.num_hidden_layers):
                assert get_cache_seq_len(past_key_values, layer_idx) <= budget
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            logical_pos += 1


@pytest.mark.parametrize("budget_ratio", [0.25, 0.5])
def test_compressed_kv_cache_memory_matches_ratio(
    tiny_qwen2_model: Qwen2ForCausalLM,
    budget_ratio: float,
) -> None:
    """c) Compressed KV-cache tensor memory matches the expected compression ratio."""
    torch.manual_seed(7)
    prompt_len = 80
    budget = int(prompt_len * budget_ratio)
    prompt = torch.randint(0, 256, (1, prompt_len), dtype=torch.long)

    _, full_metrics, full_cache = generate_with_policy(
        model=tiny_qwen2_model,
        input_ids=prompt,
        policy=FullCachePolicy(),
        max_new_tokens=1,
        return_cache=True,
    )
    _, snap_metrics, snap_cache = generate_with_policy(
        model=tiny_qwen2_model,
        input_ids=prompt,
        policy=SnapKVPolicy(budget=budget, window_size=8, kernel_size=5),
        max_new_tokens=1,
        return_cache=True,
    )

    cfg = tiny_qwen2_model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    expected_full_mb = theoretical_kv_cache_mb(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        seq_len=prompt_len,
        dtype=torch.float32,
    )
    expected_snap_mb = theoretical_kv_cache_mb(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        seq_len=budget,
        dtype=torch.float32,
    )

    assert abs(get_kv_cache_memory_mb(full_cache) - expected_full_mb) < 1e-6
    assert abs(get_kv_cache_memory_mb(snap_cache) - expected_snap_mb) < 1e-6
    assert abs(snap_metrics.kv_cache_mb / full_metrics.kv_cache_mb - budget_ratio) < 1e-5


def test_gqa_head_group_aggregation_valid_and_finite() -> None:
    """d) GQA head-group aggregation produces valid shapes and no NaNs/Infs."""
    torch.manual_seed(99)
    bsz, num_q_heads, num_kv_heads, seq_len, head_dim = 1, 16, 2, 128, 32

    # Stress-test with FP16 inputs to verify FP32 softmax prevents overflow/NaNs
    q = torch.randn(bsz, num_q_heads, seq_len, head_dim, dtype=torch.float16) * 10.0
    k = torch.randn(bsz, num_kv_heads, seq_len, head_dim, dtype=torch.float16) * 10.0

    scores = compute_windowed_importance_scores(
        query_states=q,
        key_states=k,
        window_size=32,
        apply_causal_mask=True,
    )

    assert scores.shape == (bsz, num_kv_heads, seq_len)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all(), "Importance scores contain NaN or Inf!"
    assert (scores >= 0).all(), "Softmax-aggregated scores must be non-negative."
    # Sum of probabilities across keys over 32 observation queries must equal ~32.0 per KV-head
    total_mass = scores.sum(dim=-1)
    assert torch.allclose(total_mass, torch.full_like(total_mass, 32.0), atol=1e-3)


def test_h2o_and_snapkv_preserve_kv_shapes(tiny_qwen2_model: Qwen2ForCausalLM) -> None:
    """Verify compressed K and V tensors preserve [B, num_kv_heads, budget, head_dim]."""
    torch.manual_seed(11)
    prompt = torch.randint(0, 256, (1, 60), dtype=torch.long)
    budget = 20

    for policy in [
        SnapKVPolicy(budget=budget, window_size=8, kernel_size=5),
        H2OPolicy(budget=budget, sink_size=4, recent_ratio=0.5, init_window=16),
    ]:
        _, _, cache = generate_with_policy(
            model=tiny_qwen2_model,
            input_ids=prompt,
            policy=policy,
            max_new_tokens=5,
            return_cache=True,
        )
        k0, v0 = get_layer_kv(cache, 0)
        assert k0 is not None and v0 is not None
        assert k0.shape == (1, 2, budget, 16)
        assert v0.shape == (1, 2, budget, 16)
        assert torch.isfinite(k0).all() and torch.isfinite(v0).all()


def test_logical_rope_positions_preserved_after_eviction(
    tiny_qwen2_model: Qwen2ForCausalLM,
) -> None:
    """Verify rotary_emb receives strict logical_pos (L, L+1, ...) after KV eviction."""
    torch.manual_seed(55)
    prompt_len = 64
    budget = 20
    max_new_tokens = 6
    prompt = torch.randint(0, 256, (1, prompt_len), dtype=torch.long)

    observed_pos_ids = []
    orig_rotary = tiny_qwen2_model.model.rotary_emb.forward

    def spy_rotary(x, position_ids, *args, **kwargs):
        observed_pos_ids.append(position_ids.detach().cpu().tolist())
        return orig_rotary(x, position_ids, *args, **kwargs)

    tiny_qwen2_model.model.rotary_emb.forward = spy_rotary
    try:
        gen_ids, metrics = generate_with_policy(
            model=tiny_qwen2_model,
            input_ids=prompt,
            policy=SnapKVPolicy(budget=budget, window_size=8, kernel_size=5),
            max_new_tokens=max_new_tokens,
            eos_token_id=None,
        )
    finally:
        tiny_qwen2_model.model.rotary_emb.forward = orig_rotary

    assert gen_ids.shape == (1, max_new_tokens)
    assert metrics.kv_cache_len <= budget
    # Step 0 is prefill [0..prompt_len-1]; steps 1..max_new_tokens-1 must be [prompt_len], [prompt_len+1], ...
    assert observed_pos_ids[0] == [list(range(prompt_len))]
    for step_idx in range(1, max_new_tokens):
        assert observed_pos_ids[step_idx] == [[prompt_len + step_idx - 1]]


def test_niah_question_fits_in_snapkv_observation_window() -> None:
    """Verify the NIAH question sits at the very end of the prompt within SnapKV's 32-token window."""
    from benchmarks.needle_haystack import build_niah_prompt, evaluate_passkey

    class _WhitespaceTokenizer:
        bos_token_id = 1
        chat_template = None

        def __init__(self):
            self._vocab = {"<bos>": 1}
            self._inv = {1: "<bos>"}

        def encode(self, text: str, add_special_tokens: bool = False):
            ids = []
            for tok in text.split():
                if tok not in self._vocab:
                    idx = len(self._vocab) + 1
                    self._vocab[tok] = idx
                    self._inv[idx] = tok
                ids.append(self._vocab[tok])
            return ids

        def decode(self, ids, skip_special_tokens: bool = True):
            return " ".join(self._inv[i] for i in ids if not (skip_special_tokens and i == 1))

    tok = _WhitespaceTokenizer()
    input_ids, prompt_text = build_niah_prompt(tok, target_context_len=256, depth=0.3, passkey="749201")
    assert input_ids.shape == (1, 256)
    tail_32_text = tok.decode(input_ids[0, -32:].tolist())
    assert "Question:" in tail_32_text and "passkey" in tail_32_text
    assert "749201" in prompt_text
    assert evaluate_passkey("The passkey is 749201.", "749201")
    assert not evaluate_passkey("The passkey is 123456.", "749201")
