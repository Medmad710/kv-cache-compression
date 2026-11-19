"""Needle-In-A-Haystack (NIAH) benchmark generator and exact-match evaluator."""

from __future__ import annotations

import hashlib
import re
from typing import Callable, List, Tuple

import torch

# Diverse essay-style technical & systems paragraphs (Paul Graham / systems engineering tone)
_HAYSTACK_PARAGRAPHS: List[str] = [
    (
        "The best way to build reliable software systems is to keep the core abstraction "
        "small enough that a single engineer can hold the entire dataflow in their head. "
        "When an architecture grows accidental layers of indirection, performance bugs hide "
        "in the seams between modules, and nobody can tell whether latency comes from "
        "memory bandwidth, synchronization barriers, or unnecessary copies."
    ),
    (
        "Most startups fail not because they chose the wrong programming language or cloud "
        "provider, but because they spent months polishing features nobody urgently needed. "
        "Talking to users early feels uncomfortable because their feedback rarely matches "
        "the neat mental model you drew on a whiteboard, yet that friction is the only "
        "reliable compass toward something genuinely valuable."
    ),
    (
        "Modern accelerators are overwhelmingly compute-bound during dense matrix "
        "multiplications and memory-bandwidth-bound during autoregressive token generation. "
        "Every decoded token streams the entire model weight matrix and the accumulated "
        "key-value cache across high-bandwidth memory, which is why reducing cache "
        "footprint directly translates into higher batch throughput and lower tail latency."
    ),
    (
        "Writing clearly is a forcing function for thinking clearly. Whenever a technical "
        "design document relies on vague adjectives instead of concrete measurements, it "
        "usually means the author has not yet worked out the failure modes. A crisp "
        "two-page memo with back-of-the-envelope numbers beats a forty-slide deck every time."
    ),
    (
        "Good research tastes like engineering in reverse: instead of starting from a "
        "fixed specification and finding the cleanest implementation, you start from an "
        "empirical anomaly and keep stripping away hypotheses until only the essential "
        "mechanism remains. Ablation studies are how you verify you actually found it."
    ),
    (
        "In distributed systems, clocks drift, packets arrive out of order, and disks "
        "occasionally stall for seconds without raising an error. Resilient services treat "
        "partial failure as the normal operating regime rather than an exceptional edge "
        "case, using idempotent operations, bounded retries, and explicit timeouts."
    ),
    (
        "Grouped-query attention struck a remarkably practical compromise between "
        "multi-head attention and multi-query attention. By sharing each key-value head "
        "across a small group of query heads, models retain nearly all the representational "
        "capacity of full multi-head attention while slashing KV-cache traffic by a factor "
        "of four to eight."
    ),
    (
        "Great tools disappear from conscious attention once you learn their grammar. "
        "Whether it is a compiler, a profiler, or a text editor, the hallmark of thoughtful "
        "craft is predictable latency and composable primitives that let experienced users "
        "automate repetitive work without fighting the interface."
    ),
    (
        "When profiling a neural network on a GPU, never trust wall-clock timings that omit "
        "explicit device synchronization. Asynchronous kernel launches make the CPU side "
        "look deceptively fast while the execution queue silently backs up behind a single "
        "uncoalesced memory gather or an unexpected host-to-device transfer."
    ),
    (
        "The history of computing is a recurring cycle of hardware constraints forcing "
        "clever algorithms, followed by hardware abundance making people forget those "
        "algorithms until the next scaling wall arrives. Long-context inference is simply "
        "the latest chapter in that cycle, bringing classical cache-replacement theory back "
        "to the center of machine learning systems."
    ),
    (
        "Empirical evaluation is only as trustworthy as its baselines. A new compression "
        "method that beats an unoptimized full-cache implementation in throughput might "
        "still lose to a simple sliding window with attention sinks on short-horizon tasks "
        "unless tested across diverse retrieval depths and strict memory budgets."
    ),
    (
        "Simplicity is rarely the starting point of a design; it is the result of iterating "
        "past the first working prototype and deleting every special case that turned out "
        "not to matter. If a subtle invariant must hold for correctness—such as preserving "
        "rotary position indices after cache eviction—make it explicit in the function "
        "signature rather than burying it in a side effect."
    ),
]


def deterministic_passkey(context_len: int, depth: float, seed: int = 42) -> str:
    """Generate a deterministic, unique 6-digit passkey for a (context_len, depth) cell."""
    raw = f"niah-passkey-{seed}-{context_len}-{depth:.4f}".encode("utf-8")
    digest = int(hashlib.sha256(raw).hexdigest()[:8], 16)
    code = 100000 + (digest % 900000)
    return str(code)


def build_niah_prompt(
    tokenizer,
    target_context_len: int,
    depth: float,
    passkey: str,
) -> Tuple[torch.Tensor, str]:
    """Construct a tokenized NIAH prompt with exact target token length and needle depth.

    Args:
        tokenizer: HuggingFace tokenizer for the target model.
        target_context_len: Desired total prompt length in tokens (e.g., 2048, 4096, 8192).
        depth: Relative insertion depth in ``[0.0, 1.0]`` (e.g., 0.1, 0.3, 0.5, 0.7, 0.9).
        passkey: 6-digit secret passkey string to embed.

    Returns:
        ``(input_ids, prompt_text)`` where ``input_ids`` has shape ``[1, target_context_len]``.
    """
    header = (
        "You are a precise document reader. Below is a collection of technical essays. "
        "Hidden somewhere inside is a single secret six-digit passkey.\n\n"
    )
    needle = (
        f"\n\n[IMPORTANT ARCHIVE NOTE] The secret six-digit passkey is {passkey}. "
        f"Keep this exact six-digit passkey in mind: {passkey}.\n\n"
    )
    # Keep the retrieval question concise (<25 tokens) at the VERY END of the prompt
    # so SnapKV's trailing observation window (window_size=32) captures the full query signal.
    question = (
        "\n\nQuestion: What is the secret 6-digit passkey hidden in the text? "
        "Reply with ONLY the 6-digit number."
    )

    use_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if use_chat_template:
        sentinel = "<|HAYSTACK_SENTINEL|>"
        messages = [
            {
                "role": "system",
                "content": "You extract exact passkeys from documents. Reply with only the 6-digit passkey.",
            },
            {"role": "user", "content": f"{header}{sentinel}{question}"},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prefix_text, suffix_text = rendered.split(sentinel, maxsplit=1)
        bos_ids = []
        header_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        footer_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    else:
        bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        footer_ids = tokenizer.encode(question + "\nAnswer:", add_special_tokens=False)

    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    overhead = len(bos_ids) + len(header_ids) + len(needle_ids) + len(footer_ids)
    haystack_budget = max(16, target_context_len - overhead)

    # Tokenize paragraphs until we exceed haystack_budget
    haystack_ids: List[int] = []
    idx = 0
    sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    while len(haystack_ids) < haystack_budget:
        para = _HAYSTACK_PARAGRAPHS[idx % len(_HAYSTACK_PARAGRAPHS)]
        para_ids = tokenizer.encode(para, add_special_tokens=False)
        if haystack_ids:
            haystack_ids.extend(sep_ids)
        haystack_ids.extend(para_ids)
        idx += 1

    haystack_ids = haystack_ids[:haystack_budget]
    insert_pos = int(round(haystack_budget * max(0.0, min(1.0, depth))))

    full_ids = (
        bos_ids
        + header_ids
        + haystack_ids[:insert_pos]
        + needle_ids
        + haystack_ids[insert_pos:]
        + footer_ids
    )
    # Trim or pad by at most a few tokens if needed to hit exact target_context_len
    if len(full_ids) > target_context_len:
        excess = len(full_ids) - target_context_len
        # Trim from the end of haystack before footer so needle & question stay intact
        cut_end = len(full_ids) - len(footer_ids)
        full_ids = full_ids[: max(0, cut_end - excess)] + footer_ids

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    prompt_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    return input_ids, prompt_text


def evaluate_passkey(generated_text: str, passkey: str) -> bool:
    """Exact-match evaluator checking whether the 6-digit passkey was retrieved."""
    cleaned = generated_text.strip()
    if passkey in cleaned:
        return True
    digits_only = re.sub(r"\D+", "", cleaned)
    return passkey in digits_only


def make_passkey_stop_fn(tokenizer, passkey: str) -> Callable[[torch.Tensor], bool]:
    """Early-stop predicate for ``generate_with_policy`` once passkey is emitted."""

    def _stop(generated_ids: torch.Tensor) -> bool:
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return passkey in text

    return _stop
