"""CLI benchmark runner for KV-Cache Compression on NVIDIA Tesla T4 (16 GB VRAM).

Runs:
  Experiment 1 — Needle-In-A-Haystack (NIAH):
      5 depths (10%, 30%, 50%, 70%, 90%) x 3 context lengths (2048, 4096, 8192)
      comparing Full, SlidingWindow, H2O, and SnapKV at 25% KV budget.
  Experiment 2 — KV-Cache Budget Ablation (at 4096 tokens):
      Accuracy (%) & KV-cache memory (MB) across budgets [12.5%, 25%, 50%, 100%].
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.needle_haystack import (
    build_niah_prompt,
    deterministic_passkey,
    evaluate_passkey,
    make_passkey_stop_fn,
)
from src.generator import generate_with_policy
from src.metrics import get_optimal_dtype
from src.policies import build_policy


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_niah(
    model,
    tokenizer,
    policy_name: str,
    context_len: int,
    depth: float,
    budget_ratio: float,
    max_new_tokens: int,
    seed: int,
) -> Dict[str, Any]:
    passkey = deterministic_passkey(context_len, depth, seed=seed)
    input_ids, _ = build_niah_prompt(tokenizer, context_len, depth, passkey)
    input_ids = input_ids.to(model.device)

    actual_len = int(input_ids.shape[1])
    budget = None if policy_name == "full" else max(16, int(round(actual_len * budget_ratio)))
    policy = build_policy(policy_name, budget=budget)

    stop_fn = make_passkey_stop_fn(tokenizer, passkey)
    gen_ids, metrics = generate_with_policy(
        model=model,
        input_ids=input_ids,
        policy=policy,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        stop_fn=stop_fn,
    )
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    correct = evaluate_passkey(gen_text, passkey)

    policy.reset()
    del input_ids, gen_ids, policy
    _cleanup_cuda()

    return {
        "policy": policy_name,
        "context_len": context_len,
        "actual_prompt_len": actual_len,
        "depth": depth,
        "budget_ratio": 1.0 if policy_name == "full" else budget_ratio,
        "budget_tokens": budget if budget is not None else actual_len,
        "passkey": passkey,
        "prediction": gen_text,
        "correct": bool(correct),
        "kv_cache_len": metrics.kv_cache_len,
        "kv_cache_mb": round(metrics.kv_cache_mb, 4),
        "full_kv_cache_mb": round(metrics.full_kv_cache_mb, 4),
        "compression_ratio": round(metrics.compression_ratio, 4),
        "prefill_time_s": round(metrics.prefill_time_s, 4),
        "decode_time_s": round(metrics.decode_time_s, 4),
        "decode_tokens_per_s": round(metrics.decode_tokens_per_s, 2),
        "peak_vram_mb": round(metrics.peak_vram_mb, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KV-Cache Compression Benchmark Runner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=str, default="benchmarks/results.json")
    parser.add_argument(
        "--exp1-contexts", type=int, nargs="+", default=[2048, 4096, 8192]
    )
    parser.add_argument(
        "--exp1-depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9]
    )
    parser.add_argument("--exp1-budget-ratio", type=float, default=0.25)
    parser.add_argument("--exp2-context", type=int, default=4096)
    parser.add_argument(
        "--exp2-ratios", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = get_optimal_dtype(device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    compute_cap = (
        f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
        if torch.cuda.is_available()
        else "N/A"
    )

    print(f"[Setup] Model: {args.model}")
    print(f"[Setup] Device: {gpu_name} (CC {compute_cap}) | Precision: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    _cleanup_cuda()

    policies = ["full", "sliding_window", "h2o", "snapkv"]

    # ------------------------------------------------------------------
    # Experiment 1: Needle-In-A-Haystack (5 depths x 3 context lengths)
    # ------------------------------------------------------------------
    print("\n=== Experiment 1: Needle-In-A-Haystack (25% KV Budget) ===")
    exp1_runs: List[Dict[str, Any]] = []
    # Cache (policy, context_len, depth, budget_ratio) -> record for reuse in Exp 2
    run_cache: Dict[tuple, Dict[str, Any]] = {}

    for policy_name in policies:
        for ctx_len in args.exp1_contexts:
            hits = 0
            for depth in args.exp1_depths:
                rec = run_single_niah(
                    model=model,
                    tokenizer=tokenizer,
                    policy_name=policy_name,
                    context_len=ctx_len,
                    depth=depth,
                    budget_ratio=1.0 if policy_name == "full" else args.exp1_budget_ratio,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                )
                exp1_runs.append(rec)
                key = (policy_name, ctx_len, round(depth, 4), round(rec["budget_ratio"], 4))
                run_cache[key] = rec
                hits += int(rec["correct"])
                status = "PASS" if rec["correct"] else "FAIL"
                print(
                    f"  [{policy_name:14s}] L={ctx_len:4d} depth={int(depth*100):2d}% -> "
                    f"{status} | KV={rec['kv_cache_mb']:6.2f} MB | "
                    f"prefill={rec['prefill_time_s']:.2f}s | pred={rec['prediction']!r}"
                )
            print(
                f"  -> {policy_name} @ L={ctx_len}: {hits}/{len(args.exp1_depths)} "
                f"({100.0 * hits / len(args.exp1_depths):.1f}%)"
            )

    # ------------------------------------------------------------------
    # Experiment 2: Compression Ratio Ablation at 4096 tokens
    # ------------------------------------------------------------------
    print(f"\n=== Experiment 2: KV Budget Ablation (L={args.exp2_context}) ===")
    exp2_runs: List[Dict[str, Any]] = []
    ablation_policies = ["sliding_window", "h2o", "snapkv"]

    for policy_name in ablation_policies:
        for ratio in args.exp2_ratios:
            hits = 0
            kv_mbs = []
            for depth in args.exp1_depths:
                lookup_policy = "full" if ratio >= 1.0 else policy_name
                cache_key = (
                    lookup_policy,
                    args.exp2_context,
                    round(depth, 4),
                    round(1.0 if ratio >= 1.0 else ratio, 4),
                )
                if cache_key in run_cache:
                    rec = dict(run_cache[cache_key])
                    rec["policy"] = policy_name
                    rec["budget_ratio"] = ratio
                else:
                    rec = run_single_niah(
                        model=model,
                        tokenizer=tokenizer,
                        policy_name=lookup_policy,
                        context_len=args.exp2_context,
                        depth=depth,
                        budget_ratio=ratio,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                    )
                    rec["policy"] = policy_name
                    rec["budget_ratio"] = ratio
                    run_cache[cache_key] = rec
                exp2_runs.append(rec)
                hits += int(rec["correct"])
                kv_mbs.append(rec["kv_cache_mb"])
            avg_kv = sum(kv_mbs) / len(kv_mbs)
            print(
                f"  [{policy_name:14s}] budget={ratio*100:5.1f}% -> "
                f"acc={100.0 * hits / len(args.exp1_depths):5.1f}% | avg KV={avg_kv:6.2f} MB"
            )

    # Build concise summary per policy for README / Colab table
    summary: Dict[str, Any] = {}
    for policy_name in policies:
        p_runs = [r for r in exp1_runs if r["policy"] == policy_name]
        acc = 100.0 * sum(r["correct"] for r in p_runs) / max(1, len(p_runs))
        runs_8k = [r for r in p_runs if r["context_len"] == max(args.exp1_contexts)]
        kv_8k = sum(r["kv_cache_mb"] for r in runs_8k) / max(1, len(runs_8k))
        vram_peak = max((r["peak_vram_mb"] for r in p_runs), default=0.0)
        avg_tps = sum(r["decode_tokens_per_s"] for r in p_runs) / max(1, len(p_runs))
        summary[policy_name] = {
            "niah_overall_acc_pct": round(acc, 1),
            "kv_cache_mb_at_max_ctx": round(kv_8k, 2),
            "peak_vram_mb": round(vram_peak, 1),
            "avg_decode_tps": round(avg_tps, 1),
        }

    total_time_s = round(time.perf_counter() - t_start, 1)
    payload = {
        "metadata": {
            "model_id": args.model,
            "gpu_name": gpu_name,
            "compute_capability": compute_cap,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_runtime_s": total_time_s,
            "exp1_contexts": args.exp1_contexts,
            "exp1_depths": args.exp1_depths,
            "exp1_budget_ratio": args.exp1_budget_ratio,
            "exp2_context": args.exp2_context,
            "exp2_ratios": args.exp2_ratios,
        },
        "summary": summary,
        "exp1_niah": exp1_runs,
        "exp2_ablation": exp2_runs,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Done] Saved benchmark results to {args.output} in {total_time_s:.1f}s")


if __name__ == "__main__":
    main()
