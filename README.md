# KV-Cache Compression from Scratch in PyTorch

Production-grade **Paper-to-Code** implementation of long-context **KV-Cache Eviction Policies** in PyTorch for Grouped-Query Attention (GQA) `Qwen2` / `Qwen2.5` architectures:

- **SnapKV** (*NeurIPS 2024* — *SnapKV: LLM Knows What You are Looking for Before Generation*)
- **H2O — Heavy-Hitter Oracle** (*NeurIPS 2023* — *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*)
- **Sliding Window + Attention Sinks** (*StreamingLLM* baseline)
- **Full KV-Cache** (exact uncompressed reference)

Engineered specifically to run safely within **16 GB VRAM (NVIDIA Tesla T4)** via $O(L)$ memory-safe attention importance scoring, native FP16 Tensor Core execution, FP32 importance softmax, and strict logical RoPE position preservation.

---

## Key Engineering Guarantees

| Requirement | Implementation Detail |
| :--- | :--- |
| **$O(L)$ Memory-Safe Scoring** | Never materializes an $L \times L$ attention matrix. Exact attention output runs through `F.scaled_dot_product_attention` (SDPA), while importance scoring slices only $W \ll L$ trailing observation queries against $K$ without duplicating KV-heads. |
| **FP32 Importance Softmax** | Attention logits for eviction scoring are cast to `torch.float32` prior to softmax (`F.softmax(logits, dim=-1, dtype=torch.float32)`) to eliminate FP16 overflow/NaNs on long sequences. |
| **Hardware-Aware Precision** | Automatically inspects `torch.cuda.get_device_capability(0)`: uses native `torch.float16` on Turing/Volta (`major < 8`, e.g., Tesla T4) and `torch.bfloat16` on Ampere+ (`major >= 8`). |
| **Strict RoPE Position Preservation** | After KV eviction shrinks `cache.get_seq_length()` from $L \to B$, the autoregressive loop tracks `logical_pos = L, L+1, ...` and passes explicit `position_ids` and `cache_position` so rotary phases remain exact. |
| **GQA Native Aggregation** | Aggregates query-head importance across GQA groups ($H_q = 16 \to H_{kv} = 2$ on `Qwen2.5-3B-Instruct`) before top-$k$ index selection. |

---

## Architecture: SnapKV vs. H2O Eviction

```text
Prompt KV-Cache (Length L)
[ t_0  t_1  t_2  t_3 | t_4  ...  t_i  ...  t_{L-W-1} | t_{L-W} ... t_{L-1} ]
  \________________/   \___________________________/   \_________________/
   Attention Sinks             Prefix Region             Observation Window
     (sink_size)                 (L - W)                     (W tokens)

------------------------------------------------------------------------------
1. SnapKV (One-Shot Prefill Observation + 1D Average Pooling)
------------------------------------------------------------------------------
   Q_obs (last W=32 queries) x K^T  -->  FP32 Softmax  -->  GQA Group Mean
        -->  Sum over W  -->  AvgPool1D(kernel=5, pad=2) on Prefix
        -->  Top-(B - W) Prefix Indices (sorted)  +  All W Recent Tokens

   Retained Cache (Length B):
   [ Selected Clustered Prefix Tokens (B - W) | Observation Window (W) ]

------------------------------------------------------------------------------
2. H2O (Dynamic Heavy-Hitter Accumulation Across Prefill + Decode)
------------------------------------------------------------------------------
   Prefill: Last 128 queries x K^T --> FP32 Softmax --> Init HH Score Buffer
   Decode:  Each step q_t (1 x cache_len) --> Accumulate into HH Score Buffer
   Evict:   Keep Sinks (first 4) + Top-k Heavy-Hitters + Recent Window

   Retained Cache (Length B):
   [ Sinks (4) | Top Heavy-Hitters (B - 4 - R) | Recent Tokens (R) ]
```

---

## Mathematical Formulation

Let $Q \in \mathbb{R}^{H_q \times L \times d}$ and $K, V \in \mathbb{R}^{H_{kv} \times L \times d}$ denote the post-RoPE query, key, and value tensors for a single layer with GQA group factor $G = H_q / H_{kv}$.

### 1. SnapKV Pooled Observation Scoring
Define the observation window queries $Q_{\text{obs}} = Q[:, L-W:L, :]$ (default $W=32$). For each KV-head $h \in \{1, \dots, H_{kv}\}$ and grouped query head $g \in \{1, \dots, G\}$:

$$A_{h,g} = \operatorname{softmax}_{\text{fp32}}\!\left(\frac{Q_{\text{obs}}^{(h,g)} (K^{(h)})^\top}{\sqrt{d}} + M_{\text{causal}}\right) \in \mathbb{R}^{W \times L}$$

The per-KV-head importance vector $s_h \in \mathbb{R}^{L}$ aggregates across the GQA group and observation window:

$$s_h(j) = \frac{1}{G} \sum_{g=1}^{G} \sum_{i=1}^{W} A_{h,g}(i, j)$$

SnapKV applies 1D average pooling with kernel size $k=5$ (padding $2$) over the prefix $j \in \{0, \dots, L-W-1\}$ to preserve contiguous semantic clusters:

$$\tilde{s}_h = \operatorname{AvgPool1D}_{k=5}\!\big(s_h[0 : L-W]\big)$$

The retained index set $\mathcal{I}_h$ selects the top $B - W$ prefix indices sorted in ascending order (preserving causal order) concatenated with the trailing observation window $\{L-W, \dots, L-1\}$.

### 2. H2O Heavy-Hitter Accumulation
H2O maintains a running FP32 importance buffer $S_h^{(t)} \in \mathbb{R}^{|\mathcal{C}_t|}$ per KV-head:
- **Prefill initialization ($t=0$):** Computed memory-safely via the trailing $W_{\text{init}}=128$ queries ($O(128 \cdot L)$ memory) with causal masking and GQA group aggregation.
- **Autoregressive decode ($t \ge 1$):** For the newly decoded query $q_t \in \mathbb{R}^{H_q \times 1 \times d}$:

$$S_h^{(t)} = \operatorname{concat}\!\big(S_h^{(t-1)},\, 0\big) + \frac{1}{G}\sum_{g=1}^{G} \operatorname{softmax}_{\text{fp32}}\!\left(\frac{q_t^{(h,g)} (K_t^{(h)})^\top}{\sqrt{d}}\right)$$

Whenever $|\mathcal{C}_t| > B$, H2O retains $\text{sink\_size}=4$ initial tokens, the top $B - 4 - R$ Heavy-Hitter tokens from the middle region ranked by $S_h^{(t)}$, and the most recent $R$ tokens, gathering $K$, $V$, and $S_h^{(t)}$ with the identical index set.

---

## Key Results (`Qwen/Qwen2.5-3B-Instruct`, NVIDIA Tesla T4 16 GB, FP16)

> Results below are populated directly from `notebooks/run_t4_benchmark.ipynb` on Google Colab T4.

| Policy | KV Budget | NIAH Accuracy (%) | KV-Cache @ 8K (MiB) | KV RAM Saved (%) | Peak VRAM (MiB) | Avg Decode (tok/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full KV-Cache** | 100% | TBD (T4 GPU) | TBD (T4 GPU) | 0.0% | TBD (T4 GPU) | TBD (T4 GPU) |
| **Sliding Window + Sinks** | 25% | TBD (T4 GPU) | TBD (T4 GPU) | TBD (T4 GPU) | TBD (T4 GPU) | TBD (T4 GPU) |
| **H2O (Heavy-Hitter Oracle)** | 25% | TBD (T4 GPU) | TBD (T4 GPU) | TBD (T4 GPU) | TBD (T4 GPU) | TBD (T4 GPU) |
| **SnapKV (Observation Pooled)** | 25% | TBD (T4 GPU) | TBD (T4 GPU) | TBD (T4 GPU) | TBD (T4 GPU) | TBD (T4 GPU) |

### Planned Visual Artifacts (Generated by `benchmarks/plot_results.py`)
- `assets/niah_heatmaps.png` — Side-by-side 4-panel Needle-In-A-Haystack heatmap (Depth 10%–90% vs. Context Length 2K / 4K / 8K).
- `assets/budget_ablation.png` — Accuracy (%) vs. KV-Cache Budget (12.5%–100%) curve + KV-Cache VRAM savings bar chart at 4K context.

---

## Repository Structure

```text
kv-cache-compression/
├── src/
│   ├── __init__.py
│   ├── policies.py            # FullCache, SlidingWindow, H2O, SnapKV + DynamicCache adapter
│   ├── attention_patch.py     # Reversible Qwen2Attention patch (SDPA + O(L) compression)
│   ├── generator.py           # Manual prefill + decode loop with logical RoPE tracking
│   └── metrics.py             # Exact KV-cache MiB calculator, peak VRAM tracker, dtype helper
├── tests/
│   └── test_compression.py    # Fast (<10s) pytest suite on tiny in-memory Qwen2Config
├── benchmarks/
│   ├── needle_haystack.py     # NIAH prompt builder (6-digit passkey) & exact-match evaluator
│   ├── run_benchmark.py       # CLI runner: Exp 1 (NIAH) + Exp 2 (KV Budget Ablation)
│   └── plot_results.py        # Publication-style plot generator -> assets/
├── notebooks/
│   └── run_t4_benchmark.ipynb # Turnkey 1-click Google Colab T4 notebook
├── assets/
├── requirements.txt
└── README.md
```

---

## Reproduction

### 1. Fast Local Verification (CPU, <10 seconds, Zero Downloads)
```bash
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
```

### 2. Full T4 GPU Benchmarks (1-Click Google Colab)
Open `notebooks/run_t4_benchmark.ipynb` in Google Colab with a **Tesla T4 GPU** runtime and run all 4 cells. It automatically clones the repo, runs unit tests, executes Experiment 1 (NIAH) and Experiment 2 (Budget Ablation), renders both charts inline, prints the Markdown results table, and downloads `t4_artifacts.zip`.

### 3. CLI Benchmark Execution (Linux / CUDA)
```bash
PYTHONPATH=. python benchmarks/run_benchmark.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output benchmarks/results.json

PYTHONPATH=. python benchmarks/plot_results.py \
  --input benchmarks/results.json \
  --assets-dir assets
```
