# Draft-Free Dynamic Sparse Prefill (MInference-Style) — Implementation Plan

Date: 2026-07-06
Suggested branch: `feat/sparse-prefill-draftfree`
Status: **implemented through Phase 2 (Stage-1 kernels); Phase 3 (Stage-2 Metal kernel) greenlit** — see the two Results sections at the end.
Originally planned as item B1, ranked #4 in [5x_speedup_research.md](5x_speedup_research.md) recommended order.
Companion to: [ngram_speculation_plan.md](ngram_speculation_plan.md) (template & precedent),
[MInference](https://openreview.net/forum?id=fPBACAbqSN)

## Goal

Replace SpecPrefill's draft-model scoring stage with **calibrated static per-head sparse
attention patterns** applied training-free at prefill time (MInference): each attention
head gets an offline-determined pattern class — A-shape (sink+local), vertical-slash
(periodic columns + local diagonal), or block-sparse — and prefill attention computes only
that pattern. No draft model → works on the 8 GB box; sparsity is applied *inside*
attention rather than by dropping tokens, so all tokens get KV entries (unlike
SpecPrefill's top-K% token selection).

Expected: 3–10x prefill at ≥16K context, scaling with length; ~neutral below ~8K
(sparsity overhead not worth it — gate on prompt length like `specprefill_threshold`).

Why now: SpecPrefill (`omlx/patches/specprefill.py`) exists but requires a resident draft
model running full O(L²) attention to score importance — its cost structure caps the win
and excludes low-RAM machines. The hard machinery it built (attention-module discovery and
patching, per-architecture query extractors, layer→cache mapping, RoPE position mapping)
carries over.

## Non-goals (this branch)

- Sparse *decode* attention (that is A3's compressed-KV territory).
- Trained sparse-attention models (NSA/InfLLM-v2 style) — training-free calibration only.
- Removing SpecPrefill — it stays as-is; this is a sibling mode. Deprecation decided later
  on benchmark evidence.
- Cross-layer index reuse (IndexCache-paper style; note `omlx/patches/index_cache.py`
  already does something related for DSA models — see "Later").
- VLM prefill.

## Design overview

Two components: an **offline calibration job** and a **runtime sparse-attention patch**.

```
Calibration (one-time per model, admin-panel job):
  run N (~8–16) long diverse prompts with attention capture
    (reuse specprefill's _AttentionCapture / _patch_attention_for_capture machinery,
     extended to capture attention row samples, not just queries)
  per head: fit each pattern class, pick the one with best recall of true attention mass
    at the target FLOP budget (MInference's offline search, simplified)
  emit JSON config: {layer: {head: {kind: "a_shape"|"vertical_slash"|"block_sparse",
                                     params: {...}}}}
  store under the model dir or omlx data dir (like oq_calibration_data.json precedent)

Runtime (prefill only, decode untouched):
  chunked prefill step ≥ threshold:
    per layer, replace SDPA with pattern dispatch:
      - a_shape:        mask-based SDPA over sink+local — cheap, no kernel needed
      - vertical_slash: last-64-queries estimate → top columns + diagonal slash →
                        gather-based attention (Metal kernel, see below)
      - block_sparse:   block-mean pooled QK → top blocks → block-gathered SDPA
  decode steps: stock full attention (KV cache is complete — no quality cliff)
```

Kernel strategy — do it in three escalation stages, measure between each:

1. **Stage 1, no custom kernels:** implement all three patterns with mx.fast.
   scaled_dot_product_attention + gather/masking. Vertical-slash via `mx.take` of key
   columns. This will not hit paper speedups but proves quality and the dispatch plumbing.
2. **Stage 2, `mx.fast.metal_kernel`:** custom vertical-slash sparse attention kernel
   (the pattern MInference found dominant). Precedent for custom kernels in-repo:
   `omlx/custom_kernels/glm_moe_dsa`, `minimax_m3` — follow their packaging.
3. **Stage 3 (only if needed): block-sparse FlashAttention-style kernel.**

## Phases

### Phase 0 — baseline & scaffolding

- [ ] Prefill benchmark at {4K, 8K, 16K, 32K, 64K} on reference machines (M1 8 GB with a
      small model; a bigger box if available): stock vs SpecPrefill (where a draft exists).
      Save table here — this is the bar to beat. Use `scripts/perf_bench.py` for the
      TTFT/scenario harness once `sparse_prefill_enabled` exists
      (`--ab --setting-key sparse_prefill_enabled`); add a long-prompt scenario to its
      `SCENARIOS` dict for the ≥16K lengths this feature targets (the built-in scenarios
      top out around a few hundred tokens of context).
- [ ] Verify specprefill's capture machinery works on the target model list (extractors
      exist for qwen3.5/3.6, llama, gemma4, nemotron-h — `specprefill.py:96-170`).

### Phase 1 — calibration job (offline, no runtime change)

- [ ] `omlx/speculative/` is decode-side; new home `omlx/prefill_sparse/` (or
      `omlx/patches/sparse_prefill.py` + `omlx/sparse_calibration.py`): capture-based
      calibration producing the per-head pattern JSON.
- [ ] Calibration prompt set: bundle a small mixed corpus (long doc, code file, multi-turn
      transcript) or synthesize from user-provided files; length ≥16K tokens.
- [ ] Pattern fitting: for each head, compute attention-mass recall of best-fit A-shape /
      vertical-slash / block-sparse at a fixed budget (e.g. 10% of full FLOPs); pick
      argmax. Keep the search simple (grid over slash counts / block counts).
- [ ] Report: per-model summary (pattern distribution per layer, expected recall). Cheap
      go/no-go gate: if mean recall at 10% budget is <90%, sparse prefill will hurt
      quality — stop and reassess budget before any runtime work.
- [ ] Admin-panel job button (long-running task, like existing calibration precedents;
      `oq_calibration_data.json` shows the storage pattern).

### Phase 2 — runtime dispatch, Stage-1 kernels

- [ ] Attention patch applying pattern dispatch during prefill steps only (hook the same
      seam `_patch_attention_for_capture` uses; unpatch for decode or branch on
      query-length>1 inside the wrapper — prefer the latter, it is how prefill/decode
      are usually distinguished).
- [ ] Settings (`omlx/model_settings.py`): `sparse_prefill_enabled: bool = False`,
      `sparse_prefill_threshold` (min prompt tokens, default 8192),
      `sparse_prefill_budget` (fraction, default 0.1), calibration-file path resolution.
      Validation: requires calibration file present; mutually exclusive with
      `specprefill_enabled`; allowed with ngram spec, prefix/SSD cache, chunked prefill
      (patterns apply within each chunk's rows — verify masks compose with chunked
      prefill's incremental offsets, this is a known fiddly spot).
- [ ] Quality gate: long-context QA/needle tasks, sparse vs stock — task-level accuracy
      within noise at budget 0.1; token-identity NOT expected (attention is approximated).

### Phase 3 — Stage-2 Metal kernel

- [ ] Vertical-slash gather-attention kernel via `mx.fast.metal_kernel` under
      `omlx/custom_kernels/sparse_prefill/`; numerics test vs Stage-1 reference
      implementation; benchmark speedup per layer.
- [ ] Re-run Phase 0 matrix; decision point: does Stage 3 (block-sparse kernel) pay?
      Document.

### Phase 4 — results & defaults

- [ ] Full benchmark table (this doc): TTFT at each length, stock / SpecPrefill /
      sparse-prefill, plus quality deltas. `perf_bench.py --stats-path` can surface
      calibration recall / pattern-hit stats if exposed via an admin endpoint.
- [ ] Pick per-profile defaults (enable above threshold only where calibration exists);
      keep opt-in globally.

## Risks

| Risk | Mitigation |
|---|---|
| Quality regression on tasks needing dense middle-context attention | Recall gate in Phase 1; budget knob; task-level quality gate in Phase 2; opt-in |
| Chunked-prefill interaction (patterns assume full-row visibility, chunks see growing KV) | Treat pattern coordinates as absolute positions over cache offset; dedicated unit tests with chunk sizes {512, 2048} |
| Stage-1 gather implementation slower than dense SDPA (gathers are bandwidth-hungry) | Stage-1 is a correctness vehicle only; speed claims deferred to Stage 2; threshold keeps it off short prompts |
| Calibration overfits to the calibration corpus | Mixed-domain corpus; recall measured on held-out prompt; per-head patterns are coarse (hard to overfit badly) |
| Custom Metal kernel maintenance burden (CLAUDE.md: keep upstream sync easy) | Kernel isolated in `custom_kernels/sparse_prefill/` following existing in-repo kernel packaging; zero edits to vendored attention code — patching only |

## Later (explicitly deferred)

- Cross-layer index reuse (VSPrefill/IndexCache-style) — potentially unify with
  `omlx/patches/index_cache.py`'s existing DSA index-caching.
- Compose with B2 chunk-KV reuse: sparse-attend only the recompute subset.
- Dynamic (per-prompt) pattern re-estimation for the vertical-slash column choice
  (MInference does a cheap last-64-query estimate at runtime — include in Stage 2 kernel
  if calibration-static columns underperform).
- SpecPrefill deprecation decision.

## File touchpoints

| File | Change |
|---|---|
| `omlx/patches/sparse_prefill.py` | new — runtime pattern dispatch patch |
| `omlx/sparse_calibration.py` | new — offline calibration job |
| `omlx/custom_kernels/sparse_prefill/` | new — Stage-2 Metal kernel |
| `omlx/patches/specprefill.py` | none — capture/extractor helpers imported; refactor to shared module only if imports get circular |
| `omlx/model_settings.py` | `sparse_prefill_*` settings + validation |
| `omlx/admin/` | calibration job trigger + toggle + stats |
| `scripts/perf_bench.py` | add long-prompt scenario; A/B via `--setting-key sparse_prefill_enabled` |
| `tests/test_sparse_prefill.py` | new — pattern fitting, mask correctness, chunked-prefill offsets, kernel-vs-reference numerics |

## Results (2026-07-09, Stage 1 on M1 8 GB)

Implemented on branch `feat/sparse-prefill-draftfree`:

- `omlx/patches/sparse_prefill.py` — runtime SDPA patch (layer taggers +
  monkey-patched `mlx_lm.models.base.scaled_dot_product_attention`, same seam
  as turboquant/sdpa256). Stage-1 only: block-chunked SDPA with contiguous
  sink/window slices, per-kv-head vertical-column gather, boolean per-head
  masks, dedup of overlapping regions. Vertical columns are re-estimated per
  prefill chunk from the last 64 query rows (MInference-style), so the
  calibration stores pattern *class + sink/window sizes*, not concrete columns.
- `omlx/sparse_calibration.py` — offline calibration CLI
  (`python -m omlx.sparse_calibration --model <id>`), synthetic aperiodic
  mixed corpus (prose/code/QA), per-head recall grid over
  a_shape{sink 64/256/1024} and vertical_slash{window 256/1024}, 0.90 recall
  gate. Output: `~/.omlx/sparse_prefill/<model>.json`.
- Settings: `sparse_prefill_enabled/threshold/budget/calibration_file`,
  mutually exclusive with `specprefill_enabled`; activation wired in
  `BatchedEngine` model load. 13 unit tests pass.

Deviations from plan: block_sparse pattern class deferred (MInference found
vertical-slash dominant; calibration picked vertical_slash for 48/48 heads
here); admin-panel job button deferred (CLI only); Stage-2 Metal kernel not
started.

### Calibration (Qwen3.5-0.8B-MLX-4bit, 2×16K prompts)

Hybrid gated-delta model: only 6/24 layers are full attention (8 q-heads
each). Recall @ budget: 0.1 → 0.587, 0.2 → 0.792, 0.3 → 0.889,
**0.4 → 0.941 (gate passed)**. All 48 heads fitted vertical_slash.
Small-model caveat: attention is much less sparse than in the ≥7B models
MInference reports on; the 10% budget from the paper does not transfer.

### Prefill benchmark (chunked 2048, budget 0.4, threshold 8192)

| tokens | dense s | sparse s | speedup | effective density |
|---|---|---|---|---|
| 4 088  | 5.9  | 5.9  | 1.00x (correctly gated off) | — |
| 8 184  | 10.9 | 11.0 | 1.00x (correctly gated off) | — |
| 16 356 | 23.2 | 21.6 | **1.07x** | 0.44 |
| 32 740 | 53.3 | 45.7 | **1.17x** | 0.43 |

### Quality (needle retrieval, 16K context)

Primed-continuation probe (`…"the magic checkpoint code is` → expects
`TANGERINE-42`): retrieved correctly dense AND sparse at needle positions
0.85 *and 0.3* (deep middle context — outside any window; the runtime
vertical-column estimate catches it). Free-form generation is byte-similar
between modes. Recall gate + parity: no observed quality regression.

### Assessment & next steps

Stage 1 is a working correctness vehicle with a real (small) win, but far
from the 3–10x target, for three compounding reasons on this setup:
(1) hybrid model — only 25% of layers are full attention (Amdahl caps the
ceiling at ~1.3x even with free sparse attention); (2) budget had to be 0.4,
not 0.1, for the recall gate on a 0.8B model; (3) Stage-1 gathers/masks are
bandwidth-hungry vs a fused kernel. Decision from Phase 3's decision point,
brought forward: on the M1 8 GB box with small hybrid models, the Stage-2
Metal kernel would sharpen 1.17x to maybe 1.5x — worth doing only after
benchmarking on a full-attention model (e.g. Llama-3.2-1B) where all layers
benefit and calibrated budgets can be lower. Feature stays opt-in
(`sparse_prefill_enabled=False` default), exactly as planned.

## Results (2026-07-08, full-attention model: Llama-3.2-1B-Instruct-4bit)

Follow-up to the assessment above: re-ran calibration + benchmark on a
full-attention model (all 16 layers, 32 q-heads each) to remove the hybrid
Amdahl cap and test whether lower budgets calibrate.

### Calibration (2×16K prompts)

Full-attention Llama is far sparser than the hybrid 0.8B Qwen: recall
@ budget 0.1 → 0.863 (just under gate), **0.15 → 0.902 (gate passed)**,
0.2 → 0.927. All 512 heads fitted vertical_slash at 0.15
(509 vertical_slash / 3 a_shape at 0.1). Compare Qwen needing budget 0.4.

### Prefill benchmark (chunked 2048, budget 0.15, threshold 8192)

`scripts/sparse_prefill_bench.py`, M1 8 GB:

| tokens | dense s | sparse s | speedup | effective density | needle d/s |
|---|---|---|---|---|---|
| 4 094  | 5.19  | 5.16  | 1.01x (gated off) | — | Y/Y |
| 8 189  | 10.83 | 10.82 | 1.00x (gated off) | — | Y/Y |
| 16 382 | 25.84 | 22.86 | **1.13x** | 0.203 | Y/Y |
| 32 766 | 71.14 | 59.10 | **1.20x** | 0.182 | Y/Y |

Needle retrieval (`TANGERINE-42`) correct dense and sparse at every length.

### Stage-2 decision (Phase 3 gate)

The Amdahl explanation is now eliminated: every layer is sparse-eligible and
effective density is ~0.18, yet speedup is only 1.2x. The gap between the
~5x theoretical attention-FLOP reduction and 1.2x wall-clock is therefore
almost entirely **Stage-1 implementation overhead** (per-head boolean masks,
`mx.take` column gathers, block-chunked SDPA launches) — exactly the
bandwidth-bound behaviour the risk table predicted. Conclusion: the Stage-2
fused vertical-slash Metal kernel is now clearly the binding constraint and
is worth building (Phase 3 go). Vertical-slash-only is sufficient — it won
960/1024 fitted heads across both models and 512/512 on Llama.

## Results (2026-07-08, Phase 3 / Stage 2)

Two Stage-2 implementations were built and measured on the Llama-3.2-1B
setup above (budget 0.15, threshold 8192, chunked 2048, M1 8 GB):

**1. Custom fused Metal kernel** (`omlx/custom_kernels/sparse_prefill/`,
`mx.fast.metal_kernel` JIT — no native build). Flash-style blocked design:
16 query rows per threadgroup share K/V tiles through threadgroup memory,
online softmax with lane-distributed accumulators, per-row validity masks.
Numerically correct (matches the Stage-1 reference to <5e-3 in fp16,
matches dense exactly under full coverage) but **slower than dense SDPA**:
0.6x/0.44x naive (one simdgroup per row), 0.76x/0.66x blocked, at 16K/32K.
Root cause: scalar per-lane dot products cannot compete with MLX's
simdgroup-matrix flash attention — the ~6x FLOP reduction is more than
eaten by a >6x lower FLOP throughput. Getting ahead would require
simdgroup_matrix tiles (a full flash-attention rewrite in Metal). The
kernel is kept in-repo for reference/tests but disabled by default
(`_USE_KERNEL = False`).

**2. Maskless block-SDPA path** (`_maskless_attention`, now the default
sparse path). Insight: with layer-max sink/window (a quality-safe superset
of the per-head patterns) and vertical columns estimated only over the
strictly-causal zone `[S, q_start − W + 1)`, every query block's key set
`[sink | cols | window-slice]` needs **no materialized mask** — MLX's
bottom-right-aligned `mask="causal"` handles the window diagonal exactly,
and the whole computation rides the stock fused flash-SDPA kernel. This
removed the Stage-1 boolean-mask overhead entirely:

| tokens | dense s | sparse s | speedup | effective density | needle d/s |
|---|---|---|---|---|---|
| 8 189  | 10.98 | 11.05 | 0.99x (gated off) | — | Y/Y |
| 16 382 | 26.01 | 20.10 | **1.29x** | 0.19 | Y/Y |
| 32 766 | 71.14 | 42.88 | **1.66x** | 0.20 | Y/Y |
| 65 534 | 298.2 | 120.0 | **2.49x** | 0.18 | Y/Y |

(`_QUERY_BLOCK` raised 512 → 1024: 1.64x → 1.66x at 32K.) Needle retrieval
correct dense and sparse at every length. Speedup grows with context
exactly as the O(L²) → O(L·budget·L) model predicts; extrapolating, ≥3x
lands around 96K+ tokens on this hardware.

Also fixed while validating the kernel: Stage-1's mask semantics gave
vertical_slash heads the layer-max sink/window at block granularity
(a superset) and its column dedup was block-level, dropping some columns
for late rows in a block. Both paths now implement the exact per-head
pattern (per-row dedup); tests cover kernel-vs-reference, maskless-vs-dense
saturation, chunk offsets, and first-chunk fallback (21 pass).

### Remaining gaps to the 3–10x target

- The maskless path attends the *union* pattern per layer (max sink/window),
  so effective density (~0.18–0.20) sits above the calibrated 0.15.
- Block-sparse pattern class and simdgroup-matrix custom kernel remain
  unimplemented — the latter is the only route to paper-level speedups at
  16–32K, and is a large standalone effort (essentially flash attention
  with gather in Metal).
- On ≥7B full-attention models the calibrated budget should drop well below
  0.15 (MInference's regime), improving all numbers; not testable on the
  8 GB box.

## Next steps (2026-07-08, priority order)

- [ ] **End-to-end validation through the real server** (Phase 4 precursor):
      run `scripts/perf_bench.py --ab --setting-key sparse_prefill_enabled
      --scenario long_context` against a running oMLX server with
      Llama-3.2-1B + the b0.15 calibration. Verifies BatchedEngine
      activation, chunked prefill, prefix-cache and settings validation in
      the actual serving path, and that the standalone-script speedups
      (1.29–2.49x) survive there. Surface `get_stats()` via an admin
      endpoint for `--stats-path` while at it.
- [ ] **Widen the quality gate beyond the single needle probe**: a light
      multi-question long-context QA set (10–20 questions at varied depths,
      dense-vs-sparse answer agreement) before recommending the feature as
      a per-profile default anywhere.
- [ ] **Only then, big-ticket items** (gated on access to a bigger box +
      ≥7B full-attention model, where calibrated budget should drop to
      ~0.1): simdgroup-matrix custom kernel (the only route to paper-level
      3–10x at 16–32K; measured here that scalar Metal kernels lose to
      MLX's flash SDPA), block_sparse pattern class, admin-panel
      calibration job button, SpecPrefill deprecation decision.
