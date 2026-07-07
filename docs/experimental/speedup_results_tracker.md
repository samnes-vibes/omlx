# Speedup Results Tracker — Measured Results per Branch & Combined Effect

Date created: 2026-07-06 (living document — update after every branch's Phase-final benchmark)
Companion to: [5x_speedup_research.md](5x_speedup_research.md) (estimates & combination matrix this doc verifies)

## Purpose

The research doc predicts multiplicative stacking of independent techniques. This document
is the ledger that checks those predictions against reality: **one section per feature
branch** with its measured standalone result, and a **combined-effect section** measured
with feature stacks actually enabled together — not multiplied on paper.

Rules for updating:

1. Numbers come from reproducible runs: name the script + flags, model, machine, and date.
   Median of ≥3 runs, temp=0 unless the scenario demands otherwise. Use
   `scripts/perf_bench.py` for this — it's the generalized A/B harness (generalized from
   this branch's `spec_bench.py`): `--ab --setting-key <flag> [--stats-path
   <admin/api/...>]` flips a model setting off/on via the admin API and reports
   TTFT/decode tok/s per scenario plus any feature-specific stats. Extend its `SCENARIOS`
   dict rather than writing a new per-feature script when a plan needs a scenario the
   built-in four (`summarize`, `code_edit`, `rag`, `freeform`) don't cover (e.g. long
   context, permuted RAG chunks).
2. Standalone = feature on vs *everything else off* on the same baseline.
3. Combined rows are **measured, never computed** from standalone numbers. When a measured
   stack underperforms the product of its parts, that finding goes in "Interaction notes" —
   it is the most valuable content here.
4. A no-go / regression is recorded with the same rigor as a win.

## Reference configurations

| ID | Machine | Model | Notes |
|---|---|---|---|
| REF-8GB | M1 Mac mini 8 GB | `Qwen3.5-0.8B-MLX-4bit` (VLM engine, hybrid GDN) | compute-bound decode; worst case for speculation |
| REF-BIG | *(fill in: ≥32 GB machine)* | *(fill in: mid-size model + DFlash pair)* | bandwidth-bound decode; required for DFlash/ANE work |

Baselines (fill per config before first feature measurement; re-baseline after any
upstream merge that touches the hot path):

| Config | Decode tok/s (freeform) | Prefill tok/s @8K | TTFT @8K | Date / commit |
|---|---|---|---|---|
| REF-8GB | 104.2 (spec_bench freeform, off-arm, 2026-07-05) | — | — | b499b7e |
| REF-BIG | — | — | — | — |

---

## Per-branch results

### 1. N-gram speculation — `feat/ngram-spec-decoding` — **measured**

Source: [ngram_speculation_plan.md](ngram_speculation_plan.md) "Measured results";
`scripts/spec_bench.py --ab --runs 3`, REF-8GB, 2026-07-05.

| Scenario | Speedup | Accept | Verdict |
|---|---|---|---|
| code_edit | **1.31x** | 96 % | clears breakeven |
| freeform | 0.92x | 0 % | gates cap the loss |
| summarize | 0.90x | 40 % | verify too costly on this hw/path |
| rag | 0.70x | 43 % | short responses dominated by probe cost |

Key context for stacking: REF-8GB's gdn-capture verify costs ~6.5x a plain step (breakeven
accept ~70 %). On trim-mode models / bandwidth-bound machines the economics improve
wholesale — **REF-BIG re-run is the open item** before this feature's stacking value is
known.

### 2. CacheBlend chunk-KV reuse — `feat/cacheblend-kv-reuse` — *not started*

Plan: [cacheblend_plan.md](cacheblend_plan.md). Predicted: 2–3x TTFT on RAG/agent prompts.
Benchmark with `scripts/perf_bench.py --ab --setting-key chunk_kv_reuse_enabled` once
the RAG-permuted/agent-loop/multi-turn scenarios are added to its `SCENARIOS` dict.

| Scenario (ttft_bench) | TTFT off | TTFT on | Speedup | Fidelity (answer-match) | Date |
|---|---|---|---|---|---|
| RAG permuted chunks | — | — | — | — | — |
| agent loop, stable head | — | — | — | — | — |
| multi-turn, edited middle | — | — | — | — | — |
| strict-prefix control | — | — | expect ~1.0x | — | — |

### 3. DFlash long-context verify — `feat/dflash-long-context` — *not started*

Plan: [dflash2_long_context_plan.md](dflash2_long_context_plan.md). Predicted: 3–4x decode
sustained to 32K (vs fallback ≈1x today past `dflash_max_ctx`). REF-BIG only.
`scripts/perf_bench.py`'s `--ab` flip is boolean-only; the numeric `dflash_verify_*_size`
settings need a wrapper or a boolean convenience flag — see the plan's Phase 3 note.

| Context | tok/s fallback (today) | tok/s windowed verify | Accept Δ vs full verify | Date |
|---|---|---|---|---|
| 8K | — | — | — | — |
| 16K | — | — | — | — |
| 32K | — | — | — | — |

Also record here: Phase 0 verification of the already-shipped dflash prefix cache
(turn-2+ TTFT) and ddtree-vs-adaptive verify-mode comparison.

### 4. Draft-free sparse prefill — `feat/sparse-prefill-draftfree` — *not started*

Plan: [sparse_prefill_plan.md](sparse_prefill_plan.md). Predicted: 3–10x prefill ≥16K.
Benchmark with `scripts/perf_bench.py --ab --setting-key sparse_prefill_enabled` plus a
long-prompt scenario added to `SCENARIOS`. Record calibration recall (Phase 1 go/no-go)
plus:

| Prompt len | Stock TTFT | SpecPrefill TTFT | Sparse TTFT (stage 1 / stage 2) | Quality Δ | Date |
|---|---|---|---|---|---|
| 8K | — | — | — | — | — |
| 16K | — | — | — | — | — |
| 32K | — | — | — | — | — |

### 5. Fused int4 attention — `feat/fused-int4-attention` — *not started*

Plan: [fused_int4_attention_plan.md](fused_int4_attention_plan.md). Predicted: 1.5–3x
decode at 8–32K with TurboQuant KV. `scripts/kernel_bench.py` (new, isolated kernel
timing) covers Phase 0/1 microbenchmarks; Phase 3's end-to-end A/B uses
`scripts/perf_bench.py --ab --setting-key turboquant_fused_kernel`. Record Phase-0
headroom first (dequant-path overhead vs fp16 baseline), then:

| Context | tok/s fp16 KV | tok/s TQ dequant path | tok/s TQ fused | Top-1 identity | Date |
|---|---|---|---|---|---|
| 2K | — | — | — | — | — |
| 8K | — | — | — | — | — |
| 32K | — | — | — | — | — |

### 6. Research spikes — `experiment/ane-*`, `experiment/dllm-engine` — *not started*

Plan: [research_spikes_plan.md](research_spikes_plan.md). Each spike appends go/no-go +
kill-gate measurements to its plan doc; mirror only the verdict line here:

| Spike | Kill gates passed | PoC result | Verdict | Date |
|---|---|---|---|---|
| A4 ANE draft pipeline | — | — | — | — |
| B3 ANE prefill | — | — | — | — |
| A5 dLLM engine | — | — | — | — |

---

## Combined effect (measured stacks)

Mirror of the research doc's combination matrix, with a "measured" column that only gets
filled by an actual run with the stack enabled. Predicted values copied for comparison.

| Stack | Features enabled together | Config | Predicted | **Measured** | Date / commit |
|---|---|---|---|---|---|
| Decode, 8 GB | ngram + fused int4 (TQ KV) | REF-8GB | ~3–4x (research says with MTP+layer-skip; ngram-only stack will be lower) | — | — |
| Decode, big box | dflash long-ctx + fused int4 verify | REF-BIG | ~7x | — | — |
| Decode, big box max | + ANE draft pipeline | REF-BIG | ~8–10x | — | — |
| Prefill, long ctx | sparse prefill + fused int4 | any | ~5–8x | — | — |
| Prefill, RAG/agent | cacheblend + sparse prefill on recompute set | any | ~6–10x | — | — |

### Compatibility matrix (what may legally be enabled together)

Maintained here because it gates which combined rows are even measurable. Update when
settings-validation rules change (`omlx/model_settings.py`).

| | ngram | mtp | dflash | TQ KV | fused int4 | cacheblend | sparse prefill |
|---|---|---|---|---|---|---|---|
| **ngram** | · | ✗ (mutually excl.) | ✗ | ⚠ gdn-capture path incompatible today (proxy); fused kernel may fix | plan §P2 | ✓ planned (orthogonal) | ✓ planned |
| **mtp** | | · | ✗ | ✓ | plan §P2 | ✓ | ✓ |
| **dflash** | | | · | ✗ today (fork setting) | Design B later | ✗ (own engine/cache) | ✗ (own prefill) |
| **TQ KV** | | | | · | requires | ✗ v1 (shift needs dequant) | ✓ |
| **cacheblend** | | | | | | · | ✓ later (compose) |

✓ = allowed/planned, ✗ = refused by validation, ⚠ = known issue, · = self.

### Interaction notes

*(append findings here when a measured stack diverges from the product of standalone
numbers — e.g. two features competing for the same bandwidth headroom, gate interactions,
scheduler contention. Nothing yet.)*

- 2026-07-05 (pre-existing): ngram spec × TurboQuant KV on the gdn path is detected
  incompatible at runtime (`_QuantizedStateProxy` not subscriptable in target_verify) —
  speculation stays off rather than crashing. Fused-int4 plan Phase 2 owns re-testing this.

## Bottom line (keep this current)

> **Best measured end-to-end speedup so far: 1.31x decode** (ngram spec, code_edit,
> REF-8GB, 2026-07-05) vs research target of 5–10x. All other branches unstarted;
> REF-BIG baseline missing — establishing it is the highest-leverage next measurement.
