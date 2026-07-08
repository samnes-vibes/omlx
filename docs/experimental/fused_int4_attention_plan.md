# Fused Int4 Compressed-Domain Attention Kernel — Implementation Plan

Date: 2026-07-06
Suggested branch: `feat/fused-int4-attention`
Status: **implemented (v1, verify-length path)** — see "What was learned" at the end.
(item A3, ranked #5 in [5x_speedup_research.md](5x_speedup_research.md) recommended order)
Companion to: [ngram_speculation_plan.md](ngram_speculation_plan.md) (template & precedent),
[Open-TQ-Metal (arXiv:2604.16957)](https://arxiv.org/pdf/2604.16957)

## Goal

A fused Metal SDPA kernel that attends **directly over TurboQuant 4-bit KV** instead of
dequantize-then-attend. Decode throughput on Apple Silicon ≈ bandwidth / bytes-per-token;
today `omlx/turboquant_kv.py` compresses storage 4x but the attention path materializes
fp16 KV first, so the bandwidth win is mostly lost at the moment it matters. Reference
result (Open-TQ-Metal): 48x attention speedup at 128K vs dequantize-then-attend, identical
top-1 predictions.

Expected in oMLX: 1.5–3x decode at moderate context (8–32K), growing with length; and it
**multiplies with every speculation feature** (ngram spec, MTP, DFlash verify — all their
verify forwards read the same KV).

## Non-goals (this branch)

- W4A4/W4A8 quantized-activation GEMMs (M5/NVFP4 lever — research B4, separate).
- Prefill sparse attention (B1 plan) — decode/verify SDPA only.
- Changing TurboQuant's quantization algorithm or on-disk/SSD format.
- Bit depths other than 4 in v1 (`turboquant_kv_bits` supports 2–8; kernel v1 targets 4,
  the default; others fall back to the existing path).

## Design overview

```
today:  q(fp16) × dequant(K_int4 → fp16) → scores → softmax × dequant(V_int4 → fp16)
                 └── full fp16 KV materialized: bandwidth = fp16-equivalent

target: fused kernel: load K,V as packed int4 + scales/biases,
        dequantize per-tile in registers/threadgroup memory, accumulate in fp32
        → bytes moved from device memory = int4 + metadata only (~4x less)
```

Integration points, in the order an implementer should read them:

1. `omlx/turboquant_kv.py` — the quantized cache: packed layout, group size,
   scales/biases, and the `_QuantizedStateProxy` the ngram plan flagged as
   non-subscriptable in capture paths (this kernel is the eventual fix for that class of
   incompatibility: verify forwards attending in compressed domain need no proxy dequant).
2. `omlx/patches/turboquant_attention.py` — the existing attention patch where
   dequant-then-attend happens today; the kernel replaces its inner SDPA call.
3. `omlx/custom_kernels/glm_moe_dsa`, `minimax_m3` — in-repo packaging precedent for
   custom Metal kernels (`mx.fast.metal_kernel` usage, dispatch/fallback conventions).

Kernel shape decisions:

- **Decode-first: q_len ∈ [1, ~32].** One kernel specialization covers plain decode
  (q_len=1) and speculative verify (q_len = K+1 ≤ ~17 for ngram/MTP, 16 for DFlash
  blocks). This is the small-q regime where a hand kernel beats generic SDPA most.
- GQA-aware: KV heads < Q heads (all target models); tile over KV heads, broadcast to the
  query-head group.
- Softmax numerics in fp32 with running-max (flash-style single pass over KV).
- Quantization metadata (per-group scale/bias) loaded once per tile.
- Causal masking for the verify case (q_len>1); no arbitrary-mask support in v1.

Acceptance bar borrowed from the paper: **top-1 predictions identical** to the
dequantize-then-attend path on greedy decode (bitwise logit identity is not achievable and
not required).

## Phases

### Phase 0 — quantify the headroom (no kernel code)

- [ ] Instrument the current TurboQuant attention path: time SDPA+dequant per decode step
      at context {2K, 8K, 32K} vs fp16-KV baseline on the reference machine(s). If
      TurboQuant decode is not measurably slower than the theoretical int4-bandwidth
      bound, the kernel's ceiling is small — record the measured headroom here first.
- [ ] Read Open-TQ-Metal's published kernel (if code is available) and the in-repo kernel
      precedents; write a one-page layout note (tile sizes, threadgroup memory budget for
      M1-class GPUs) appended to this doc before implementation.

### Phase 1 — reference + kernel skeleton

- [ ] `omlx/custom_kernels/tq_attention/`: pure-MLX reference implementation
      (dequant-in-tiles emulation) that defines the exact expected numerics, + the
      `mx.fast.metal_kernel` implementation for q_len=1, 4-bit, one group size.
- [ ] Numerics tests: kernel vs reference vs existing dequant path — max logit deviation
      bound, top-1 identity over randomized (seeded) Q/KV at context {128, 2K, 8K}
      including GQA shapes of the supported model list.
- [ ] Microbenchmark script (`scripts/kernel_bench.py`): kernel vs current path, ns/step
      vs context length; the deliverable is the crossover point. (Purpose-built micro
      timing, not `scripts/perf_bench.py` — that harness measures end-to-end server
      TTFT/decode tok/s, not isolated kernel latency; use it in Phase 3 instead.)

### Phase 2 — verify-length support + integration

- [ ] Extend kernel to q_len ≤ 32 with causal mask (covers ngram/MTP/DFlash verify).
- [ ] Wire into `omlx/patches/turboquant_attention.py`: dispatch to the fused kernel when
      (bits==4, supported head dims/group size, q_len ≤ 32); silently fall back to the
      existing path otherwise. No new user-facing setting needed beyond a kill switch:
      `turboquant_fused_kernel: bool = True` in `omlx/model_settings.py` (default on when
      turboquant is on — fallback safety makes this reasonable; flip default to off if
      Phase 3 finds surprises).
- [ ] Resolve/retire the `_QuantizedStateProxy` capture incompatibility for the ngram-spec
      gdn path if the fused kernel makes it moot (see ngram plan "What was learned" #5) —
      at minimum, re-test that combination and update the compatibility matrix.

### Phase 3 — end-to-end validation

- [ ] Greedy A/B, temp=0: fused vs dequant path token-identity over the
      `scripts/perf_bench.py` scenario suite (`--ab --setting-key
      turboquant_fused_kernel`; per the acceptance bar; document any near-tie divergences
      the way the ngram plan did).
- [ ] End-to-end decode benchmark: {2K, 8K, 32K} context × {plain decode, ngram spec on}
      × {fused on/off}; the headline table for this doc. `perf_bench.py --stats-path` can
      report ngram accept stats alongside the fused-kernel toggle for the stacked case.
- [ ] Long-run stability: 30-min sustained generation, watch memory (`memory_monitor`)
      and numerical drift (perplexity spot check).
- [ ] M1 (8 GB) and at least one bandwidth-bound machine (M3/M4-class) — the win should be
      *larger* on bandwidth-bound hardware; confirm the ngram plan's follow-up hypothesis.

## Risks

| Risk | Mitigation |
|---|---|
| `mx.fast.metal_kernel` constraints (no simdgroup matrix ops in some MLX versions; threadgroup memory limits on base chips) force a slow kernel | Phase 0 layout note before coding; reference impl defines correctness independent of kernel strategy; worst case ship q_len=1 only |
| TurboQuant layout (group size / packing) awkward for coalesced loads | Kernel may require one supported group-size config; validate against `turboquant_kv.py` defaults first, extend later |
| Numerics: fp32-accum fused path diverges from dequant path at near-ties | Top-1-identity bar with measured divergence rate, same reporting convention as ngram plan learned-item #4 |
| MLX version churn breaks the kernel API | Pin/guard on mlx version at patch-apply time; fallback path always present |
| Win smaller than paper (paper measured 128K; oMLX realistic contexts 8–32K) | Phase 0 headroom measurement sets expectations before the effort is sunk; crossover-point microbenchmark gates integration |

## Later (explicitly deferred)

- 2/3/8-bit kernel variants (`turboquant_kv_bits` full matrix).
- Prefill-length q (chunked prefill over compressed KV).
- DFlash Design-B composition: full-context compressed verify
  ([dflash2_long_context_plan.md](dflash2_long_context_plan.md)).
- Fused small-batch verify GEMM (research A3's second extension).

## File touchpoints

| File | Change |
|---|---|
| `omlx/custom_kernels/tq_attention/` | new — reference impl + Metal kernel + dispatch |
| `omlx/patches/turboquant_attention.py` | dispatch to fused kernel with fallback |
| `omlx/turboquant_kv.py` | expose packed buffers/metadata accessors the kernel needs (read-only additions) |
| `omlx/model_settings.py` | `turboquant_fused_kernel` kill switch |
| `scripts/kernel_bench.py` | new — microbenchmarks |
| `scripts/perf_bench.py` | Phase 3 end-to-end A/B (`--setting-key turboquant_fused_kernel`) |
| `tests/test_tq_attention_kernel.py` | new — numerics, GQA shapes, fallback dispatch |

## What was learned (implementation, 2026-07-08)

1. **The plan's premise was half-stale.** The pinned mlx-vlm already fuses
   q_len=1 decode in the compressed domain (`_fused_mse_decode_kernel`, 1-pass
   ≤2K tokens, 2-pass beyond) and covers q_len>1 prefill for the *Prod* key
   codec via `prefill_attention`. The real gap was exactly the verify regime:
   q_len ∈ [2, 32] with the default integer-bit **MSE** codec fell through to
   dequantize + fp16 SDPA (full fp16 KV materialization + O(T·D²) inverse
   rotation per verify step). v1 targets only that gap.
2. **Implementation shape.** Two-kernel design mirroring mlx-vlm's own
   prefill structure: a new multi-query MSE score kernel
   (`omlx/custom_kernels/tq_attention/fast.py` — unpack each packed key once
   per token, loop the R·L query rows) + mlx-vlm's existing
   `_single_tile_value_weighted_sum_kernel` for values, causal mask + fp32
   softmax as MLX ops in between. Pure-MLX reference in `reference.py`
   defines the numerics. No mlx-vlm changes needed.
3. **Numerics bar met.** Fused output matches dequantize-then-attend on the
   same quantized state to <2e-2 max deviation (fp16) with >99.9% per-vector
   argmax agreement — they are mathematically identical
   (q·(norm·R⁻¹cb) = (Rq)·cb·norm), differing only in accumulation order.
4. **Profitability gate instead of blanket q_len ≤ 32.** The value kernel
   unrolls R·L accumulators and register-spills past R·L ≈ 64; the score
   kernel is latency-bound and linear in L. Measured (M-class, 32q/4kv
   GQA-8, D=128, `scripts/kernel_bench.py`): **2.3–2.4x at L=2, ~1.7x at
   L=4, ~1.2–1.3x at L=8 for T ≥ 8K; loses beyond R·L=64 or at T<4K with
   R·L>16.** Dispatch gates on (R·L ≤ 64) and (T ≥ 4096 or R·L ≤ 16);
   everything else silently falls back to the existing path.
   `force=True` bypasses the gate for tests/benchmarks.
5. **Fractional bits work.** turboquant_kv_bits=2.5 etc. split into
   integer-bit MSE codecs (key=floor, value=ceil), so the kernel supports
   them; only genuinely non-MSE codec configs fall back.
6. **Kill switch** is `turboquant_fused_kernel` (default True), classified
   model-specific in profiles, wired at patch-apply time in
   `engine/batched.py` and `engine/vlm.py`.
7. **Remaining headroom (Phase 2 candidates):** L ≥ 16 (DFlash-16 on GQA-8
   models is outside the gate) needs either query-chunked value dispatch or
   a flash-style fused kernel with threadgroup-memory scores; the score
   kernel's one-simdgroup-per-(token, repeat) layout has poor occupancy and
   is the next thing to attack. Phase 3 end-to-end A/B
   (`perf_bench.py --ab --setting-key turboquant_fused_kernel`) not yet run.
