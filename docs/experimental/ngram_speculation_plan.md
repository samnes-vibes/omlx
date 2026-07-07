# N-gram / Prompt-Lookup Speculative Decoding — Implementation Plan

Date: 2026-07-05 (results appended same day — see "Measured results" at the end)
Branch: `feat/ngram-spec-decoding`
Status: **implemented** (`omlx/speculative/ngram.py`, `omlx/patches/ngram_spec.py`,
`scripts/perf_bench.py`; enabled per-model via `ngram_spec_enabled`)
Companion to: [5x_speedup_research.md](5x_speedup_research.md) (item A2.1, ranked #1 in recommended order),
[low_ram_perf_optimization_map.md](low_ram_perf_optimization_map.md) (items 0, 2a, 2d)

## Goal

Draft-model-free speculative decoding: match the most recent generated n-gram against the
prompt + generation history, propose the continuation as draft tokens, verify with the target
model in one forward pass. Expected 2–3x decode throughput on workloads where output echoes
input (code editing, summarization, RAG, agent/tool loops); ~neutral on freeform chat.

Why this first (see research doc for the full argument):

- **Zero extra memory** — no draft model, no trained checkpoint. Works on the 8 GB M1
  reference box where DFlash is impossible, and on every model architecture.
- **Worst case is ~no-op** — when the n-gram misses, decode proceeds normally. No quality
  risk: verification makes output distribution-identical to normal decoding.
- **The hard half already exists** — `omlx/patches/mlx_lm_mtp/batch_generator.py` implements
  multi-token verify forwards, stochastic acceptance (`min(1, p_target/p_draft)` + residual
  distribution), cache rollback, and acceptance stats. This feature adds a new *draft source*
  and generalizes the verify length; it does not build a new speculation engine.

## Non-goals (this branch)

- Tree/multi-candidate drafting, suffix automatons (SuffixDecoding), datastore-backed lookup.
  Linear single-candidate drafts only; extensions noted in "Later" below.
- Layer-skip self-speculation and the full cascade (research item A2.3) — separate branch.
- VLM engine path (`vlm_mtp`) — text `BatchGenerator` path first.
- Upstreaming to mlx-lm.

## Design overview

```
scheduler decode step
  └─ BatchGenerator.next()  (patched, as MTP does today)
       ├─ NgramProposer.propose(recent_tokens) → [t1..tK] | None
       │     (hash map over prompt + generated tokens, built at admission,
       │      appended to as tokens are emitted)
       ├─ hit:  verify forward over [next_main, t1..tK]  (K+1 tokens)
       │        → accept longest valid prefix (greedy: exact match;
       │           sampled: stochastic acceptance per token)
       │        → rollback KV/logit rows past the accepted prefix
       └─ miss: normal 1-token decode step (or MTP cycle if mtp_enabled)
```

Key differences from the existing MTP patch, which this generalizes:

1. **Variable-length verify.** MTP verifies exactly 2 tokens (`[next_main, draft]`,
   `n_confirmed=1`). N-gram hits produce K=4–16 draft tokens, so the verify path, cache
   rollback, and row-realignment logic must handle `n_confirmed ∈ [1, K+1]`. This is the
   main engineering risk (see Risks).
2. **Draft source is free and CPU-side.** No MTP-head forward, no GPU cost when drafting.
   Propose/miss decisions happen on the Python side between steps.

### NgramProposer sketch

- Per-request structure built once at admission from prompt tokens (`O(prompt_len)`),
  extended incrementally as tokens are emitted.
- Map from n-gram (tuple of token ids, for n = `max_n` down to `min_n`) → most recent
  position of that n-gram. On propose: take the last `n` emitted tokens, look up, return the
  `K` tokens following the match position. Prefer longest n first (fewer false continuations).
- Defaults to start with (tune in Phase 4): `min_n=2`, `max_n=4`, `K=8`.
- Memory: one dict per request, ~O(context_len) small tuples — negligible next to KV.

## Phases

### Phase 0 — prerequisites (partly done)

- [ ] Commit the environment fixes already on this branch's working tree
      (`.python-version`, `omlx/_transformers_compat.py`, `omlx/__init__.py` import) as a
      separate commit — dev workflow does not function on a clean install without them.
- [ ] Capture baseline: PP/TG admin-benchmark run for `Qwen3.5-0.8B-MLX-4bit` on the
      reference machine; save results next to this doc.

### Phase 1 — acceptance metrics surfaced in the benchmark

The MTP patch already tracks `cycles / accepts / rejects / draft_emits / bonus_emits /
backbone_ms` (`batch_generator.py:432-445`) but the admin benchmark does not report them.

- [ ] Expose the stats struct through the engine (`BatchedEngine` → scheduler → generator)
      per request and cumulatively.
- [ ] Add accept-rate + emitted-tokens-per-cycle + speculation-overhead columns to the admin
      benchmark output; make them speculation-source-agnostic (MTP now, n-gram next).
- [ ] Log line on request completion mirroring the DFlash one:
      `Ngram spec: 502 tokens, accept=61%, 2.4 tok/cycle, 74 tok/s`.

Deliverable: MTP on/off comparison measurable (closes perf-map item 2d's tooling gap) before
any n-gram code lands.

### Phase 2 — NgramProposer (pure Python, no GPU changes)

- [ ] `omlx/speculative/ngram.py`: proposer class as sketched above + unit tests
      (hit/miss/incremental-extend/eviction correctness, determinism).
- [ ] Wire construction into request admission; feed emitted tokens from the response loop.
- [ ] No behavior change yet: log would-be-proposals + hypothetical hit rate behind a debug
      flag. Run against the Phase 4 workloads to validate hit rates *before* touching the
      verify path (cheap go/no-go gate: if hit rates on target workloads are <20%, stop and
      reassess `min_n/max_n/K` or the feature itself).

### Phase 3 — variable-length verify integration

- [ ] Generalize the MTP patch's 2-token verify to K+1 tokens: verify forward, acceptance
      loop (greedy prefix match; stochastic acceptance when a sampler is set — reuse the
      existing `min(1, p_t/p_d)` machinery), KV rollback of rejected rows (extend the
      `cache_rollback.py` trim path), batch row realignment.
- [ ] Draft-source arbitration per step: n-gram hit → n-gram draft; miss and `mtp_enabled` →
      MTP cycle; else plain decode. (Do **not** stack n-gram drafts on top of MTP drafts in
      the same step in v1.)
- [ ] Batch handling v1: speculation only for batch rows that are eligible under the same
      alignment constraints the MTP patch enforces (`_batch_rows_aligned_for_mtp` analog);
      ineligible rows take the plain path. Mixed per-row verify lengths are out of scope —
      cap the whole batch's verify length to the minimum proposed K to keep rows aligned.
- [ ] Settings: `ngram_spec_enabled: bool = False`, `ngram_spec_min_n / max_n / max_draft`
      in `omlx/model_settings.py` following the `mtp_*` pattern; admin UI toggle under
      Experimental Features; `requires_reload` on change.
      Compatibility matrix: allowed with `turboquant_kv_enabled` (verify is a normal forward);
      allowed with `mtp_enabled` (arbitration above); refused with `dflash_enabled`
      (DFlashEngine bypasses BatchGenerator entirely).

### Phase 4 — validation & tuning

- [ ] Benchmark scenarios (extend admin benchmark or a script under `scripts/`):
      (a) summarization of a 2–4K document, (b) code-edit ("rewrite this function")
      round-trips, (c) RAG-style answer-with-quotes, (d) freeform chat as the neutrality
      control.
- [ ] Sweep `min_n/max_n/K` on the reference machine; pick defaults; document results here.
- [ ] Accept/rollback correctness A/B: temp=0 output must be token-identical with the
      feature on vs off across the scenario suite (same seed, greedy).
- [ ] Reference-machine check (M1 8 GB): confirm net-positive wall clock. The MTP cost-model
      comment (`batch_generator.py:31-35`) warns compute-bound small-model decode makes a
      2-token verify cost ~2x a 1-token step; n-gram's long free drafts should clear the bar
      where MTP couldn't, but this must be measured, not assumed. If K+1-token verify
      forwards regress TTFT under concurrency, gate speculation on batch size 1–2.

## Risks

| Risk | Mitigation |
|---|---|
| Variable-length verify breaks the heavily monkey-patched BatchGenerator internals (row realignment, `filter`/`extend` passthroughs, `omlx/scheduler.py:600-1070`) | Phase 3 lands behind the off-by-default setting; temp=0 token-identity A/B is the merge gate; keep the 2-token MTP path untouched as fallback |
| Verify forward cost > speculation gain on compute-bound configs (small models on M1/M2 base) | Phase 2 hit-rate gate + Phase 4 wall-clock measurement before enabling by default anywhere; per-model setting stays opt-in |
| Upstream-sync friction (CLAUDE.md: avoid changes that complicate merging upstream) | All logic in new files (`omlx/speculative/ngram.py`) + the existing patch modules; no edits to vendored/upstream-mirroring code paths beyond the already-patched surfaces |
| Hit rate too low outside echo-heavy workloads → feature looks dead in default benchmarks | Report accept-rate per scenario (Phase 1 metrics) so neutral-on-chat / fast-on-RAG reads as designed behavior, not noise |

## Later (explicitly deferred)

- Suffix-automaton / SuffixDecoding-style datastore for higher hit rates and O(1) longest-match.
- Cascade with MTP within a single step (n-gram for long echoes, MTP head for the miss case).
- Cross-request corpus lookup (shared datastore over recent completions).
- Tree verification of multiple candidate continuations.

## File touchpoints

| File | Change |
|---|---|
| `omlx/speculative/ngram.py` | new — proposer + stats |
| `omlx/patches/mlx_lm_mtp/batch_generator.py` | generalize verify length; draft-source arbitration |
| `omlx/patches/mlx_lm_mtp/cache_rollback.py` | K-token rollback |
| `omlx/model_settings.py` | `ngram_spec_*` settings + compatibility validation |
| `omlx/scheduler.py` | proposer lifecycle (admission/emit/cleanup), stats plumbing |
| `omlx/admin/benchmark.py`, `admin/routes.py`, dashboard templates/js | metrics + toggle |
| `tests/test_ngram_proposer.py`, `tests/test_ngram_spec_decoding.py` | new |

---

## Measured results (2026-07-05, reference machine)

Hardware: M1 Mac mini 8 GB (compute-bound decode — the *worst case* for
speculative decoding per the MTP cost-model note). Model:
`Qwen3.5-0.8B-MLX-4bit` via VLM engine (hybrid GDN → gdn rollback mode).
`scripts/spec_bench.py --ab --runs 3`, temp=0, median of 3:

| scenario  | off tok/s | on tok/s | speedup | accept | notes |
|-----------|-----------|----------|---------|--------|-------|
| code_edit | 110.0 | 144.6 | **1.31x** | 96 % | 4.8 tokens/forward on hit streaks |
| freeform  | 104.2 | 95.5  | 0.92x | 0 %  | gates disable spec after ~4 cycles |
| summarize | 107.6 | 96.7  | 0.90x | 40 % | verify too costly at this accept rate (see below) |
| rag       | 208.6 | 145.3 | 0.70x | 43 % | 16-token responses; dominated by probe + first cycles |

### What was learned

1. **The verify forward is expensive on this path**: a 9-token gdn-capture
   verify measures ~60 ms vs ~9.6 ms for a plain step (~6.5x, not the ~2x a
   bandwidth-bound machine would pay). Two independent reasons: M1 decode is
   compute-bound, and the capture forward takes mlx-vlm's `target_verify`
   kernels (dequantized verify linears). Breakeven acceptance is therefore
   ~70 % here — only the code_edit-class workloads clear it. On trim-mode
   models (plain transformers) and on M3/M4-class machines both factors
   shrink, so the economics improve wholesale; re-benchmark there.
2. **The miss path must cost exactly zero.** Proposal misses delegate to the
   pre-patch stock step, and lookups anchor at the last *emitted* token
   (whose host-side id is free) so a miss adds no GPU sync. Earlier versions
   that ran their own plain step, or synced the pending sample every step,
   lost 25 % throughput on freeform from broken async pipelining alone.
3. **Adaptive gates are mandatory**: exponential backoff on consecutive
   zero-accept cycles, plus a measured cycle-cost vs plain-step-cost EMA
   gate (two strikes → speculation off for the request). These cap the
   worst-case regression to the first few cycles of a request.
4. **Greedy identity**: bit-exact in trim mode (asserted by unit tests across
   draft lengths, stop tokens, max_tokens). On the hybrid gdn path the
   capture forward computes logits through higher-precision verify kernels,
   so argmax can differ from the stock path at near-ties (~1 divergence per
   few hundred tokens observed; each continuation is self-consistent and a
   valid greedy output of the same model). This is the same caveat that
   applies to the mlx-vlm MTP/EAGLE paths using those kernels.
5. TurboQuant KV + gdn capture is incompatible (`_QuantizedStateProxy` is not
   subscriptable in the target_verify attention path); detected up front and
   speculation stays off rather than crashing.

### Follow-ups

- Benchmark on bandwidth-bound hardware (M3/M4/Max) and on a trim-mode
  model — both remove the dominant cost factor measured here.
- Investigate a non-capture verify for the all-accepted fast path, or a
  cheaper gdn rollback that avoids `target_verify` kernels.
- Adaptive draft length (scale K with rolling acceptance).
- Phase 1 acceptance metrics in the admin dashboard (stats endpoint exists:
  `GET /admin/api/ngram-spec/stats`).
