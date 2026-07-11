# DFlash 2.0 — Long-Context Verify (Remove the Context Ceiling) — Implementation Plan

Date: 2026-07-06 (revised 2026-07-11: fork→patch-module correction, A3-landed update, new Phase 0.5)
Suggested branch: `feat/dflash-long-context`
Status: **planned** (item A1.1+A1.2, ranked #3 in [5x_speedup_research.md](5x_speedup_research.md) recommended order)
Companion to: [dflash_mlx_integration.md](dflash_mlx_integration.md),
[ngram_speculation_plan.md](ngram_speculation_plan.md) (template & precedent)

## Scope correction vs the research doc

The research doc listed four sub-items (A1.1–A1.4). Since it was written, the repo already
gained several of them — an implementer must NOT redo these:

- **A1.2 prefix caching: largely done.** `dflash_in_memory_cache` (L1 RAM),
  `dflash_ssd_cache` (L2 spill) settings exist and are wired in `omlx/engine/dflash.py`.
  Remaining work here is verification/benchmarking only (Phase 0).
- **A1.3 tree/adaptive verification: exposed.** `dflash_verify_mode` supports
  `"dflash" | "adaptive" | "ddtree"` (default "adaptive"). Remaining work: benchmark
  ddtree vs adaptive, pick per-profile defaults (Phase 3).
- **Sink/window draft KV: exposed.** `dflash_draft_window_size` / `dflash_draft_sink_size`
  already configure the *draft*.

**What this branch is actually about — the one big missing piece: A1.1, the context
ceiling.** `dflash_max_ctx` (default behavior falls back to `BatchedEngine` past the
threshold, see `omlx/engine/dflash.py:127,186`) throws away the entire 3–4x speedup
exactly where decode is slowest. The fix: apply the attention-sink + sliding-window trick
to the **target verify pass**, so verification cost stops growing with context, and retire
the fallback.

## Goal

DFlash speedup active at all context lengths (target: ≥32K) instead of only below
`dflash_max_ctx`. Verify-pass attention over sink + window KV instead of full context.
Expected: converts "3–4x on short prompts" into "3–4x always"; vs today's long-context
fallback path this is effectively >5x.

## Non-goals (this branch)

- Changes to draft-side behavior (already windowed).
- Temperature-aware stochastic acceptance for DFlash (research A1.4) — separate, smaller
  branch; the acceptance logic lives in the vendored dflash-mlx fork.
- New verify algorithms beyond what `verify_mode` already exposes.
- Applying sink+window to *normal* (non-DFlash) decode — that changes model quality
  semantics and is a different feature (StreamingLLM-style), out of scope.

## Design overview

The verify pass is a K-token forward through the target model attending to the full KV
cache. Two candidate designs, decided by a Phase 1 measurement:

**Design A — windowed verify KV (StreamingLLM semantics for verify only).** Target keeps
its full KV cache for correctness of future prefill/branching, but the verify forward
attends only to sink (first S tokens) + window (last W tokens). Implementation: a masked
SDPA or a gather of the sink+window KV slices into a compact buffer per verify call.
Quality caveat: verified tokens are then *accepted against a windowed approximation* of the
target distribution — output is no longer distribution-identical to full-context decode.
This must be validated empirically (paper precedent: sink+window preserves perplexity for
recent-token prediction, which is exactly the verify workload).

**Design B — quantized full-context verify.** Keep full-context verify but make it cheap:
target KV stored via TurboQuant (`omlx/turboquant_kv.py`) and verify attends over
compressed KV. Exact distribution, but the bandwidth win requires the fused int4 attention
kernel (research item A3, [fused int4 plan](fused_int4_attention_plan.md)) to avoid
dequantize-then-attend. If A3 lands first, Design B may dominate; the plans are
deliberately composable.

**Update 2026-07-11 — A3 has landed** (`fused_int4_attention_plan.md` work merged: commits
`9d07fbb`, `17a19bf`). The doc's original assumption ("Design B may dominate if A3 lands
first") is now testable, not hypothetical. Before committing to Design A only, run Phase
0.5 below to check whether Design B is now cheap.

**Recommendation: implement Design A behind a setting as the default path; use Phase 0.5 to
decide whether Design B is now competitive enough to implement instead/also.** Design A is
self-contained and available now regardless of A3.

```
DFlashEngine decode cycle (per block)
  draft: sink+window KV (already shipped)
  verify: full model forward over 16-token block
     └─ NEW: attention per layer restricted to [0:S] ∪ [L-W:L] of target KV
        (S = dflash_verify_sink_size, W = dflash_verify_window_size)
  accept/rollback: unchanged
  fallback: dflash_max_ctx retained as an escape hatch, default flips to None (unlimited)
     when verify windowing is enabled
```

**Correction 2026-07-11 — dflash-mlx is not our fork.** `pyproject.toml:108` pins
`dflash-mlx` straight from `git+https://github.com/bstnxbt/dflash-mlx@9ca0028...` (upstream,
not `jundot/dflash-mlx` mentioned in `dflash_mlx_integration.md` — that reference is stale
for what's actually installed). Forking it just for this feature would create an ongoing
upstream-sync burden the CLAUDE.md instructions explicitly want avoided.

Where the code actually lives: this repo already has a proven pattern for altering
dflash-mlx behavior without forking — `omlx/patches/dflash_lifecycle.py` monkey-patches
dflash-mlx's class-level hooks at runtime and restores them on `DFlashEngine.stop()`
(wrap/backup/idempotency-flag pattern). As of dflash-mlx 0.1.10 there is also a clean seam
to hang a patch on: `TargetOps.verify_block` / `verify_tree_block`
(`dflash_mlx/engine/target_ops.py`), implemented per model family in
`dflash_mlx/engine/target_qwen_gdn.py` and `dflash_mlx/engine/target_gemma4.py`. Implement
windowed verify as a new `omlx/patches/dflash_verify_window.py` following the
`dflash_lifecycle.py` wrap/restore pattern, patching `verify_block`/`verify_tree_block` for
the family backends actually in use, rather than touching a fork. Only settings plumbing
goes through `omlx/engine/dflash.py` / `omlx/model_settings.py`.

## Phases

### Phase 0 — baseline & A1.2 verification (no code)

- [ ] Benchmark matrix on a ≥32 GB machine (DFlash needs two models): context lengths
      {2K, 4K, 8K, 16K, 32K} × {dflash on (below ceiling), fallback (above ceiling)}.
      Record decode tok/s and the exact context where fallback kicks in today.
- [ ] Verify the existing L1/L2 prefix cache: multi-turn scenario, confirm turn-2+ TTFT
      drop and log hit stats. If it underperforms the research doc's expectation, file
      findings here — but do not expand scope into cache work on this branch.
- [ ] Measure verify-pass cost vs context length in isolation (time the block-verify
      forward at each length) — this is the curve the feature must flatten.

### Phase 0.5 — Design A vs Design B decision check (new, 2026-07-11)

- [x] Checked whether DFlash's target KV is already stored via TurboQuant int4
      (`omlx/turboquant_kv.py`) on the verify path: it is not. Neither `omlx/engine/dflash.py`
      nor the installed `dflash_mlx` package references `turboquant`/`tq_attention`
      anywhere; `turboquant_kv_enabled` has no dflash wiring today.
- [x] **Decision: Design A.** Target KV is not quantized under DFlash, so Design B would
      need that storage-format wiring in addition to the fused int4 kernel that already
      landed (A3) — clearly more work than Design A's monkeypatch-only approach.
      Implemented Design A below; Design B remains a later option (see "Later" section).

### Phase 1 — windowed verify as an omlx patch module

- [x] New `omlx/patches/dflash_verify_window.py`, mirroring `dflash_lifecycle.py`'s
      wrap/backup/idempotency-flag/restore structure. Patch target: the module-level SDPA
      dispatch dflash-mlx's verify forward actually calls —
      `target_qwen_gdn._gqa_reshape_sdpa` and `target_gemma4._gemma4_full_gqa_sdpa` — plus
      `TargetOps.verify_block` / `verify_tree_block` on both backends to scope activation to
      verify calls only (contextvar, not a global flag, so it's safe across concurrent
      generations).
- [x] Implemented as a **mask+gather at the fully-materialized KV, post `cache.update_and_fetch`**,
      not a per-layer cache slice: RoPE is already applied and cached at each key's original
      position by the time this hook runs, so gathering sink+window indices from the fetched
      `keys`/`values` arrays needs no re-rotation — positions are preserved by construction.
      Falls back untouched (passthrough) whenever `sink+window >= kv_len`, matching "windowed
      verify == full verify when S+W ≥ context" exactly, and whenever the window is smaller
      than the query block it is widened so the newest verify tokens can see their own
      just-written keys.
- [x] Unit tests in `tests/test_dflash_verify_window.py`: passthrough-when-covers-full-context,
      gather/trim correctness (sink block + window block indices, causal mask shape and
      structure), window auto-widening, activation scoping (contextvar only set during
      `verify_block`/`verify_tree_block`, cleared after), install/restore idempotency and
      round-trip cleanliness. All existing `test_dflash_*` suites (110 tests) still pass.
- [ ] Acceptance-decision divergence measurement (W=1024, S=64 vs full verify, quantified) —
      needs a real target/draft model pair on ≥32 GB hardware; not runnable in this
      dev sandbox. Deferred to Phase 3 alongside the quality gate.
- [ ] Handle rotating/hybrid layer types the same way the draft path does; refuse (keep
      full verify) on architectures where slicing is not implemented rather than crashing.
      Current patch only covers the two full-attention SDPA dispatch points used by
      Qwen-GDN and Gemma4 — needs a real-model smoke test to confirm no third path is missed.
- [x] Teardown wired: `restore_dflash_verify_window_patch()` called alongside
      `restore_dflash_class_patches()` in both `DFlashEngine.stop()` and
      `_evict_dflash_and_start_fallback()` (`omlx/engine/dflash.py`).

### Phase 2 — omlx plumbing

- [x] `omlx/model_settings.py`: `dflash_verify_window_size: Optional[int]` and
      `dflash_verify_sink_size: Optional[int]` (None = full verify, current behavior),
      following the existing `dflash_draft_*` pattern; docstrings added. Classified in
      `omlx/model_profiles.py` `MODEL_SPECIFIC_PROFILE_FIELDS` (a pre-existing test,
      `test_all_model_settings_fields_classified`, enforces every new field is
      classified — caught this automatically).
- [x] `omlx/admin/routes.py`: request model fields, PUT handler (same 0/None/negative
      normalization as `dflash_draft_window_size`/`dflash_draft_sink_size`), reset-to-default
      block, and diffusion-model unsupported-field sanitization list all updated.
- [x] Admin UI: fields added to `_modal_model_settings.html` (Verify window size / Verify
      sink size, next to the existing Verify mode select) and `dashboard.js` (defaults on
      load, payload building, profile-fields list, reset-to-default in two places). Also
      found and fixed `omlx/engine_pool.py`'s reload-fingerprint builder, which did not
      include the two new fields — without that, changing them via UI/API would not have
      triggered a model reload (settings change silently ineffective until restart).
- [x] `omlx/engine/dflash.py`: reads the two settings in `__init__`, calls
      `configure_verify_window(sink, window)` in `start()` right after installing the
      lifecycle wrap, and the startup log line now prints `verify_window=window=…,sink=…`
      or `verify_window=off`.
- [ ] `dflash_max_ctx` default-to-unlimited-when-windowing-is-set behavior — not implemented;
      today the two settings are independent (a caller must still explicitly set
      `dflash_max_ctx=None`/unlimited to actually reach long-context territory where
      windowed verify matters). Small follow-up in `DFlashEngine.__init__`.
- [ ] Keep the runtime-context passthrough (`verify_config`, `dflash.py:338`) working —
      per-request overrides if dflash-mlx supports them. Not investigated this pass.

### Phase 3 — validation & defaults

- [ ] Quality gate: long-context needle/summary tasks (16K–32K) with windowed verify on
      vs full verify — acceptance rate, output quality (task-level, not token-identity:
      Design A is approximate by construction). Bar: acceptance-rate drop <5 % absolute
      and no visible task regression at W=1024/S=64; else sweep W ∈ {1024, 2048, 4096}.
- [ ] Re-run the Phase 0 matrix; the deliverable table for this doc: tok/s at 8K/16K/32K,
      dflash-windowed vs today's fallback. `scripts/perf_bench.py` covers the scenario/TTFT
      harness, but `--setting-key` expects a boolean toggle (`--ab` flips it off/on) while
      `dflash_verify_window_size`/`sink_size` are numeric — either add a boolean
      `dflash_verify_windowing_enabled` convenience flag to gate on/off cleanly for `--ab`,
      or drive the sweep with a small wrapper script that calls `set_setting_enabled`'s
      underlying PUT with numeric values directly and reuses `run_pass`/`print_comparison`.
- [ ] Benchmark `verify_mode=ddtree` vs `adaptive` at long context while at it; record
      results and set profile defaults if ddtree wins.
- [ ] Memory check: gathered sink+window KV buffer is per-verify-call transient; confirm
      no growth in steady state (`memory_monitor` stats before/after a 32K run).

## Risks

| Risk | Mitigation |
|---|---|
| Windowed verify changes accepted-token distribution → subtle long-context quality loss | Explicit divergence measurement (Phase 1) + task-level gate (Phase 3); setting is opt-in, full verify remains default until data says otherwise |
| Middle-context information matters for the *next* block's draft even if verify windows fine | Draft already runs windowed and delivers 3–4x — the draft side is unchanged, so no new risk there; only acceptance decisions change |
| Fork divergence from upstream dflash-mlx grows | Isolate windowed verify behind one flag in the fork; keep the full-verify path untouched as default |
| Gather cost per verify call eats the win at moderate context | Measure at 4K–8K in Phase 3; if slicing costs more than it saves below ~8K, auto-enable windowing only above a context threshold |
| Requires ≥32 GB dev/bench machine (M1 8 GB reference box cannot run DFlash) | Plan explicitly assumes big-box hardware; note in doc so an 8 GB agent doesn't attempt local benchmarks |

## Later (explicitly deferred)

- Design B: TurboQuant compressed-domain full-context verify (compose with A3 kernel).
- Temperature-aware stochastic acceptance (research A1.4).
- Per-request adaptive W based on measured acceptance.

## File touchpoints

| File | Change |
|---|---|
| `omlx/patches/dflash_verify_window.py` (new) | sink+window verify attention patch on `TargetOps.verify_block`/`verify_tree_block` — the core change, no fork needed |
| `omlx/engine/dflash.py` | plumb `dflash_verify_window_size`/`sink_size`; max_ctx default logic; startup log line; call the new patch module's install/restore alongside `dflash_lifecycle.py`'s |
| `omlx/model_settings.py` | new settings + docstrings |
| `omlx/admin/` (routes, templates) | settings fields |
| `scripts/perf_bench.py` | long-context benchmark scenarios (extend `SCENARIOS`, see Phase 3 note on numeric settings) |
| fork tests / `tests/` | windowed-verify equivalence + divergence tests |
