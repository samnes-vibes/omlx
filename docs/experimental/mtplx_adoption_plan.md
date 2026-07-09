# MTPLX Adoption Research: Multi-Depth MTP Drafting + Automation

Date: 2026-07-07
Branch: `feat/mtp-multi-depth`
Status: **Phase 1 + Phase 3 implemented; Phase 2 spike implemented**
(2026-07-09) — see implementation notes at the end.
Phase 1 benchmarking (step 4) **blocked on hardware**: the dev machine is an
M1 base / 8 GB / <8 GB free disk, which can hold neither Qwen 3.6 27B nor
Qwen 3.5 9B, and the only local MTP-family checkpoint
(mlx-community Qwen3.5-0.8B-4bit) ships no mtp.* weights. Run step 4 on the
usual rig with `scripts/perf_bench.py --sweep-values` (added for this).
Upstream reference: [youssofal/MTPLX](https://github.com/youssofal/MTPLX) (Apache-2.0)
Companion to: [5x_speedup_research.md](5x_speedup_research.md), [ngram_speculation_plan.md](ngram_speculation_plan.md)
Measured outcomes ledger: [speedup_results_tracker.md](speedup_results_tracker.md)

## TL;DR

MTPLX's *core* MTP technique is the same thing oMLX already ships in
`omlx/patches/mlx_lm_mtp/`: native MTP-head speculation with exact
Leviathan & Chen rejection sampling + residual correction. **Adopting its
inference core would gain nothing.** The real, adoptable deltas are:

| MTPLX feature | oMLX status | Benefit type | Verdict |
|---|---|---|---|
| Multi-depth drafting (chained MTP head, depth k) | **depth-1 only** (1 draft + bonus, ceiling ~1.74×) | **Speedup** (MTPLX reports 2.24× on Qwen 3.6 27B @ temp 0.6) | **Adopt — P1** |
| `mtplx tune` (per-machine draft-depth auto-tuning) | none; MTP is a boolean (`mtp_enabled`) | Automation; also fixes the known compute-bound M1/M2 net-negative case | **Adopt — P2** |
| `mtplx inspect` (4-tier MTP compatibility classification) | implicit — errors surface at load time via `model_settings` validation | Automation / UX | **Adopt (light) — P3** |
| BF16-MTP-head policy (quantized MTP weights collapse acceptance 79-85% → 5-11%) | partially: per-key norm-convention repair in `qwen35_model.py`; no explicit quantization guard | Robustness | **Adopt (cheap check) — P3** |
| Forge (train MTP adapters for models *without* heads; HF→MLX convert; verify; publish) | none | Capability + automation | **Skip for now** — training internals are not open (spec documents only the CLI contract), heavy lift, and oMLX already covers head-less models via ngram-spec and DFlash |
| Thermal-pinned benchmark modes (Sustained/Burst) | admin benchmark exists (`omlx/admin/benchmark.py`), no fan pinning | Measurement hygiene | Optional nicety inside P2 |

## Why omlx's current MTP is capped

`batch_generator.py`'s cycle: one 2-token verify forward + one MTP-head
forward emits **1 + p** tokens per cycle (p = accept rate). Even at p = 1
that is ~1.74× — below MTPLX's measured 2.24×. The gap is draft depth:
with the MTP head applied autoregressively k times, a cycle verifies
k+1 positions in one forward and emits up to k+1 tokens. Expected tokens
per cycle become 1 + p + p² + … + pᵏ (with per-position acceptance p),
so at p ≈ 0.85 depth 3 yields ~3.2 tokens/cycle vs 1.85 at depth 1.

Caveat already documented in `batch_generator.py`: on compute-bound
single-stream Apple Silicon (M1/M2 base/Pro) the (k+1)-token verify
forward is *not* nearly free, so deeper drafts can be net-negative.
This is exactly why depth must be tuned per machine (P2), including
depth 0 = off. Wins concentrate on M3/M4, MoE backbones, and batched
decode — the same regime where depth-1 MTP already wins (#1097/#1311).

## What MTPLX actually is (for the record)

- Native macOS app + CLI + OpenAI/Anthropic-compatible server for Apple
  Silicon (M1+, macOS 14+), Apache-2.0.
- Uses **built-in MTP heads** shipped with Qwen 3.5/3.6 (and Gemma 4) —
  no external draft model. Exact rejection sampling (Leviathan & Chen,
  with residual correction) ⇒ output distribution identical to standard
  decoding at any temperature.
- Claim: 2.24× decode TPS on Qwen 3.6 27B @ temp 0.6.
- `mtplx tune --model … --retune`: benchmarks each draft depth on the
  user's machine with fans pinned, keeps autoregressive as baseline,
  persists the winning depth.
- `mtplx inspect`: classifies a model as verified / arch-compatible-
  unverified / incompatible / no-MTP-heads without running it.
- Forge: download → convert → **calibrate** (adapter training; internals
  undocumented, only progress-file contract with `loss`/`ppl` fields) →
  verify (throughput + acceptance across depths, measured before/after
  on the user's hardware) → publish to Hub. Hard policy: quantizing MTP
  weights collapses acceptance to 5-11% (vs 79-85% BF16); requires
  `--allow-degraded-mtp` to override.

## Implementation plan

Default-off, additive, no scheduler/cache-layer changes — same
integration philosophy as the existing MTP patch (all logic stays inside
`omlx/patches/mlx_lm_mtp/`).

### Phase 1 — Multi-depth drafting (speedup, the main prize)

Goal: `mtp_draft_depth: int = 1` in `ModelSettings` (1 = current
behavior, so default is bit-identical to today).

1. **Draft chain** (`qwen35_model.py`, `deepseek_v4_model.py`):
   generalize `mtp_forward` so the head can be applied autoregressively
   k times — each step feeds the previous draft token's embedding +
   the head's own hidden state. Qwen 3.5/3.6 and DeepSeek ship a single
   MTP block, so depth >1 reuses the same head with its own tiny KV/SSM
   state; that state must be included in the existing
   `rollback_state` snapshot machinery (`cache_rollback.py`) so a
   mid-chain rejection restores both backbone *and* head state.
2. **Verify step** (`batch_generator.py`): widen the 2-token verify
   forward to k+1 tokens `[next_main, d₁ … dₖ]`. Acceptance walks
   left-to-right: greedy path compares argmax per position; stochastic
   path applies `min(1, p_target/p_draft)` per position and residual-
   samples at the first rejection (standard multi-token Leviathan &
   Chen — the depth-1 code is the k=1 special case). `cache.trim(n)`
   already only moves `_idx`, so trimming a variable number of rejected
   positions needs no PagedCacheManager changes — verify this with a
   test rather than assuming it.
3. **Identity contracts**: extend `tests/test_mlx_lm_mtp_patch.py` —
   greedy identity at depths 1–4, stochastic distribution test, batch
   `extend`/`filter` state-drop at depth >1, SSM rollback across a
   mid-chain rejection.
4. **Measure** on the usual rig and record in
   [speedup_results_tracker.md](speedup_results_tracker.md): depth
   1/2/3/4 × {Qwen 3.6 27B, Qwen 3.5 9B} × temp {0, 0.6}. Success
   criterion: any depth >1 beats depth 1 by ≥10% on at least one
   model/temp cell; otherwise stop here and skip Phase 2.

Effort: ~2–4 days. Risk: SSM/conv rollback correctness at depth >1 is
the sharp edge (GatedDeltaNet state snapshots are currently taken once
per cycle, not per chain step).

### Phase 2 — Auto-tune (`omlx` analog of `mtplx tune`)

Goal: `mtp_draft_depth: "auto"` resolves to a per-(model, machine)
tuned value, cached on disk.

1. Reuse `omlx/admin/benchmark.py` internals: short decode benchmark at
   depth 0 (plain autoregressive baseline), then 1…4, fixed prompt set,
   ≥2 repeats, discard warmup. Persist
   `{model_id, hardware_id, depth, tps}` under `~/.omlx/` (same pattern
   as other cached tuning artifacts); `hardware_id` from
   `platform` + chip string.
2. Trigger: lazily on first MTP-enabled load with `"auto"`, or
   explicitly via an admin endpoint (`POST /admin/mtp/tune`) so the
   dashboard can drive it. Skip fan-pinning; instead interleave
   depth trials round-robin so thermal drift biases all depths equally.
3. Pick `argmax(tps)`; if the winner is depth 0, resolve to MTP-off for
   this machine — this converts the documented M1/M2 net-negative
   failure mode from a footnote into automated behavior.

Effort: ~1–2 days on top of Phase 1.

### Phase 3 — Inspect + BF16 head guard (cheap robustness)

1. `omlx.utils.model_loading`: at load, classify MTP support from
   config/weights only (heads present? architecture in the verified
   set?) and log one clear line; expose in the admin models listing.
   No new CLI needed.
2. Guard: if MTP weights are detected as quantized (dtype/`quant_config`
   on `mtp.*` keys), log a warning citing the acceptance-collapse
   finding and — if measured acceptance over the first N cycles falls
   below a floor (e.g. 15%) — auto-drop to standard decode via the
   existing `_drop_mtp_state` fallback path. The per-key norm repair in
   `qwen35_model.py` already handles the related mixed-convention bug;
   this adds the runtime safety net.

Effort: ~0.5–1 day.

### Explicitly out of scope

- **Forge-style adapter training.** The public repo documents only the
  frontend↔backend CLI contract, not the training recipe; reproducing
  it means designing MTP-head distillation from scratch. For head-less
  models oMLX already has ngram-spec (free) and DFlash (3–4×). Revisit
  only if a specific head-less model becomes a priority target.
- Porting any MTPLX code wholesale — it's a parallel server stack
  (own app, own vllm_metal fork); only the *ideas* above transfer.
  License is Apache-2.0, so selective code reading/porting is
  permitted with attribution if ever needed.

## Interaction matrix note

Multi-depth MTP inherits the existing MTP exclusivity rules in
`model_settings.py` (mutually exclusive with `dflash_enabled`,
`turboquant_kv_enabled`, `vlm_mtp_enabled`); depth is a parameter of the
existing feature, not a new row in the tracker matrix.

## Phase 1 implementation notes (2026-07-09)

Landed on `feat/mtp-multi-depth`. Deltas vs the plan sketch above:

- **Setting**: `mtp_draft_depth: int = 1` (1–8, validated in
  `ModelSettings.__post_init__`; classified model-specific in
  `model_profiles.py`). Depth is stamped per model instance at load
  (`_omlx_mtp_draft_depth`, same pattern as `_omlx_mtp_decode_enabled`).
- **Chain drafting** is anchored + chained, not replayed: step 1 fuses the
  *backbone's* pre-norm hidden at the anchor's predecessor with the
  confirmed anchor token (identical to the old `_step_mtp`); steps 2..k
  feed the head's own pre-norm hidden + previous draft back in via a new
  `mtp_forward_hidden` hook (`qwen35_model.py`, MTPModule gained
  `return_hidden`). Presence of `mtp_forward_hidden` is the capability
  gate for depth > 1.
- **Verify** widened to `[next_main, d₁..dₖ]` with `n_confirmed=1`;
  acceptance walks left-to-right (greedy compare / per-position
  Leviathan-Chen with residual sampling at the first rejection). One
  batched sampler call over (k+1, vocab).
- **Partial rollback**: `GatedDeltaNet.__call__` now processes the draft
  suffix per-token and stores `rollback_state` as a *list* of (conv, ssm)
  snapshots — index j = state after confirmed + j accepted drafts.
  `_restore_or_trim_caches(cache, accepted, total_drafts)` restores
  `snaps[accepted]` / trims `total - accepted`. The rotating-cache MTP
  undo arms for any verify size ≥ 2 (was == 2). As the plan predicted,
  `cache.trim(n)` needed no PagedCacheManager changes (test-verified).
- **Head-cache hygiene**: chained entries whose drafts get rejected are
  counted in `state.stale_head_entries` and trimmed at the next chain
  refill; depth 1 always trims 0, preserving the old behavior exactly.
- **DeepSeek-V4 clamps to depth 1** (no `mtp_forward_hidden`; its head
  cache is a RotatingKVCache whose speculative entries can't be safely
  trimmed once rotated). The mlx-vlm runtime path also stays at depth 1.
- **Tests** (`tests/test_mlx_lm_mtp_patch.py`): end-to-end greedy identity
  at depths 1–4 against plain autoregressive decode on a tiny
  random-weight patched Qwen3.5 (linear+full attention mix, so SSM
  snapshots and partial rollback actually execute); an oracle-head
  coverage guard forcing full/partial/zero-accept cycles; unit tests for
  snapshot-list restore, legacy tuple compat, atomic refusal, depth
  clamping, and settings validation. Stochastic distribution testing is
  deferred to the benchmarking step.
- **Stats**: per-sequence log line now reports accepted/drafted positions
  (`drafts a/b`), the per-cycle accept counters kept for full-accepts.

Next: plan step 4 — measure depth 1/2/3/4 on real checkpoints via
`scripts/perf_bench.py --setting-key mtp_draft_depth --sweep-values 1,2,3,4`,
record in [speedup_results_tracker.md](speedup_results_tracker.md); go/no-go
for Phase 2 auto-tune. Blocked on hardware as of 2026-07-09 (see Status).

## Phase 3 implementation notes (2026-07-09)

Landed on `feat/mtp-multi-depth` alongside benchmark plumbing. Deltas vs
the plan sketch:

- **3.1 (inspect/classification)** turned out to already exist:
  `_mtp_compat_for_model` in `omlx/admin/routes.py` performs the full
  config + model_type + mtp.*-weights classification and the admin models
  listing exposes `mtp_compatible` / `mtp_compatibility_reason`. Load-time
  logging also existed. No new code needed beyond the guard below.
- **3.2 (BF16 head guard)**: `_mtp_weights_quantized` in
  `omlx/utils/model_loading.py` detects quantized MTP heads from the
  safetensors index (an `mtp.*…​.scales` key; no shard I/O), warns at load,
  and stamps `_omlx_mtp_head_quantized` on the model instance (via a
  process-wide construction flag mirroring `set_mtp_draft_depth`).
  At runtime `_maybe_disable_low_acceptance_mtp` (`batch_generator.py`)
  checks after every verify cycle: acceptance < 15% after ≥ 64 drafted
  positions flips `_omlx_mtp_decode_enabled` off on the model instance —
  the next cycle finds the batch ineligible, drops MTP state, and decode
  continues on the standard path (sticky until reload). The warning names
  the quantized head as the likely cause when the stamp is set.
- **Benchmark plumbing for step 4**: `mtp_draft_depth` was missing from
  the admin API's settings request model (settable only via
  model_settings.json on disk) — added with 1–8 validation.
  `scripts/perf_bench.py` gained `--sweep-values` (comma-separated JSON
  values for `--setting-key`, one pass per value, comparison table vs the
  first value).
- **Tests**: quantized-index detection (incl. nested prefixes and the
  preload-dispatch stamp/reset), acceptance-floor guard (disable, hint
  text, min-draft gate, healthy-rate no-op, unstamped no-op, inner
  `language_model` stamp).

## Phase 2 spike implementation notes (2026-07-09)

`omlx/admin/mtp_tune.py` + `POST /admin/api/models/{id}/mtp-tune` +
`mtp_draft_depth: "auto"`. Deltas vs the plan sketch:

- **No reload per depth**: trials re-stamp the live model instance's
  per-instance markers (`_omlx_mtp_draft_depth`,
  `_omlx_mtp_decode_enabled`) between requests instead of driving the
  settings/reload path. The dispatch reads the stamps at chain-refill
  time, so between-request flips are safe, each trial is cheap, and the
  planned round-robin interleaving (thermal fairness) costs nothing.
  Consequence: the model must be loaded with `mtp_enabled=true` to tune
  (the head is attached at load); the endpoint 400s otherwise.
- **Depth grid**: 0..4 by default (0 = plain autoregressive baseline);
  models without `mtp_forward_hidden` (DeepSeek-V4, VLM runtime) sweep
  {0, 1}. One `max_tokens=128`, temp-0 decode per trial, 2 repeats,
  median tps, `argmax` wins; warmup run first.
- **Persistence**: `<base_path>/mtp_tune.json` keyed by
  `{model_dir_name: {hardware_id: {depth, tps_by_depth, tuned_at}}}`;
  `hardware_id` = platform machine + `machdep.cpu.brand_string` slug.
- **Resolution**: `ModelSettings.mtp_draft_depth` now accepts `"auto"`
  (admin API too). At load, `maybe_apply_pre_load_patches` resolves it
  from the store: untuned → depth 1 (info log points at the endpoint),
  winner 0 → MTP decode disabled for this machine (the documented M1/M2
  net-negative case, now automated), else the tuned depth.
- Rather than `omlx/admin/benchmark.py` internals (full unload/load
  orchestration + SSE events), the tuner uses `engine.stream_generate`
  directly and trusts the engine-reported `generation_tps` — the same
  metric benchmark.py prefers.
- **Not done (spike cuts)**: lazy tune on first `"auto"` load (explicit
  endpoint only — a surprise multi-minute benchmark inside a load path
  seemed hostile), dashboard UI, staleness/invalidations of tune results
  when checkpoints change.
- **Tests** (`tests/test_mtp_tune.py`): store round-trip incl. depth 0 and
  corrupt file, argmax + stamp restore on a fake engine, depth-0 winner,
  hook-less depth clamp, mtp-disabled rejection, `"auto"` validation, and
  end-to-end load-time resolution via `OMLX_BASE_PATH` (tuned, depth-0,
  untuned).

Real-model validation of the tuner (and the Phase 1 depth benchmark it
automates) still needs the usual rig — same hardware blocker as step 4.
