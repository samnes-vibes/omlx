# Model Optimization Advisor: per-model recommendations in the admin UI

Date: 2026-07-10
Branch: `feat/mtp-multi-depth`
Status: **Phase 1 implemented** (2026-07-10); **Phase 2a (measured
tune-store recommendations) and Phase 3 (apply-as-profile) implemented**
(2026-07-11) — see implementation notes at the end. **Phase 4 (capability
matrix) implemented** (2026-07-11); **Phase 5 (live MTP automation
status) implemented** (2026-07-11); **Phase 6 (benefit ordering)
implemented** (2026-07-11). **Phase 2b (generic A/B trial engine) and
Phase 7 (guided wizard flow) implemented** (2026-07-11) — see the design
sections below and the implementation notes at the end.
Companion to: [mtplx_adoption_plan.md](mtplx_adoption_plan.md) (the MTP
auto-tune endpoint this surfaces), [5x_speedup_research.md](5x_speedup_research.md)

## Problem

The dashboard knows a lot about what each model *could* do — per-feature
compatibility classification (`mtp_compatible`, `dflash_compatible`, with
reasons), mutual-exclusion rules, quantized-MTP-head detection, and now a
per-machine MTP depth tuner — but all of it is passive. The settings modal
greys out impossible toggles and the benchmark tab prints numbers, yet
nothing tells the operator *"this model would decode faster if you flipped
X"*, and nothing applies it. Every optimization is discovered by reading
docs or logs.

## Approach

A thin advisor layer over existing signals, not a new analysis engine:

1. **Rule engine** (`omlx/admin/recommendations.py`): a pure function from
   (model info, model settings, compat probes, tune-store state) to a list
   of recommendation dicts. Deterministic, no I/O of its own — inputs are
   gathered by the endpoint, so rules are unit-testable in isolation.
2. **Endpoint**: `GET /admin/api/models/{model_id}/recommendations`
   collects the inputs (reusing `_mtp_compat_for_model`,
   `_dflash_compat_for_model`, `_mtp_weights_quantized`,
   `load_tuned_depth`) and runs the rules.
3. **UI**: a "Recommendations" panel at the top of the settings modal's
   Experimental section. Loads when the modal opens; each item has a
   severity badge, an explanation, and — when the fix is a settings
   change — an **Apply** button that PUTs just that key (the settings
   endpoint only applies explicitly-sent fields, so single-key PUTs are
   safe). The MTP tune recommendation gets a **Run tune** button that
   drives `POST /admin/api/models/{id}/mtp-tune` and re-loads the list.

### Recommendation shape

```json
{
  "id": "mtp-enable",
  "severity": "suggest",        // "info" | "suggest" | "warn"
  "title": "...",               // short, i18n-keyed in the UI by id
  "detail": "...",              // backend-composed explanation (English)
  "action": {"type": "settings", "payload": {"mtp_enabled": true}}
  // or {"type": "mtp_tune"} — UI renders the tune button
  // or null — advice only (e.g. quantized MTP head)
}
```

### Phase 1 rules (static + tune-store, no measurement)

| id | condition | severity | action |
|---|---|---|---|
| `mtp-enable` | MTP-compatible, `mtp_enabled` off, no conflicting speculative path on | suggest | set `mtp_enabled: true` |
| `mtp-quantized-head` | MTP on, `mtp.*` weights quantized | warn | none (advice: BF16 checkpoint; acceptance-floor guard will auto-disable) |
| `mtp-tune` | MTP on, no tune result for this (model, machine) | suggest | run tune endpoint |
| `mtp-use-auto` | MTP on, tuned winner exists, `mtp_draft_depth` ≠ `"auto"` | suggest (warn when winner is 0 — MTP is net-negative here) | set `mtp_draft_depth: "auto"` |
| `dflash-candidate` | DFlash-compatible, nothing speculative on, not MTP-compatible | info | none (needs a draft model choice — link only) |

(An `ngram-spec` rule was sketched but dropped: this codebase has no
`ngram_spec_enabled` setting — that flag exists only in an unrelated
installed config on the dev machine.)

Ordering: warn > suggest > info; stable within severity. Rules never
recommend anything the settings validation would reject (mutual
exclusions are part of the conditions).

### Phase 2 — measured recommendations

**2a (implemented 2026-07-11):** recommendations carry the tune-store
numbers. `load_tune_entry()` exposes the full per-(model, machine) entry;
the rule engine distills it into an optional `measured` field
(`tps_by_depth`, `winner_depth`/`winner_tps`, `baseline_tps`, `gain_pct`
vs depth 0, `tuned_at`) attached to `mtp-use-auto`, plus a new
`mtp-tuned-optimal` info rule ("measured +N% at depth d") when depth is
already `"auto"` and the tuned winner is > 0. The UI renders the
depth×tps row (winner bolded) under any recommendation with `measured`.

**2b (implemented 2026-07-11, design below):** generic A/B trials — not DFlash-only —
so any "suggest" recommendation can be upgraded to a measured claim before
the operator applies it. Requires a background-run pattern (SSE progress
like the benchmark tab) once trials take minutes. See detailed design
under [Phase 2b](#phase-2b--generic-ab-trial-engine-detailed-design-proposed).

### Phase 3 — apply-as-profile (implemented 2026-07-11)

"Apply all" writes the recommended settings as a profile
(`optimized-<date>`, suffixed `-2`, `-3`… on collision) via the existing
profiles machinery instead of mutating base settings, making the whole
change auditable and revertable in one click.
`POST /admin/api/models/{id}/recommendations/apply-all` re-runs the rule
engine server-side, merges settings-action payloads
(`collect_settings_payload`), saves the profile (description lists the
applied rule ids) and applies it — so `active_profile_name` is stamped
and the profiles UI shows the change. The UI button appears in the
recommendations panel header whenever at least one settings-action
recommendation is present; 400 when nothing is actionable.

### Out of scope

- Model-download recommendations (already exist, different feature).
- Cross-model advice ("use this other checkpoint instead") — except the
  factual quantized-MTP-head note.
- Auto-applying anything without a click.

## Gap review (2026-07-11) — proposed Phases 4–6

The review revealed two gap categories: (a) the advisor only covers MTP
(+ a DFlash reference), even though main already has half a dozen other
optimization settings; (b) the advisor tells you *what* to enable, but
doesn't explain *why something isn't possible* and doesn't guide model
selection — for MTP specifically it's unclear what's automatic and what
requires a specific checkpoint.

### Coverage gaps (rules missing for existing settings)

| Setting (main) | Missing rule | Note |
|---|---|---|
| `turboquant_kv_enabled/bits` | Recommend KV quantization when the model is large relative to RAM or contexts are long; the fused int4 verify path benefits directly | metric data: memory monitor + request stats |
| `dflash_max_ctx` | **Warn**: DFlash enabled but traffic exceeds the threshold → the whole speedup is lost to fallback | from request stats; link to the dflash2-long-context plan |
| `dflash_draft_quant_*` | Recommend an 8-bit draft once measured α-neutrality exists | depends on [mirror_sd_adoption_plan.md](mirror_sd_adoption_plan.md) P2 |
| `dflash_verify_mode` | adaptive vs. ddtree recommendation based on measured data | Phase 2b-style A/B |
| `specprefill_enabled` | For a MoE model with long prompts: recommend SpecPrefill (+ draft-model selection guidance) | MoE detection from config |
| `vlm_mtp_enabled` | No rule at all right now. Gemma4-VLM / Qwen3.5-VLM → recommend + state which drafter checkpoint is needed | this is the worst "how would I know" gap |

Branch-only features (ngram, CacheBlend, sparse prefill draft-free)
stay out of the advisor until merged — the rule gets added in the same
PR as the feature flag (new working rule).

### Phase 4 — capability matrix: "what can this model do and why not" (proposed)

Problem: MTP compatibility is two-part — (1) the config declares MTP heads
(`mtp_num_hidden_layers` / `num_nextn_predict_layers`,
[model_loading.py:496](omlx/utils/model_loading.py#L496)) and (2) model_type
∈ {qwen3_5\*, qwen3_6\*, deepseek_v4\*}
([model_loading.py:606](omlx/utils/model_loading.py#L606)) — but the UI just
grays out the toggle. The operator can't see *which* condition failed or
what to do about it.

Implementation:

1. **Endpoint** `GET /admin/api/models/{id}/capabilities`: for each
   feature (MTP, DFlash, VLM-MTP, SpecPrefill, TurboQuant KV) a
   status enum + explanation:
   - `active` — enabled and working
   - `available` — can be enabled from here (an advisor rule drives the recommendation)
   - `needs-config` — possible but requires a choice (e.g. a DFlash draft
     or VLM drafter path)
   - `needs-different-checkpoint` — the architecture would support it, this
     checkpoint doesn't (e.g. quantization stripped the `mtp.*` weights, or
     there's no BF16 head) — the explanation states *what to load*: known
     pairs from the table below
   - `unsupported` — the architecture doesn't support it; state the reason
     (model_type not on the list / no MTP heads in config)
   The compat probes (`_mtp_compat_for_model`, `_dflash_compat_for_model`,
   `_mtp_weights_quantized`) already return (bool, reason) — the work is
   mostly structuring the reasons into an enum + breaking out the missing
   conditions.
2. **Checkpoint catalog** (`omlx/admin/feature_catalog.py`): static
   data on known pairs — DFlash target→draft (z-lab collection:
   Qwen3/3.5 4B–27B, LLaMA-3.1-8B), VLM-MTP target→drafter
   (Gemma4-VLM→`gemma-4-…-assistant`, Qwen3.5/3.6-VLM→`qwen3_5_mtp`),
   MTP-eligible architectures + a note on BF16-head variants.
   The `needs-config`/`needs-different-checkpoint` statuses link here.
3. **UI**: a "Capabilities" row per feature in the settings modal
   (status badge + explanation + catalog link) instead of a tooltip
   on the grayed-out toggle.

### Phase 5 — visibility into MTP's automation (proposed)

What's automatic and what's manual is currently only readable from the
code/docs. Document and surface it:

| Part | Automatic? |
|---|---|
| `mtp_enabled` | **Manual** (advisor recommends, Apply button) |
| Draft depth | Automatic when `mtp_draft_depth="auto"` + tune has run; otherwise manual |
| Running tune | Manual (Run tune button) — not run automatically |
| Quantized-head guard | Automatic (prevents a bad path) |
| Acceptance-floor auto-disable | Automatic (disables MTP mid-run if α collapses) |
| Depth clamp for unsupported models | Automatic (clamps to 1) |

Implementation: (1) this table at the README/doc level; (2) **live status**
in the capabilities endpoint: when MTP is `active`, show the effective
state — depth in use, tuned winner, and in particular *"auto-disabled: α
below floor at …"* if the guard has tripped (currently only visible in
logs). Runtime-state source: the engine exposes its latest MTP state into
the stats structure, which the endpoint reads.

### Phase 6 — benefit ordering: "biggest win first" (proposed)

The current ordering is severity-based. Add an estimated benefit:

1. **Measured when available**: the tune store's `gain_pct` (Phase 2a) and
   future A/B results (Phase 2b, mirror-sd P2) → `estimated_gain: {"measured": N}`.
2. **Heuristic when not**: a rough class per rule from context data —
   e.g. MoE + long prompts → SpecPrefill "high"; dense + short prompts
   → MTP "high", KV-quant "low". Clearly marked as an estimate
   (`estimated_gain: {"class": "high", "basis": "heuristic"}`).
3. UI orders: measured > heuristic-high > … ; panel heading shows
   "Biggest estimated gain: X (+N% measured)".

Still a non-goal: nothing gets enabled without a click; the heuristic must
not present numbers, only classes — numbers only from measured data (same
principle as Phase 2a).

Proposed order: **Phase 4 first** (removes the biggest source of confusion
and produces the structures 5–6 use), then 5 (cheap, mostly surfacing
existing state), then 6 (depends on measurement data accumulating).

## Phase 2b — generic A/B trial engine: detailed design (proposed)

### Problem

Two measurement mechanisms exist today and don't talk to each other:

- `omlx/admin/mtp_tune.py` + `POST /admin/api/models/{id}/mtp-tune`:
  **synchronous** — the request blocks for the whole round-robin sweep
  (depths 0..4, 2 repeats, 128 tokens/trial ⇒ tens of seconds). Good
  measurement discipline (round-robin against thermal drift, restores the
  model's stamps in a `finally`), but no progress reporting and hard-coded
  to one axis (MTP draft depth).
- `omlx/admin/benchmark.py` + `POST /api/bench/start` /
  `GET /api/bench/{id}/stream`: **background task + SSE**
  (`BenchmarkRun` dataclass, append-only `events` log, `asyncio.Condition`
  for replay-then-wait subscribers, single-run guard via
  `get_active_run()`). Good delivery model, but only measures raw
  prompt-length/batch-size throughput — it has no concept of "setting A
  vs. setting B".

Neither can answer "would enabling DFlash actually help *this* model on
*this* machine, before I commit to it" — the recommendations panel can
only assert `mtp-use-auto`/`mtp-tuned-optimal` claims today because those
piggyback on the MTP tuner; every other "suggest" rule (`mtp-enable`,
`dflash-candidate`, and the Phase-4-gap rules for `specprefill_enabled`,
`turboquant_kv_enabled`, `dflash_draft_quant_*`, `dflash_verify_mode`) is
heuristic-only forever unless something can run the comparison.

### Approach

Take `mtp_tune.py`'s measurement discipline (round-robin, restore original
state in `finally`, warmup trial, median-of-repeats, `tps_by_depth`-style
data) and `benchmark.py`'s delivery model (background task, `Run`
dataclass with SSE `events`, single-run guard), and generalize the *axis*
being swept from "MTP draft depth" to "arbitrary settings dict".

1. **`omlx/admin/ab_trial.py`** (new): `ABTrialRun` dataclass, same shape
   as `BenchmarkRun` (`trial_id`, `status`, `events`, `cond`, `terminal`,
   `task`) plus:
   ```python
   variants: list[dict]   # e.g. [{"label": "current", "settings": {...}},
                           #       {"label": "candidate", "settings": {...}}]
   prompt: str = _DEFAULT_PROMPT   # reuse mtp_tune._PROMPT by default
   repeats: int = 2
   max_tokens: int = 128
   results: dict[str, list[float]] = {}   # label -> tps samples
   ```
   `run_ab_trial(run, engine_pool)`: for each repeat, round-robin over
   `variants`, apply that variant's settings to the *live* engine instance
   the same way `mtp_tune._set_trial_depth` stamps per-instance markers
   for MTP — for settings that aren't stampable on a live instance (e.g.
   `dflash_enabled`, `turboquant_kv_enabled`, which are read at model-load
   time, not per-request), fall back to **reloading the model** between
   variants via the existing settings-apply + engine-pool reload path.
   This makes trials involving those settings much more expensive (model
   reload, ~seconds–tens of seconds each) than a pure MTP-depth sweep —
   the UI must show this cost estimate before the operator confirms (see
   Phase 7 below), and the endpoint should refuse >2 variants when any
   settings key requires a reload, to keep worst-case bounded.
   Emits `progress` events per (repeat, variant) like the benchmark tab,
   a final `result` event with `{variant_label: {tps_median, samples}}`
   plus `gain_pct` between the two variants, and restores original
   settings/stamps in `finally` regardless of outcome.
2. **Endpoint** `POST /admin/api/models/{id}/recommendations/{rec_id}/ab-trial`:
   looks up the recommendation by id (re-runs `_build_recommendations_for`
   server-side so the payload can't be spoofed), builds `variants` as
   `[{"label": "current", "settings": <settings.model_dump()>},
     {"label": "candidate", "settings": <current + collect_settings_payload([rec])>}]`,
   creates an `ABTrialRun`, starts it as an `asyncio.create_task` (mirrors
   `start_benchmark`), returns `{"trial_id": ...}`. Rejects 409 if another
   trial *or* a throughput benchmark is running on this model (shared
   `get_active_run()`-style guard — both stress the same engine instance).
3. **Stream** `GET /admin/api/models/{id}/ab-trial/{trial_id}/stream`:
   same SSE shape as `/bench/{id}/stream` (replay `events` from offset 0,
   then wait on `cond`), so the frontend can reuse `connectBenchSSE`'s
   parsing logic with a different URL rather than writing a second SSE
   client.
4. **Recommendation payload**: once a trial completes, its `gain_pct`
   is written back onto the recommendation's `measured` field the next
   time `build_recommendations` runs for this model — i.e. persist trial
   results the same way `mtp_tune.py` persists to `mtp_tune.json`, keyed
   by `(model, hardware_id, rec_id, settings_hash)` so a stale trial
   result doesn't silently get reused after the operator changes
   something else. New store: `ab_trials.json`, same
   read/write pattern as `mtp_tune.tune_store_path`.

### Data shape (SSE `result` event)

```json
{
  "type": "result",
  "rec_id": "dflash-candidate",
  "variants": {
    "current":   {"tps_median": 41.2, "samples": [40.8, 41.6]},
    "candidate": {"tps_median": 58.9, "samples": [58.1, 59.7]}
  },
  "gain_pct": 43.0,
  "reload_required": true,
  "elapsed_s": 34.1
}
```

### Out of scope (2b)

- Trials that need a *choice* first (DFlash draft model, VLM drafter
  checkpoint) — the operator must resolve `needs-config` (Phase 4) before
  a candidate settings dict even exists to trial.
- Multi-variant sweeps beyond current-vs-candidate (MTP depth sweeps stay
  on the existing `mtp-tune` endpoint; this is specifically for the
  advisor's binary "should I flip this?" case).
- Accuracy/quality regression checks — this measures throughput only,
  same as `mtp_tune.py` and the benchmark tab today.

## Phase 7 — guided wizard flow (proposed)

### Problem

The settings modal today is flat and toggle-first: Profiles row →
Basic settings → Advanced settings → Experimental section, where the
Experimental section itself stacks three independent information sources
(capability matrix, recommendations panel, then five separate toggle
cards for TurboQuant KV / IndexCache / SpecPrefill / DFlash / MTP / VLM
MTP). An operator who wants "just tell me what to turn on" has to read
the capability badge, cross-reference the recommendation card, then
scroll to the matching toggle card to actually apply anything that isn't
a single-key settings PUT (e.g. picking a DFlash draft model). Nothing
walks them through it in order, and nothing offers to *prove* a
recommendation before committing to it.

### Approach

Not a replacement UI — an **optional guided rail overlaid on the existing
recommendations panel**, so the toggle cards keep working for manual
fine-tuning and the wizard doesn't become a second UI to maintain
alongside them (this is a third-party-maintained fork per `CLAUDE.md`;
minimize surface that diverges from upstream's modal structure).

1. **Entry point**: a "Guide me" button next to the existing
   "Apply all" button in the recommendations panel header (only shown
   when `modelRecs.length > 0`). Opens a step sequence *within* the same
   modal (Alpine `x-show` step index), not a separate modal, reusing
   `modelRecs` — no new data fetch.
2. **Step sequence** (one recommendation at a time, in the existing
   severity → estimated-gain order from Phase 6 — so the wizard visits
   `warn` items first, and within a tier the biggest `estimated_gain`
   first):
   - **Explain**: title + detail (already have this) + capability-matrix
     cross-reference when the rec's feature is `needs-config` (link to
     the relevant toggle card instead of a bare Apply button — Phase 4's
     catalog data answers "what do I need to load/pick" right here
     instead of the operator discovering it three cards down).
   - **Optional compare**: when the rec's action is a `settings` PUT *and*
     the model isn't already mid-benchmark, show a secondary "Compare
     before/after (~30s)" button that calls the Phase 2b `ab-trial`
     endpoint and streams progress inline (reuse `connectBenchSSE`'s SSE
     parsing against the new stream URL). Skippable — default path is
     still direct Apply, matching the existing one-click principle
     (Phase 3's non-goal: nothing applies without an explicit click, and
     the *comparison* is opt-in on top of that, not a forced gate).
   - **Decide**: Apply (single-key PUT, same call as today) / Skip / Undo
     if a trial's candidate variant was left active (trials should leave
     the model on the "current" variant when they finish per Phase 2b's
     `finally`-restore — Undo here is a safety net, not the primary path).
   - Advance to the next recommendation; **Finish** step shows a summary
     (applied / skipped / measured-gain list) and offers "Apply all
     remaining" (existing endpoint) as an escape hatch for anyone who
     stops reading partway through.
3. **State**: purely client-side (`wizardStep`, `wizardRecIndex`) — no new
   backend state beyond what Phase 2b introduces; `modelRecs` is reloaded
   after each Apply exactly like `applyRecommendation()` does today, so
   the wizard and the flat panel never disagree about what's left.
4. **First-run nudge (optional, cheap)**: when a model's settings modal
   is opened for the first time and `modelRecs` contains at least one
   `warn`/`suggest`, auto-suggest the "Guide me" button with a one-time
   highlight (localStorage flag per `model_id`, not a server-side
   first-run concept) instead of a forced wizard — respects the
   non-goal that nothing happens without a click.

### Sequencing dependency

Phase 7's "optional compare" step is inert without Phase 2b (the button
would have nothing to call), so **2b ships first**; Phase 7 is UI-only
once the trial endpoint exists, and degrades gracefully (compare button
hidden) if 2b isn't done yet — i.e. Phase 7 can start once Phase 4 exists
(for the "needs-config" cross-reference) even before 2b lands, with the
compare step arriving as a follow-up.

### Out of scope (7)

- A separate onboarding modal for *new installs* (this is per-model,
  triggered from the existing settings modal, not a first-launch tour).
- Reordering or removing the flat toggle cards — they stay as the manual
  fine-tuning path.
- Wizard support for recommendations with no `action` (advice-only rows
  like `mtp-quantized-head`) beyond just displaying them in the Explain
  step — there's nothing to Apply or Compare there.

## Phase 2b + Phase 7 implementation notes (2026-07-11)

Implemented per the designs above, with these deviations/decisions:

- **`omlx/admin/ab_trial.py`**: as designed (`ABTrialRun`, round-robin
  runner, `ab_trials.json` store keyed by (model, hardware, rec_id,
  settings_hash)). The stampable set is just `mtp_draft_depth` (and only
  while `mtp_enabled` is already on — the head must be attached); every
  other key goes through the engine pool's **existing transient
  `runtime_settings` variant mechanism** (`get_engine(...,
  runtime_settings=candidate)`), which reloads per variant switch without
  mutating persisted settings — no new reload plumbing was needed, and the
  `finally` restore is just one more `get_engine()` with persisted
  settings. `"auto"` depth in a variant resolves through
  `load_tuned_depth`. A fresh candidate load gets its own 16-token warmup
  so shader compile doesn't bill the first sample.
- **Endpoints** in `routes.py`: `POST .../recommendations/{rec_id}/ab-trial`
  (re-runs the rule engine server-side; 404 unknown rec, 400 actionless
  rec, 409 when another trial *or* throughput bench is running) and
  `GET .../ab-trial/{trial_id}/stream` (same replay-then-attach SSE loop
  as the bench stream). Trial results feed back via a new
  `RecommendationContext.ab_trial_results` (hash-filtered by the endpoint
  with `load_trial_results`); `_attach_ab_trial_measurement` upgrades a
  rec to `measured: {gain_pct, baseline_tps, candidate_tps, trial_at,
  source: "ab_trial"}` — tune-store measurements win when both exist, and
  the Phase 6 gain pill picks the number up unchanged.
- **Wizard (P7)**, UI-only as designed: "Guide me" button in the
  recommendations panel header opens an inline step card (Alpine state,
  no new fetch). It walks a **snapshot** of `modelRecs` taken at start,
  so the flat panel reloading after each Apply never shifts step order.
  Steps: explain (severity badge + title + gain pill + detail) →
  optional "Compare before/after" (starts the 2b trial, streams progress
  via a second `EventSource`, shows current/candidate tok/s + gain) →
  Apply / Skip. Advice-only rows show a "Next" button; the `mtp_tune`
  action runs the existing tuner. Finish step summarizes
  applied/skipped and offers "Apply all remaining" (existing P3
  endpoint). First-run nudge = amber ring on "Guide me" via a
  per-model localStorage flag. The Undo affordance from the design was
  dropped: trials always restore state in `finally`, so there is nothing
  for the wizard to undo.
- Tests: `tests/test_ab_trial.py` (variant classification, store
  round-trip + hash staleness, stamp-path and reload-path runs, event
  shape, error path) and `TestAbTrialMeasured` in
  `tests/test_model_recommendations.py`. Not verified against a live
  model on this machine — the trial endpoints follow the same engine
  contracts as `mtp_tune.py`/`benchmark.py`, but a real end-to-end run
  is still worth doing when a model is loaded.

## Phase 6 implementation notes (2026-07-11)

- `omlx/admin/recommendations.py`: `_attach_estimated_gain(rec)` tags every
  recommendation with an `estimated_gain` field — `{"measured": pct}` when
  the rec already carries a `measured.gain_pct` (i.e. `mtp-use-auto` /
  `mtp-tuned-optimal`, reusing the Phase 2a numbers, no new measurement
  path), else `{"class": "high"|"medium"|"low", "basis": "heuristic"}` from
  a small per-rule-id table (`mtp-enable` → high, `dflash-candidate` →
  medium). Advice-only/procedural rules (`mtp-quantized-head`, `mtp-tune`)
  get no estimate — there's nothing to quantify yet.
- Ordering stays **severity-first** (unchanged Phase 1 contract, still
  covered by `TestOrderingAndEmpty`); `estimated_gain` only breaks ties
  within a severity tier via `_gain_sort_key` (measured, ranked by
  descending %, before heuristic class, before no-estimate). This matches
  the plan's non-goal: heuristics never invent numbers, they only rank.
- `best_estimated_gain(recs)` picks the single biggest-estimated-gain rec
  for the panel heading; new `GET .../recommendations` field `best_gain`
  (endpoint in `routes.py`) carries `{id, title, estimated_gain}` or
  `None`.
- UI (`_modal_model_settings.html` + `dashboard.js`): panel header shows
  "Biggest estimated gain: `<title>` (`<value>`)" when `best_gain` is
  present; each rec row gets a small pill via `formatEstimatedGain()`
  (`"+30.0% measured"` or `"high (estimate)"`). i18n keys added to
  `en.json` and synced with `normalize_i18n.py` (fallback-by-value-copy,
  same as the rest of the panel).
- Tests: `TestEstimatedGain` in `tests/test_model_recommendations.py`
  (heuristic tagging, measured tagging, no-estimate cases, `best_estimated_
  gain` ranking and empty/None cases).

## Phase 5 implementation notes (2026-07-11)

- `omlx/patches/mlx_lm_mtp/batch_generator.py`: `_maybe_disable_low_acceptance_mtp`
  now stamps `_omlx_mtp_auto_disabled_reason` / `_omlx_mtp_auto_disabled_at`
  on the model instance (alongside the existing `_omlx_mtp_decode_enabled =
  False` flip) instead of leaving the trip only visible in the log line.
  New public accessor `mtp_runtime_status(model)` reads the per-instance
  markers (`_model_mtp_decode_enabled`, `_mtp_draft_depth` — the latter
  already applies the depth-1 clamp for models without
  `mtp_forward_hidden`) and returns `None` when the model was never
  MTP-stamped, else `{decode_active, effective_depth, auto_disabled,
  auto_disabled_reason, auto_disabled_at}`. `auto_disabled` requires both
  a stamped reason *and* `decode_active is False` — the guard is sticky
  for the life of the loaded instance, so this can't disagree with itself.
- `omlx/admin/capabilities.py`: `CapabilityContext` gained `mtp_runtime`
  (the dict above, or `None` when the model isn't loaded),
  `mtp_tuned_winner_depth`, `mtp_tuned_at`. `_mtp_live_block(ctx)` shapes
  these into a `live` sub-dict attached to the mtp capability whenever
  `mtp_enabled` is true (both the plain "active" branch and the
  quantized-head branch, since a quantized head staying "active" is
  exactly the case most likely to trip the guard).
- `omlx/admin/routes.py` (`_build_capabilities_for`): when settings have
  `mtp_enabled`, reads `entry.engine._model` (only present if the model
  is actually loaded — untouched otherwise, so an unloaded model reports
  `live.loaded: false` rather than erroring) and calls
  `mtp_runtime_status`; separately reads `load_tune_entry` (already used
  by the recommendations endpoint) for the tuned winner + timestamp,
  independent of whether the model is currently loaded.
- UI: capability card for MTP renders effective depth + tuned winner
  inline, and an amber "Auto-disabled: <reason>" line when the guard has
  tripped — the reason string already carries the quantized-head hint
  when applicable, so no separate UI branch is needed for that.
- Tests: `TestMtpLiveStatus` in `tests/test_model_capabilities.py` (data
  shaping); `TestMtpRuntimeStatus` + a new stamp-on-trip case in
  `tests/test_mlx_lm_mtp_patch.py` (guard + accessor).
- Deliberate scope: only native MTP (`batch_generator.py`'s guard). VLM
  MTP has its own acceptance logging (`scheduler.py::_log_vlm_mtp_stats`)
  but no auto-disable guard today, so there's no live state to surface
  there yet — out of scope for this pass.

## Phase 4 implementation notes (2026-07-11)

- `omlx/admin/capabilities.py`: pure `build_capabilities(ctx)` over a
  structured `CapabilityContext` (model facts + settings) → one dict per
  feature (mtp, dflash, vlm_mtp, specprefill, turboquant_kv) with the
  five-state status enum, reason, and optional catalog hints.
- `omlx/admin/feature_catalog.py`: static known-pairs data (MTP-capable
  architectures, DFlash target→draft, VLM-MTP drafters) + lookup helpers.
- Endpoint `GET /admin/api/models/{id}/capabilities` in `routes.py`
  (`_build_capabilities_for`): parses config.json once (model_type,
  vision_config, MoE via `_config_is_moe`), breaks the MTP probe into its
  individual conditions (`_has_mtp_heads` → whitelist → weight tensors →
  quantized head) so the engine can distinguish
  `needs-different-checkpoint` from `unsupported`.
- UI: "Capabilities" panel above the recommendations panel in the
  settings modal (status badge color-coded per state, reason text,
  catalog bullet list); loaded on modal open alongside recommendations.
  i18n keys added and synced with `normalize_i18n.py`.
- Deliberate nuances: quantized MTP head reports
  `needs-different-checkpoint` when MTP is off but stays `active` (with
  the BF16 warning in the reason) when already on; paroquant overrides
  both MTP and DFlash to `unsupported`; TurboQuant notes the MTP mutex.
- Tests: `tests/test_model_capabilities.py` (matrix shape + per-feature
  status transitions, 21 cases).

## Phase 1 implementation notes (2026-07-10)

- `omlx/admin/recommendations.py`: `build_recommendations(ctx)` pure rule
  engine over a small `RecommendationContext` dataclass; endpoint
  `GET /admin/api/models/{model_id}/recommendations` in `routes.py`
  gathers compat probes + settings + `load_tuned_depth` and returns
  `{"recommendations": [...]}`.
- UI (`_modal_model_settings.html` + `dashboard.js`): recommendations
  panel at the top of the Experimental section; fetched on modal open
  (non-diffusion models only). Apply = single-key PUT to the settings
  endpoint + models refresh + recommendations reload. `mtp_tune` action
  renders a Run-tune button with a running state; the tps-by-depth table
  from the tune response is shown inline once finished.
- i18n: severity labels + panel strings added to `en.json` and synced to
  the other locales with `scripts/normalize_i18n.py` (untranslated keys
  fall back to the English string by value-copy, matching repo practice).
- Tests (`tests/test_model_recommendations.py`): one test per rule firing
  and per suppression condition, ordering, and the no-recommendations
  case.
