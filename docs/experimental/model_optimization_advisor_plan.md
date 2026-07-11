# Model Optimization Advisor: per-model recommendations in the admin UI

Date: 2026-07-10
Branch: `feat/mtp-multi-depth`
Status: **Phase 1 implemented** (2026-07-10); **Phase 2a (measured
tune-store recommendations) and Phase 3 (apply-as-profile) implemented**
(2026-07-11) — see implementation notes at the end. Phase 2b (A/B trials
+ SSE) not started. **Phase 4 (capability matrix) implemented**
(2026-07-11); Phases 5–6 proposed (gap review 2026-07-11).
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

**2b (not started):** DFlash A/B trials (perf_bench-style, driven through
the engine like `mtp_tune.py`) so "suggest" items can be upgraded to
measured claims. Requires a background-run pattern (SSE progress like the
benchmark tab) once trials take minutes.

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
