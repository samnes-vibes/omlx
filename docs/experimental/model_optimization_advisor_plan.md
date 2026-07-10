# Model Optimization Advisor: per-model recommendations in the admin UI

Date: 2026-07-10
Branch: `feat/mtp-multi-depth`
Status: **Phase 1 implemented** (2026-07-10) — see implementation notes at
the end. Phases 2–3 not started.
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

### Phase 2 — measured recommendations (not started)

Extend the advisor with numbers: after an `mtp-tune` run, include the
depth×tps table in the recommendation detail; add ngram-spec / DFlash A/B
trials (perf_bench-style, driven through the engine like `mtp_tune.py`)
so "suggest" items can be upgraded to "measured +N%" claims. Requires a
background-run pattern (SSE progress like the benchmark tab) once trials
take minutes.

### Phase 3 — apply-as-profile (not started)

"Apply all" writes the recommended settings as a profile
(`optimized-<date>`) via the existing profiles API instead of mutating
base settings, making the whole change auditable and revertable in one
click.

### Out of scope

- Model-download recommendations (already exist, different feature).
- Cross-model advice ("use this other checkpoint instead") — except the
  factual quantized-MTP-head note.
- Auto-applying anything without a click.

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
