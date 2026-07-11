# SPDX-License-Identifier: Apache-2.0
"""Per-model optimization recommendations (Model Optimization Advisor P1).

A pure rule engine over signals the server already computes elsewhere:
feature compatibility probes, model settings, the quantized-MTP-head
detector, and the MTP depth-tune store. The endpoint in ``routes.py``
gathers those inputs into a :class:`RecommendationContext`; this module
only decides what to say. See
``docs/experimental/model_optimization_advisor_plan.md``.

Recommendation dict shape (consumed by the settings-modal panel):

    {
      "id": "mtp-enable",
      "severity": "info" | "suggest" | "warn",
      "title": str,       # short label (English; UI may map id -> i18n)
      "detail": str,      # one-paragraph explanation
      "action": {"type": "settings", "payload": {...}}   # single-key PUT
                | {"type": "mtp_tune"}                    # run tune endpoint
                | None                                    # advice only
      "measured": {...} | absent   # P2: tune-store numbers backing the
                                   # advice (tps_by_depth, gain_pct, ...)
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SEVERITY_ORDER = {"warn": 0, "suggest": 1, "info": 2}

# Phase 6: rough benefit class for rules with no measured tune-store number
# to quote yet. Deliberately coarse (no invented percentages) — only rules
# whose action plausibly moves decode speed get a class at all.
_HEURISTIC_GAIN = {
    "mtp-enable": "high",
    "dflash-candidate": "medium",
    "ngram-spec-enable": "medium",
    "sparse-prefill-enable": "medium",
    "chunk-kv-reuse-candidate": "low",
}
_GAIN_CLASS_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass
class RecommendationContext:
    """Inputs for the rule engine, gathered by the endpoint."""

    mtp_compatible: bool = False
    mtp_compatibility_reason: str = ""
    dflash_compatible: bool = False
    mtp_enabled: bool = False
    mtp_draft_depth: Any = 1  # int or "auto"
    dflash_enabled: bool = False
    vlm_mtp_enabled: bool = False
    turboquant_kv_enabled: bool = False
    specprefill_enabled: bool = False
    mtp_head_quantized: bool = False
    # None = never tuned on this machine; 0 = tuned, MTP-off wins.
    mtp_tuned_depth: int | None = None
    # Full tune-store entry ({"depth", "tps_by_depth", "tuned_at"}) when
    # available — lets measured rules quote numbers, not just the winner.
    mtp_tune_entry: dict | None = None
    # P2b: completed A/B trial results for the model's *current* settings
    # (hash-filtered by the endpoint), keyed by rec id. Shape per entry:
    # {"gain_pct", "variants", "trial_at", "settings_hash", ...}.
    ab_trial_results: dict[str, dict] | None = None
    # Integration-branch features (ngram spec, CacheBlend chunk-KV reuse,
    # draft-free sparse prefill) — rules added alongside the feature merge.
    ngram_spec_enabled: bool = False
    chunk_kv_reuse_enabled: bool = False
    sparse_prefill_enabled: bool = False
    # Whether a sparse-prefill calibration file exists for this model on
    # this machine (enabling without one leaves prefill dense).
    sparse_prefill_calibrated: bool = False


def _any_speculative_on(ctx: RecommendationContext) -> bool:
    return (
        ctx.mtp_enabled
        or ctx.dflash_enabled
        or ctx.vlm_mtp_enabled
        or ctx.ngram_spec_enabled
    )


def _tune_measurement(ctx: RecommendationContext) -> dict | None:
    """Distill the tune-store entry into a ``measured`` payload, or None.

    Gain is the winner's tps relative to the depth-0 (MTP-off) baseline;
    omitted when the sweep has no depth-0 sample to compare against.
    """
    entry = ctx.mtp_tune_entry
    if not entry or not isinstance(entry.get("tps_by_depth"), dict):
        return None
    try:
        tps_by_depth = {
            int(k): float(v) for k, v in entry["tps_by_depth"].items()
        }
        winner = int(entry["depth"])
    except Exception:
        return None
    if winner not in tps_by_depth:
        return None
    measured: dict = {
        "winner_depth": winner,
        "winner_tps": round(tps_by_depth[winner], 1),
        "tps_by_depth": {
            str(d): round(t, 1) for d, t in sorted(tps_by_depth.items())
        },
    }
    if entry.get("tuned_at"):
        measured["tuned_at"] = entry["tuned_at"]
    baseline = tps_by_depth.get(0)
    if baseline and baseline > 0:
        measured["baseline_tps"] = round(baseline, 1)
        measured["gain_pct"] = round(
            (tps_by_depth[winner] - baseline) / baseline * 100, 1
        )
    return measured


def _attach_ab_trial_measurement(rec: dict, ctx: RecommendationContext) -> None:
    """P2b: back a rec with its stored A/B trial result, when one exists.

    Tune-store measurements (already on ``measured``) win — they carry the
    richer per-depth table. Only trials with a computable ``gain_pct``
    upgrade the rec; a half-failed trial stays invisible.
    """
    if rec.get("measured"):
        return
    entry = (ctx.ab_trial_results or {}).get(rec["id"])
    if not isinstance(entry, dict) or entry.get("gain_pct") is None:
        return
    measured: dict = {
        "gain_pct": float(entry["gain_pct"]),
        "source": "ab_trial",
    }
    variants = entry.get("variants")
    if isinstance(variants, dict):
        cur = (variants.get("current") or {}).get("tps_median")
        cand = (variants.get("candidate") or {}).get("tps_median")
        if cur is not None:
            measured["baseline_tps"] = cur
        if cand is not None:
            measured["candidate_tps"] = cand
    if entry.get("trial_at"):
        measured["trial_at"] = entry["trial_at"]
    rec["measured"] = measured


def _attach_estimated_gain(rec: dict) -> None:
    """P6: tag ``rec`` with ``estimated_gain`` — measured beats heuristic.

    Measured numbers come from the tune store (already on ``measured.
    gain_pct`` when present). Everything else gets a class from
    ``_HEURISTIC_GAIN``, or no estimate at all (advice-only rules like the
    quantized-head warning, or the tune rule itself before it has run).
    """
    measured = rec.get("measured")
    if measured and "gain_pct" in measured:
        rec["estimated_gain"] = {"measured": measured["gain_pct"]}
        return
    cls = _HEURISTIC_GAIN.get(rec["id"])
    if cls:
        rec["estimated_gain"] = {"class": cls, "basis": "heuristic"}


def _gain_sort_key(rec: dict) -> tuple[int, float]:
    """Lower sorts first: measured (by descending %) > heuristic class > none."""
    gain = rec.get("estimated_gain")
    if not gain:
        return (2, 0.0)
    if "measured" in gain:
        return (0, -float(gain["measured"]))
    return (1, float(_GAIN_CLASS_RANK.get(gain.get("class"), 3)))


def best_estimated_gain(recs: list[dict]) -> dict | None:
    """Pick the single biggest-estimated-gain recommendation (P6 heading).

    Used for the panel's "Biggest estimated gain: ..." summary; returns
    None when nothing carries an estimate. Measured numbers always outrank
    heuristic classes.
    """
    candidates = [r for r in recs if r.get("estimated_gain")]
    if not candidates:
        return None
    best = min(candidates, key=_gain_sort_key)
    return {
        "id": best["id"],
        "title": best["title"],
        "estimated_gain": best["estimated_gain"],
    }


def build_recommendations(ctx: RecommendationContext) -> list[dict]:
    """Run the Phase 1 rules; returns warn-first, stable within severity."""
    recs: list[dict] = []

    if (
        ctx.mtp_compatible
        and not ctx.mtp_enabled
        and not ctx.dflash_enabled
        and not ctx.vlm_mtp_enabled
        and not ctx.ngram_spec_enabled
        and not ctx.turboquant_kv_enabled
    ):
        recs.append(
            {
                "id": "mtp-enable",
                "severity": "suggest",
                "title": "Enable native MTP",
                "detail": (
                    "This model ships usable MTP heads but mtp_enabled is "
                    "off. Native MTP speculation typically speeds up decode "
                    "with output identical to standard decoding."
                ),
                "action": {"type": "settings", "payload": {"mtp_enabled": True}},
            }
        )

    if ctx.mtp_enabled and ctx.mtp_head_quantized:
        recs.append(
            {
                "id": "mtp-quantized-head",
                "severity": "warn",
                "title": "MTP head weights are quantized",
                "detail": (
                    "Quantized mtp.* weights are known to collapse "
                    "speculative acceptance (79-85% BF16 vs 5-11% int4). If "
                    "measured acceptance stays below the floor, MTP "
                    "auto-disables for this model. Prefer a checkpoint that "
                    "keeps the MTP head in BF16."
                ),
                "action": None,
            }
        )

    if ctx.mtp_enabled and ctx.mtp_tuned_depth is None:
        recs.append(
            {
                "id": "mtp-tune",
                "severity": "suggest",
                "title": "Tune MTP draft depth for this machine",
                "detail": (
                    "The best draft depth (including MTP-off) depends on "
                    "this machine's compute/bandwidth balance. Run the "
                    "depth tuner once; afterwards mtp_draft_depth \"auto\" "
                    "resolves to the measured winner."
                ),
                "action": {"type": "mtp_tune"},
            }
        )

    if (
        ctx.mtp_enabled
        and ctx.mtp_tuned_depth is not None
        and ctx.mtp_draft_depth != "auto"
    ):
        winner_off = ctx.mtp_tuned_depth == 0
        measured = _tune_measurement(ctx)
        recs.append(
            {
                "id": "mtp-use-auto",
                "severity": "warn" if winner_off else "suggest",
                "title": (
                    "MTP measured net-negative on this machine"
                    if winner_off
                    else "Use the tuned MTP draft depth"
                ),
                "detail": (
                    (
                        "The depth tuner measured plain autoregressive decode "
                        "as fastest on this machine (winner: depth 0). Set "
                        'mtp_draft_depth to "auto" so MTP turns itself off '
                        "here while staying available on stronger machines."
                    )
                    if winner_off
                    else (
                        f"A tune result exists for this machine (winner: depth "
                        f"{ctx.mtp_tuned_depth}) but mtp_draft_depth is pinned "
                        f'to {ctx.mtp_draft_depth!r}. Set it to "auto" to use '
                        "the measured winner."
                    )
                ),
                "action": {
                    "type": "settings",
                    "payload": {"mtp_draft_depth": "auto"},
                },
                **({"measured": measured} if measured else {}),
            }
        )

    if (
        ctx.mtp_enabled
        and ctx.mtp_draft_depth == "auto"
        and ctx.mtp_tuned_depth is not None
        and ctx.mtp_tuned_depth > 0
    ):
        measured = _tune_measurement(ctx)
        if measured is not None and "gain_pct" in measured:
            recs.append(
                {
                    "id": "mtp-tuned-optimal",
                    "severity": "info",
                    "title": (
                        f"MTP tuned: measured {measured['gain_pct']:+.1f}% "
                        f"at depth {measured['winner_depth']}"
                    ),
                    "detail": (
                        f"The depth tuner measured "
                        f"{measured['winner_tps']:.1f} tok/s at depth "
                        f"{measured['winner_depth']} vs "
                        f"{measured['baseline_tps']:.1f} tok/s without MTP "
                        f"on this machine ({measured['gain_pct']:+.1f}%). "
                        'mtp_draft_depth is "auto", so this depth is already '
                        "in use. Re-run the tuner if the hardware or "
                        "checkpoint changes."
                    ),
                    "action": None,
                    "measured": measured,
                }
            )

    if (
        ctx.dflash_compatible
        and not ctx.mtp_compatible
        and not _any_speculative_on(ctx)
    ):
        recs.append(
            {
                "id": "dflash-candidate",
                "severity": "info",
                "title": "DFlash speculative decoding available",
                "detail": (
                    "This model has no MTP heads but is DFlash-compatible. "
                    "With a suitable draft model configured, DFlash can "
                    "speed up decode substantially. Pick a draft model in "
                    "the DFlash section below to try it."
                ),
                "action": None,
            }
        )

    # N-gram / prompt-lookup speculation: draft-model-free, one click. Only
    # offered when native MTP isn't an option for this checkpoint (MTP wins
    # when available) and no other speculative path is on — the paths are
    # mutually exclusive in settings validation.
    if not ctx.mtp_compatible and not _any_speculative_on(ctx):
        recs.append(
            {
                "id": "ngram-spec-enable",
                "severity": "suggest",
                "title": "Enable n-gram speculative decoding",
                "detail": (
                    "This checkpoint has no usable MTP heads, but n-gram / "
                    "prompt-lookup speculation needs no draft model: drafts "
                    "come from the request's own tokens and are verified by "
                    "the model, so output is unchanged. Strongest on "
                    "echo-heavy workloads (summarization, code edits, RAG); "
                    "roughly neutral on freeform prose."
                ),
                "action": {
                    "type": "settings",
                    "payload": {"ngram_spec_enabled": True},
                },
            }
        )

    # Draft-free sparse prefill: needs an offline calibration file — enabling
    # it without one leaves prefill dense, so the rule only offers the flag
    # once calibration exists, and otherwise explains how to produce it.
    if not ctx.sparse_prefill_enabled and not ctx.specprefill_enabled:
        if ctx.sparse_prefill_calibrated:
            recs.append(
                {
                    "id": "sparse-prefill-enable",
                    "severity": "suggest",
                    "title": "Enable sparse prefill (calibration found)",
                    "detail": (
                        "A sparse-prefill calibration file exists for this "
                        "model on this machine. Enabling it applies "
                        "calibrated per-head sparse attention to long "
                        "prefills (≥8K tokens by default), cutting "
                        "time-to-first-token without dropping tokens; decode "
                        "is untouched."
                    ),
                    "action": {
                        "type": "settings",
                        "payload": {"sparse_prefill_enabled": True},
                    },
                }
            )
        else:
            recs.append(
                {
                    "id": "sparse-prefill-calibrate",
                    "severity": "info",
                    "title": "Sparse prefill available after calibration",
                    "detail": (
                        "Draft-free sparse prefill can cut long-prompt "
                        "time-to-first-token, but this model has no "
                        "calibration file on this machine yet. Run "
                        "`python -m omlx.sparse_calibration --model "
                        "<model>` once, then enable "
                        "sparse_prefill_enabled."
                    ),
                    "action": None,
                }
            )

    # CacheBlend chunk-KV reuse: prefill-side, workload-dependent (helps
    # RAG/agent prompts that repeat content at shifted positions), and
    # incompatible with DFlash and TurboQuant KV.
    if (
        not ctx.chunk_kv_reuse_enabled
        and not ctx.dflash_enabled
        and not ctx.turboquant_kv_enabled
    ):
        recs.append(
            {
                "id": "chunk-kv-reuse-candidate",
                "severity": "info",
                "title": "Chunk KV reuse for repeated-content prompts",
                "detail": (
                    "CacheBlend-style chunk KV reuse (experimental) reuses "
                    "precomputed KV for prompt chunks even when they move "
                    "position, cutting prefill on RAG and agent-loop "
                    "workloads that repeat content at shifted offsets. "
                    "Neutral for prompts without repeated chunks."
                ),
                "action": {
                    "type": "settings",
                    "payload": {"chunk_kv_reuse_enabled": True},
                },
            }
        )

    for rec in recs:
        _attach_ab_trial_measurement(rec, ctx)
        _attach_estimated_gain(rec)

    recs.sort(
        key=lambda r: (_SEVERITY_ORDER.get(r["severity"], 9), _gain_sort_key(r))
    )
    return recs


def collect_settings_payload(recs: list[dict]) -> tuple[dict, list[str]]:
    """Merge every settings-action payload into one dict (P3 apply-all).

    Returns ``(payload, rec_ids)``. Rules recommend disjoint keys by
    construction (mutually exclusive conditions), so a plain merge in
    severity order is safe; a later duplicate key would win.
    """
    payload: dict = {}
    ids: list[str] = []
    for rec in recs:
        action = rec.get("action") or {}
        if action.get("type") != "settings":
            continue
        payload.update(action.get("payload") or {})
        ids.append(rec["id"])
    return payload, ids
