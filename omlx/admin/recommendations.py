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
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SEVERITY_ORDER = {"warn": 0, "suggest": 1, "info": 2}


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


def _any_speculative_on(ctx: RecommendationContext) -> bool:
    return ctx.mtp_enabled or ctx.dflash_enabled or ctx.vlm_mtp_enabled


def build_recommendations(ctx: RecommendationContext) -> list[dict]:
    """Run the Phase 1 rules; returns warn-first, stable within severity."""
    recs: list[dict] = []

    if (
        ctx.mtp_compatible
        and not ctx.mtp_enabled
        and not ctx.dflash_enabled
        and not ctx.vlm_mtp_enabled
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

    recs.sort(key=lambda r: _SEVERITY_ORDER.get(r["severity"], 9))
    return recs
