# SPDX-License-Identifier: Apache-2.0
"""Per-model feature capability matrix (Model Optimization Advisor P4).

A pure function from structured model facts to a list of capability
dicts — one per optimization feature — answering "what can this model do
and why not", instead of a grayed-out toggle. The endpoint in
``routes.py`` gathers the inputs into a :class:`CapabilityContext`.
See ``docs/experimental/model_optimization_advisor_plan.md`` (Phase 4).

Capability dict shape (consumed by the settings-modal panel):

    {
      "feature": "mtp",            # mtp | dflash | vlm_mtp | specprefill | turboquant_kv
      "label": "Native MTP",
      "status": "active" | "available" | "needs-config"
                | "needs-different-checkpoint" | "unsupported",
      "reason": str,               # why this status (English)
      "catalog": [str, ...] | None # what to load/configure, when relevant
    }
"""

from __future__ import annotations

from dataclasses import dataclass

from .feature_catalog import (
    dflash_drafts_for,
    mtp_architecture_notes,
    vlm_mtp_drafter_for,
)

STATUSES = (
    "active",
    "available",
    "needs-config",
    "needs-different-checkpoint",
    "unsupported",
)


@dataclass
class CapabilityContext:
    """Structured model facts, gathered by the endpoint."""

    model_type: str | None = None
    has_vision: bool = False
    is_moe: bool = False
    is_paroquant: bool = False
    paroquant_reason: str = ""
    # MTP condition breakdown (each only meaningful if the previous holds)
    has_mtp_heads: bool = False
    mtp_arch_supported: bool = False
    mtp_weights_present: bool = False
    mtp_head_quantized: bool = False
    # DFlash probe result
    dflash_supported: bool = False
    dflash_reason: str = ""
    # Current settings
    mtp_enabled: bool = False
    dflash_enabled: bool = False
    dflash_draft_model: str | None = None
    vlm_mtp_enabled: bool = False
    vlm_mtp_draft_model: str | None = None
    specprefill_enabled: bool = False
    specprefill_draft_model: str | None = None
    turboquant_kv_enabled: bool = False
    # Live runtime state (Phase 5), only populated when the model is
    # loaded — None fields mean "not loaded" / "not applicable" rather
    # than a specific value.
    mtp_runtime: dict | None = None
    mtp_tuned_winner_depth: int | None = None
    mtp_tuned_at: str | None = None


def _cap(feature: str, label: str, status: str, reason: str, catalog=None) -> dict:
    return {
        "feature": feature,
        "label": label,
        "status": status,
        "reason": reason,
        "catalog": catalog,
    }


def _mtp_live_block(ctx: CapabilityContext) -> dict:
    """Effective runtime state for the MTP row (Phase 5).

    ``mtp_runtime`` is None when the model isn't currently loaded — the
    settings say MTP is enabled, but nothing has run yet on this engine
    instance, so there's no decode/depth/guard state to report.
    """
    runtime = ctx.mtp_runtime or {}
    return {
        "loaded": ctx.mtp_runtime is not None,
        "decode_active": runtime.get("decode_active"),
        "effective_depth": runtime.get("effective_depth"),
        "auto_disabled": runtime.get("auto_disabled", False),
        "auto_disabled_reason": runtime.get("auto_disabled_reason"),
        "auto_disabled_at": runtime.get("auto_disabled_at"),
        "tuned_winner_depth": ctx.mtp_tuned_winner_depth,
        "tuned_at": ctx.mtp_tuned_at,
    }


def _mtp_capability(ctx: CapabilityContext) -> dict:
    label = "Native MTP"
    if ctx.is_paroquant:
        return _cap("mtp", label, "unsupported", ctx.paroquant_reason)
    if not ctx.has_mtp_heads:
        return _cap(
            "mtp",
            label,
            "unsupported",
            "The model config declares no MTP head layers "
            "(mtp_num_hidden_layers / num_nextn_predict_layers). Only "
            "checkpoints of MTP-capable architectures can use native MTP.",
            catalog=mtp_architecture_notes(),
        )
    if not ctx.mtp_arch_supported:
        return _cap(
            "mtp",
            label,
            "unsupported",
            f"model_type={ctx.model_type!r} is not on the MTP whitelist "
            "(supported: qwen3_5*, qwen3_6*, deepseek_v4*), even though the "
            "config declares MTP heads.",
            catalog=mtp_architecture_notes(),
        )
    if not ctx.mtp_weights_present:
        return _cap(
            "mtp",
            label,
            "needs-different-checkpoint",
            "The architecture supports MTP and the config declares heads, "
            "but this conversion stripped the mtp.* weight tensors. "
            "Re-convert from HF with an MTP-preserving converter, or use a "
            "checkpoint that ships them.",
            catalog=mtp_architecture_notes(),
        )
    if ctx.mtp_head_quantized:
        cap = _cap(
            "mtp",
            label,
            "needs-different-checkpoint" if not ctx.mtp_enabled else "active",
            "MTP is usable but the mtp.* head weights are quantized, which "
            "is known to collapse acceptance (79-85% BF16 vs 5-11% int4); "
            "the acceptance-floor guard may auto-disable it. Prefer a "
            "checkpoint with a BF16 MTP head.",
            catalog=mtp_architecture_notes(),
        )
        if ctx.mtp_enabled:
            cap["live"] = _mtp_live_block(ctx)
        return cap
    if ctx.mtp_enabled:
        cap = _cap("mtp", label, "active", "Enabled; draft+verify in use.")
        cap["live"] = _mtp_live_block(ctx)
        return cap
    return _cap(
        "mtp",
        label,
        "available",
        "This checkpoint ships usable MTP heads — can be enabled from here.",
    )


def _dflash_capability(ctx: CapabilityContext) -> dict:
    label = "DFlash"
    if ctx.is_paroquant:
        return _cap("dflash", label, "unsupported", ctx.paroquant_reason)
    if not ctx.dflash_supported:
        return _cap(
            "dflash",
            label,
            "unsupported",
            ctx.dflash_reason or "Not supported by the DFlash backend.",
        )
    drafts = dflash_drafts_for(ctx.model_type)
    if ctx.dflash_enabled and ctx.dflash_draft_model:
        return _cap(
            "dflash",
            label,
            "active",
            f"Enabled with draft model {ctx.dflash_draft_model}.",
        )
    if ctx.dflash_enabled:
        return _cap(
            "dflash",
            label,
            "needs-config",
            "dflash_enabled is set but no draft model is configured — "
            "DFlash cannot run without one.",
            catalog=drafts or None,
        )
    if ctx.dflash_draft_model:
        return _cap(
            "dflash",
            label,
            "available",
            "A draft model is configured; DFlash can be enabled from here.",
        )
    return _cap(
        "dflash",
        label,
        "needs-config",
        "The target architecture is DFlash-compatible, but a matching "
        "draft checkpoint must be chosen first (dflash_draft_model).",
        catalog=drafts or None,
    )


def _vlm_mtp_capability(ctx: CapabilityContext) -> dict:
    label = "VLM MTP"
    if not ctx.has_vision:
        return _cap(
            "vlm_mtp",
            label,
            "unsupported",
            "Text-only model — VLM MTP applies to vision-language models; "
            "use native MTP or DFlash instead.",
        )
    drafter = vlm_mtp_drafter_for(ctx.model_type)
    if drafter is None:
        return _cap(
            "vlm_mtp",
            label,
            "unsupported",
            f"No known VLM-MTP drafter for model_type={ctx.model_type!r} "
            "(supported targets: Gemma4 VLMs, Qwen3.5/3.6 VLMs).",
        )
    hint = (
        f"drafter kind {drafter['drafter_kind']}, e.g. {drafter['example']}"
    )
    if ctx.vlm_mtp_enabled:
        return _cap(
            "vlm_mtp",
            label,
            "active",
            f"Enabled with drafter {ctx.vlm_mtp_draft_model}.",
        )
    if ctx.vlm_mtp_draft_model:
        return _cap(
            "vlm_mtp",
            label,
            "available",
            "A drafter is configured; VLM MTP can be enabled from here.",
        )
    return _cap(
        "vlm_mtp",
        label,
        "needs-config",
        f"This VLM supports MTP speculation via an external drafter "
        f"({hint}) — set vlm_mtp_draft_model first.",
        catalog=[f"{drafter['drafter_kind']}: {drafter['example']}"],
    )


def _specprefill_capability(ctx: CapabilityContext) -> dict:
    label = "SpecPrefill"
    if not ctx.is_moe:
        return _cap(
            "specprefill",
            label,
            "unsupported",
            "SpecPrefill targets MoE models (attention-based sparse "
            "prefill); this model is dense.",
        )
    if ctx.specprefill_enabled:
        return _cap("specprefill", label, "active", "Enabled.")
    if ctx.specprefill_draft_model:
        return _cap(
            "specprefill",
            label,
            "available",
            "A draft model is configured; SpecPrefill can be enabled from "
            "here.",
        )
    return _cap(
        "specprefill",
        label,
        "needs-config",
        "MoE model — SpecPrefill can cut prefill cost on long prompts, but "
        "needs a draft model that shares this model's tokenizer "
        "(specprefill_draft_model).",
    )


def _turboquant_capability(ctx: CapabilityContext) -> dict:
    label = "TurboQuant KV"
    if ctx.is_paroquant:
        return _cap("turboquant_kv", label, "unsupported", ctx.paroquant_reason)
    if ctx.turboquant_kv_enabled:
        return _cap("turboquant_kv", label, "active", "Enabled.")
    if ctx.mtp_enabled or ctx.vlm_mtp_enabled:
        return _cap(
            "turboquant_kv",
            label,
            "available",
            "Available, but mutually exclusive with the active MTP path — "
            "enabling it requires turning MTP off.",
        )
    return _cap(
        "turboquant_kv",
        label,
        "available",
        "KV-cache quantization (2-8 bit) can be enabled from here; most "
        "useful for long contexts or when RAM is tight.",
    )


def build_capabilities(ctx: CapabilityContext) -> list[dict]:
    """Capability matrix in fixed feature order."""
    return [
        _mtp_capability(ctx),
        _dflash_capability(ctx),
        _vlm_mtp_capability(ctx),
        _specprefill_capability(ctx),
        _turboquant_capability(ctx),
    ]
