# SPDX-License-Identifier: Apache-2.0
"""Static catalog of known feature/checkpoint pairings (advisor Phase 4).

Data only — the capability engine (``capabilities.py``) attaches these
entries to ``needs-config`` / ``needs-different-checkpoint`` statuses so
the operator sees *what to load*, not just that something is missing.
See ``docs/experimental/model_optimization_advisor_plan.md`` (Phase 4).
"""

from __future__ import annotations

# Architectures whose MTP heads oMLX can actually drive (mirrors the
# whitelist in omlx.utils.model_loading._is_mtp_compatible). The note is
# shown when a checkpoint fails the weight/quantization checks.
MTP_ARCHITECTURES: list[dict] = [
    {
        "model_type_prefix": "qwen3_5",
        "note": "Qwen3.5 — needs a conversion that preserves mtp.* tensors "
        "(mlx-lm PR 990 path); keep the MTP head in BF16.",
    },
    {
        "model_type_prefix": "qwen3_6",
        "note": "Qwen3.6 — same converter requirements as Qwen3.5.",
    },
    {
        "model_type_prefix": "deepseek_v4",
        "note": "DeepSeek-V4-Flash — Blaizzy/mlx-lm fork PR 15 path.",
    },
]

# Known-good DFlash target→draft pairings (z-lab HF collection; see
# docs/experimental/dflash_mlx_integration.md).
DFLASH_DRAFT_CATALOG: list[dict] = [
    {
        "target_model_type_substring": "qwen",
        "drafts": [
            "z-lab/DFlash-Qwen3-4B",
            "z-lab/DFlash-Qwen3-8B",
            "z-lab/DFlash-Qwen3.5-27B",
        ],
    },
    {
        "target_model_type_substring": "gemma4",
        "drafts": ["gemma-4 -assistant checkpoints (gemma4_assistant)"],
    },
]

# VLM-MTP drafter per target family (mlx-vlm f96138e+; both resolve to
# draft_kind="mtp" in mlx-vlm — see ModelSettings.vlm_mtp_draft_model).
VLM_MTP_DRAFTERS: list[dict] = [
    {
        "target_model_type_prefix": "gemma4",
        "drafter_kind": "gemma4_assistant",
        "example": "gemma-4-26B-A4B-it-assistant",
    },
    {
        "target_model_type_prefix": "qwen3_5",
        "drafter_kind": "qwen3_5_mtp",
        "example": "Qwen3.5-VL MTP drafter checkpoint",
    },
    {
        "target_model_type_prefix": "qwen3_6",
        "drafter_kind": "qwen3_5_mtp",
        "example": "Qwen3.6-VL MTP drafter checkpoint",
    },
]


def dflash_drafts_for(model_type: str | None) -> list[str]:
    """Known draft checkpoints for a DFlash-compatible target, or []."""
    mt = (model_type or "").lower()
    for entry in DFLASH_DRAFT_CATALOG:
        if entry["target_model_type_substring"] in mt:
            return list(entry["drafts"])
    return []


def vlm_mtp_drafter_for(model_type: str | None) -> dict | None:
    """Known VLM-MTP drafter entry for a target model_type, or None."""
    mt = (model_type or "").lower()
    for entry in VLM_MTP_DRAFTERS:
        if mt.startswith(entry["target_model_type_prefix"]):
            return dict(entry)
    return None


def mtp_architecture_notes() -> list[str]:
    """Human-readable list of MTP-capable architectures for catalog links."""
    return [e["note"] for e in MTP_ARCHITECTURES]
