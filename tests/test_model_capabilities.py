# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-model capability matrix (advisor Phase 4)."""

from __future__ import annotations

from omlx.admin.capabilities import (
    STATUSES,
    CapabilityContext,
    build_capabilities,
)


def _cap(caps, feature):
    return next(c for c in caps if c["feature"] == feature)


def _mtp_ready(**kw):
    defaults = dict(
        model_type="qwen3_5",
        has_mtp_heads=True,
        mtp_arch_supported=True,
        mtp_weights_present=True,
    )
    defaults.update(kw)
    return CapabilityContext(**defaults)


class TestMatrixShape:
    def test_all_features_present_with_valid_statuses(self):
        caps = build_capabilities(CapabilityContext())
        assert [c["feature"] for c in caps] == [
            "mtp",
            "dflash",
            "vlm_mtp",
            "specprefill",
            "turboquant_kv",
        ]
        for c in caps:
            assert c["status"] in STATUSES
            assert c["reason"]


class TestMtpCapability:
    def test_no_heads_is_unsupported_with_catalog(self):
        cap = _cap(build_capabilities(CapabilityContext()), "mtp")
        assert cap["status"] == "unsupported"
        assert cap["catalog"]  # points at MTP-capable architectures

    def test_heads_but_wrong_arch_is_unsupported(self):
        ctx = CapabilityContext(model_type="llama", has_mtp_heads=True)
        cap = _cap(build_capabilities(ctx), "mtp")
        assert cap["status"] == "unsupported"
        assert "whitelist" in cap["reason"]

    def test_stripped_weights_is_needs_different_checkpoint(self):
        ctx = _mtp_ready(mtp_weights_present=False)
        cap = _cap(build_capabilities(ctx), "mtp")
        assert cap["status"] == "needs-different-checkpoint"

    def test_quantized_head_flags_checkpoint_when_disabled(self):
        ctx = _mtp_ready(mtp_head_quantized=True)
        cap = _cap(build_capabilities(ctx), "mtp")
        assert cap["status"] == "needs-different-checkpoint"
        assert "BF16" in cap["reason"]

    def test_quantized_head_stays_active_when_enabled(self):
        ctx = _mtp_ready(mtp_head_quantized=True, mtp_enabled=True)
        cap = _cap(build_capabilities(ctx), "mtp")
        assert cap["status"] == "active"
        assert "BF16" in cap["reason"]

    def test_available_and_active(self):
        assert _cap(build_capabilities(_mtp_ready()), "mtp")["status"] == "available"
        assert (
            _cap(build_capabilities(_mtp_ready(mtp_enabled=True)), "mtp")["status"]
            == "active"
        )

    def test_paroquant_wins(self):
        ctx = _mtp_ready(is_paroquant=True, paroquant_reason="paroquant model")
        cap = _cap(build_capabilities(ctx), "mtp")
        assert cap["status"] == "unsupported"
        assert cap["reason"] == "paroquant model"


class TestMtpLiveStatus:
    """Phase 5: runtime status surfaced when MTP is enabled."""

    def test_no_live_block_when_not_enabled(self):
        cap = _cap(build_capabilities(_mtp_ready()), "mtp")
        assert "live" not in cap

    def test_not_loaded_yet(self):
        ctx = _mtp_ready(mtp_enabled=True, mtp_runtime=None)
        cap = _cap(build_capabilities(ctx), "mtp")
        assert cap["live"]["loaded"] is False
        assert cap["live"]["auto_disabled"] is False

    def test_loaded_and_healthy_reports_depth(self):
        ctx = _mtp_ready(
            mtp_enabled=True,
            mtp_runtime={
                "decode_active": True,
                "effective_depth": 3,
                "auto_disabled": False,
                "auto_disabled_reason": None,
                "auto_disabled_at": None,
            },
            mtp_tuned_winner_depth=3,
            mtp_tuned_at="2026-07-11T10:00:00",
        )
        cap = _cap(build_capabilities(ctx), "mtp")
        live = cap["live"]
        assert live["loaded"] is True
        assert live["decode_active"] is True
        assert live["effective_depth"] == 3
        assert live["auto_disabled"] is False
        assert live["tuned_winner_depth"] == 3
        assert live["tuned_at"] == "2026-07-11T10:00:00"

    def test_auto_disabled_surfaces_reason(self):
        ctx = _mtp_ready(
            mtp_enabled=True,
            mtp_runtime={
                "decode_active": False,
                "effective_depth": 1,
                "auto_disabled": True,
                "auto_disabled_reason": "MTP acceptance 8.0% ... below the 15% floor",
                "auto_disabled_at": "2026-07-11T09:00:00",
            },
        )
        cap = _cap(build_capabilities(ctx), "mtp")
        live = cap["live"]
        assert live["auto_disabled"] is True
        assert "floor" in live["auto_disabled_reason"]
        assert live["auto_disabled_at"] == "2026-07-11T09:00:00"

    def test_quantized_head_active_also_carries_live_block(self):
        ctx = _mtp_ready(
            mtp_head_quantized=True,
            mtp_enabled=True,
            mtp_runtime={
                "decode_active": False,
                "effective_depth": 1,
                "auto_disabled": True,
                "auto_disabled_reason": "collapsed acceptance",
                "auto_disabled_at": "2026-07-11T09:00:00",
            },
        )
        cap = _cap(build_capabilities(ctx), "mtp")
        assert cap["status"] == "active"
        assert cap["live"]["auto_disabled"] is True


class TestDflashCapability:
    def test_unsupported_arch(self):
        ctx = CapabilityContext(dflash_supported=False, dflash_reason="nope")
        cap = _cap(build_capabilities(ctx), "dflash")
        assert cap["status"] == "unsupported"
        assert cap["reason"] == "nope"

    def test_compatible_without_draft_needs_config_with_catalog(self):
        ctx = CapabilityContext(model_type="qwen3", dflash_supported=True)
        cap = _cap(build_capabilities(ctx), "dflash")
        assert cap["status"] == "needs-config"
        assert cap["catalog"]  # known qwen drafts

    def test_draft_configured_is_available_then_active(self):
        ctx = CapabilityContext(
            model_type="qwen3", dflash_supported=True, dflash_draft_model="d"
        )
        assert _cap(build_capabilities(ctx), "dflash")["status"] == "available"
        ctx.dflash_enabled = True
        assert _cap(build_capabilities(ctx), "dflash")["status"] == "active"

    def test_enabled_without_draft_is_needs_config(self):
        ctx = CapabilityContext(
            model_type="qwen3", dflash_supported=True, dflash_enabled=True
        )
        assert _cap(build_capabilities(ctx), "dflash")["status"] == "needs-config"


class TestVlmMtpCapability:
    def test_text_only_is_unsupported(self):
        cap = _cap(build_capabilities(CapabilityContext()), "vlm_mtp")
        assert cap["status"] == "unsupported"

    def test_vision_unknown_family_is_unsupported(self):
        ctx = CapabilityContext(model_type="llava", has_vision=True)
        assert _cap(build_capabilities(ctx), "vlm_mtp")["status"] == "unsupported"

    def test_gemma4_vlm_needs_config_names_drafter(self):
        ctx = CapabilityContext(model_type="gemma4", has_vision=True)
        cap = _cap(build_capabilities(ctx), "vlm_mtp")
        assert cap["status"] == "needs-config"
        assert "gemma4_assistant" in cap["reason"]
        assert cap["catalog"]

    def test_qwen35_vlm_drafter_configured_then_enabled(self):
        ctx = CapabilityContext(
            model_type="qwen3_5_vl",
            has_vision=True,
            vlm_mtp_draft_model="drafter",
        )
        assert _cap(build_capabilities(ctx), "vlm_mtp")["status"] == "available"
        ctx.vlm_mtp_enabled = True
        assert _cap(build_capabilities(ctx), "vlm_mtp")["status"] == "active"


class TestSpecprefillCapability:
    def test_dense_is_unsupported(self):
        cap = _cap(build_capabilities(CapabilityContext()), "specprefill")
        assert cap["status"] == "unsupported"

    def test_moe_statuses(self):
        ctx = CapabilityContext(is_moe=True)
        assert _cap(build_capabilities(ctx), "specprefill")["status"] == "needs-config"
        ctx.specprefill_draft_model = "d"
        assert _cap(build_capabilities(ctx), "specprefill")["status"] == "available"
        ctx.specprefill_enabled = True
        assert _cap(build_capabilities(ctx), "specprefill")["status"] == "active"


class TestTurboquantCapability:
    def test_paroquant_is_unsupported(self):
        ctx = CapabilityContext(is_paroquant=True, paroquant_reason="paro")
        assert _cap(build_capabilities(ctx), "turboquant_kv")["status"] == "unsupported"

    def test_mutex_note_when_mtp_active(self):
        ctx = CapabilityContext(mtp_enabled=True)
        cap = _cap(build_capabilities(ctx), "turboquant_kv")
        assert cap["status"] == "available"
        assert "mutually exclusive" in cap["reason"]

    def test_default_available_then_active(self):
        assert (
            _cap(build_capabilities(CapabilityContext()), "turboquant_kv")["status"]
            == "available"
        )
        ctx = CapabilityContext(turboquant_kv_enabled=True)
        assert _cap(build_capabilities(ctx), "turboquant_kv")["status"] == "active"
