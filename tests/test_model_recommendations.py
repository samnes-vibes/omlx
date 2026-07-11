# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-model optimization recommendation rules."""

from __future__ import annotations

from omlx.admin.recommendations import (
    RecommendationContext,
    build_recommendations,
)


def _ids(recs):
    return [r["id"] for r in recs]


class TestMtpEnableRule:
    def test_fires_for_compatible_disabled_model(self):
        recs = build_recommendations(RecommendationContext(mtp_compatible=True))
        assert "mtp-enable" in _ids(recs)
        rec = next(r for r in recs if r["id"] == "mtp-enable")
        assert rec["action"] == {
            "type": "settings",
            "payload": {"mtp_enabled": True},
        }

    def test_suppressed_when_already_enabled(self):
        recs = build_recommendations(
            RecommendationContext(mtp_compatible=True, mtp_enabled=True)
        )
        assert "mtp-enable" not in _ids(recs)

    def test_suppressed_by_conflicting_features(self):
        for conflict in (
            "dflash_enabled",
            "vlm_mtp_enabled",
            "turboquant_kv_enabled",
        ):
            recs = build_recommendations(
                RecommendationContext(mtp_compatible=True, **{conflict: True})
            )
            assert "mtp-enable" not in _ids(recs), conflict

    def test_suppressed_when_incompatible(self):
        recs = build_recommendations(RecommendationContext(mtp_compatible=False))
        assert "mtp-enable" not in _ids(recs)


class TestQuantizedHeadRule:
    def test_fires_only_when_mtp_on_and_quantized(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True, mtp_enabled=True, mtp_head_quantized=True
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-quantized-head")
        assert rec["severity"] == "warn"
        assert rec["action"] is None

    def test_suppressed_when_mtp_off(self):
        recs = build_recommendations(
            RecommendationContext(mtp_compatible=True, mtp_head_quantized=True)
        )
        assert "mtp-quantized-head" not in _ids(recs)


class TestTuneRules:
    def test_untuned_suggests_tune(self):
        recs = build_recommendations(
            RecommendationContext(mtp_compatible=True, mtp_enabled=True)
        )
        rec = next(r for r in recs if r["id"] == "mtp-tune")
        assert rec["action"] == {"type": "mtp_tune"}

    def test_tuned_pinned_depth_suggests_auto(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth=2,
                mtp_tuned_depth=3,
            )
        )
        assert "mtp-tune" not in _ids(recs)
        rec = next(r for r in recs if r["id"] == "mtp-use-auto")
        assert rec["severity"] == "suggest"
        assert rec["action"]["payload"] == {"mtp_draft_depth": "auto"}

    def test_tuned_winner_zero_escalates_to_warn(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth=1,
                mtp_tuned_depth=0,
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-use-auto")
        assert rec["severity"] == "warn"

    def test_auto_already_set_is_quiet(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth="auto",
                mtp_tuned_depth=3,
            )
        )
        assert "mtp-use-auto" not in _ids(recs)
        assert "mtp-tune" not in _ids(recs)


_TUNE_ENTRY = {
    "depth": 2,
    "tps_by_depth": {"0": 100.0, "1": 120.0, "2": 130.0},
    "tuned_at": "2026-07-10T12:00:00",
}


class TestMeasuredRecommendations:
    def test_use_auto_carries_measured_numbers(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth=1,
                mtp_tuned_depth=2,
                mtp_tune_entry=_TUNE_ENTRY,
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-use-auto")
        m = rec["measured"]
        assert m["winner_depth"] == 2
        assert m["baseline_tps"] == 100.0
        assert m["gain_pct"] == 30.0
        assert m["tps_by_depth"]["2"] == 130.0
        assert m["tuned_at"] == _TUNE_ENTRY["tuned_at"]

    def test_use_auto_without_entry_has_no_measured(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth=1,
                mtp_tuned_depth=2,
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-use-auto")
        assert "measured" not in rec

    def test_tuned_optimal_fires_on_auto_with_gain(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth="auto",
                mtp_tuned_depth=2,
                mtp_tune_entry=_TUNE_ENTRY,
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-tuned-optimal")
        assert rec["severity"] == "info"
        assert rec["action"] is None
        assert "+30.0%" in rec["title"]

    def test_tuned_optimal_suppressed_when_winner_zero(self):
        entry = {"depth": 0, "tps_by_depth": {"0": 100.0, "1": 90.0}}
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth="auto",
                mtp_tuned_depth=0,
                mtp_tune_entry=entry,
            )
        )
        assert "mtp-tuned-optimal" not in _ids(recs)

    def test_tuned_optimal_suppressed_without_baseline(self):
        entry = {"depth": 2, "tps_by_depth": {"1": 120.0, "2": 130.0}}
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth="auto",
                mtp_tuned_depth=2,
                mtp_tune_entry=entry,
            )
        )
        assert "mtp-tuned-optimal" not in _ids(recs)

    def test_malformed_entry_is_ignored(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth=1,
                mtp_tuned_depth=2,
                mtp_tune_entry={"depth": 2, "tps_by_depth": "garbage"},
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-use-auto")
        assert "measured" not in rec


class TestDflashCandidateRule:
    def test_fires_for_dflash_only_model(self):
        recs = build_recommendations(
            RecommendationContext(dflash_compatible=True)
        )
        rec = next(r for r in recs if r["id"] == "dflash-candidate")
        assert rec["severity"] == "info"
        assert rec["action"] is None

    def test_suppressed_when_mtp_compatible(self):
        recs = build_recommendations(
            RecommendationContext(dflash_compatible=True, mtp_compatible=True)
        )
        assert "dflash-candidate" not in _ids(recs)

    def test_suppressed_when_something_speculative_on(self):
        recs = build_recommendations(
            RecommendationContext(dflash_compatible=True, dflash_enabled=True)
        )
        assert "dflash-candidate" not in _ids(recs)


class TestOrderingAndEmpty:
    def test_warn_sorts_first(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_head_quantized=True,
                mtp_tuned_depth=None,
            )
        )
        severities = [r["severity"] for r in recs]
        assert severities == sorted(
            severities, key=lambda s: {"warn": 0, "suggest": 1, "info": 2}[s]
        )
        assert recs[0]["id"] == "mtp-quantized-head"

    def test_nothing_to_say(self):
        assert build_recommendations(RecommendationContext()) == []
