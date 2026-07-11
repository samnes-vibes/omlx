# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-model optimization recommendation rules."""

from __future__ import annotations

from omlx.admin.recommendations import (
    RecommendationContext,
    best_estimated_gain,
    build_recommendations,
    collect_settings_payload,
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


class TestCollectSettingsPayload:
    def test_merges_only_settings_actions(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth=1,
                mtp_tuned_depth=2,
                mtp_head_quantized=True,  # warn, no action — must be skipped
                chunk_kv_reuse_enabled=True,
                sparse_prefill_enabled=True,
                sparse_prefill_calibrated=True,
            )
        )
        payload, ids = collect_settings_payload(recs)
        assert payload == {"mtp_draft_depth": "auto"}
        assert ids == ["mtp-use-auto"]

    def test_untuned_model_yields_enable_only(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                chunk_kv_reuse_enabled=True,
                sparse_prefill_enabled=True,
                sparse_prefill_calibrated=True,
            )
        )
        payload, ids = collect_settings_payload(recs)
        assert payload == {"mtp_enabled": True}
        assert ids == ["mtp-enable"]

    def test_empty_when_nothing_actionable(self):
        recs = build_recommendations(
            RecommendationContext(
                dflash_compatible=True,
                ngram_spec_enabled=True,  # suppresses the actionable specs
                chunk_kv_reuse_enabled=True,
                sparse_prefill_enabled=True,
                sparse_prefill_calibrated=True,
            )
        )
        payload, ids = collect_settings_payload(recs)
        assert payload == {}
        assert ids == []


class TestEstimatedGain:
    def test_mtp_enable_gets_heuristic_high(self):
        recs = build_recommendations(RecommendationContext(mtp_compatible=True))
        rec = next(r for r in recs if r["id"] == "mtp-enable")
        assert rec["estimated_gain"] == {"class": "high", "basis": "heuristic"}

    def test_dflash_candidate_gets_heuristic_medium(self):
        recs = build_recommendations(RecommendationContext(dflash_compatible=True))
        rec = next(r for r in recs if r["id"] == "dflash-candidate")
        assert rec["estimated_gain"] == {"class": "medium", "basis": "heuristic"}

    def test_quantized_head_warning_has_no_estimate(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True, mtp_enabled=True, mtp_head_quantized=True
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-quantized-head")
        assert "estimated_gain" not in rec

    def test_use_auto_measured_gain_matches_tune_store(self):
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
        assert rec["estimated_gain"] == {"measured": 30.0}

    def test_use_auto_without_measurement_has_no_estimate(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth=1,
                mtp_tuned_depth=2,
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-use-auto")
        assert "estimated_gain" not in rec

    def test_best_estimated_gain_prefers_measured_over_heuristic(self):
        # Built by hand: the rule engine's mutually-exclusive conditions mean
        # a measured rec and a heuristic rec never fire together in practice,
        # but best_estimated_gain must still rank measured first if they did.
        recs = [
            {"id": "dflash-candidate", "title": "DFlash available",
             "estimated_gain": {"class": "medium", "basis": "heuristic"}},
            {"id": "mtp-use-auto", "title": "Use tuned depth",
             "estimated_gain": {"measured": 30.0}},
        ]
        best = best_estimated_gain(recs)
        assert best["id"] == "mtp-use-auto"
        assert best["estimated_gain"] == {"measured": 30.0}

    def test_best_estimated_gain_none_when_nothing_estimated(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_head_quantized=True,
                chunk_kv_reuse_enabled=True,
                sparse_prefill_enabled=True,
                sparse_prefill_calibrated=True,
            )
        )
        assert best_estimated_gain(recs) is None

    def test_best_estimated_gain_empty_list(self):
        assert best_estimated_gain([]) is None


class TestAbTrialMeasured:
    """P2b: stored A/B trial results upgrade heuristic recs to measured."""

    _entry = {
        "settings_hash": "h",
        "gain_pct": 43.0,
        "variants": {
            "current": {"tps_median": 41.2, "samples": [40.8, 41.6]},
            "candidate": {"tps_median": 58.9, "samples": [58.1, 59.7]},
        },
        "trial_at": "2026-07-11T10:00:00",
    }

    def test_trial_result_attaches_measured(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                ab_trial_results={"mtp-enable": self._entry},
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-enable")
        assert rec["measured"]["gain_pct"] == 43.0
        assert rec["measured"]["source"] == "ab_trial"
        assert rec["measured"]["baseline_tps"] == 41.2
        assert rec["measured"]["candidate_tps"] == 58.9
        assert rec["estimated_gain"] == {"measured": 43.0}

    def test_tune_store_measurement_wins_over_trial(self):
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                mtp_enabled=True,
                mtp_draft_depth=1,
                mtp_tuned_depth=2,
                mtp_tune_entry={
                    "depth": 2,
                    "tps_by_depth": {"0": 10.0, "2": 13.0},
                },
                ab_trial_results={"mtp-use-auto": self._entry},
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-use-auto")
        assert rec["measured"].get("source") != "ab_trial"
        assert rec["measured"]["winner_depth"] == 2

    def test_trial_without_gain_is_ignored(self):
        entry = {**self._entry, "gain_pct": None}
        recs = build_recommendations(
            RecommendationContext(
                mtp_compatible=True,
                ab_trial_results={"mtp-enable": entry},
            )
        )
        rec = next(r for r in recs if r["id"] == "mtp-enable")
        assert "measured" not in rec
        assert rec["estimated_gain"]["basis"] == "heuristic"

    def test_no_results_means_no_change(self):
        recs = build_recommendations(
            RecommendationContext(mtp_compatible=True, ab_trial_results={})
        )
        rec = next(r for r in recs if r["id"] == "mtp-enable")
        assert "measured" not in rec


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
        # Every feature already on (or resolved) — no advice left to give.
        assert (
            build_recommendations(
                RecommendationContext(
                    ngram_spec_enabled=True,
                    chunk_kv_reuse_enabled=True,
                    sparse_prefill_enabled=True,
                    sparse_prefill_calibrated=True,
                )
            )
            == []
        )


class TestIntegrationFeatureRules:
    """Rules for the integration-branch features (ngram, sparse, chunk-KV)."""

    def test_ngram_fires_when_mtp_unavailable_and_nothing_on(self):
        recs = build_recommendations(RecommendationContext())
        rec = next(r for r in recs if r["id"] == "ngram-spec-enable")
        assert rec["severity"] == "suggest"
        assert rec["action"] == {
            "type": "settings",
            "payload": {"ngram_spec_enabled": True},
        }
        assert rec["estimated_gain"] == {"class": "medium", "basis": "heuristic"}

    def test_ngram_suppressed_when_mtp_compatible(self):
        recs = build_recommendations(RecommendationContext(mtp_compatible=True))
        assert not any(r["id"] == "ngram-spec-enable" for r in recs)

    def test_ngram_suppressed_when_any_speculative_on(self):
        for flag in (
            "mtp_enabled",
            "dflash_enabled",
            "vlm_mtp_enabled",
            "ngram_spec_enabled",
        ):
            recs = build_recommendations(RecommendationContext(**{flag: True}))
            assert not any(r["id"] == "ngram-spec-enable" for r in recs), flag

    def test_mtp_enable_suppressed_when_ngram_on(self):
        recs = build_recommendations(
            RecommendationContext(mtp_compatible=True, ngram_spec_enabled=True)
        )
        assert not any(r["id"] == "mtp-enable" for r in recs)

    def test_sparse_prefill_enable_when_calibrated(self):
        recs = build_recommendations(
            RecommendationContext(sparse_prefill_calibrated=True)
        )
        rec = next(r for r in recs if r["id"] == "sparse-prefill-enable")
        assert rec["severity"] == "suggest"
        assert rec["action"]["payload"] == {"sparse_prefill_enabled": True}
        assert not any(r["id"] == "sparse-prefill-calibrate" for r in recs)

    def test_sparse_prefill_calibrate_advice_when_not_calibrated(self):
        recs = build_recommendations(RecommendationContext())
        rec = next(r for r in recs if r["id"] == "sparse-prefill-calibrate")
        assert rec["severity"] == "info"
        assert rec["action"] is None

    def test_sparse_prefill_suppressed_when_on_or_specprefill(self):
        recs = build_recommendations(
            RecommendationContext(
                sparse_prefill_enabled=True, sparse_prefill_calibrated=True
            )
        )
        assert not any(r["id"].startswith("sparse-prefill") for r in recs)
        recs = build_recommendations(
            RecommendationContext(specprefill_enabled=True)
        )
        assert not any(r["id"].startswith("sparse-prefill") for r in recs)

    def test_chunk_kv_reuse_fires_with_action(self):
        recs = build_recommendations(RecommendationContext())
        rec = next(r for r in recs if r["id"] == "chunk-kv-reuse-candidate")
        assert rec["severity"] == "info"
        assert rec["action"]["payload"] == {"chunk_kv_reuse_enabled": True}

    def test_chunk_kv_reuse_suppressed_by_dflash_and_turboquant(self):
        for flag in ("dflash_enabled", "turboquant_kv_enabled", "chunk_kv_reuse_enabled"):
            recs = build_recommendations(RecommendationContext(**{flag: True}))
            assert not any(
                r["id"] == "chunk-kv-reuse-candidate" for r in recs
            ), flag
