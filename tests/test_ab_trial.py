# SPDX-License-Identifier: Apache-2.0
"""Tests for the generic A/B settings trial engine (advisor P2b)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from omlx.admin.ab_trial import (
    changed_keys,
    create_trial_run,
    load_trial_results,
    requires_reload,
    run_ab_trial,
    save_trial_result,
    settings_hash,
    trial_store_path,
)
from omlx.admin.mtp_tune import hardware_id


def _variants(current: dict, payload: dict) -> list[dict]:
    return [
        {"label": "current", "settings": current},
        {"label": "candidate", "settings": {**current, **payload}},
    ]


class TestVariantClassification:
    def test_changed_keys(self):
        variants = _variants({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert changed_keys(variants) == {"b", "c"}

    def test_depth_change_with_mtp_on_is_stampable(self):
        current = {"mtp_enabled": True, "mtp_draft_depth": 1}
        assert not requires_reload({"mtp_draft_depth"}, current)

    def test_depth_change_with_mtp_off_requires_reload(self):
        current = {"mtp_enabled": False, "mtp_draft_depth": 1}
        assert requires_reload({"mtp_draft_depth"}, current)

    def test_load_time_key_requires_reload(self):
        current = {"mtp_enabled": False}
        assert requires_reload({"mtp_enabled"}, current)
        assert requires_reload({"turboquant_kv_enabled"}, current)


class TestTrialStore:
    def test_round_trip_hash_filtered(self, tmp_path):
        h = settings_hash({"mtp_enabled": True})
        save_trial_result(
            "model-a",
            "mtp-use-auto",
            {"settings_hash": h, "gain_pct": 12.5, "trial_at": "t"},
            base_path=tmp_path,
        )
        assert load_trial_results("model-a", h, base_path=tmp_path) == {
            "mtp-use-auto": {"settings_hash": h, "gain_pct": 12.5, "trial_at": "t"}
        }
        # Different current settings -> stale entry is not returned.
        assert load_trial_results("model-a", "other-hash", base_path=tmp_path) == {}
        assert load_trial_results("model-b", h, base_path=tmp_path) == {}

    def test_store_file_shape(self, tmp_path):
        save_trial_result(
            "model-a", "rec-x", {"settings_hash": "h", "gain_pct": 1.0},
            base_path=tmp_path,
        )
        data = json.loads(trial_store_path(tmp_path).read_text())
        assert data["model-a"][hardware_id()]["rec-x"]["gain_pct"] == 1.0

    def test_corrupt_store_returns_empty(self, tmp_path):
        trial_store_path(tmp_path).write_text("{not json")
        assert load_trial_results("model-a", "h", base_path=tmp_path) == {}

    def test_settings_hash_stable_and_sensitive(self):
        a = settings_hash({"x": 1, "y": 2})
        assert a == settings_hash({"y": 2, "x": 1})
        assert a != settings_hash({"x": 1, "y": 3})


class _FakeModel:
    def __init__(self, tps_by_depth):
        self._omlx_mtp_decode_enabled = True
        self._omlx_mtp_draft_depth = 1
        self._tps_by_depth = tps_by_depth


class _FakeEngine:
    """Reports tps from the stamped depth (stamp path) or a fixed rate
    per loaded variant (reload path)."""

    def __init__(self, tps_by_depth=None, fixed_tps=None):
        self._model = _FakeModel(tps_by_depth or {})
        self._fixed_tps = fixed_tps

    async def stream_generate(self, **kwargs):
        if self._fixed_tps is not None:
            tps = self._fixed_tps
        else:
            model = self._model
            depth = (
                model._omlx_mtp_draft_depth
                if model._omlx_mtp_decode_enabled
                else 0
            )
            tps = model._tps_by_depth[depth]
        yield SimpleNamespace(
            generation_tps=tps,
            completion_tokens=kwargs.get("max_tokens", 0),
        )


class _StampPathPool:
    def __init__(self, engine):
        self._engine = engine

    async def get_engine(self, model_id, force_lm=False, runtime_settings=None):
        return self._engine


class _ReloadPathPool:
    """Returns a slower engine for persisted settings and a faster one when
    a runtime_settings variant is requested; counts variant reloads."""

    def __init__(self):
        self.variant_loads = 0
        self.persisted_loads = 0

    async def get_engine(self, model_id, force_lm=False, runtime_settings=None):
        if runtime_settings is not None:
            self.variant_loads += 1
            return _FakeEngine(fixed_tps=60.0)
        self.persisted_loads += 1
        return _FakeEngine(fixed_tps=40.0)


class _FakeSettingsManager:
    def __init__(self, settings_obj):
        self._settings = settings_obj

    def get_settings(self, model_id):
        # Copy semantics like the real manager.
        return SimpleNamespace(**vars(self._settings))


class TestRunAbTrial:
    async def test_stamp_path_measures_and_persists(self, tmp_path):
        engine = _FakeEngine(tps_by_depth={0: 10.0, 1: 20.0, 2: 30.0})
        current = {"mtp_enabled": True, "mtp_draft_depth": 1}
        run = create_trial_run(
            model_id="m",
            model_key="model-a",
            rec_id="mtp-use-auto",
            variants=_variants(current, {"mtp_draft_depth": 2}),
            current_hash=settings_hash(current),
            reload_needed=False,
        )
        run.base_path = tmp_path
        await run_ab_trial(run, _StampPathPool(engine), _FakeSettingsManager(None))

        assert run.status == "completed"
        result = run.result
        assert result["variants"]["current"]["tps_median"] == 20.0
        assert result["variants"]["candidate"]["tps_median"] == 30.0
        assert result["gain_pct"] == 50.0
        # Original stamps restored.
        assert engine._model._omlx_mtp_draft_depth == 1
        assert engine._model._omlx_mtp_decode_enabled is True
        # Persisted and hash-filtered readable.
        stored = load_trial_results(
            "model-a", settings_hash(current), base_path=tmp_path
        )
        assert stored["mtp-use-auto"]["gain_pct"] == 50.0

    async def test_stamp_path_events_shape(self, tmp_path):
        engine = _FakeEngine(tps_by_depth={0: 10.0, 1: 20.0, 2: 30.0})
        current = {"mtp_enabled": True, "mtp_draft_depth": 1}
        run = create_trial_run(
            model_id="m",
            model_key="model-a",
            rec_id="mtp-use-auto",
            variants=_variants(current, {"mtp_draft_depth": 2}),
            current_hash=settings_hash(current),
            reload_needed=False,
        )
        run.base_path = tmp_path
        await run_ab_trial(run, _StampPathPool(engine), _FakeSettingsManager(None))

        types = [e["type"] for e in run.events]
        assert types[0] == "start"
        assert types[-1] == "result"
        assert types.count("progress") == run.repeats * 2
        assert run.terminal is True

    async def test_reload_path_uses_runtime_settings_variant(self, tmp_path):
        pool = _ReloadPathPool()
        current = {"mtp_enabled": False, "mtp_draft_depth": 1}
        settings_obj = SimpleNamespace(mtp_enabled=False, mtp_draft_depth=1)
        run = create_trial_run(
            model_id="m",
            model_key="model-a",
            rec_id="mtp-enable",
            variants=_variants(current, {"mtp_enabled": True}),
            current_hash=settings_hash(current),
            reload_needed=True,
        )
        run.base_path = tmp_path
        await run_ab_trial(run, pool, _FakeSettingsManager(settings_obj))

        assert run.status == "completed"
        assert run.result["variants"]["current"]["tps_median"] == 40.0
        assert run.result["variants"]["candidate"]["tps_median"] == 60.0
        assert run.result["gain_pct"] == 50.0
        assert run.result["reload_required"] is True
        assert pool.variant_loads == run.repeats
        # finally-restore acquires the persisted engine once more.
        assert pool.persisted_loads >= run.repeats + 1

    async def test_error_is_reported_and_terminal(self, tmp_path):
        class _BrokenPool:
            async def get_engine(self, *a, **k):
                raise RuntimeError("boom")

        current = {"mtp_enabled": True, "mtp_draft_depth": 1}
        run = create_trial_run(
            model_id="m",
            model_key="model-a",
            rec_id="mtp-use-auto",
            variants=_variants(current, {"mtp_draft_depth": 2}),
            current_hash=settings_hash(current),
            reload_needed=False,
        )
        run.base_path = tmp_path
        await run_ab_trial(run, _BrokenPool(), _FakeSettingsManager(None))
        assert run.status == "error"
        assert run.events[-1]["type"] == "error"
        assert run.terminal is True
        assert load_trial_results(
            "model-a", settings_hash(current), base_path=tmp_path
        ) == {}
