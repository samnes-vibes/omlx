# SPDX-License-Identifier: Apache-2.0
"""Tests for MTP draft-depth auto-tuning (mtp_tune + "auto" resolution)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from omlx.admin.mtp_tune import (
    hardware_id,
    load_tuned_depth,
    run_mtp_tune,
    save_tune_result,
    tune_store_path,
)
from omlx.model_settings import ModelSettings


class TestTuneStore:
    def test_hardware_id_is_stable_slug(self):
        hid = hardware_id()
        assert hid and hid == hardware_id()
        assert hid == hid.lower()
        assert " " not in hid

    def test_round_trip(self, tmp_path):
        save_tune_result("model-a", 3, {0: 10.0, 3: 20.0}, base_path=tmp_path)
        assert load_tuned_depth("model-a", base_path=tmp_path) == 3
        assert load_tuned_depth("model-b", base_path=tmp_path) is None

    def test_depth_zero_round_trip(self, tmp_path):
        save_tune_result("model-a", 0, {0: 30.0, 1: 20.0}, base_path=tmp_path)
        assert load_tuned_depth("model-a", base_path=tmp_path) == 0

    def test_store_file_shape(self, tmp_path):
        save_tune_result("model-a", 2, {1: 15.5}, base_path=tmp_path)
        data = json.loads(tune_store_path(tmp_path).read_text())
        entry = data["model-a"][hardware_id()]
        assert entry["depth"] == 2
        assert entry["tps_by_depth"] == {"1": 15.5}
        assert "tuned_at" in entry

    def test_corrupt_store_returns_none(self, tmp_path):
        tune_store_path(tmp_path).write_text("{not json")
        assert load_tuned_depth("model-a", base_path=tmp_path) is None


class _FakeModel:
    """Minimal stand-in exposing the per-instance MTP stamps + chain hook."""

    def __init__(self, tps_by_depth):
        self._omlx_mtp_decode_enabled = True
        self._omlx_mtp_draft_depth = 1
        self._tps_by_depth = tps_by_depth

    def mtp_forward_hidden(self):  # capability gate for depth > 1
        raise NotImplementedError


class _FakeEngine:
    def __init__(self, tps_by_depth):
        self._model = _FakeModel(tps_by_depth)

    async def stream_generate(self, **kwargs):
        model = self._model
        depth = (
            model._omlx_mtp_draft_depth if model._omlx_mtp_decode_enabled else 0
        )
        yield SimpleNamespace(
            generation_tps=model._tps_by_depth[depth],
            completion_tokens=kwargs.get("max_tokens", 0),
        )


class TestRunMtpTune:
    async def test_picks_argmax_and_persists(self, tmp_path):
        engine = _FakeEngine({0: 10.0, 1: 14.0, 2: 18.0, 3: 17.0, 4: 12.0})
        result = await run_mtp_tune(
            engine, "model-a", repeats=2, max_tokens=8, base_path=tmp_path
        )
        assert result["winner_depth"] == 2
        assert set(result["tps_by_depth"]) == {"0", "1", "2", "3", "4"}
        assert load_tuned_depth("model-a", base_path=tmp_path) == 2
        # Original stamps restored after the sweep.
        assert engine._model._omlx_mtp_decode_enabled is True
        assert engine._model._omlx_mtp_draft_depth == 1

    async def test_depth_zero_can_win(self, tmp_path):
        engine = _FakeEngine({0: 30.0, 1: 20.0, 2: 18.0, 3: 15.0, 4: 12.0})
        result = await run_mtp_tune(
            engine, "model-a", repeats=1, max_tokens=8, base_path=tmp_path
        )
        assert result["winner_depth"] == 0
        assert load_tuned_depth("model-a", base_path=tmp_path) == 0

    async def test_no_chain_hook_limits_to_depth_1(self, tmp_path):
        engine = _FakeEngine({0: 10.0, 1: 14.0})
        # Replace the model with one lacking mtp_forward_hidden: the tuner
        # must limit the sweep to {0, 1}.
        engine._model = SimpleNamespace(
            _omlx_mtp_decode_enabled=True,
            _omlx_mtp_draft_depth=1,
            _tps_by_depth={0: 10.0, 1: 14.0},
        )
        result = await run_mtp_tune(
            engine, "model-a", repeats=1, max_tokens=8, base_path=tmp_path
        )
        assert set(result["tps_by_depth"]) == {"0", "1"}
        assert result["winner_depth"] == 1

    async def test_rejects_mtp_disabled_model(self, tmp_path):
        engine = _FakeEngine({0: 10.0})
        engine._model._omlx_mtp_decode_enabled = False
        with pytest.raises(ValueError, match="mtp_enabled"):
            await run_mtp_tune(engine, "model-a", base_path=tmp_path)

    async def test_rejects_engine_without_model(self, tmp_path):
        with pytest.raises(ValueError, match="no loaded model"):
            await run_mtp_tune(
                SimpleNamespace(_model=None), "model-a", base_path=tmp_path
            )


class TestAutoSetting:
    def test_auto_is_valid(self):
        assert ModelSettings(mtp_draft_depth="auto").mtp_draft_depth == "auto"

    def test_bogus_string_rejected(self):
        with pytest.raises(ValueError, match="auto"):
            ModelSettings(mtp_draft_depth="fast")

    def test_int_range_still_enforced(self):
        with pytest.raises(ValueError):
            ModelSettings(mtp_draft_depth=9)


class TestAutoResolutionAtLoad:
    def _write_config(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5", "mtp_num_hidden_layers": 1})
        )

    def test_tuned_depth_resolves(self, tmp_path, monkeypatch):
        from omlx.patches.mlx_lm_mtp import is_mtp_active, mtp_draft_depth
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "base"))
        self._write_config(tmp_path)
        save_tune_result(tmp_path.name, 3, {3: 20.0}, base_path=tmp_path / "base")
        maybe_apply_pre_load_patches(
            str(tmp_path),
            model_settings=ModelSettings(mtp_enabled=True, mtp_draft_depth="auto"),
        )
        assert is_mtp_active() is True
        assert mtp_draft_depth() == 3

    def test_depth_zero_winner_disables_mtp(self, tmp_path, monkeypatch):
        from omlx.patches.mlx_lm_mtp import is_mtp_active
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "base"))
        self._write_config(tmp_path)
        save_tune_result(tmp_path.name, 0, {0: 30.0}, base_path=tmp_path / "base")
        maybe_apply_pre_load_patches(
            str(tmp_path),
            model_settings=ModelSettings(mtp_enabled=True, mtp_draft_depth="auto"),
        )
        assert is_mtp_active() is False

    def test_untuned_auto_falls_back_to_depth_1(self, tmp_path, monkeypatch):
        from omlx.patches.mlx_lm_mtp import is_mtp_active, mtp_draft_depth
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "base"))
        self._write_config(tmp_path)
        maybe_apply_pre_load_patches(
            str(tmp_path),
            model_settings=ModelSettings(mtp_enabled=True, mtp_draft_depth="auto"),
        )
        assert is_mtp_active() is True
        assert mtp_draft_depth() == 1
