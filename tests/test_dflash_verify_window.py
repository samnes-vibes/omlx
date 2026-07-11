# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.dflash_verify_window (long-context plan, Phase 1)."""

from __future__ import annotations

import pytest


@pytest.fixture
def _clear_window_state():
    from omlx.patches import dflash_verify_window as vw

    vw._SDPA_BACKUP.clear()
    vw._TARGET_OPS_BACKUP.clear()
    vw.configure_verify_window(None, None)
    yield
    vw._SDPA_BACKUP.clear()
    vw._TARGET_OPS_BACKUP.clear()
    vw.configure_verify_window(None, None)


def test_configure_verify_window_none_disables():
    from omlx.patches import dflash_verify_window as vw

    vw.configure_verify_window(None, 1024)
    assert vw.get_configured_window() is None
    vw.configure_verify_window(64, None)
    assert vw.get_configured_window() is None
    vw.configure_verify_window(64, 1024)
    assert vw.get_configured_window() == (64, 1024)
    vw.configure_verify_window(None, None)


def test_gather_sink_window_passthrough_when_covers_full_context():
    import mlx.core as mx

    from omlx.patches.dflash_verify_window import _gather_sink_window

    keys = mx.zeros((1, 2, 100, 8))
    values = mx.zeros((1, 2, 100, 8))
    queries = mx.zeros((1, 2, 4, 8))
    g_keys, g_values, g_mask = _gather_sink_window(
        queries, keys, values, "causal", sink=64, window=1024
    )
    # sink+window (1088) >= kv_len (100) -> untouched passthrough.
    assert g_keys is keys
    assert g_values is values
    assert g_mask == "causal"


def test_gather_sink_window_trims_and_builds_causal_mask():
    import mlx.core as mx

    from omlx.patches.dflash_verify_window import _gather_sink_window

    kv_len, sink, window, q_len = 100, 8, 20, 4
    keys = mx.arange(kv_len).reshape(1, 1, kv_len, 1).astype(mx.float32)
    values = mx.arange(kv_len).reshape(1, 1, kv_len, 1).astype(mx.float32)
    queries = mx.zeros((1, 1, q_len, 8))

    g_keys, g_values, g_mask = _gather_sink_window(
        queries, keys, values, "causal", sink=sink, window=window
    )

    assert g_keys.shape[2] == sink + window
    assert g_values.shape[2] == sink + window
    # Sink block keeps the original first `sink` positions' values (0..sink-1).
    assert g_keys[0, 0, :sink, 0].tolist() == list(range(sink))
    # Window block keeps the last `window` positions (kv_len-window..kv_len-1).
    expected_window = list(range(kv_len - window, kv_len))
    assert g_keys[0, 0, sink:, 0].tolist() == expected_window

    assert g_mask.shape == (q_len, sink + window)
    # Every query (all near the tail) must see the entire sink block: sink
    # positions are always in the past relative to any verify query here.
    assert bool(mx.all(g_mask[:, :sink]).item())
    # Window causal structure: last query row sees the whole window (it's the
    # newest token); the first query row should see strictly fewer window
    # keys than the last row.
    assert int(g_mask[-1, sink:].sum().item()) > int(g_mask[0, sink:].sum().item())


def test_gather_sink_window_expands_window_to_cover_query_block():
    import mlx.core as mx

    from omlx.patches.dflash_verify_window import _gather_sink_window

    # window (2) smaller than q_len (4): must be widened so the newest
    # queries aren't masked away from their own just-written keys.
    kv_len, sink, window, q_len = 50, 4, 2, 4
    keys = mx.zeros((1, 1, kv_len, 8))
    values = mx.zeros((1, 1, kv_len, 8))
    queries = mx.zeros((1, 1, q_len, 8))

    g_keys, _, g_mask = _gather_sink_window(
        queries, keys, values, "causal", sink=sink, window=window
    )
    assert g_keys.shape[2] == sink + q_len
    # The last query must be able to attend to its own position.
    assert bool(g_mask[-1, -1].item())


def _make_fake_sdpa_module():
    from types import SimpleNamespace

    class FakeTargetOps:
        def verify_block(self, **kwargs):
            return "verified", kwargs

        def verify_tree_block(self, **kwargs):
            return "tree_verified", kwargs

    calls: list = []

    def fake_gqa_reshape_sdpa(queries, keys, values, *, scale, mask, cache=None):
        calls.append({"keys_len": int(keys.shape[2]), "mask": mask})
        return "output"

    module = SimpleNamespace(
        _gqa_reshape_sdpa=fake_gqa_reshape_sdpa,
        QwenGdnTargetOps=FakeTargetOps,
    )
    return module, calls


def test_wrap_sdpa_passthrough_when_no_active_window():
    import mlx.core as mx

    from omlx.patches.dflash_verify_window import _wrap_sdpa

    module, calls = _make_fake_sdpa_module()
    wrapped = _wrap_sdpa(module._gqa_reshape_sdpa)

    keys = mx.zeros((1, 1, 100, 8))
    values = mx.zeros((1, 1, 100, 8))
    queries = mx.zeros((1, 1, 4, 8))
    wrapped(queries, keys, values, scale=1.0, mask="causal")

    assert len(calls) == 1
    assert calls[0]["keys_len"] == 100
    assert calls[0]["mask"] == "causal"


def test_wrap_verify_method_activates_window_only_during_call(_clear_window_state):
    from omlx.patches.dflash_verify_window import (
        _active_window,
        _wrap_verify_method,
        configure_verify_window,
    )

    module, _ = _make_fake_sdpa_module()

    seen_inside: list = []

    class Ops:
        def verify_block(self, **kwargs):
            seen_inside.append(_active_window.get())
            return "ok"

    _wrap_verify_method(Ops, "verify_block")
    configure_verify_window(64, 128)

    assert _active_window.get() is None
    result = Ops().verify_block(target_model=None, verify_ids=None, target_cache=[])
    assert result == "ok"
    assert seen_inside == [(64, 128)]
    # Context is scoped to the call — cleared again afterward.
    assert _active_window.get() is None


def test_install_and_restore_roundtrip(_clear_window_state):
    """install/restore must never raise, whether or not dflash-mlx is
    importable, and must leave no patched state behind afterward.
    """
    from omlx.patches.dflash_verify_window import (
        _SDPA_BACKUP,
        _TARGET_OPS_BACKUP,
        install_dflash_verify_window_patch,
        restore_dflash_verify_window_patch,
    )

    install_dflash_verify_window_patch()
    install_dflash_verify_window_patch()  # idempotent second call
    restore_dflash_verify_window_patch()
    assert _SDPA_BACKUP == {}
    assert _TARGET_OPS_BACKUP == {}
