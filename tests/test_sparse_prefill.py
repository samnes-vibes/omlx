# SPDX-License-Identifier: Apache-2.0
"""Tests for draft-free sparse prefill (omlx.patches.sparse_prefill)."""

import json

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.sparse_prefill import (
    HeadPattern,
    _patterns_from_calibration,
    _sparse_path_applies,
    _STATE,
    estimate_vertical_columns,
    sparse_prefill_attention,
)


def _rand_qkv(K=2048, L=512, Hq=8, Hkv=4, D=64, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, Hq, L, D)).astype(mx.float16)
    k = mx.random.normal((1, Hkv, K, D)).astype(mx.float16)
    v = mx.random.normal((1, Hkv, K, D)).astype(mx.float16)
    return q, k, v, D**-0.5


def _dense_causal(q, k, v, scale):
    K = k.shape[-2]
    L = q.shape[-2]
    mask = mx.arange(K)[None, :] <= (K - L + mx.arange(L))[:, None]
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask[None, None]
    )


class TestSparseAttention:
    def test_full_coverage_matches_dense(self):
        """sink+window covering everything must reproduce dense attention."""
        q, k, v, scale = _rand_qkv()
        heads = [HeadPattern("a_shape", sink=2048, window=2048)] * 8
        out = sparse_prefill_attention(q, k, v, scale, heads, budget=1.0)
        ref = _dense_causal(q, k, v, scale)
        assert mx.abs(out - ref).max().item() < 1e-3

    def test_vslash_approximates_dense(self):
        """vertical_slash at a moderate budget stays close to dense on
        random (near-uniform-attention) inputs."""
        q, k, v, scale = _rand_qkv(K=1024, L=256)
        heads = [HeadPattern("vertical_slash", sink=64, window=128)] * 8
        out = sparse_prefill_attention(q, k, v, scale, heads, budget=0.8)
        ref = _dense_causal(q, k, v, scale)
        assert bool(mx.isfinite(out).all().item())
        assert mx.abs(out - ref).mean().item() < 0.05

    def test_a_shape_matches_numpy_reference(self):
        """Sparse a_shape output equals a numpy masked-softmax reference."""
        q, k, v, scale = _rand_qkv(K=512, L=512, Hq=4, Hkv=2, D=32)
        S, W = 32, 96
        heads = [HeadPattern("a_shape", sink=S, window=W)] * 4
        out = np.array(
            sparse_prefill_attention(q, k, v, scale, heads, budget=0.5,
                                     query_block=128),
            dtype=np.float32,
        )

        qn = np.array(q, dtype=np.float32)
        kn = np.repeat(np.array(k, dtype=np.float32), 2, axis=1)
        vn = np.repeat(np.array(v, dtype=np.float32), 2, axis=1)
        K = kn.shape[-2]
        pos = np.arange(K)  # queries occupy all K positions here (K == L)
        col = np.arange(K)
        allowed = (col[None, :] <= pos[:, None]) & (
            (col[None, :] < S) | (col[None, :] > pos[:, None] - W)
        )
        scores = qn @ kn.transpose(0, 1, 3, 2) * scale
        scores = np.where(allowed[None, None], scores, -np.inf)
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        ref = weights @ vn
        assert np.abs(out - ref).max() < 5e-3

    def test_chunked_prefill_offsets(self):
        """Later chunk (K > L) attends at correct absolute positions."""
        # Process the same 1024 keys as one 1024-query chunk vs the last
        # 512 queries only (cache offset 512); outputs for those rows match.
        q, k, v, scale = _rand_qkv(K=1024, L=1024)
        heads = [HeadPattern("a_shape", sink=64, window=256)] * 8
        full = sparse_prefill_attention(q, k, v, scale, heads, budget=0.3,
                                        query_block=256)
        tail = sparse_prefill_attention(
            q[..., 512:, :], k, v, scale, heads, budget=0.3, query_block=256
        )
        assert mx.abs(full[..., 512:, :] - tail).max().item() < 5e-3

    def test_no_double_counting(self):
        """Vertical columns overlapping sink/window must not be counted twice.

        sink=256 + window=512 already cover every causal key at K=768, so
        the vertical columns forced in by budget 1.5 are pure duplicates.
        With correct dedup the output equals dense attention exactly; any
        double counting corrupts the softmax normalization.
        """
        q, k, v, scale = _rand_qkv(K=768, L=768)
        heads = [HeadPattern("vertical_slash", sink=256, window=512)] * 8
        out = sparse_prefill_attention(q, k, v, scale, heads, budget=1.5)
        ref = _dense_causal(q, k, v, scale)
        assert mx.abs(out - ref).max().item() < 1e-2


class TestEstimator:
    def test_finds_planted_column(self):
        """A key column that every query attends to must rank in the top set."""
        mx.random.seed(1)
        D, K, L = 32, 1024, 256
        q = mx.random.normal((1, 4, L, D)).astype(mx.float16)
        k = mx.random.normal((1, 2, K, D)).astype(mx.float16) * 0.01
        v = mx.random.normal((1, 2, K, D)).astype(mx.float16)
        # Plant column 300: strongly aligned with all queries' mean direction
        q_mean = q.mean(axis=(0, 1, 2))
        k_np = np.array(k)
        k_np[:, :, 300, :] = np.array(q_mean) * 5.0
        k = mx.array(k_np).astype(mx.float16)
        cols = estimate_vertical_columns(q, k, D**-0.5, n_cols=16)
        assert cols.shape == (2, 16)
        for g in range(2):
            assert 300 in cols[g].tolist()


class TestGating:
    def setup_method(self):
        self._saved = (_STATE.enabled, dict(_STATE.patterns), _STATE.current_layer)
        _STATE.enabled = True
        _STATE.patterns = {0: [HeadPattern("a_shape", 64, 512)]}
        _STATE.current_layer = 0

    def teardown_method(self):
        _STATE.enabled, _STATE.patterns, _STATE.current_layer = (
            self._saved[0],
            self._saved[1],
            self._saved[2],
        )

    def _q(self, B=1, L=512):
        return mx.zeros((B, 8, L, 64))

    def test_applies_on_causal_prefill(self):
        assert _sparse_path_applies(self._q(), None, "causal", None)
        assert _sparse_path_applies(self._q(), None, None, None)

    def test_rejects_decode_and_short(self):
        assert not _sparse_path_applies(self._q(L=1), None, None, None)
        assert not _sparse_path_applies(self._q(L=16), None, None, None)

    def test_rejects_batched_and_masked(self):
        assert not _sparse_path_applies(self._q(B=2), None, "causal", None)
        assert not _sparse_path_applies(self._q(), None, mx.zeros((1, 1)), None)
        assert not _sparse_path_applies(self._q(), None, "causal", mx.zeros((8,)))

    def test_rejects_unconfigured_layer(self):
        _STATE.current_layer = 5
        assert not _sparse_path_applies(self._q(), None, "causal", None)

    def test_rejects_quantized_cache(self):
        class QCache:
            bits = 4

        assert not _sparse_path_applies(self._q(), QCache(), "causal", None)


class TestCalibrationFormat:
    def test_roundtrip(self, tmp_path):
        data = {
            "version": 1,
            "budget": 0.1,
            "layers": {
                "0": [{"kind": "a_shape", "sink": 64, "window": 900}],
                "3": [{"kind": "vertical_slash", "sink": 64, "window": 256}],
            },
        }
        p = tmp_path / "calib.json"
        p.write_text(json.dumps(data))
        from omlx.patches.sparse_prefill import load_calibration

        patterns = _patterns_from_calibration(load_calibration(p))
        assert patterns[0][0].kind == "a_shape"
        assert patterns[3][0].window == 256


class TestSettings:
    def test_mutual_exclusion_with_specprefill(self):
        from omlx.model_settings import ModelSettings

        with pytest.raises(ValueError):
            ModelSettings(sparse_prefill_enabled=True, specprefill_enabled=True)
        # Each alone is fine
        ModelSettings(sparse_prefill_enabled=True)
        ModelSettings(specprefill_enabled=True)
