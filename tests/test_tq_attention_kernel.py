# SPDX-License-Identifier: Apache-2.0
"""Numerics and dispatch tests for the fused TurboQuant verify-attention kernel.

Acceptance bar (see docs/experimental/fused_int4_attention_plan.md): the fused
compressed-domain path must match dequantize-then-attend on the same quantized
state up to fp accumulation order — they are mathematically identical, so we
assert tight allclose plus argmax agreement per output vector.
"""

import pytest

mx = pytest.importorskip("mlx.core")
tq = pytest.importorskip("mlx_vlm.turboquant")

from omlx.custom_kernels.tq_attention import (  # noqa: E402
    fused_verify_attention,
    reference_verify_attention,
    set_enabled,
)

METAL = tq._metal_available()


def _make_cache(B, n_kv_heads, T, D, bits=4, seed=0):
    cache = tq.TurboQuantKVCache(bits=float(bits), seed=seed)
    mx.random.seed(seed)
    keys = mx.random.normal((B, n_kv_heads, T, D)).astype(mx.float16)
    values = mx.random.normal((B, n_kv_heads, T, D)).astype(mx.float16)
    cache.update_and_fetch(keys, values)
    return cache


def _dequant_sdpa(cache, queries, scale, causal=True):
    k, v = cache.dequantize()
    B, nq, L, D = queries.shape
    T = k.shape[2]
    n_rep = nq // k.shape[1]
    k = mx.repeat(k, n_rep, axis=1)
    v = mx.repeat(v, n_rep, axis=1)
    scores = (queries.astype(mx.float32) * scale) @ k.astype(
        mx.float32
    ).transpose(0, 1, 3, 2)
    if causal:
        cols = mx.arange(T)[None, :]
        rows = (T - L) + mx.arange(L)[:, None]
        scores = mx.where(cols <= rows, scores, mx.finfo(mx.float32).min)
    w = mx.softmax(scores, axis=-1)
    return (w @ v.astype(mx.float32)).astype(queries.dtype)


@pytest.mark.skipif(not METAL, reason="Metal not available")
@pytest.mark.parametrize("L", [2, 4, 17, 32])
@pytest.mark.parametrize(
    "B,n_q,n_kv,T,D",
    [
        (1, 8, 8, 128, 128),  # MHA
        (1, 32, 4, 2048, 128),  # GQA 8x
        (2, 16, 2, 512, 64),  # batch + GQA, D=64
        (1, 8, 8, 8192, 128),  # long context
    ],
)
def test_fused_matches_dequant_sdpa(L, B, n_q, n_kv, T, D):
    cache = _make_cache(B, n_kv, T, D)
    mx.random.seed(L * 1000 + T)
    q = mx.random.normal((B, n_q, L, D)).astype(mx.float16)
    scale = D**-0.5

    fused = fused_verify_attention(
        cache, q, *cache.state, scale=scale, mask="causal", force=True
    )
    assert fused is not None, "fused path unexpectedly unsupported"

    expected = _dequant_sdpa(cache, q, scale)
    f32, e32 = fused.astype(mx.float32), expected.astype(mx.float32)
    max_dev = mx.max(mx.abs(f32 - e32)).item()
    assert max_dev < 2e-2, f"max deviation {max_dev}"
    # Top-1 identity per output vector (proxy for the greedy-token bar)
    agree = mx.mean(
        (mx.argmax(f32, axis=-1) == mx.argmax(e32, axis=-1)).astype(mx.float32)
    ).item()
    assert agree > 0.999, f"argmax agreement {agree}"


@pytest.mark.skipif(not METAL, reason="Metal not available")
def test_fused_matches_reference():
    cache = _make_cache(1, 4, 1024, 128)
    q = mx.random.normal((1, 16, 8, 128)).astype(mx.float16)
    scale = 128**-0.5
    fused = fused_verify_attention(
        cache, q, *cache.state, scale=scale, mask=None, force=True
    )
    ref = reference_verify_attention(cache, q, *cache.state, scale=scale, mask=None)
    assert fused is not None and ref is not None
    max_dev = mx.max(
        mx.abs(fused.astype(mx.float32) - ref.astype(mx.float32))
    ).item()
    assert max_dev < 2e-2


@pytest.mark.skipif(not METAL, reason="Metal not available")
def test_array_mask_additive():
    cache = _make_cache(1, 4, 256, 128)
    L, T = 4, 256
    q = mx.random.normal((1, 16, L, 128)).astype(mx.float16)
    scale = 128**-0.5
    cols = mx.arange(T)[None, :]
    rows = (T - L) + mx.arange(L)[:, None]
    addmask = mx.where(cols <= rows, 0.0, mx.finfo(mx.float32).min)
    fused = fused_verify_attention(
        cache, q, *cache.state, scale=scale, mask=addmask, force=True
    )
    causal = fused_verify_attention(
        cache, q, *cache.state, scale=scale, mask="causal", force=True
    )
    assert fused is not None and causal is not None
    assert mx.max(mx.abs(fused - causal)).item() < 1e-3


@pytest.mark.skipif(not METAL, reason="Metal not available")
def test_dispatch_fallbacks():
    cache = _make_cache(1, 4, 128, 128)
    scale = 128**-0.5
    ks, vs = cache.state

    # q_len out of range
    q1 = mx.random.normal((1, 16, 1, 128)).astype(mx.float16)
    assert fused_verify_attention(cache, q1, ks, vs, scale=scale) is None
    q40 = mx.random.normal((1, 16, 40, 128)).astype(mx.float16)
    assert fused_verify_attention(cache, q40, ks, vs, scale=scale) is None

    # unsupported string mask
    q = mx.random.normal((1, 16, 4, 128)).astype(mx.float16)
    assert fused_verify_attention(cache, q, ks, vs, scale=scale, mask="sliding") is None

    # fractional turboquant bits split into integer-bit MSE codecs
    # (key=floor, value=ceil) — still supported, must not return None
    frac = _make_cache(1, 4, 128, 128, bits=2.5)
    fks, fvs = frac.state
    assert (
        fused_verify_attention(frac, q, fks, fvs, scale=scale, force=True)
        is not None
    )

    # empty cache falls back
    empty = tq.TurboQuantKVCache(bits=4.0, seed=0)
    empty.key_codec = cache.key_codec
    empty.value_codec = cache.value_codec
    eks = tq.TurboQuantMSEState(
        ks.norms[:, :, :0], ks.indices[:, :, :0]
    )
    evs = tq.TurboQuantMSEState(
        vs.norms[:, :, :0], vs.indices[:, :, :0]
    )
    assert fused_verify_attention(empty, q, eks, evs, scale=scale) is None

    # kill switch
    set_enabled(False)
    try:
        assert fused_verify_attention(cache, q, ks, vs, scale=scale) is None
    finally:
        set_enabled(True)


@pytest.mark.skipif(not METAL, reason="Metal not available")
def test_profitability_gate():
    # R=4, T=128 (< 4096): rl = 4*L; L=4 (rl=16) dispatches, L=8 (rl=32) not
    cache = _make_cache(1, 4, 128, 128)
    ks, vs = cache.state
    scale = 128**-0.5
    q4 = mx.random.normal((1, 16, 4, 128)).astype(mx.float16)
    assert fused_verify_attention(cache, q4, ks, vs, scale=scale) is not None
    q8 = mx.random.normal((1, 16, 8, 128)).astype(mx.float16)
    assert fused_verify_attention(cache, q8, ks, vs, scale=scale) is None
    # force bypasses the gate
    assert fused_verify_attention(cache, q8, ks, vs, scale=scale, force=True) is not None
