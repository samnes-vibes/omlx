# SPDX-License-Identifier: Apache-2.0
"""Pure-MLX reference for verify-length attention over TurboQuant MSE states.

Defines the exact expected numerics of the fused Metal path in fast.py:
scores computed in the rotated compressed domain (never materializing fp16
KV), fp32 softmax, weighted sum over codebook values, inverse rotation last.

Mathematically identical to dequantize-then-attend on the same quantized
state: score(q, k̂) = q · (norm · R⁻¹ cb[idx]) = (R q) · cb[idx] · norm.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


def _apply_verify_mask(scores: mx.array, mask, total_tokens: int, q_len: int):
    """Apply causal/array mask to (B, H, R, L, T) scores. None = unsupported."""
    if mask is None or (isinstance(mask, str) and mask == "causal"):
        # Verify semantics: query l is the (T - L + l)-th token overall.
        cols = mx.arange(total_tokens)[None, :]
        rows = (total_tokens - q_len) + mx.arange(q_len)[:, None]
        causal = cols <= rows  # (L, T), broadcasts over (B, H, R)
        return mx.where(causal, scores, mx.finfo(scores.dtype).min)
    if isinstance(mask, mx.array):
        if mask.ndim == scores.ndim - 1:
            mask = mx.expand_dims(mask, axis=2)
        if mask.dtype == mx.bool_:
            return mx.where(mask, scores, mx.finfo(scores.dtype).min)
        return scores + mask.astype(scores.dtype)
    return None


def reference_verify_attention(
    cache,
    queries: mx.array,
    keys_state,
    values_state,
    scale: float = 1.0,
    mask=None,
) -> Optional[mx.array]:
    """Compressed-domain attention for q_len > 1 using only codec MLX ops.

    Args mirror TurboQuantKVCache.decode_attention. Returns None when the
    codec/state combination is unsupported (caller must fall back).
    """
    try:
        from mlx_vlm.turboquant import (
            TurboQuantMSEState,
            _state_length,
            _TurboQuantMSECodec,
        )
    except ImportError:
        return None

    keys_state = getattr(keys_state, "_state", keys_state)
    values_state = getattr(values_state, "_state", values_state)

    if not (
        isinstance(cache.key_codec, _TurboQuantMSECodec)
        and isinstance(cache.value_codec, _TurboQuantMSECodec)
        and isinstance(keys_state, TurboQuantMSEState)
        and isinstance(values_state, TurboQuantMSEState)
    ):
        return None

    B, n_q_heads, L, D = queries.shape
    n_kv_heads = keys_state.norms.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    T = _state_length(keys_state)
    if T == 0:
        return None

    grouped = (queries * scale).reshape(B, n_kv_heads, n_repeats, L, D)
    scores = cache.key_codec.score(grouped.astype(mx.float32), keys_state)
    scores = _apply_verify_mask(scores, mask, T, L)
    if scores is None:
        return None
    weights = mx.softmax(scores, axis=-1)
    output = cache.value_codec.weighted_sum(weights, values_state)
    value_dim = cache.value_codec.dim
    return output.reshape(B, n_q_heads, L, value_dim).astype(queries.dtype)
