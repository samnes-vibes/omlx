# SPDX-License-Identifier: Apache-2.0
"""Fused Metal path for verify-length (2 ≤ q_len ≤ 32) TurboQuant attention.

Two-kernel design, mirroring mlx-vlm's prefill_attention structure for the
Prod codec but for the default integer-bit MSE key codec:

  1. Multi-query MSE score kernel (here): unpack each key's packed codebook
     indices ONCE per token, loop over the R·L query rows — scores land as
     (B·H·R, L, T) fp32 without materializing fp16 keys.
  2. Causal mask + fp32 softmax as MLX ops (L ≤ 32, so scores are small).
  3. Value weighted sum via mlx-vlm's _single_tile_value_weighted_sum_kernel
     over packed int values, inverse rotation applied once at the end.

Bytes moved from device memory ≈ packed int4 + norms — the point of the
exercise. Dispatch returns None on any unsupported configuration; the
attention patch falls back to the existing dequantize+SDPA path.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import mlx.core as mx

from .reference import _apply_verify_mask

logger = logging.getLogger(__name__)

_ENABLED = True
_MAX_VERIFY_LEN = 32


def set_enabled(flag: bool) -> None:
    global _ENABLED
    _ENABLED = bool(flag)


def is_enabled() -> bool:
    return _ENABLED


@lru_cache(maxsize=None)
def _multi_query_mse_score_kernel(
    key_bits: int, repeat_count: int, num_queries: int, dims_per_lane: int
):
    """Score kernel: grid (32 lanes, R, B·H·T); unpack key once, loop L queries."""
    if key_bits <= 0 or repeat_count < 1 or num_queries < 1:
        return None

    mask = (1 << key_bits) - 1

    return mx.fast.metal_kernel(
        name=f"omlx_mq_mse_score_k{key_bits}_r{repeat_count}_l{num_queries}",
        input_names=["q_rot", "key_norms", "key_packed", "key_codebook"],
        output_names=["out"],
        source=f"""
            auto lane = thread_position_in_grid.x;
            auto ri = thread_position_in_grid.y;
            auto n = thread_position_in_grid.z;
            auto tc = key_norms_shape[2];
            auto kh = key_norms_shape[1];
            auto b = n / (kh * tc);
            auto rem = n % (kh * tc);
            auto h = rem / tc;
            auto t = rem % tc;
            if (ri >= {repeat_count}) return;

            auto kt = key_packed + ((b*kh+h)*tc+t) * KPackedWidth;
            float kn = static_cast<float>(key_norms[(b*kh+h)*tc+t]);

            // Unpack this token's key codebook values ONCE
            float kc[{dims_per_lane}];
            for (int i=0, d=lane; d < Dim; i++, d+=32) {{
                int bo = d * {key_bits};
                uint idx = (kt[bo>>5] >> (bo&31));
                if (((bo&31)+{key_bits}) > 32)
                    idx |= kt[(bo>>5)+1] << ({key_bits} - ((bo&31)+{key_bits}-32));
                kc[i] = key_codebook[idx & {mask}u];
            }}

            // Loop over L queries, reusing the unpacked key
            auto bq = (b*kh+h) * {repeat_count} + ri;
            for (int l = 0; l < {num_queries}; l++) {{
                float ps = 0.0f;
                for (int i=0, d=lane; d < Dim; i++, d+=32)
                    ps += static_cast<float>(q_rot[(bq*{num_queries}+l)*Dim+d]) * kc[i];
                float s = simd_sum(ps) * kn;
                if (lane == 0)
                    out[bq*{num_queries}*tc + l*tc + t] = s;
            }}
        """,
    )


def fused_verify_attention(
    cache,
    queries: mx.array,
    keys_state,
    values_state,
    scale: float = 1.0,
    mask=None,
    force: bool = False,
) -> Optional[mx.array]:
    """Compressed-domain attention for 2 ≤ q_len ≤ 32 over MSE-codec states.

    Returns None on unsupported configuration (fractional bits, non-MSE
    codec, q_len out of range, string masks other than "causal", Metal
    unavailable) or when outside the measured profitability region —
    caller falls back to the existing path. force=True skips the
    profitability gate (tests/benchmarks only).
    """
    if not _ENABLED:
        return None
    try:
        from mlx_vlm.turboquant import (
            TurboQuantMSEState,
            _metal_available,
            _single_tile_value_weighted_sum_kernel,
            _state_length,
            _TurboQuantMSECodec,
        )
    except ImportError:
        return None

    if not _metal_available():
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

    key_bits = cache.key_codec.bits
    val_bits = cache.value_codec.bits
    if key_bits != int(key_bits) or val_bits != int(val_bits):
        return None
    key_bits = int(key_bits)
    val_bits = int(val_bits)
    if key_bits <= 0 or val_bits <= 0:
        return None

    B, n_q_heads, L, D = queries.shape
    if not (1 < L <= _MAX_VERIFY_LEN):
        return None
    n_kv_heads = keys_state.norms.shape[1]
    if n_q_heads % n_kv_heads != 0:
        return None
    n_repeats = n_q_heads // n_kv_heads
    T = _state_length(keys_state)
    if T == 0:
        return None

    # Profitability gate (measured on M-class GPUs, scripts/kernel_bench.py):
    # the value kernel unrolls R·L accumulator registers and spills past 64,
    # and at short contexts the dequant path's fixed cost is already low.
    # Outside this region the existing dequantize+SDPA path is faster.
    if not force:
        rl = n_repeats * L
        if rl > 64:
            return None
        if T < 4096 and rl > 16:
            return None

    value_dim = cache.value_codec.dim
    dims_per_lane = (D + 31) // 32
    val_dims_per_lane = (value_dim + 31) // 32
    score_kernel = _multi_query_mse_score_kernel(
        key_bits, n_repeats, L, dims_per_lane
    )
    val_kernel = _single_tile_value_weighted_sum_kernel(
        val_bits, n_repeats * L, val_dims_per_lane
    )
    if score_kernel is None or val_kernel is None:
        return None

    dtype = queries.dtype
    grouped = (queries * scale).reshape(B, n_kv_heads, n_repeats, L, D)
    q_rot = cache.key_codec.prepare_queries(grouped).reshape(
        B * n_kv_heads * n_repeats, L, D
    )

    scores = score_kernel(
        inputs=[
            q_rot,
            keys_state.norms,
            keys_state.indices,
            cache.key_codec.codebook,
        ],
        template=[
            ("Dim", D),
            ("KPackedWidth", keys_state.indices.shape[-1]),
        ],
        grid=(32, n_repeats, B * n_kv_heads * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B * n_kv_heads * n_repeats, L, T)],
        output_dtypes=[mx.float32],
    )[0].reshape(B, n_kv_heads, n_repeats, L, T)

    scores = _apply_verify_mask(scores, mask, T, L)
    if scores is None:
        return None

    weights = mx.softmax(scores, axis=-1).reshape(
        B * n_kv_heads, n_repeats * L, T
    )

    tok_tile_size = 1024
    num_tok_tiles = (T + tok_tile_size - 1) // tok_tile_size
    out_tiled = val_kernel(
        inputs=[
            weights,
            values_state.norms,
            values_state.indices,
            cache.value_codec.codebook,
        ],
        template=[
            ("Dim", value_dim),
            ("RepeatCount", n_repeats * L),
            ("TokTileSize", tok_tile_size),
            ("DimsPerLane", val_dims_per_lane),
            ("PackedWidth", values_state.indices.shape[-1]),
        ],
        grid=(value_dim, 1, B * n_kv_heads * num_tok_tiles),
        threadgroup=(value_dim, 1, 1),
        output_shapes=[
            (B * n_kv_heads * num_tok_tiles, n_repeats * L, value_dim)
        ],
        output_dtypes=[mx.float32],
    )[0]

    out_tiled = out_tiled.reshape(
        B * n_kv_heads, num_tok_tiles, n_repeats * L, value_dim
    )
    out_rotated = (
        mx.sum(out_tiled, axis=1) if num_tok_tiles > 1 else out_tiled.squeeze(1)
    )
    out_rotated = out_rotated.reshape(B, n_kv_heads, n_repeats, L, value_dim)
    output = cache.value_codec._rotate_inverse(out_rotated)
    return output.reshape(B, n_q_heads, L, value_dim).astype(dtype)
