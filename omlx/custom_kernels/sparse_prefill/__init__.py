# SPDX-License-Identifier: Apache-2.0
"""Stage-2 fused vertical-slash sparse attention kernel (JIT Metal).

Flash-style blocked design: each threadgroup handles RB consecutive query
rows of one query head (one simdgroup per row). Candidate key positions —
sink prefix, local window band (union over the row block), and per-kv-head
vertical columns — are streamed through threadgroup memory in tiles of 32,
so K/V global traffic is shared by all RB rows. Per row, an online softmax
runs with the output accumulator distributed across lanes (DH/32 dims per
lane). Per-row validity (causality, own window band, column dedup against
sink/window) is applied as a -inf score, so rows in a block may have
different allowed sets.

Built with mx.fast.metal_kernel (no native extension), per the plan's
Stage-2 strategy; packaging mirrors omlx/custom_kernels/glm_moe_dsa in
spirit (availability probe + graceful fallback) without the csrc build.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

# Query rows per threadgroup (simdgroups per threadgroup).
_ROWS_PER_BLOCK = 16
# Keys per threadgroup-memory tile (== simd width; one key per lane).
_TILE_KEYS = 32

_SOURCE = """
    const uint lane = thread_position_in_threadgroup.x;
    const uint sg = thread_position_in_threadgroup.y;   // row within block
    const uint blk = threadgroup_position_in_grid.y;
    const uint h = threadgroup_position_in_grid.z;

    const int L = q_shape[1];
    const int K = k_shape[1];
    const int Hq = q_shape[0];
    const int Hkv = k_shape[0];
    const int C = cols_shape[1];
    const int group = Hq / Hkv;
    const int kvh = (int)h / group;
    constexpr int TK = 32;           // keys per tile
    constexpr int DPL = DH / 32;     // output dims owned per lane

    const int r = (int)(blk * RB + sg);
    const bool row_valid = r < L;
    const int p = K - L + metal::min(r, L - 1);  // absolute row position
    const int p_min = K - L + (int)(blk * RB);
    const int p_max = K - L + metal::min((int)(blk * RB) + RB - 1, L - 1);

    const int S = sink[h];
    const int W = window[h];
    const float scale = scaleb[0];

    const device T* kbase = k + (size_t)kvh * K * DH;
    const device T* vbase = v + (size_t)kvh * K * DH;

    threadgroup T Ktile[TK * DH];
    threadgroup T Vtile[TK * DH];
    threadgroup int pos_tile[TK];
    threadgroup float Qs[RB * DH];

    // Each simdgroup loads its own (scaled) query row
    {
        const device T* qrow = q + ((size_t)h * L + metal::min(r, L - 1)) * DH;
        for (int d = (int)lane; d < DH; d += 32)
            Qs[sg * DH + d] = (float)qrow[d] * scale;
    }

    float m = -INFINITY;
    float l = 0.0f;
    float acc[DPL];
    for (int j = 0; j < DPL; ++j) acc[j] = 0.0f;

    const int sink_n = metal::min(S, p_max + 1);
    const int ws_min = metal::max(S, p_min - W + 1);  // window union start
    const int win_n = metal::max(0, p_max + 1 - ws_min);
    const int col_n = has_cols[h] ? C : 0;
    const int ws_row = metal::max(S, p - W + 1);      // this row's window

    const int tid = (int)(sg * 32 + lane);

    for (int phase = 0; phase < 3; ++phase) {
      const int n = phase == 0 ? sink_n : (phase == 1 ? win_n : col_n);
      for (int base = 0; base < n; base += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          const int i = base + (int)lane;
          int c = -1;
          if (i < n)
            c = phase == 0 ? i : (phase == 1 ? ws_min + i : cols[kvh * C + i]);
          pos_tile[lane] = c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < TK * DH; e += 32 * RB) {
          const int c = pos_tile[e / DH];
          Ktile[e] = (c >= 0) ? kbase[(size_t)c * DH + (e % DH)] : (T)0;
          Vtile[e] = (c >= 0) ? vbase[(size_t)c * DH + (e % DH)] : (T)0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int c = pos_tile[lane];
        bool valid = row_valid && c >= 0 && c <= p;
        if (phase == 1) valid = valid && (c >= ws_row);
        // Column dedup: allowed only where NOT covered by own sink/window
        if (phase == 2) valid = valid && (c >= S) && (c < ws_row);
        float dot = -INFINITY;
        if (valid) {
          dot = 0.0f;
          for (int d = 0; d < DH; ++d)
            dot += Qs[sg * DH + d] * (float)Ktile[lane * DH + d];
        }

        const float tile_max = metal::simd_max(dot);
        if (tile_max > -INFINITY) {
          const float m_new = metal::max(m, tile_max);
          const float corr = metal::exp(m - m_new);  // 0 when m == -inf
          const float w = (dot > -INFINITY) ? metal::exp(dot - m_new) : 0.0f;
          l = l * corr + metal::simd_sum(w);
          for (int j = 0; j < DPL; ++j) acc[j] *= corr;
          for (int kk = 0; kk < TK; ++kk) {
            const float wk = metal::simd_shuffle(w, (ushort)kk);
            if (wk != 0.0f) {
              for (int j = 0; j < DPL; ++j)
                acc[j] += wk * (float)Vtile[kk * DH + (int)lane * DPL + j];
            }
          }
          m = m_new;
        }
      }
    }

    if (row_valid && l > 0.0f) {
      device T* orow = out + ((size_t)h * L + r) * DH;
      for (int j = 0; j < DPL; ++j)
        orow[(int)lane * DPL + j] = (T)(acc[j] / l);
    }
"""

_KERNELS: Dict[Tuple, object] = {}
_DISABLED = False


def is_available() -> bool:
    return not _DISABLED and mx.default_device().type == mx.DeviceType.gpu


def supports_head_dim(d: int) -> bool:
    return d % 32 == 0 and d <= 128


def _get_kernel():
    key = "vertical_slash_attention"
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=key,
            input_names=["q", "k", "v", "cols", "sink", "window", "has_cols", "scaleb"],
            output_names=["out"],
            source=_SOURCE,
        )
        _KERNELS[key] = kernel
    return kernel


def vertical_slash_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cols: Optional[mx.array],
    sink: mx.array,
    window: mx.array,
    has_cols: mx.array,
    scale: float,
) -> mx.array:
    """Fused sparse prefill attention.

    queries: (1, Hq, L, D); keys/values: (1, Hkv, K, D), causal alignment
    with the chunk's rows in the last L key positions. cols: (Hkv, C) int32
    sorted vertical column indices (or None). sink/window/has_cols: (Hq,)
    int32 per-query-head pattern parameters. Returns (1, Hq, L, D).
    """
    global _DISABLED
    B, Hq, L, D = queries.shape
    Hkv = keys.shape[1]
    if B != 1:
        raise ValueError("vertical_slash_attention requires batch size 1")
    if not supports_head_dim(D):
        raise ValueError(f"unsupported head dim {D}")

    q = queries[0]
    k = keys[0]
    v = values[0]
    if cols is None or cols.shape[-1] == 0:
        cols = mx.zeros((Hkv, 1), dtype=mx.int32)
        has_cols = mx.zeros((Hq,), dtype=mx.int32)

    rb = _ROWS_PER_BLOCK
    n_blocks = (L + rb - 1) // rb
    kernel = _get_kernel()
    try:
        (out,) = kernel(
            inputs=[
                q,
                k,
                v,
                cols,
                sink.astype(mx.int32),
                window.astype(mx.int32),
                has_cols.astype(mx.int32),
                mx.array([scale], dtype=mx.float32),
            ],
            template=[("T", q.dtype), ("DH", D), ("RB", rb)],
            grid=(32, rb * n_blocks, Hq),
            threadgroup=(32, rb, 1),
            output_shapes=[(Hq, L, D)],
            output_dtypes=[q.dtype],
        )
    except Exception:
        _DISABLED = True
        logger.warning(
            "sparse_prefill Metal kernel failed; disabling fused path",
            exc_info=True,
        )
        raise
    return out[None]
