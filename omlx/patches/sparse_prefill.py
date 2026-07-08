# SPDX-License-Identifier: Apache-2.0
"""Draft-free dynamic sparse prefill (MInference-style).

Applies calibrated static per-head sparse attention patterns at prefill
time. Each attention head has an offline-determined pattern class:

  - a_shape:        attend to sink (first S keys) + local window (last W)
  - vertical_slash: sink + local window + top-C "vertical" key columns,
                    re-estimated per prefill chunk from the last few query
                    rows (MInference's cheap runtime estimate)

Unlike SpecPrefill this needs no draft model and drops no tokens: sparsity
is applied *inside* attention, so every token gets a KV entry and decode
runs stock full attention over a complete cache.

Stage-1 implementation (no custom Metal kernels): query-block iteration
with contiguous sink/window slices plus a per-kv-head gather of vertical
columns, computed via mx.fast.scaled_dot_product_attention with a boolean
mask. See docs/experimental/sparse_prefill_plan.md.

Activation seam mirrors turboquant_attention: monkey-patch
mlx_lm.models.base.scaled_dot_product_attention, with a per-layer tagger
wrapper (precedent: specprefill._AttentionCapture) so the patched SDPA
knows which layer's pattern config applies.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

# Number of trailing query rows used for the runtime vertical-column
# estimate (MInference uses the last 64 queries).
_N_ESTIMATE_ROWS = 64
# Query rows processed per sparse-attention block.
_QUERY_BLOCK = 1024
# Minimum q_len for the sparse path; shorter forwards (decode, speculative
# verify blocks) fall through to dense attention.
_MIN_SPARSE_QLEN = 128

DEFAULT_THRESHOLD = 8192
DEFAULT_BUDGET = 0.1

# Default location for calibration files written by omlx.sparse_calibration.
CALIBRATION_DIR = Path.home() / ".omlx" / "sparse_prefill"


@dataclass
class HeadPattern:
    """Per-head sparse pattern parameters (token counts, absolute)."""

    kind: str  # "a_shape" | "vertical_slash"
    sink: int
    window: int


class _SparsePrefillState:
    """Module-level runtime state installed by activate_sparse_prefill."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.threshold: int = DEFAULT_THRESHOLD
        self.budget: float = DEFAULT_BUDGET
        # layer_idx -> list[HeadPattern] (one per query head)
        self.patterns: Dict[int, List[HeadPattern]] = {}
        self.current_layer: Optional[int] = None
        # Stats for the admin API / debugging
        self.sparse_calls: int = 0
        self.dense_calls: int = 0
        self.keys_attended: int = 0
        self.keys_total: int = 0

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "budget": self.budget,
            "layers_configured": len(self.patterns),
            "sparse_calls": self.sparse_calls,
            "dense_calls": self.dense_calls,
            "keys_attended": self.keys_attended,
            "keys_total": self.keys_total,
            "effective_density": (
                self.keys_attended / self.keys_total if self.keys_total else None
            ),
        }


_STATE = _SparsePrefillState()
_PATCHED = False
_ORIGINAL_SDPA = None

# Preferred fast path: maskless block SDPA (_maskless_attention), which
# rides MLX's fused flash kernel. The custom Metal kernel (Stage-2,
# omlx/custom_kernels/sparse_prefill) is numerically validated but measured
# *slower* than dense SDPA on M1 (scalar dot products cannot compete with
# simdgroup-matrix flash attention) — kept for reference/tests, off by
# default. See the plan doc's Stage-2 results.
_USE_MASKLESS = True
_USE_KERNEL = False
_KERNEL_MAX_HEAD_DIM = 128


def get_stats() -> Dict[str, Any]:
    return _STATE.stats()


# ---------------------------------------------------------------------------
# Calibration file loading
# ---------------------------------------------------------------------------


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "--").replace(" ", "_")


def default_calibration_path(model_name: str) -> Path:
    return CALIBRATION_DIR / f"{sanitize_model_name(model_name)}.json"


def load_calibration(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "layers" not in data:
        raise ValueError(f"Invalid sparse-prefill calibration file: {path}")
    return data


def _patterns_from_calibration(data: Dict[str, Any]) -> Dict[int, List[HeadPattern]]:
    patterns: Dict[int, List[HeadPattern]] = {}
    for layer_key, heads in data["layers"].items():
        patterns[int(layer_key)] = [
            HeadPattern(
                kind=h["kind"],
                sink=int(h["sink"]),
                window=int(h["window"]),
            )
            for h in heads
        ]
    return patterns


# ---------------------------------------------------------------------------
# Layer tagging (so the SDPA patch knows which layer is executing)
# ---------------------------------------------------------------------------


class _LayerTagger:
    """Wraps an attention module to record its layer index in _STATE.

    Same delegation pattern as specprefill._AttentionCapture.
    """

    def __init__(self, original, layer_idx: int):
        self._original = original
        self._layer_idx = layer_idx

    def __call__(self, *args, **kwargs):
        prev = _STATE.current_layer
        _STATE.current_layer = self._layer_idx
        try:
            return self._original(*args, **kwargs)
        finally:
            _STATE.current_layer = prev

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install_layer_taggers(model) -> None:
    from .specprefill import (
        _find_attention_layers,
        _get_attn_module,
        _set_attn_module,
    )

    for layer_idx, layer in _find_attention_layers(model):
        attn = _get_attn_module(layer)
        if isinstance(attn, _LayerTagger):
            continue
        _set_attn_module(layer, _LayerTagger(attn, layer_idx))


def _remove_layer_taggers(model) -> None:
    from .specprefill import (
        _find_attention_layers,
        _get_attn_module,
        _set_attn_module,
    )

    for _layer_idx, layer in _find_attention_layers(model):
        attn = _get_attn_module(layer)
        if isinstance(attn, _LayerTagger):
            _set_attn_module(layer, attn._original)


# ---------------------------------------------------------------------------
# Stage-1 sparse attention
# ---------------------------------------------------------------------------


def estimate_vertical_columns(
    queries: mx.array,
    keys: mx.array,
    scale: float,
    n_cols: int,
    n_rows: int = _N_ESTIMATE_ROWS,
) -> mx.array:
    """Estimate the top-n_cols vertical key columns per kv head.

    Uses softmax attention of the last ``n_rows`` query rows against all
    keys, averaged over rows and over the query heads of each kv group
    (MInference's last-64-queries estimate).

    Returns (n_kv_heads, n_cols) int32 column indices, sorted ascending.
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = keys.shape[1]
    K = keys.shape[-2]
    group = n_q_heads // n_kv_heads
    n_rows = min(n_rows, L)

    q_est = queries[..., L - n_rows :, :]
    # (B, Hkv, G, n_rows, D) x (B, Hkv, 1, D, K) -> (B, Hkv, G, n_rows, K)
    q_est = q_est.reshape(B, n_kv_heads, group, n_rows, D)
    scores = (q_est @ keys[:, :, None].transpose(0, 1, 2, 4, 3)) * scale

    # Causal mask for the estimate rows (they are the last n_rows queries)
    pos = mx.arange(K - n_rows, K)  # absolute positions of estimate rows
    col = mx.arange(K)
    causal = col[None, :] <= pos[:, None]  # (n_rows, K)
    scores = mx.where(causal[None, None, None], scores, mx.array(-mx.inf))

    weights = mx.softmax(scores, axis=-1)
    col_mass = weights.mean(axis=(0, 2, 3))  # (Hkv, K)

    n_cols = min(n_cols, K)
    top = mx.argpartition(col_mass, kth=K - n_cols, axis=-1)[..., K - n_cols :]
    return mx.sort(top, axis=-1).astype(mx.int32)


def _capped_row_sum(p0: int, p1: int, cap: int) -> int:
    """Sum of min(p + 1, cap) for absolute row positions p in [p0, p1)."""
    if cap <= 0:
        return 0
    hi = min(cap, p1)  # rows with p + 1 <= cap have p < cap
    n_ramp = max(0, hi - p0)
    ramp = (p0 + 1 + hi) * n_ramp // 2 if n_ramp else 0
    return ramp + (p1 - p0 - n_ramp) * cap


def _maskless_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    head_patterns: List[HeadPattern],
    budget: float,
    query_block: int,
) -> Optional[mx.array]:
    """Maskless sparse path: per query block, run stock (flash) SDPA over
    [sink | vertical cols | window slice] with mask="causal".

    Uses the layer-max sink/window (superset of every head's own pattern,
    so recall only improves) and estimates vertical columns over the strictly
    causal zone [S, q_start - W + 1) only — every key column before the
    window slice is then valid for all rows, and the bottom-right-aligned
    "causal" mask handles the window diagonal exactly. No boolean mask is
    materialized and MLX's fused SDPA kernel does all the work.

    Returns None when preconditions fail (first rows would need sink
    masking), falling back to the Stage-1 masked path.
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = keys.shape[1]
    K = keys.shape[-2]

    S = max(h.sink for h in head_patterns)
    W = max(h.window for h in head_patterns)
    q_start = K - L
    w0_global = max(S, q_start - W + 1)
    if q_start < S or w0_global <= S:
        return None  # sink/window not fully causal or no column zone

    budget_keys = int(budget * K)
    n_cols = max(0, budget_keys - S - W)
    n_cols = min(n_cols, w0_global - S)

    if n_cols > 0 and any(h.kind == "vertical_slash" for h in head_patterns):
        # Column estimate restricted to [S, w0_global): all causal for the
        # estimate rows (last rows of the chunk), so no mask is needed.
        n_rows = min(_N_ESTIMATE_ROWS, L)
        group = n_q_heads // n_kv_heads
        q_est = queries[..., L - n_rows :, :].reshape(
            B, n_kv_heads, group, n_rows, D
        )
        k_zone = keys[..., S:w0_global, :]
        scores = (q_est @ k_zone[:, :, None].transpose(0, 1, 2, 4, 3)) * scale
        col_mass = mx.softmax(scores, axis=-1).mean(axis=(0, 2, 3))  # (Hkv, Z)
        Z = w0_global - S
        top = mx.argpartition(col_mass, kth=Z - n_cols, axis=-1)[..., Z - n_cols :]
        cols = mx.sort(top, axis=-1).astype(mx.int32) + S  # (Hkv, C)
        col_idx = cols[None, :, :, None]
        k_cols = mx.take_along_axis(keys, col_idx, axis=2)
        v_cols = mx.take_along_axis(values, col_idx, axis=2)
    else:
        n_cols = 0
        k_cols = v_cols = None

    k_sink = keys[..., :S, :]
    v_sink = values[..., :S, :]

    out_blocks = []
    for r0 in range(0, L, query_block):
        r1 = min(r0 + query_block, L)
        p0 = q_start + r0
        p1 = q_start + r1
        w0 = max(S, p0 - W + 1)

        parts_k = [k_sink]
        parts_v = [v_sink]
        if k_cols is not None:
            parts_k.append(k_cols)
            parts_v.append(v_cols)
        parts_k.append(keys[..., w0:p1, :])
        parts_v.append(values[..., w0:p1, :])

        out_b = mx.fast.scaled_dot_product_attention(
            queries[..., r0:r1, :],
            mx.concatenate(parts_k, axis=2),
            mx.concatenate(parts_v, axis=2),
            scale=scale,
            mask="causal",
        )
        out_blocks.append(out_b)
        _STATE.keys_attended += (S + n_cols + (p1 - w0)) * (r1 - r0)

    _STATE.keys_total += K * L
    return mx.concatenate(out_blocks, axis=2)


def _fused_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    head_patterns: List[HeadPattern],
    cols: Optional[mx.array],
    n_cols: int,
) -> Optional[mx.array]:
    """Stage-2 fused Metal kernel path; returns None on unavailability."""
    global _USE_KERNEL
    from ..custom_kernels import sparse_prefill as _kernels

    if not _kernels.is_available():
        return None

    B, n_q_heads, L, _D = queries.shape
    K = keys.shape[-2]
    sink_arr = mx.array([h.sink for h in head_patterns], dtype=mx.int32)
    window_arr = mx.array([h.window for h in head_patterns], dtype=mx.int32)
    has_cols = mx.array(
        [int(h.kind == "vertical_slash" and n_cols > 0) for h in head_patterns],
        dtype=mx.int32,
    )
    try:
        out = _kernels.vertical_slash_attention(
            queries, keys, values, cols, sink_arr, window_arr, has_cols, scale
        )
    except Exception:
        _USE_KERNEL = False
        logger.warning(
            "fused sparse prefill kernel failed; using Stage-1 path",
            exc_info=True,
        )
        return None

    # Analytic per-head attended-key estimate (upper bound, ignores
    # column/sink/window dedup — same coarse spirit as the Stage-1 count).
    p0 = K - L
    attended = 0
    for h in head_patterns:
        cap = h.sink + h.window + (n_cols if h.kind == "vertical_slash" else 0)
        attended += _capped_row_sum(p0, K, cap)
    # Same units as the Stage-1 count: head-averaged keys per (row, layer).
    _STATE.keys_attended += attended // n_q_heads
    _STATE.keys_total += K * L
    return out


def sparse_prefill_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    head_patterns: List[HeadPattern],
    budget: float,
    query_block: int = _QUERY_BLOCK,
) -> mx.array:
    """Stage-1 sparse attention over a causal prefill chunk.

    queries: (1, Hq, L, D); keys/values: (1, Hkv, K, D) with the chunk's
    rows occupying the last L key positions (standard causal alignment,
    valid under chunked prefill where K grows with the cache offset).
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = keys.shape[1]
    K = keys.shape[-2]
    group = n_q_heads // n_kv_heads

    sink_arr = mx.array([h.sink for h in head_patterns], dtype=mx.int32)
    window_arr = mx.array([h.window for h in head_patterns], dtype=mx.int32)
    is_vs = mx.array(
        [h.kind == "vertical_slash" for h in head_patterns], dtype=mx.bool_
    )

    sink_star = max(h.sink for h in head_patterns)
    window_star = max(h.window for h in head_patterns)

    # Vertical columns: shared estimate per kv head, sized by the largest
    # column budget among this layer's vertical_slash heads.
    budget_keys = int(budget * K)
    n_cols = 0
    for h in head_patterns:
        if h.kind == "vertical_slash":
            n_cols = max(n_cols, budget_keys - h.sink - h.window)
    n_cols = max(0, min(n_cols, K - sink_star))

    if _USE_MASKLESS:
        out = _maskless_attention(
            queries, keys, values, scale, head_patterns, budget, query_block
        )
        if out is not None:
            return out

    if n_cols > 0:
        cols = estimate_vertical_columns(queries, keys, scale, n_cols)  # (Hkv, C)
    else:
        cols = None

    if _USE_KERNEL and D % 32 == 0 and D <= _KERNEL_MAX_HEAD_DIM:
        out = _fused_attention(
            queries, keys, values, scale, head_patterns, cols, n_cols
        )
        if out is not None:
            return out

    if cols is not None:
        # (B, Hkv, C, D) gathers of the vertical columns
        col_idx = cols[None, :, :, None]
        k_cols = mx.take_along_axis(keys, col_idx, axis=2)
        v_cols = mx.take_along_axis(values, col_idx, axis=2)
        colpos_cols = cols  # (Hkv, C)
    else:
        k_cols = v_cols = None
        colpos_cols = None

    k_sink = keys[..., :sink_star, :]
    v_sink = values[..., :sink_star, :]
    colpos_sink = mx.arange(sink_star)

    out_blocks = []
    q_start = K - L  # absolute position of the first query row
    for r0 in range(0, L, query_block):
        r1 = min(r0 + query_block, L)
        lb = r1 - r0
        p0 = q_start + r0
        p1 = q_start + r1  # exclusive

        w_start = max(sink_star, p0 - window_star + 1)
        k_win = keys[..., w_start:p1, :]
        v_win = values[..., w_start:p1, :]
        colpos_win = mx.arange(w_start, p1)

        parts_k = [k_sink]
        parts_v = [v_sink]
        if k_cols is not None:
            parts_k.append(k_cols)
            parts_v.append(v_cols)
        parts_k.append(k_win)
        parts_v.append(v_win)

        k_b = mx.concatenate(parts_k, axis=2)
        v_b = mx.concatenate(parts_v, axis=2)

        # Build absolute column positions, per kv head: (Hkv, Ctot)
        if colpos_cols is not None:
            colpos = mx.concatenate(
                [
                    mx.broadcast_to(colpos_sink[None], (n_kv_heads, sink_star)),
                    colpos_cols,
                    mx.broadcast_to(
                        colpos_win[None], (n_kv_heads, colpos_win.shape[0])
                    ),
                ],
                axis=1,
            )
            # Region flags: vertical-column entries are only valid where they
            # are not already covered by sink or window (avoid double counting)
            c_tot = colpos.shape[1]
            region_cols = mx.concatenate(
                [
                    mx.zeros((sink_star,), dtype=mx.bool_),
                    mx.ones((n_cols,), dtype=mx.bool_),
                    mx.zeros((colpos_win.shape[0],), dtype=mx.bool_),
                ]
            )
        else:
            colpos = mx.broadcast_to(
                mx.concatenate([colpos_sink, colpos_win])[None],
                (n_kv_heads, sink_star + colpos_win.shape[0]),
            )
            c_tot = colpos.shape[1]
            region_cols = mx.zeros((c_tot,), dtype=mx.bool_)

        pos_r = mx.arange(p0, p1)  # (lb,) absolute row positions

        # Mask shape target: (1, Hq, lb, Ctot); build as (Hkv, G, lb, Ctot)
        cp = colpos[:, None, None, :]  # (Hkv, 1, 1, Ctot)
        rows = pos_r[None, None, :, None]  # (1, 1, lb, 1)

        causal = cp <= rows

        sink_h = sink_arr.reshape(n_kv_heads, group, 1, 1)
        window_h = window_arr.reshape(n_kv_heads, group, 1, 1)
        vs_h = is_vs.reshape(n_kv_heads, group, 1, 1)

        # base: this position lies in the head's own sink or window. The
        # shared sink*/window* slices may contain positions beyond a head's
        # own sizes — those are only reachable as vertical columns. A
        # vertical-column entry is allowed exactly when the same position
        # is NOT in base (else the physical sink/window entry covers it and
        # keeping both would double count the key in the softmax).
        base = (cp < sink_h) | (cp > rows - window_h)
        in_cols = region_cols[None, None, None, :]
        allowed = causal & mx.where(in_cols, vs_h & ~base, base)
        mask = allowed.reshape(1, n_q_heads, lb, c_tot)

        out_b = mx.fast.scaled_dot_product_attention(
            queries[..., r0:r1, :], k_b, v_b, scale=scale, mask=mask
        )
        out_blocks.append(out_b)
        _STATE.keys_attended += c_tot * lb

    _STATE.keys_total += K * L

    return mx.concatenate(out_blocks, axis=2)


# ---------------------------------------------------------------------------
# SDPA patch
# ---------------------------------------------------------------------------


def _sparse_path_applies(queries, cache, mask, sinks) -> bool:
    if not _STATE.enabled or _STATE.current_layer is None:
        return False
    if _STATE.current_layer not in _STATE.patterns:
        return False
    if sinks is not None:
        return False
    if cache is not None and hasattr(cache, "bits"):
        return False  # quantized cache path (mlx-lm QuantizedKVCache)
    if mask is not None and not (isinstance(mask, str) and mask == "causal"):
        return False  # batched/padded prefill masks: stay dense for correctness
    B, _H, L, _D = queries.shape
    if B != 1 or L < _MIN_SPARSE_QLEN:
        return False
    return True


def apply_sparse_prefill_patch() -> bool:
    """Monkey-patch mlx_lm.models.base.scaled_dot_product_attention."""
    global _PATCHED, _ORIGINAL_SDPA
    if _PATCHED:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    original_sdpa = mlx_base.scaled_dot_product_attention
    _ORIGINAL_SDPA = original_sdpa

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask,
        sinks=None,
    ) -> mx.array:
        if _sparse_path_applies(queries, cache, mask, sinks):
            # TurboQuant wraps key/value states in proxies carrying _state;
            # those need the quantized path — stay dense there.
            K = keys.shape[-2] if not hasattr(keys, "_state") else 0
            if K >= _STATE.threshold:
                try:
                    out = sparse_prefill_attention(
                        queries,
                        keys,
                        values,
                        scale,
                        _STATE.patterns[_STATE.current_layer],
                        _STATE.budget,
                    )
                    _STATE.sparse_calls += 1
                    return out
                except Exception:
                    logger.warning(
                        "sparse prefill attention failed; falling back to dense",
                        exc_info=True,
                    )
        _STATE.dense_calls += 1
        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Patch model modules that imported the symbol at import time
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("mlx_lm.models."):
            continue
        if getattr(mod, "scaled_dot_product_attention", None) is original_sdpa:
            setattr(mod, "scaled_dot_product_attention", patched_sdpa)

    _PATCHED = True
    logger.info("Sparse prefill attention patch applied")
    return True


# ---------------------------------------------------------------------------
# Activation entry point
# ---------------------------------------------------------------------------


def activate_sparse_prefill(model, model_settings, model_name: str) -> bool:
    """Load calibration, tag layers, and install the SDPA patch.

    Returns True when sparse prefill is active for this model.
    """
    calib_path = getattr(model_settings, "sparse_prefill_calibration_file", None)
    path = Path(calib_path) if calib_path else default_calibration_path(model_name)
    if not path.exists():
        logger.warning(
            "Sparse prefill enabled but no calibration file at %s — "
            "run `python -m omlx.sparse_calibration --model %s` first; "
            "prefill stays dense.",
            path,
            model_name,
        )
        return False

    data = load_calibration(path)
    _STATE.patterns = _patterns_from_calibration(data)
    _STATE.budget = float(
        getattr(model_settings, "sparse_prefill_budget", None)
        or data.get("budget", DEFAULT_BUDGET)
    )
    _STATE.threshold = int(
        getattr(model_settings, "sparse_prefill_threshold", None) or DEFAULT_THRESHOLD
    )
    _install_layer_taggers(model)
    apply_sparse_prefill_patch()
    _STATE.enabled = True
    logger.info(
        "Sparse prefill active: %d layers, budget=%.2f, threshold=%d (from %s)",
        len(_STATE.patterns),
        _STATE.budget,
        _STATE.threshold,
        path,
    )
    return True


def deactivate_sparse_prefill(model=None) -> None:
    """Disable the sparse path (patch stays installed but inert)."""
    _STATE.enabled = False
    _STATE.patterns = {}
    if model is not None:
        _remove_layer_taggers(model)
