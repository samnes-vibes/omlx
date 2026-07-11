# SPDX-License-Identifier: Apache-2.0
"""Sink+window attention for the DFlash verify pass (long-context plan, Phase 1).

See docs/experimental/dflash2_long_context_plan.md. The verify forward normally
attends over the *entire* target KV cache, so its cost grows with context and
dflash-mlx's fused GQA fast path degrades until ``dflash_max_ctx`` kicks the
whole engine over to the BatchedEngine fallback — losing the 3-4x speedup
exactly where decode is slowest.

This module monkey-patches the module-level SDPA dispatch functions dflash-mlx
calls for the verify forward's full-attention layers —
``dflash_mlx.engine.target_qwen_gdn._gqa_reshape_sdpa`` and
``dflash_mlx.engine.target_gemma4._gemma4_full_gqa_sdpa`` — so that, while a
verify call is in flight, only the attention-sink prefix (first S positions)
and a trailing window (last W positions) of the already-cached, already-RoPE'd
keys/values are attended to. No re-rotation is needed: the gathered keys keep
whatever absolute position they were computed at, exactly like dflash-mlx's
existing draft-side window (``dflash_draft_window_size``/``dflash_draft_sink_size``).

Activation is scoped to the verify call only: ``TargetOps.verify_block`` /
``verify_tree_block`` on the Qwen-GDN and Gemma4 backends are wrapped to set a
context-local (sink, window) pair for the duration of that one forward. Draft
forwards and prefill are untouched.

Same wrap/backup/idempotency-flag/restore shape as ``dflash_lifecycle.py`` so
this can be installed and torn down alongside it.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Callable, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_active_window: contextvars.ContextVar[Optional[tuple[int, int]]] = (
    contextvars.ContextVar("dflash_verify_window", default=None)
)

# Process-wide default (sink_size, window_size); None = full verify (unchanged
# behavior). Set once from model_settings before DFlashEngine.start() loads
# the models. A None pair or either bound of it means "no windowing" for that
# call.
_configured_window: Optional[tuple[int, int]] = None

# (module, attr_name) -> original function, so restore can put it back.
_SDPA_BACKUP: dict[tuple[Any, str], Callable[..., Any]] = {}
_TARGET_OPS_BACKUP: dict[type, dict[str, Callable[..., Any]]] = {}


def configure_verify_window(
    sink_size: Optional[int], window_size: Optional[int]
) -> None:
    """Set the process-wide sink/window sizes used by verify calls.

    ``None`` for either value disables windowing (verify attends full
    context, today's behavior) — this is the default until a model's
    settings opt in.
    """
    global _configured_window
    if sink_size is None or window_size is None:
        _configured_window = None
    else:
        _configured_window = (max(0, int(sink_size)), max(0, int(window_size)))


def get_configured_window() -> Optional[tuple[int, int]]:
    return _configured_window


def _gather_sink_window(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    mask: Any,
    sink: int,
    window: int,
) -> tuple[mx.array, mx.array, Any]:
    kv_len = int(keys.shape[2])
    q_len = int(queries.shape[2])
    # Window must cover at least the verify block itself, else the newest
    # queries would be masked from their own just-written keys.
    window = max(window, q_len)
    if sink + window >= kv_len:
        # Nothing to trim — sink+window already covers full context.
        return keys, values, mask

    window_start = kv_len - window
    gather_idx = mx.concatenate(
        [mx.arange(0, sink), mx.arange(window_start, kv_len)]
    )
    g_keys = mx.take(keys, gather_idx, axis=2)
    g_values = mx.take(values, gather_idx, axis=2)

    q_pos = mx.arange(kv_len - q_len, kv_len)[:, None]
    sink_pos = mx.arange(0, sink)[None, :]
    window_pos = mx.arange(window_start, kv_len)[None, :]
    gathered_mask = mx.concatenate([sink_pos <= q_pos, window_pos <= q_pos], axis=-1)
    return g_keys, g_values, gathered_mask


def _wrap_sdpa(orig_fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        scale: float,
        mask: Any,
        cache: Optional[Any] = None,
    ) -> mx.array:
        window = _active_window.get()
        if window is None:
            return orig_fn(queries, keys, values, scale=scale, mask=mask, cache=cache)
        sink, win = window
        g_keys, g_values, g_mask = _gather_sink_window(
            queries, keys, values, mask, sink, win
        )
        return orig_fn(queries, g_keys, g_values, scale=scale, mask=g_mask, cache=cache)

    return wrapped


def _wrap_verify_method(
    cls: type, method_name: str
) -> None:
    original = getattr(cls, method_name, None)
    if original is None:
        return

    def wrapped(self, **kwargs: Any) -> Any:
        window = _configured_window
        if window is None:
            return original(self, **kwargs)
        token = _active_window.set(window)
        try:
            return original(self, **kwargs)
        finally:
            _active_window.reset(token)

    _TARGET_OPS_BACKUP.setdefault(cls, {})[method_name] = original
    setattr(cls, method_name, wrapped)


def _patch_module_sdpa(mod: Any, attr_name: str) -> bool:
    key = (mod, attr_name)
    if key in _SDPA_BACKUP:
        return False
    original = getattr(mod, attr_name)
    _SDPA_BACKUP[key] = original
    setattr(mod, attr_name, _wrap_sdpa(original))
    return True


def install_dflash_verify_window_patch() -> bool:
    """Monkey-patch dflash-mlx's verify-pass SDPA dispatch to support sink+window.

    Safe to call repeatedly (idempotent — checked via ``_SDPA_BACKUP``).
    Windowing only actually activates once ``configure_verify_window`` has
    been called with non-None sizes; until then this patch is a no-op
    passthrough. Returns True if at least one backend was patched.
    """
    patched_any = False

    try:
        from dflash_mlx.engine import target_qwen_gdn as _qwen_gdn
    except ImportError:
        logger.debug("dflash_mlx.engine.target_qwen_gdn not importable")
    else:
        if _patch_module_sdpa(_qwen_gdn, "_gqa_reshape_sdpa"):
            _wrap_verify_method(_qwen_gdn.QwenGdnTargetOps, "verify_block")
            _wrap_verify_method(_qwen_gdn.QwenGdnTargetOps, "verify_tree_block")
            patched_any = True

    try:
        from dflash_mlx.engine import target_gemma4 as _gemma4
    except ImportError:
        logger.debug("dflash_mlx.engine.target_gemma4 not importable")
    else:
        if _patch_module_sdpa(_gemma4, "_gemma4_full_gqa_sdpa"):
            _wrap_verify_method(_gemma4.Gemma4TargetOps, "verify_block")
            _wrap_verify_method(_gemma4.Gemma4TargetOps, "verify_tree_block")
            patched_any = True

    if patched_any:
        logger.debug("dflash verify-window patch installed")
    return patched_any


def restore_dflash_verify_window_patch() -> None:
    """Revert the SDPA dispatch functions and TargetOps methods to pre-patch state."""
    for (mod, attr_name), original in list(_SDPA_BACKUP.items()):
        setattr(mod, attr_name, original)
    _SDPA_BACKUP.clear()

    for cls, methods in list(_TARGET_OPS_BACKUP.items()):
        for method_name, original in methods.items():
            setattr(cls, method_name, original)
    _TARGET_OPS_BACKUP.clear()

    logger.info("dflash verify-window patch restored")
