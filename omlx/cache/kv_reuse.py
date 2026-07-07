# SPDX-License-Identifier: Apache-2.0
"""
CacheBlend-style non-prefix KV reuse — chunking, content-hash, RoPE-shift
and cache-assembly primitives.

Phase 1 (see docs/experimental/cacheblend_plan.md): chunking + content
addressing only. Phase 2 adds the two pieces needed to actually reuse a
hit chunk at a new position:

  - `shift_kv_rope`: re-position an already-RoPE-encoded K tensor by a
    pure delta rotation, with no access to the original raw (pre-RoPE)
    tensor. Built on `omlx.patches.specprefill`'s manual RoPE helpers
    (imported, not duplicated — see that module for the tested base
    cases this relies on).
  - `plan_chunk_prefill`: given chunks paired with their (possible)
    content-hash hits, decides — in prompt order — which spans need a
    real forward pass (miss chunks in full; the first `recompute_pct`
    of every hit chunk, to repair cross-chunk attention) and which spans
    can be spliced in directly from shifted, stored KV. This is a pure
    planning function: it returns a step list, it does not touch a
    model or a cache. Executing the plan against a live `BatchedEngine`
    cache is Phase 3 wiring (see the plan doc's Phase 3 section).

Chunk identity is `hash(token_ids) + compat signature`, deliberately
excluding position and parent-block lineage. This is what lets a chunk be
recognized as reusable regardless of where it lands in a future prompt,
unlike `paged_cache.compute_block_hash`'s prefix-chain hash which is
position-dependent by design.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

try:
    import mlx.core as mx

    from omlx.patches.specprefill import (
        _find_attention_layers,
        _get_attn_module,
        _get_dims,
        manual_rope,
        manual_rope_with_freqs,
    )

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

_DEFAULT_MIN_CHUNK_TOKENS = 256
_DEFAULT_RECOMPUTE_PCT = 0.15


def compute_content_hash(
    token_ids: Sequence[int],
    *,
    model_name: str = "",
    cache_signature: str = "",
) -> bytes:
    """
    Position-independent content hash for a chunk of tokens.

    Unlike `paged_cache.compute_block_hash`, this takes no parent hash and
    no position: the same token sequence always hashes the same way no
    matter where it appears in a prompt. `cache_signature` should be the
    same compat-signature string already used to gate reuse elsewhere
    (`PagedSSDCacheManager`'s `_cache_compat_signature`) so a chunk saved
    under one model/cache layout is never handed back under another.

    Args:
        token_ids: Token IDs making up this chunk.
        model_name: Model name, isolates caches between different models.
        cache_signature: Compat signature (dtype/layer-types/block-size/etc).

    Returns:
        32-byte SHA-256 digest.
    """
    hasher = hashlib.sha256()
    # NUL separators keep the variable-length fields unambiguous
    # (model_name="ab" + sig="c" must not collide with "a" + "bc").
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(cache_signature.encode("utf-8"))
    hasher.update(b"\x00")
    # int() coercion: the save path and the lookup path must produce the
    # same digest even if one of them hands in numpy ints (whose str() is
    # "np.int64(5)" on numpy>=2), or every lookup silently misses.
    hasher.update(str(tuple(int(t) for t in token_ids)).encode("utf-8"))
    return hasher.digest()


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a prompt's tokens, identified by content hash."""

    start: int
    end: int
    token_ids: tuple[int, ...]
    content_hash: bytes

    @property
    def token_count(self) -> int:
        return self.end - self.start


def _sanitize_boundaries(boundaries: Sequence[int], total_tokens: int) -> list[int]:
    """Dedup, sort, clamp to [0, total_tokens), and guarantee a leading 0."""
    cleaned = sorted({b for b in boundaries if 0 <= b < total_tokens})
    if not cleaned or cleaned[0] != 0:
        cleaned.insert(0, 0)
    return cleaned


def chunk_tokens(
    tokens: Sequence[int],
    *,
    message_token_offsets: Sequence[int] | None = None,
    min_chunk_tokens: int = _DEFAULT_MIN_CHUNK_TOKENS,
    model_name: str = "",
    cache_signature: str = "",
) -> list[Chunk]:
    """
    Split a token sequence into content-addressed chunks.

    Boundary source, in priority order:
      1. `message_token_offsets` — token offsets where each chat-template
         message begins (plumbed from the request layer). Stable under
         appends: adding a new trailing message never changes the token
         offsets of earlier ones, so earlier chunks' hashes are untouched.
      2. Fallback: fixed-size blocks of `min_chunk_tokens` tokens. Stable
         under trailing appends for the same reason (only the final,
         partial chunk is affected).

    Editing an earlier message's content does shift every boundary after
    the edit (its token count changes) — that is expected, not a bug: the
    chunks after an edit are supposed to miss so they get recomputed, and
    chunks entirely before it still match verbatim.

    Args:
        tokens: Full prompt token sequence.
        message_token_offsets: Optional explicit chunk-start offsets.
        min_chunk_tokens: Fixed chunk size used when no explicit offsets
            are given.
        model_name: Passed through to `compute_content_hash`.
        cache_signature: Passed through to `compute_content_hash`.

    Returns:
        List of `Chunk`s covering `tokens` end to end, in order.
    """
    total = len(tokens)
    if total == 0:
        return []

    if message_token_offsets is not None:
        starts = _sanitize_boundaries(message_token_offsets, total)
    else:
        if min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be positive")
        starts = list(range(0, total, min_chunk_tokens))

    chunks = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else total
        chunk_token_ids = tuple(tokens[start:end])
        content_hash = compute_content_hash(
            chunk_token_ids, model_name=model_name, cache_signature=cache_signature
        )
        chunks.append(
            Chunk(start=start, end=end, token_ids=chunk_token_ids, content_hash=content_hash)
        )
    return chunks


def shift_kv_rope(keys: "mx.array", delta: "int | mx.array", rope_module: Any) -> "mx.array":
    """
    Re-position an already RoPE-encoded K tensor via a pure delta rotation.

    RoPE encodes position as a 2D rotation: K_p = R(p) @ K_raw. Rotations
    at the same frequency compose additively (R(a) @ R(b) = R(a + b)), so
    re-encoding at a new position p' needs no access to the raw (pre-RoPE)
    key at all:

        K_p' = R(p') @ K_raw = R(p' - p) @ R(p) @ K_raw = R(p' - p) @ K_p

    which is exactly `manual_rope`/`manual_rope_with_freqs` applied to the
    already-encoded `keys` with `positions = delta`. This is what lets a
    chunk's stored KV be reused at a different prompt position: load it,
    shift it by `actual_position - stored_position`, splice it in.

    Args:
        keys: Stored K tensor, shape (B, n_heads, L, head_dim), already
            RoPE-encoded at its original (stored) position(s).
        delta: `actual_position - stored_position`. Either a single
            int/float applied to every position (the common case: a whole
            chunk moves by a constant offset since it stays contiguous),
            or a per-token `mx.array` of shape (L,) for non-uniform shifts.
        rope_module: The target attention layer's `.rope` object (read
            for its dims/base/scale or custom-freqs parameterization;
            never mutated).

    Returns:
        Re-positioned K tensor, same shape as `keys`.
    """
    if not HAS_MLX:
        raise RuntimeError("shift_kv_rope requires mlx, which is not installed")

    length = keys.shape[2]
    if isinstance(delta, mx.array):
        if delta.ndim != 1 or delta.shape[0] != length:
            raise ValueError(
                f"per-token delta must have shape ({length},), got {delta.shape}"
            )
        positions = delta.astype(mx.float32)
    else:
        if delta == 0:
            return keys
        positions = mx.full((length,), float(delta))

    dims = _get_dims(rope_module)
    if hasattr(rope_module, "_freqs"):
        # pre_scale is deliberately fixed at 1.0: it was already baked into
        # `keys` the one time the chunk was originally encoded (see the
        # derivation above — re-scaling it here would double-apply it).
        return manual_rope_with_freqs(keys, positions, dims, rope_module._freqs, pre_scale=1.0)

    base = getattr(rope_module, "base", 10000.0)
    scale = getattr(rope_module, "scale", 1.0)
    return manual_rope(keys, positions, dims, base=base, scale=scale)


@dataclass(frozen=True)
class ChunkPrefillStep:
    """
    One step of an ordered chunk-prefill plan (see `plan_chunk_prefill`).

    `kind` is one of:
      - "full_prefill": chunk missed the content-hash store; run a normal
        forward over the whole chunk, at its real position.
      - "recompute": leading span of a hit chunk (`recompute_pct` of its
        tokens); run a real forward at the real position (attending to
        whatever came before, already resolved by earlier steps) to
        repair cross-chunk attention drift. Static v1 selection — see
        the "Later" section of cacheblend_plan.md for dynamic selection.
      - "reuse": trailing span of a hit chunk; splice in the stored KV
        after shifting it by `rope_delta`, no forward pass needed.
    """

    kind: str
    chunk: Chunk
    token_start: int
    token_end: int
    # Original position of THIS SPAN's first token (i.e. the hit's
    # content_position + token_start, not the chunk's content_position) --
    # see `rope_delta` below for why the token_start offset must already be
    # folded in here.
    stored_position: int | None = None
    # Content-addressed block hash this span's KV was loaded from (Phase 3
    # execution needs this to call PagedSSDCacheManager.load_block()).
    stored_block_hash: bytes | None = None

    def __post_init__(self):
        if self.kind not in ("full_prefill", "recompute", "reuse"):
            raise ValueError(f"invalid ChunkPrefillStep kind: {self.kind!r}")
        if not (0 <= self.token_start < self.token_end <= self.chunk.token_count):
            raise ValueError(
                f"invalid span [{self.token_start}:{self.token_end}) for chunk of "
                f"{self.chunk.token_count} tokens"
            )
        if self.kind == "reuse" and self.stored_position is None:
            raise ValueError("reuse steps must carry stored_position")

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.chunk.token_ids[self.token_start : self.token_end]

    @property
    def position_start(self) -> int:
        return self.chunk.start + self.token_start

    @property
    def position_end(self) -> int:
        return self.chunk.start + self.token_end

    @property
    def rope_delta(self) -> int | None:
        """`actual_position - stored_position`, for `shift_kv_rope`. None for
        steps that run a real forward pass (nothing to shift).

        Both `position_start` and `stored_position` already include the
        span's `token_start` offset within the chunk (reuse spans are always
        a chunk's tail, i.e. token_start == the recompute span's length), so
        the two offsets cancel and this is a constant shift for the whole
        span: `chunk.start - hit.content_position`. Do NOT subtract the raw
        `hit.content_position` here without adding token_start to it first,
        or the shift overshoots by `token_start` positions.
        """
        if self.stored_position is None:
            return None
        return self.position_start - self.stored_position


def chunks_with_hits(chunks: Sequence[Chunk], ssd_index) -> list[tuple[Chunk, Any | None]]:
    """
    Pair each chunk with a content-hash hit, if any.

    When multiple stored entries share a chunk's content hash (the same
    content was seen at more than one position previously), the first one
    `ssd_index` returns is used — `PagedSSDCacheIndex.get_by_content_hash`
    returns most-recently-accessed first, so the default pick is the entry
    least likely to be evicted.

    Args:
        chunks: Chunks computed for the current request.
        ssd_index: A `PagedSSDCacheIndex` (duck-typed; must expose
            `get_by_content_hash`).

    Returns:
        List of (chunk, hit_metadata_or_None) pairs, same order as `chunks`.
        `hit_metadata` exposes at least `.content_position`.
    """
    result = []
    for chunk in chunks:
        hits = ssd_index.get_by_content_hash(chunk.content_hash)
        result.append((chunk, hits[0] if hits else None))
    return result


def plan_chunk_prefill(
    chunk_hits: Sequence[tuple[Chunk, Any | None]],
    recompute_pct: float = _DEFAULT_RECOMPUTE_PCT,
) -> list[ChunkPrefillStep]:
    """
    Decide, in prompt order, how to prefill a chunked prompt given which
    chunks already hit the content-hash store.

    For a miss, the whole chunk needs a real forward pass — there is
    nothing to reuse. For a hit, the first `ceil(recompute_pct *
    chunk.token_count)` tokens (at least 1, capped at the chunk length)
    get a real forward pass to repair cross-chunk attention (the recompute
    step comes first so its causal context is whatever preceding steps
    already resolved); the remaining tail is spliced in directly from
    stored KV via `shift_kv_rope`, with no forward pass at all.

    This function only plans; it does not execute anything against a
    model or cache (see the module docstring / Phase 3 in
    cacheblend_plan.md for the execution side).

    Args:
        chunk_hits: Output of `chunks_with_hits` (or an equivalent list of
            (chunk, hit_or_None) pairs).
        recompute_pct: Fraction of each hit chunk's tokens to recompute.

    Returns:
        Ordered list of `ChunkPrefillStep`, covering every chunk end to end.
    """
    if not 0.0 < recompute_pct <= 1.0:
        raise ValueError("recompute_pct must be in (0, 1]")

    steps: list[ChunkPrefillStep] = []
    for chunk, hit in chunk_hits:
        if hit is None:
            steps.append(
                ChunkPrefillStep(
                    kind="full_prefill", chunk=chunk, token_start=0, token_end=chunk.token_count
                )
            )
            continue

        # Always >= 1 (chunks are never empty), so a hit chunk always gets
        # a recompute step, possibly covering the whole chunk.
        recompute_count = min(
            chunk.token_count, max(1, math.ceil(chunk.token_count * recompute_pct))
        )
        steps.append(
            ChunkPrefillStep(
                kind="recompute", chunk=chunk, token_start=0, token_end=recompute_count
            )
        )
        if recompute_count < chunk.token_count:
            steps.append(
                ChunkPrefillStep(
                    kind="reuse",
                    chunk=chunk,
                    token_start=recompute_count,
                    token_end=chunk.token_count,
                    # + recompute_count: this span starts partway through the
                    # chunk, so its original position is offset by the same
                    # amount (see the rope_delta docstring).
                    stored_position=hit.content_position + recompute_count,
                    stored_block_hash=getattr(hit, "block_hash", None),
                )
            )
    return steps


def count_potential_hits(chunks: Sequence[Chunk], ssd_index) -> tuple[int, int]:
    """
    Would-hit telemetry: how many of `chunks` are already present in the
    content-hash index, without loading or touching anything.

    Used by the Phase 1 store path to log a go/no-go signal (see the
    cacheblend plan's Phase 1 hit-rate gate) before any load/RoPE-shift
    logic exists.

    Args:
        chunks: Chunks computed for the current request.
        ssd_index: A `PagedSSDCacheIndex` (duck-typed to avoid a hard
            import cycle; must expose `get_by_content_hash`).

    Returns:
        (would_hit_count, total_count).
    """
    would_hit = sum(
        1 for chunk in chunks if ssd_index.get_by_content_hash(chunk.content_hash)
    )
    return would_hit, len(chunks)


def chunks_with_manager_hits(
    tokens: Sequence[int], ssd_manager: Any, block_size: int
) -> list[tuple[Chunk, Any | None]]:
    """
    Chunk `tokens` at the SSD cache's own block granularity and pair each
    chunk with a content-hash hit looked up through `ssd_manager`.

    This chunks at `block_size` (the paged-cache block size the SSD store
    actually saves under -- `PagedSSDCacheManager.save_block`'s
    `content_position`/`content_hash` are computed per physical block), NOT
    `chunk_kv_min_chunk_tokens`: content hashes are block-scoped, so looking
    up hits at any other chunk granularity would simply never match anything
    that was ever stored.

    Args:
        tokens: Token span to chunk and probe (e.g. the prefix-cache-trimmed
            remainder of a request's prompt).
        ssd_manager: A `PagedSSDCacheManager` (duck-typed; must expose
            `find_content_hash_hit(token_ids) -> PagedSSDBlockMetadata | None`,
            which recomputes the content hash using the manager's own
            model/cache-signature expectations -- callers never need to
            supply `model_name`/`cache_signature` themselves).
        block_size: SSD paged-cache block size (`config.paged_cache_block_size`).

    Returns:
        List of (chunk, hit_metadata_or_None) pairs, in prompt order.
    """
    chunks = chunk_tokens(tokens, min_chunk_tokens=block_size)
    return [(c, ssd_manager.find_content_hash_hit(list(c.token_ids))) for c in chunks]


def is_chunk_reuse_eligible(cache: Sequence[Any]) -> bool:
    """
    Whether every layer's live cache object is a plain (non-rotating)
    KVCache -- the only kind `execute_chunk_prefill_plan` supports in v1.

    Rotating/sliding-window caches (`RotatingKVCache`) only retain a trailing
    window of KV, so "splice a shifted block back in" is not well-defined
    once a layer has started evicting -- see the cacheblend plan's
    "RoPE-only positional encodings assumed" design note. Duck-typed on
    class name (rather than importing `mlx_lm.models.cache.KVCache`) so this
    stays a cheap check callers can run before deciding whether to even
    attempt a chunk-reuse plan.
    """
    return all(type(c).__name__ == "KVCache" for c in cache)


def execute_chunk_prefill_plan(
    model: Any, cache: Sequence[Any], steps: Sequence[ChunkPrefillStep], ssd_manager: Any
) -> int:
    """
    Execute a chunk-prefill plan against a live model + KVCache list.

    For "full_prefill"/"recompute" steps, runs a real forward pass over the
    step's token span (advancing `cache` the normal way). For "reuse" steps,
    loads the step's stored block from `ssd_manager`, RoPE-shifts its K by
    `step.rope_delta`, and splices the shifted K + as-is V directly onto each
    layer's cache -- no forward pass. Steps are executed in order, so a
    later step's forward pass or splice correctly attends over/positions
    after everything executed before it (this is what makes recompute-then-
    reuse-then-recompute-... work like one contiguous prefill).

    Callers MUST have already confirmed `is_chunk_reuse_eligible(cache)` --
    this function assumes every cache entry supports direct `.keys`/
    `.values`/`.offset` assignment (see `_splice_reuse_step`).

    Args:
        model: The loaded mlx-lm model (called as `model(tokens, cache=cache)`
            for forward-pass steps).
        cache: Per-layer `KVCache` list, already at the correct starting
            offset for `steps[0]` (e.g. the prefix-cache-restored cache).
        steps: Ordered plan from `plan_chunk_prefill`.
        ssd_manager: A `PagedSSDCacheManager`, used to `load_block()` reuse
            steps' stored KV.

    Returns:
        Total number of tokens actually forwarded through the model
        (full_prefill + recompute spans; reuse spans are not forwarded).

    Raises:
        RuntimeError: mlx is not installed, or a reuse step's stored block
            was evicted between planning and execution (the caller should
            catch this and fall back to an ordinary full prefill).
    """
    if not HAS_MLX:
        raise RuntimeError("execute_chunk_prefill_plan requires mlx, which is not installed")

    attn_layers = _find_attention_layers(model)
    forwarded = 0
    for step in steps:
        if step.kind == "reuse":
            _splice_reuse_step(cache, step, ssd_manager, attn_layers)
        else:
            token_arr = mx.array(step.token_ids)[None]
            model(token_arr, cache=cache)
            mx.eval([c.state for c in cache])
            forwarded += len(step.token_ids)
    return forwarded


def _splice_reuse_step(
    cache: Sequence[Any],
    step: ChunkPrefillStep,
    ssd_manager: Any,
    attn_layers: list[tuple[int, Any]],
) -> None:
    """Load, shift, and splice one "reuse" step's KV onto `cache` in place."""
    block_data = ssd_manager.load_block(step.stored_block_hash)
    if block_data is None:
        raise RuntimeError(
            "chunk-reuse block "
            f"{step.stored_block_hash.hex()[:16] if step.stored_block_hash else '?'} "
            "was evicted between planning and execution"
        )
    delta = step.rope_delta
    span_start, span_end = step.token_start, step.token_end
    for layer_idx, layer in attn_layers:
        attn = _get_attn_module(layer)
        stored_keys, stored_values = block_data[layer_idx]
        span_keys = stored_keys[:, :, span_start:span_end, :]
        span_values = stored_values[:, :, span_start:span_end, :]
        shifted_keys = shift_kv_rope(span_keys, delta, attn.rope)

        layer_cache = cache[layer_idx]
        valid_keys = layer_cache.keys[..., : layer_cache.offset, :]
        valid_values = layer_cache.values[..., : layer_cache.offset, :]
        layer_cache.keys = mx.concatenate([valid_keys, shifted_keys], axis=2)
        layer_cache.values = mx.concatenate([valid_values, span_values], axis=2)
        layer_cache.offset = layer_cache.keys.shape[2]
    mx.eval([c.state for c in cache])


# ---------------------------------------------------------------------------
# Stats (mirrors omlx.patches.ngram_spec's _TOTALS accumulator pattern)
# ---------------------------------------------------------------------------

_TOTALS_LOCK = threading.Lock()
_TOTALS: dict[str, float] = {}


def record_chunk_reuse_attempt(
    *, chunks_total: int, chunks_hit: int, tokens_reused: int, tokens_forwarded: int
) -> None:
    """Accumulate one request's chunk-reuse outcome into cumulative totals
    (admin dashboard / benchmark consumption via `get_kv_reuse_totals`)."""
    with _TOTALS_LOCK:
        _TOTALS["requests"] = _TOTALS.get("requests", 0) + 1
        _TOTALS["chunks_total"] = _TOTALS.get("chunks_total", 0) + chunks_total
        _TOTALS["chunks_hit"] = _TOTALS.get("chunks_hit", 0) + chunks_hit
        _TOTALS["tokens_reused"] = _TOTALS.get("tokens_reused", 0) + tokens_reused
        _TOTALS["tokens_forwarded"] = _TOTALS.get("tokens_forwarded", 0) + tokens_forwarded


def get_kv_reuse_totals(reset: bool = False) -> dict[str, float]:
    """Cumulative chunk-KV-reuse counters (admin stats / benchmarks)."""
    with _TOTALS_LOCK:
        snapshot = dict(_TOTALS)
        if reset:
            _TOTALS.clear()
    return snapshot
