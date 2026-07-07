# SPDX-License-Identifier: Apache-2.0
"""
Tests for CacheBlend-style KV reuse (chunking + content addressing, and the
Phase 2 RoPE-shift / cache-assembly primitives; see
docs/experimental/cacheblend_plan.md).
"""

from pathlib import Path

import pytest

from omlx.cache.kv_reuse import (
    Chunk,
    ChunkPrefillStep,
    chunk_tokens,
    chunks_with_hits,
    chunks_with_manager_hits,
    compute_content_hash,
    count_potential_hits,
    execute_chunk_prefill_plan,
    is_chunk_reuse_eligible,
    plan_chunk_prefill,
    shift_kv_rope,
)
from omlx.cache.paged_ssd_cache import (
    PagedSSDBlockMetadata,
    PagedSSDCacheIndex,
    PagedSSDCacheManager,
)
from omlx.model_settings import ModelSettings


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


class TestComputeContentHash:
    def test_deterministic(self):
        tokens = [1, 2, 3, 4, 5]
        h1 = compute_content_hash(tokens, model_name="m", cache_signature="s")
        h2 = compute_content_hash(tokens, model_name="m", cache_signature="s")
        assert h1 == h2
        assert isinstance(h1, bytes) and len(h1) == 32

    def test_different_tokens_differ(self):
        h1 = compute_content_hash([1, 2, 3], model_name="m")
        h2 = compute_content_hash([1, 2, 4], model_name="m")
        assert h1 != h2

    def test_model_name_isolates_cache(self):
        h1 = compute_content_hash([1, 2, 3], model_name="model-a")
        h2 = compute_content_hash([1, 2, 3], model_name="model-b")
        assert h1 != h2

    def test_cache_signature_isolates_cache(self):
        h1 = compute_content_hash([1, 2, 3], cache_signature="sig-a")
        h2 = compute_content_hash([1, 2, 3], cache_signature="sig-b")
        assert h1 != h2

    def test_position_independent(self):
        # Same content, no positional argument exists at all — this is the
        # whole point vs. compute_block_hash's parent-chain hash.
        tokens = tuple(range(10))
        h1 = compute_content_hash(tokens, model_name="m", cache_signature="s")
        h2 = compute_content_hash(list(tokens), model_name="m", cache_signature="s")
        assert h1 == h2

    def test_field_boundaries_unambiguous(self):
        # Adjacent variable-length fields must not collide when the split
        # point between them moves.
        h1 = compute_content_hash([1], model_name="ab", cache_signature="c")
        h2 = compute_content_hash([1], model_name="a", cache_signature="bc")
        assert h1 != h2
        h3 = compute_content_hash([1], model_name="x", cache_signature="")
        h4 = compute_content_hash([1], model_name="", cache_signature="x")
        assert h3 != h4

    def test_numpy_int_tokens_hash_like_plain_ints(self):
        # The save path and lookup path must agree on the digest even if
        # one of them passes numpy integer scalars.
        np = pytest.importorskip("numpy")
        plain = [1, 2, 3]
        h1 = compute_content_hash(plain, model_name="m", cache_signature="s")
        h2 = compute_content_hash(
            list(np.array(plain, dtype=np.int64)), model_name="m", cache_signature="s"
        )
        assert h1 == h2


class TestChunkTokens:
    def test_empty_tokens(self):
        assert chunk_tokens([]) == []

    def test_fixed_size_fallback_covers_all_tokens(self):
        tokens = list(range(600))
        chunks = chunk_tokens(tokens, min_chunk_tokens=256)
        assert [c.start for c in chunks] == [0, 256, 512]
        assert [c.end for c in chunks] == [256, 512, 600]
        assert chunks[-1].token_count == 88
        # Chunks concatenate back to the original sequence.
        assert sum((list(c.token_ids) for c in chunks), []) == tokens

    def test_fixed_size_exact_multiple(self):
        tokens = list(range(512))
        chunks = chunk_tokens(tokens, min_chunk_tokens=256)
        assert [c.start for c in chunks] == [0, 256]
        assert [c.end for c in chunks] == [256, 512]

    def test_invalid_min_chunk_tokens_raises(self):
        with pytest.raises(ValueError):
            chunk_tokens([1, 2, 3], min_chunk_tokens=0)

    def test_message_boundaries_used_when_given(self):
        tokens = list(range(100))
        chunks = chunk_tokens(tokens, message_token_offsets=[0, 30, 70])
        assert [(c.start, c.end) for c in chunks] == [(0, 30), (30, 70), (70, 100)]

    def test_message_boundaries_sanitized(self):
        tokens = list(range(100))
        # Unsorted, duplicate, out-of-range, missing leading 0.
        chunks = chunk_tokens(tokens, message_token_offsets=[50, 50, 999, -1, 20])
        assert [(c.start, c.end) for c in chunks] == [(0, 20), (20, 50), (50, 100)]

    def test_determinism(self):
        tokens = list(range(300))
        c1 = chunk_tokens(tokens, min_chunk_tokens=128, model_name="m", cache_signature="s")
        c2 = chunk_tokens(tokens, min_chunk_tokens=128, model_name="m", cache_signature="s")
        assert c1 == c2

    def test_boundary_stability_under_trailing_append(self):
        """Appending a new trailing message must not change earlier chunks'
        boundaries or content hashes — only a new trailing chunk appears."""
        base_tokens = list(range(300))
        extended_tokens = base_tokens + list(range(300, 350))

        base_chunks = chunk_tokens(base_tokens, message_token_offsets=[0, 100, 250])
        extended_chunks = chunk_tokens(
            extended_tokens, message_token_offsets=[0, 100, 250, 300]
        )

        # Every chunk that existed before the append is untouched: same
        # span, same tokens, same content hash.
        for before, after in zip(base_chunks, extended_chunks[:-1]):
            assert before == after

    def test_edited_middle_message_changes_hash_after_edit_only(self):
        """Editing one message shifts every boundary after it (expected),
        but chunks entirely before the edit are unaffected."""
        head = list(range(0, 50))
        middle_a = list(range(100, 110))  # 10 tokens
        middle_b = list(range(100, 125))  # 25 tokens — same "content id" range,
        # different length, simulating an edited reply
        tail = list(range(200, 220))

        tokens_a = head + middle_a + tail
        tokens_b = head + middle_b + tail
        offsets_a = [0, len(head), len(head) + len(middle_a)]
        offsets_b = [0, len(head), len(head) + len(middle_b)]

        chunks_a = chunk_tokens(tokens_a, message_token_offsets=offsets_a)
        chunks_b = chunk_tokens(tokens_b, message_token_offsets=offsets_b)

        # Head chunk (before the edit) is byte-identical.
        assert chunks_a[0] == chunks_b[0]
        # Middle/tail chunks differ (different span and/or content).
        assert chunks_a[1:] != chunks_b[1:]

    def test_chunk_content_hash_matches_compute_content_hash(self):
        tokens = list(range(50))
        chunks = chunk_tokens(
            tokens, min_chunk_tokens=20, model_name="m", cache_signature="s"
        )
        for c in chunks:
            assert c.content_hash == compute_content_hash(
                c.token_ids, model_name="m", cache_signature="s"
            )


class TestCountPotentialHits:
    class _FakeIndex:
        def __init__(self, hit_hashes):
            self._hit_hashes = set(hit_hashes)

        def get_by_content_hash(self, content_hash):
            return [object()] if content_hash in self._hit_hashes else []

    def test_no_hits(self):
        chunks = chunk_tokens(list(range(300)), min_chunk_tokens=100)
        would_hit, total = count_potential_hits(chunks, self._FakeIndex([]))
        assert (would_hit, total) == (0, len(chunks))

    def test_partial_hits(self):
        chunks = chunk_tokens(list(range(300)), min_chunk_tokens=100)
        hit_hashes = {chunks[0].content_hash, chunks[2].content_hash}
        would_hit, total = count_potential_hits(chunks, self._FakeIndex(hit_hashes))
        assert (would_hit, total) == (2, 3)


class TestPagedSSDBlockMetadataContentHash:
    def _make_metadata(self, **overrides):
        defaults = dict(
            block_hash=b"a" * 32,
            file_path=Path("/tmp/a.safetensors"),
            file_size=100,
            token_count=16,
            created_at=1.0,
            last_access=1.0,
            num_layers=4,
        )
        defaults.update(overrides)
        return PagedSSDBlockMetadata(**defaults)

    def test_round_trip_with_content_hash(self):
        meta = self._make_metadata(content_hash=b"c" * 32, content_position=256)
        restored = PagedSSDBlockMetadata.from_dict(meta.to_dict())
        assert restored.content_hash == meta.content_hash
        assert restored.content_position == meta.content_position

    def test_default_content_hash_is_none(self):
        meta = self._make_metadata()
        assert meta.content_hash is None
        assert meta.content_position == 0
        assert "content_hash" not in meta.to_dict()

    def test_backward_compat_missing_content_hash_key(self):
        """Old on-disk metadata (written before this field existed) must
        still load, with content_hash defaulting to None."""
        meta = self._make_metadata(content_hash=b"c" * 32, content_position=5)
        old_style_dict = meta.to_dict()
        del old_style_dict["content_hash"]
        del old_style_dict["content_position"]
        restored = PagedSSDBlockMetadata.from_dict(old_style_dict)
        assert restored.content_hash is None
        assert restored.content_position == 0


class TestPagedSSDCacheIndexContentHash:
    def _make_metadata(self, block_hash, content_hash, content_position=0):
        return PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=Path(f"/tmp/{block_hash.hex()}.safetensors"),
            file_size=100,
            token_count=16,
            created_at=1.0,
            last_access=1.0,
            num_layers=4,
            content_hash=content_hash,
            content_position=content_position,
        )

    def test_get_by_content_hash_across_positions(self):
        idx = PagedSSDCacheIndex(max_size_bytes=10_000_000)
        content_hash = b"c" * 32
        m1 = self._make_metadata(b"a" * 32, content_hash, content_position=0)
        m2 = self._make_metadata(b"b" * 32, content_hash, content_position=256)
        idx.add(m1)
        idx.add(m2)

        hits = idx.get_by_content_hash(content_hash)
        assert {h.block_hash for h in hits} == {m1.block_hash, m2.block_hash}

    def test_get_by_content_hash_orders_by_recency(self):
        idx = PagedSSDCacheIndex(max_size_bytes=10_000_000)
        content_hash = b"c" * 32
        older = self._make_metadata(b"a" * 32, content_hash)
        newer = self._make_metadata(b"b" * 32, content_hash)
        older.last_access = 1.0
        newer.last_access = 2.0
        idx.add(older)
        idx.add(newer)

        hits = idx.get_by_content_hash(content_hash)
        assert [h.block_hash for h in hits] == [newer.block_hash, older.block_hash]

    def test_remove_cleans_up_secondary_index(self):
        idx = PagedSSDCacheIndex(max_size_bytes=10_000_000)
        content_hash = b"c" * 32
        m1 = self._make_metadata(b"a" * 32, content_hash)
        m2 = self._make_metadata(b"b" * 32, content_hash)
        idx.add(m1)
        idx.add(m2)

        idx.remove(b"a" * 32)
        assert [h.block_hash for h in idx.get_by_content_hash(content_hash)] == [
            b"b" * 32
        ]

        idx.remove(b"b" * 32)
        assert idx.get_by_content_hash(content_hash) == []
        assert content_hash not in idx._content_index

    def test_no_content_hash_not_indexed(self):
        idx = PagedSSDCacheIndex(max_size_bytes=10_000_000)
        m1 = self._make_metadata(b"a" * 32, content_hash=None)
        idx.add(m1)
        assert idx._content_index == {}

    def test_unknown_content_hash_returns_empty(self):
        idx = PagedSSDCacheIndex(max_size_bytes=10_000_000)
        assert idx.get_by_content_hash(b"missing" * 4 + b"x" * 4) == []


@pytest.mark.skipif(not _has_mlx(), reason="requires MLX")
class TestPagedSSDCacheManagerContentHash:
    def test_save_block_with_content_token_ids_sets_metadata(self, tmp_path):
        import mlx.core as mx

        # would_hit_content mirrors save_block's compat-signature inputs via
        # the manager's self._expected_* fields (production callers, e.g.
        # scheduler.py, always pass these at construction); a manager
        # constructed without them would compute a different signature than
        # a save call that passes its own model_name/num_layers explicitly.
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            expected_model_name="test-model",
            expected_num_layers=1,
            expected_block_size=64,
        )
        block_hash = b"content_hash_test_block"
        cache_data = [(mx.zeros((1, 4, 8, 8)), mx.zeros((1, 4, 8, 8)))]
        token_ids = list(range(64))

        assert manager.would_hit_content(token_ids) is False

        assert manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=64,
            model_name="test-model",
            layer_cache_types=["KVCache"],
            content_token_ids=token_ids,
            content_position=128,
        )

        meta = manager._index.get(block_hash)
        assert meta is not None
        assert meta.content_hash is not None
        assert meta.content_position == 128

        assert manager.would_hit_content(token_ids) is True
        assert manager.would_hit_content(list(range(64, 128))) is False

    def test_save_block_without_content_token_ids_leaves_none(self, tmp_path):
        import mlx.core as mx

        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )
        block_hash = b"no_content_hash_block__"
        cache_data = [(mx.zeros((1, 4, 8, 8)), mx.zeros((1, 4, 8, 8)))]

        assert manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=64,
            model_name="test-model",
            layer_cache_types=["KVCache"],
        )

        meta = manager._index.get(block_hash)
        assert meta is not None
        assert meta.content_hash is None


@pytest.mark.skipif(not _has_mlx(), reason="requires MLX")
class TestShiftKVRope:
    """Phase 2: RoPE delta-shift correctness (cacheblend_plan.md)."""

    class _FakeRope:
        """Standard RoPE (base/scale), no custom freqs."""

        def __init__(self, dims, base=10000.0, scale=1.0):
            self.dims = dims
            self.base = base
            self.scale = scale

    class _FakeFreqsRope:
        """Custom-freqs RoPE variant (Llama3/Yarn/SuScaled-style)."""

        def __init__(self, dims, freqs, pre_scale=1.0):
            self.dims = dims
            self._freqs = freqs
            self.mscale = pre_scale

    def test_requires_mlx_flag_checked(self):
        import omlx.cache.kv_reuse as kv_reuse_mod

        assert kv_reuse_mod.HAS_MLX is True

    def test_zero_delta_is_identity(self):
        import mlx.core as mx

        rope = self._FakeRope(dims=64)
        k = mx.random.normal((1, 4, 5, 64))
        shifted = shift_kv_rope(k, 0, rope)
        assert mx.allclose(shifted, k, atol=1e-5)

    def test_matches_direct_encoding_standard_rope(self):
        """Shifting K encoded at p by delta must equal directly encoding
        the same raw tensor at p + delta (the composition property RoPE
        shifting relies on)."""
        import mlx.core as mx

        from omlx.patches.specprefill import manual_rope

        B, n_heads, L, dims = 1, 4, 6, 64
        raw = mx.random.normal((B, n_heads, L, dims))
        rope = self._FakeRope(dims=dims, base=10000.0, scale=1.0)

        stored_position = 100
        delta = 37
        k_stored = manual_rope(raw, mx.full((L,), float(stored_position)), dims, base=rope.base, scale=rope.scale)

        shifted = shift_kv_rope(k_stored, delta, rope)
        direct = manual_rope(
            raw, mx.full((L,), float(stored_position + delta)), dims, base=rope.base, scale=rope.scale
        )
        assert mx.allclose(shifted, direct, atol=1e-4)

    def test_matches_direct_encoding_freqs_rope_with_prescale(self):
        """Same composition property for the custom-freqs branch, including
        a non-1.0 pre_scale — shift_kv_rope must NOT re-apply pre_scale."""
        import mlx.core as mx

        from omlx.patches.specprefill import manual_rope_with_freqs

        B, n_heads, L, dims = 1, 2, 4, 32
        raw = mx.random.normal((B, n_heads, L, dims))
        # manual_rope_with_freqs computes inv_freq = 1/freqs internally, so
        # `freqs` here is the raw base**exponent, matching real RoPE modules'
        # `_freqs` attribute convention (not pre-inverted).
        freqs = 10000.0 ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        pre_scale = 1.3
        rope = self._FakeFreqsRope(dims=dims, freqs=freqs, pre_scale=pre_scale)

        stored_position = 10
        delta = -4  # chunk moved earlier in the new prompt
        k_stored = manual_rope_with_freqs(
            raw, mx.full((L,), float(stored_position)), dims, freqs, pre_scale=pre_scale
        )

        shifted = shift_kv_rope(k_stored, delta, rope)
        direct = manual_rope_with_freqs(
            raw, mx.full((L,), float(stored_position + delta)), dims, freqs, pre_scale=pre_scale
        )
        assert mx.allclose(shifted, direct, atol=1e-4)

    def test_wrong_length_delta_array_raises(self):
        import mlx.core as mx

        rope = self._FakeRope(dims=32)
        k = mx.random.normal((1, 2, 4, 32))  # L == 4
        with pytest.raises(ValueError, match="per-token delta"):
            shift_kv_rope(k, mx.array([1.0, 2.0]), rope)

    def test_per_token_delta_array(self):
        """delta may be a per-token mx.array, not just a scalar."""
        import mlx.core as mx

        from omlx.patches.specprefill import manual_rope

        dims = 32
        raw = mx.random.normal((1, 2, 3, dims))
        rope = self._FakeRope(dims=dims)
        stored_positions = mx.array([5.0, 10.0, 20.0])
        deltas = mx.array([1.0, -2.0, 3.0])

        k_stored = manual_rope(raw, stored_positions, dims, base=rope.base, scale=rope.scale)
        shifted = shift_kv_rope(k_stored, deltas, rope)
        direct = manual_rope(raw, stored_positions + deltas, dims, base=rope.base, scale=rope.scale)
        assert mx.allclose(shifted, direct, atol=1e-4)

    def test_partial_rotation_dims_less_than_head_dim(self):
        """Only the first `dims` dims are rotated; the pass-through tail
        must survive the shift untouched."""
        import mlx.core as mx

        head_dim, dims = 128, 64
        rope = self._FakeRope(dims=dims)
        k = mx.random.normal((1, 2, 4, head_dim))
        k_stored = shift_kv_rope(k, 5, rope)  # encode-from-nothing at delta=5 is just a rotation
        shifted_again = shift_kv_rope(k_stored, 3, rope)
        assert mx.allclose(shifted_again[..., dims:], k[..., dims:], atol=1e-5)


class TestChunksWithHits:
    class _FakeMetadata:
        def __init__(self, content_position):
            self.content_position = content_position

    class _FakeIndex:
        def __init__(self, hits_by_hash):
            self._hits_by_hash = hits_by_hash

        def get_by_content_hash(self, content_hash):
            return self._hits_by_hash.get(content_hash, [])

    def test_no_hits_all_none(self):
        chunks = chunk_tokens(list(range(200)), min_chunk_tokens=100)
        pairs = chunks_with_hits(chunks, self._FakeIndex({}))
        assert [hit for _, hit in pairs] == [None] * len(chunks)

    def test_hit_picks_first_match(self):
        chunks = chunk_tokens(list(range(200)), min_chunk_tokens=100)
        meta_a = self._FakeMetadata(content_position=42)
        meta_b = self._FakeMetadata(content_position=99)
        index = self._FakeIndex({chunks[0].content_hash: [meta_a, meta_b]})
        pairs = chunks_with_hits(chunks, index)
        assert pairs[0][1] is meta_a
        assert pairs[1][1] is None


class TestPlanChunkPrefill:
    class _FakeMetadata:
        def __init__(self, content_position, block_hash=b"\xab" * 32):
            self.content_position = content_position
            self.block_hash = block_hash

    def test_all_miss_produces_full_prefill_steps(self):
        chunks = chunk_tokens(list(range(300)), min_chunk_tokens=100)
        pairs = [(c, None) for c in chunks]
        steps = plan_chunk_prefill(pairs)
        assert [s.kind for s in steps] == ["full_prefill"] * 3
        for chunk, step in zip(chunks, steps):
            assert step.token_start == 0
            assert step.token_end == chunk.token_count
            assert step.stored_position is None
            assert step.rope_delta is None

    def test_all_hit_produces_recompute_then_reuse(self):
        chunks = chunk_tokens(list(range(200)), min_chunk_tokens=100)
        hits = [self._FakeMetadata(content_position=0), self._FakeMetadata(content_position=500)]
        pairs = list(zip(chunks, hits))
        steps = plan_chunk_prefill(pairs, recompute_pct=0.15)

        assert [s.kind for s in steps] == ["recompute", "reuse", "recompute", "reuse"]

        recompute_0, reuse_0, recompute_1, reuse_1 = steps
        # ceil(100 * 0.15) == 15
        assert recompute_0.token_start == 0 and recompute_0.token_end == 15
        assert reuse_0.token_start == 15 and reuse_0.token_end == 100
        # reuse_0's first token was originally stored at content_position (0)
        # + its own token_start (15) within the chunk.
        assert reuse_0.stored_position == 15
        assert reuse_0.rope_delta == reuse_0.position_start - 15
        # chunk 1 spans [100:200) in the prompt, stored originally at 500
        assert recompute_1.position_start == 100
        assert reuse_1.stored_position == 500 + 15
        assert reuse_1.rope_delta == reuse_1.position_start - (500 + 15)
        # rope_delta must be the SAME constant for every span of a hit chunk
        # (the chunk moves as a rigid block) -- this is what the bug being
        # guarded against would violate.
        assert reuse_1.rope_delta == recompute_1.chunk.start - 500

    def test_recompute_pct_capped_at_chunk_length_drops_reuse_step(self):
        chunks = chunk_tokens(list(range(10)), min_chunk_tokens=10)
        pairs = [(chunks[0], self._FakeMetadata(content_position=0))]
        steps = plan_chunk_prefill(pairs, recompute_pct=1.0)
        assert [s.kind for s in steps] == ["recompute"]
        assert steps[0].token_end == 10

    def test_recompute_pct_rounds_up_to_at_least_one_token(self):
        chunks = chunk_tokens(list(range(3)), min_chunk_tokens=3)
        pairs = [(chunks[0], self._FakeMetadata(content_position=0))]
        steps = plan_chunk_prefill(pairs, recompute_pct=0.01)
        assert steps[0].kind == "recompute"
        assert steps[0].token_end == 1  # ceil(3 * 0.01) == 1, not 0

    def test_mixed_hit_and_miss(self):
        chunks = chunk_tokens(list(range(300)), min_chunk_tokens=100)
        pairs = [
            (chunks[0], None),
            (chunks[1], self._FakeMetadata(content_position=1000)),
            (chunks[2], None),
        ]
        steps = plan_chunk_prefill(pairs, recompute_pct=0.15)
        assert [s.kind for s in steps] == ["full_prefill", "recompute", "reuse", "full_prefill"]

    def test_invalid_recompute_pct_raises(self):
        chunks = chunk_tokens(list(range(10)), min_chunk_tokens=10)
        pairs = [(chunks[0], None)]
        with pytest.raises(ValueError):
            plan_chunk_prefill(pairs, recompute_pct=0.0)
        with pytest.raises(ValueError):
            plan_chunk_prefill(pairs, recompute_pct=1.5)

    def test_step_covers_whole_chunk_end_to_end(self):
        """recompute + reuse spans must partition the chunk with no gaps
        or overlaps."""
        chunks = chunk_tokens(list(range(256)), min_chunk_tokens=256)
        pairs = [(chunks[0], self._FakeMetadata(content_position=0))]
        steps = plan_chunk_prefill(pairs, recompute_pct=0.15)
        spans = [(s.token_start, s.token_end) for s in steps]
        assert spans[0][0] == 0
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            assert prev_end == next_start
        assert spans[-1][1] == chunks[0].token_count


class TestChunkPrefillStepValidation:
    def test_invalid_kind_raises(self):
        chunk = chunk_tokens(list(range(10)), min_chunk_tokens=10)[0]
        with pytest.raises(ValueError):
            ChunkPrefillStep(kind="bogus", chunk=chunk, token_start=0, token_end=10)

    def test_empty_span_raises(self):
        chunk = chunk_tokens(list(range(10)), min_chunk_tokens=10)[0]
        with pytest.raises(ValueError):
            ChunkPrefillStep(kind="full_prefill", chunk=chunk, token_start=5, token_end=5)

    def test_span_past_chunk_end_raises(self):
        chunk = chunk_tokens(list(range(10)), min_chunk_tokens=10)[0]
        with pytest.raises(ValueError):
            ChunkPrefillStep(kind="full_prefill", chunk=chunk, token_start=0, token_end=11)

    def test_reuse_without_stored_position_raises(self):
        chunk = chunk_tokens(list(range(10)), min_chunk_tokens=10)[0]
        with pytest.raises(ValueError):
            ChunkPrefillStep(kind="reuse", chunk=chunk, token_start=0, token_end=10)


class TestChunksWithManagerHits:
    class _FakeManager:
        def __init__(self, hits_by_tokens):
            self._hits_by_tokens = hits_by_tokens

        def find_content_hash_hit(self, token_ids):
            return self._hits_by_tokens.get(tuple(token_ids))

    def test_chunks_at_block_size_and_pairs_hits(self):
        tokens = list(range(20))
        hit = object()
        manager = self._FakeManager({tuple(tokens[0:10]): hit})
        pairs = chunks_with_manager_hits(tokens, manager, block_size=10)
        assert len(pairs) == 2
        assert pairs[0][1] is hit
        assert pairs[1][1] is None


@pytest.mark.skipif(not _has_mlx(), reason="requires mlx")
class TestExecuteChunkPrefillPlan:
    """Phase 3: live splice/forward execution against a fake model + cache.

    Uses fakes rather than a real mlx-lm model: this exercises the
    executor's own bookkeeping (step ordering, offset advancement, forward
    vs. splice dispatch) which is what Phase 3 wiring adds -- RoPE-shift
    numerical correctness is already covered by TestShiftKVRope.
    """

    class _FakeRope:
        def __init__(self, dims, base=10000.0, scale=1.0):
            self.dims = dims
            self.base = base
            self.scale = scale

    class _FakeAttn:
        def __init__(self, rope):
            self.rope = rope

    class _FakeLayer:
        def __init__(self, rope):
            self.self_attn = TestExecuteChunkPrefillPlan._FakeAttn(rope)

    class _FakeKVCache:
        """Named to NOT be "KVCache" on purpose in most tests, since
        `is_chunk_reuse_eligible` is tested separately by class name and
        `execute_chunk_prefill_plan` itself doesn't re-check eligibility
        (callers must)."""

        def __init__(self, n_heads, head_dim):
            import mlx.core as mx

            self.keys = mx.zeros((1, n_heads, 0, head_dim))
            self.values = mx.zeros((1, n_heads, 0, head_dim))
            self.offset = 0

        @property
        def state(self):
            return (self.keys, self.values)

    class _FakeModel:
        """Forward pass appends a constant K/V per token -- this exercises
        offset bookkeeping and step ordering, not attention numerics."""

        def __init__(self, n_layers, n_heads, head_dim):
            self.layers = [
                TestExecuteChunkPrefillPlan._FakeLayer(
                    TestExecuteChunkPrefillPlan._FakeRope(head_dim)
                )
                for _ in range(n_layers)
            ]
            self._n_heads = n_heads
            self._head_dim = head_dim

        def __call__(self, tokens, cache):
            import mlx.core as mx

            n = tokens.shape[1]
            for c in cache:
                new_k = mx.ones((1, self._n_heads, n, self._head_dim))
                new_v = mx.ones((1, self._n_heads, n, self._head_dim))
                c.keys = mx.concatenate([c.keys, new_k], axis=2)
                c.values = mx.concatenate([c.values, new_v], axis=2)
                c.offset = c.keys.shape[2]

    class _FakeSSDManager:
        def __init__(self, blocks):
            self._blocks = blocks

        def load_block(self, block_hash):
            return self._blocks.get(block_hash)

    def test_is_chunk_reuse_eligible_by_class_name(self):
        class KVCache:
            pass

        class RotatingKVCache:
            pass

        assert is_chunk_reuse_eligible([KVCache(), KVCache()])
        assert not is_chunk_reuse_eligible([KVCache(), RotatingKVCache()])

    def test_all_miss_forwards_everything_no_reuse(self):
        model = self._FakeModel(n_layers=2, n_heads=2, head_dim=8)
        cache = [self._FakeKVCache(2, 8), self._FakeKVCache(2, 8)]
        chunks = chunk_tokens(list(range(20)), min_chunk_tokens=10)
        pairs = [(c, None) for c in chunks]
        steps = plan_chunk_prefill(pairs)

        forwarded = execute_chunk_prefill_plan(model, cache, steps, ssd_manager=None)

        assert forwarded == 20
        assert cache[0].offset == 20
        assert cache[1].offset == 20

    def test_reuse_step_splices_without_forward(self):
        import mlx.core as mx

        n_heads, head_dim = 2, 8
        model = self._FakeModel(n_layers=2, n_heads=n_heads, head_dim=head_dim)
        cache = [self._FakeKVCache(n_heads, head_dim) for _ in range(2)]

        chunks = chunk_tokens(list(range(20)), min_chunk_tokens=20)
        block_hash = b"\x11" * 32

        class _Hit:
            content_position = 100

        hit = _Hit()
        hit.block_hash = block_hash
        pairs = [(chunks[0], hit)]
        # ceil(20 * 0.5) == 10: recompute[0:10), reuse[10:20)
        steps = plan_chunk_prefill(pairs, recompute_pct=0.5)
        assert [s.kind for s in steps] == ["recompute", "reuse"]

        stored_len = chunks[0].token_count
        stored_keys = mx.ones((1, n_heads, stored_len, head_dim)) * 2.0
        stored_values = mx.ones((1, n_heads, stored_len, head_dim)) * 3.0
        blocks = {block_hash: [(stored_keys, stored_values) for _ in range(2)]}
        ssd_manager = self._FakeSSDManager(blocks)

        forwarded = execute_chunk_prefill_plan(model, cache, steps, ssd_manager)

        assert forwarded == 10  # only the recompute span went through model()
        assert cache[0].offset == 20  # 10 forwarded + 10 spliced
        # V is spliced through unchanged (no RoPE shift applies to V).
        assert mx.allclose(cache[0].values[:, :, 10:20, :], stored_values[:, :, 10:20, :])

    def test_reuse_step_missing_block_raises(self):
        n_heads, head_dim = 2, 8
        model = self._FakeModel(n_layers=1, n_heads=n_heads, head_dim=head_dim)
        cache = [self._FakeKVCache(n_heads, head_dim)]
        chunks = chunk_tokens(list(range(20)), min_chunk_tokens=20)

        class _Hit:
            content_position = 0

        hit = _Hit()
        hit.block_hash = b"\x22" * 32
        pairs = [(chunks[0], hit)]
        steps = plan_chunk_prefill(pairs, recompute_pct=0.5)
        ssd_manager = self._FakeSSDManager({})  # empty -> load_block misses

        with pytest.raises(RuntimeError):
            execute_chunk_prefill_plan(model, cache, steps, ssd_manager)


class TestModelSettingsChunkKVReuse:
    """Phase 2: chunk_kv_reuse_enabled settings validation (see
    docs/experimental/cacheblend_plan.md, mirrors ngram_spec's
    TestModelSettingsValidation pattern in tests/test_ngram_spec.py)."""

    def test_defaults_off(self):
        s = ModelSettings()
        assert s.chunk_kv_reuse_enabled is False
        assert s.chunk_kv_recompute_pct is None
        assert s.chunk_kv_min_chunk_tokens is None

    def test_roundtrip(self):
        s = ModelSettings(
            chunk_kv_reuse_enabled=True,
            chunk_kv_recompute_pct=0.25,
            chunk_kv_min_chunk_tokens=128,
        )
        restored = ModelSettings.from_dict(s.to_dict())
        assert restored.chunk_kv_reuse_enabled is True
        assert restored.chunk_kv_recompute_pct == 0.25
        assert restored.chunk_kv_min_chunk_tokens == 128

    @pytest.mark.parametrize("other", ["dflash_enabled", "turboquant_kv_enabled"])
    def test_mutually_exclusive_with_incompatible_paths(self, other):
        with pytest.raises(ValueError, match="chunk_kv_reuse_enabled"):
            ModelSettings(chunk_kv_reuse_enabled=True, **{other: True})

    def test_allowed_with_ngram_spec(self):
        # Orthogonal: chunk KV reuse is prefill-side, ngram spec is decode-side.
        s = ModelSettings(chunk_kv_reuse_enabled=True, ngram_spec_enabled=True)
        assert s.chunk_kv_reuse_enabled and s.ngram_spec_enabled

    def test_invalid_recompute_pct_raises(self):
        with pytest.raises(ValueError, match="chunk_kv_recompute_pct"):
            ModelSettings(chunk_kv_reuse_enabled=True, chunk_kv_recompute_pct=0.0)
        with pytest.raises(ValueError, match="chunk_kv_recompute_pct"):
            ModelSettings(chunk_kv_reuse_enabled=True, chunk_kv_recompute_pct=1.5)

    def test_invalid_min_chunk_tokens_raises(self):
        with pytest.raises(ValueError, match="chunk_kv_min_chunk_tokens"):
            ModelSettings(chunk_kv_reuse_enabled=True, chunk_kv_min_chunk_tokens=0)

    def test_validation_skipped_when_disabled(self):
        # Bogus values on an unrelated/disabled feature shouldn't raise.
        s = ModelSettings(chunk_kv_reuse_enabled=False, chunk_kv_recompute_pct=99.0)
        assert s.chunk_kv_recompute_pct == 99.0
