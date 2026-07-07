# CacheBlend-Style Non-Prefix KV Reuse — Implementation Plan

Date: 2026-07-06 (Phase 3 landed 2026-07-07)
Branch: `feat/cacheblend-kv-reuse`
Status: **Phase 0-3 code-complete, default-off, not yet quality-measured on a
live model.** Phases 0-2 as before (chunking, content-addressed storage,
RoPE delta-shift, planner, settings). Phase 3 added the live execution path:
scheduler arbitration, admin UI toggle + stats, and the off-path regression
tests — see Phase 3 below for exactly what shipped and the "minimal isolated
hook" scoping decision behind it, and "What's NOT done yet" for the real gap
(the quality A/B gate needs a running model + real workload, which is outside
what a coding session can do blind). Item B2, ranked #2 in
[5x_speedup_research.md](5x_speedup_research.md) recommended order.
Companion to: [ngram_speculation_plan.md](ngram_speculation_plan.md) (template & precedent),
[CacheBlend paper](https://arxiv.org/pdf/2405.16444)

**Bug fixed while wiring Phase 3:** `ChunkPrefillStep.rope_delta` computed the
shift using the chunk's raw `content_position` instead of
`content_position + token_start` for reuse spans (which always start
partway through a chunk, at `recompute_count`). This overshot the RoPE shift
by `token_start` positions for every reuse step. No test caught it because
`TestPlanChunkPrefill` asserted the property against its own (buggy)
definition rather than independently-derived semantics. Fixed in
`ChunkPrefillStep`/`plan_chunk_prefill` (`omlx/cache/kv_reuse.py`); the fix
and a corrected/broadened test are in the same commit as Phase 3.

## Goal

Reuse precomputed per-chunk KV **regardless of position in the prompt**, recomputing only a
small selected subset (~10–15%) of tokens to repair cross-chunk attention. Targets the
workloads where oMLX's strict-prefix cache fails: RAG prompts with reordered chunks,
system-prompt + tool-schema + file combinations, agent loops that shuffle context.
Expected: 2–3x TTFT on RAG/agent prompts; near-cold-prefill elimination on multi-turn with
reordered context.

Why this second (after n-gram spec):

- **The storage layer already exists.** `omlx/cache/paged_ssd_cache.py` is a
  content-addressed per-block KV store (`PagedSSDBlockMetadata`, `PagedSSDCacheIndex`,
  block-hash keyed safetensors files). CacheBlend needs exactly this, minus the
  prefix-chain constraint on the hash.
- **The positional re-encoding primitive already exists.** `omlx/patches/specprefill.py`
  has `manual_rope` / `manual_rope_with_freqs` / `_PositionMappedRoPE` — applying stored KV
  at a new position is a RoPE shift on K, which this code already implements for sparse
  prefill's chunk selection. Reuse it; do not rewrite it.
- Quality is preserved by construction: the selective-recompute pass recomputes exactly the
  tokens whose attention deviates most from the full-prefill result (per the paper), and
  the knob (`recompute_pct`) trades TTFT vs fidelity explicitly.

## Non-goals (this branch)

- Cross-*model* or cross-*quantization* KV reuse (the SSD cache's compat-signature checks,
  `_cache_compat_signature`, stay authoritative).
- Reuse across different chunk *contents* (no fuzzy matching; exact content-hash only).
- Combining with SpecPrefill/B1 sparse prefill in the same pass (noted in "Later").
- VLM/multimodal prompts (vision features have their own cache; text-only first).
- DFlash engine path — `BatchedEngine` prefill path only.

## Design overview

```
prompt arrives
  └─ chunker: split prompt into chunks at stable boundaries
       (message boundaries from the chat template; fallback: fixed 256-token blocks)
  └─ per-chunk content hash (token ids only, NOT position → key difference vs prefix cache)
  └─ lookup each chunk in the chunk-KV store (RAM index → SSD, reusing paged_ssd_cache)
       ├─ all miss → normal prefill; store per-chunk KV with position-independent keys
       ├─ hit(s):
       │    1. load stored KV for hit chunks
       │    2. RoPE-shift K from stored position → actual position (_PositionMappedRoPE / manual_rope)
       │    3. select recompute set: first S tokens of every chunk + tokens with highest
       │       KV deviation proxy (paper: layer-1 attention deviation; v1 proxy: fixed
       │       "first 15% of each non-initial chunk" — measure, then refine)
       │    4. one prefill forward over the recompute set only, with the loaded KV as
       │       context (positions mapped; same mechanism as specprefill's selected-chunk prefill)
       │    5. splice recomputed KV rows over the loaded ones → hand cache to decode as usual
       └─ first chunk at position 0 with exact prefix match → existing prefix cache wins (cheaper)
```

Key design decisions:

1. **Chunk identity = hash(token_ids of chunk) + model-compat signature.** Position is
   deliberately excluded. Store the *original* position in metadata so the load path knows
   the required RoPE shift delta.
2. **Layered storage, reuse not fork:** add a parallel index keyed by content hash next to
   `PagedSSDCacheIndex`'s prefix-chain hash, or extend `PagedSSDBlockMetadata` with a
   `content_hash` field and a secondary lookup map. Prefer the latter — one store, two
   indexes, shared eviction/budget (`parse_size`, LRU machinery stays).
3. **Selective recompute v1 is static** (chunk-head tokens), because the paper's dynamic
   selection needs per-layer attention deviation which costs an extra partial forward.
   Phase 4 measures whether static selection at 15% holds quality; only then invest in
   dynamic selection.
4. **RoPE-only positional encodings assumed.** Models with non-RoPE or hybrid position
   schemes (gdn/ssm hybrid layers — see ngram plan's rollback-mode split) refuse the
   feature at settings-validation time. Rotating/window caches
   (`_rotating_subclass.PrefillReadyRotatingKVCache`) are excluded in v1: shifted reuse
   inside a sliding window is not well-defined.

## Phases

### Phase 0 — measurement harness first

- [x] Reuse `scripts/perf_bench.py` (generalized from the ngram-spec branch's
      `spec_bench.py`; see [speedup_results_tracker.md](speedup_results_tracker.md)) for
      the A/B harness: `--ab --setting-key chunk_kv_reuse_enabled --stats-path
      admin/api/kv-reuse/stats` once the setting and stats endpoint exist. Its 4 built-in
      scenarios (`summarize`, `code_edit`, `rag`, `freeform`) already cover RAG/agent-style
      echo-heavy prompts; add the TTFT-specific scenarios this feature needs (RAG with
      permuted chunk order, agent loop with stable head + varying tail, multi-turn with an
      edited middle message, strict-prefix control expecting ~0 gain) as new entries in its
      `SCENARIOS` dict rather than a separate script.
      Done: `--setting-key`/`--stats-path` are now generic CLI flags (default to the ngram
      values for backward compat); added `rag_permuted`, `agent_loop`, `multi_turn_edit`,
      `prefix_control` scenarios, each supporting a `variants` list so the harness cycles
      through several prompts per scenario instead of repeating one fixed prompt (needed so
      permuted/edited-turn scenarios actually exercise non-prefix reuse rather than hitting
      a byte-identical repeat).
- [ ] Record baselines on the reference machine; save results in this doc. (Deferred —
      needs a running server + model; no chunk-reuse feature exists yet to A/B against, so
      this is now a Phase 3 exit-criterion measurement instead.)

### Phase 1 — chunk store (no behavior change)

- [x] Chunker: `omlx/cache/kv_reuse.py::chunk_tokens` — message-boundary split when
      `message_token_offsets` is supplied (plumbed 2026-07-07 from the engine chat
      layer through `Request.message_token_offsets`; see the Phase 3 measurement
      entry's "message-boundary chunking plumbed end to end"),
      fixed-size (`min_chunk_tokens`, default 256) fallback otherwise. Unit tests in
      `tests/test_kv_reuse.py`: determinism, boundary stability under trailing appends,
      and the edited-middle-message case (chunks before an edit stay byte-identical;
      chunks after it change, which is expected).
- [x] `content_hash`/`content_position` fields on `PagedSSDBlockMetadata` (additive
      `.get(key, default)` load, matching the existing per-field backward-compat pattern —
      no schema-version bump needed since the on-disk safetensors layout is unchanged) and
      a secondary `content_hash -> {block_hash}` map on `PagedSSDCacheIndex`
      (`get_by_content_hash`, kept in sync by `add`/`remove`).
- [x] Store path: `PagedSSDCacheManager.save_block` takes optional `content_token_ids` +
      `content_position`, computes the content hash internally (reusing its own
      `cache_signature`, so callers never duplicate that logic) and stores it on the
      block's metadata. `BlockAwarePrefixCache.store_cache` passes each full block's
      tokens + offset through on both `save_block` call sites (write-back and
      write-through). Would-hit telemetry: `PagedSSDCacheManager.would_hit_content` +
      `BlockAwarePrefixCache._content_hash_would_hit` / `_content_hash_candidates`
      counters, exposed via `get_stats_dict()`. The <30% go/no-go read needs a live
      workload run (Phase 0's deferred baseline) — code path is in place and tested but no
      number is recorded yet.

### Phase 2 — load + RoPE shift + splice (the core)

- [x] K rotation: `shift_kv_rope(keys, delta, rope_module)` in `omlx/cache/kv_reuse.py`,
      built on `specprefill.manual_rope` / `manual_rope_with_freqs` (imported, not copied;
      mlx import guarded with the same try/except-`HAS_MLX` pattern as
      `paged_ssd_cache.py`, since `kv_reuse.py` must stay importable without mlx installed).
      Re-derives a stored, already-RoPE-encoded K at a new position via a pure delta
      rotation (`R(p') = R(p'-p) @ R(p)`, so no access to the raw pre-RoPE tensor is
      needed) — `pre_scale` is deliberately fixed at 1.0 in the shift, since it was already
      baked into the stored K once at original encoding time; re-applying it would
      double-scale. Unit tests in `tests/test_kv_reuse.py::TestShiftKVRope`: zero-delta
      identity, composition-property match against direct encoding at `p+delta` for both
      the standard base/scale RoPE branch and the custom-`_freqs` branch (incl. non-1.0
      `pre_scale`), per-token delta arrays, and partial-rotation (`dims < head_dim`)
      pass-through preservation.
- [x] Cache assembly: `plan_chunk_prefill(chunks_with_hits, recompute_pct)` in
      `omlx/cache/kv_reuse.py` — a pure planning function (`ChunkPrefillStep` list, no
      model/cache access) that decides, in prompt order, which spans get a real forward
      pass (miss chunks in full; the leading `recompute_pct` of every hit chunk, to repair
      cross-chunk attention — static v1 selection per the design overview) and which spans
      splice in directly via `shift_kv_rope` with no forward pass at all
      (`ChunkPrefillStep.rope_delta`). `chunks_with_hits(chunks, ssd_index)` pairs chunks
      with their content-hash hit (if any) as the planner's input. Executing the plan
      against a live model/cache — the part that actually mutates a `KVCache`'s
      keys/values arrays and runs the recompute forward passes — is Phase 3 (arbitration
      in the prefill path), since it needs the scheduler/`BatchedEngine` context this
      module deliberately has no dependency on. Unit tests:
      `tests/test_kv_reuse.py::TestPlanChunkPrefill` (all-miss, all-hit, mixed, pct
      rounding/capping, span partitioning with no gaps/overlaps) and
      `TestChunkPrefillStepValidation`.
- [x] Settings in `omlx/model_settings.py` following the `ngram_spec_*` pattern:
      `chunk_kv_reuse_enabled: bool = False`, `chunk_kv_recompute_pct` (default 0.15),
      `chunk_kv_min_chunk_tokens` (default 256). Compatibility validation in
      `ModelSettings.__post_init__`: refuse with `dflash_enabled` (own engine/cache) and
      `turboquant_kv_enabled` (v1 cannot RoPE-shift quantized KV without a dequant
      round-trip — noted in Later), allowed with `ngram_spec_enabled` (orthogonal: prefill
      vs decode). Rotating-cache-model refusal isn't a static settings check (it depends on
      the loaded model's architecture, not a flag combination) — deferred to the Phase 3
      compat-signature/arbitration gate alongside the existing per-architecture checks
      there. Tests in `tests/test_kv_reuse.py::TestModelSettingsChunkKVReuse`, mirroring
      `test_ngram_spec.py::TestModelSettingsValidation`.
- [ ] Admin UI toggle under Experimental Features; `requires_reload` on change. **Deferred
      to Phase 3, deliberately**: the setting currently only gates validation — nothing
      reads it at prefill time — so a toggle would silently do nothing when switched on.
      Land it together with the Phase 3 arbitration call site so the switch is truthful the
      moment it appears.

### Phase 3 — end-to-end wiring & correctness gate

**Scoping decision (asked up front, before writing this): "minimal isolated
hook" vs. full integration into the multi-step chunked-prefill state
machine.** Chose minimal: a single new branch at the request-dispatch point
that only engages for requests that would already take the single-shot
external-prefill path; the multi-step `_begin_prefill`/`_step_prefill_chunk`/
`_advance_chunked_prefills` state machine (used for very long prompts spread
across multiple `step()` calls) is completely untouched. This means chunk
reuse currently only fires for prompts short enough to prefill in one
synchronous burst — a real v1 limitation, not an oversight; extending it to
the chunked state machine is future work (see below).

- [x] Arbitration in the prefill path: `Scheduler._do_prefill_with_chunk_reuse`
      (`omlx/scheduler.py`, right after `_do_external_prefill`), called instead of
      `_do_external_prefill` at the "Normal (non-chunked) full prefill path" dispatch site
      when `scheduler._chunk_kv_reuse_enabled` is set (propagated from
      `model_settings.chunk_kv_reuse_enabled` in `BatchedEngine.start()`, mirroring the
      TurboQuant/ngram-spec propagation pattern). Every early-out (VLM request, TurboQuant
      enabled, no SSD cache manager, single-token prefill, ineligible cache, zero
      content-hash hits, or any exception) delegates to the **unmodified**
      `_do_external_prefill` with the **original** `existing_cache` — never a partially
      mutated working copy. To make that guarantee hold even when a plan fails partway
      through splicing, `_do_prefill_with_chunk_reuse` operates on shallow-copied per-layer
      cache objects (`copy.copy(c) for c in existing_cache`) when a restored cache was
      passed in, since `execute_chunk_prefill_plan` reassigns `.keys`/`.values`/`.offset` on
      the cache objects it's given — without the copy, a failure after step 2 of a 4-step
      plan would leave the caller's own `existing_cache` half-spliced. This is the actual
      "prefix cache stays byte-identical when the flag is off" merge gate; it's now covered
      by a dedicated regression suite (`tests/test_scheduler.py::TestChunkKVReusePrefillDispatch`,
      9 tests) that exercises every early-out and the partial-failure/copy-safety property
      directly, without needing a real model or live GPU inference.
- [x] Execution primitive: `execute_chunk_prefill_plan(model, cache, steps, ssd_manager)`
      (`omlx/cache/kv_reuse.py`) — walks a `plan_chunk_prefill` plan in order, running a real
      `model(tokens, cache=cache)` forward for `full_prefill`/`recompute` spans and, for
      `reuse` spans, loading the stored block via `ssd_manager.load_block(block_hash)`,
      RoPE-shifting K via the existing `shift_kv_rope`/`specprefill._find_attention_layers`
      machinery, and splicing shifted-K + as-is-V directly onto each layer's cache
      (`layer_cache.keys = mx.concatenate(...)`, no forward pass) — the same
      direct-field-assignment pattern `prefix_cache.py`'s own splice code already uses.
      `is_chunk_reuse_eligible(cache)` gates out any request with a rotating/sliding-window
      cache layer (duck-typed on class name) before a plan is even attempted. Covered by
      `tests/test_kv_reuse.py::TestExecuteChunkPrefillPlan` using a fake model + fake KVCache
      (exercises step ordering / offset bookkeeping / forward-vs-splice dispatch; RoPE
      shift numerical correctness is `TestShiftKVRope`'s job, already covered in Phase 2).
- [x] SSD-lookup granularity fix: chunk hashes must be computed at the SSD cache's own
      **block** granularity (`config.paged_cache_block_size`), not
      `chunk_kv_min_chunk_tokens` — `PagedSSDCacheManager.save_block` hashes per physical
      block, so a lookup chunked any other way would simply never match anything ever
      stored. Added `PagedSSDCacheManager.find_content_hash_hit()` (mirrors the existing
      `would_hit_content`, but returns the metadata — block_hash + content_position — needed
      to actually load and splice, not just a bool) and `kv_reuse.chunks_with_manager_hits()`
      to pair chunks-at-block-size with manager hits.
- [x] Admin UI toggle (`chunk_kv_reuse_enabled`, `chunk_kv_recompute_pct`,
      `chunk_kv_min_chunk_tokens` — the settings/request-schema/conflict-validation/
      requires-reload glue in `omlx/admin/routes.py`, the toggle block in
      `_modal_model_settings.html`, and the field wiring in `dashboard.js`, all mirroring the
      n-gram-spec precedent) and `GET /api/kv-reuse/stats` (mirrors
      `/api/ngram-spec/stats`; backed by `kv_reuse.record_chunk_reuse_attempt`/
      `get_kv_reuse_totals`, a module-level counters dict following the same pattern as
      `patches/ngram_spec.py`'s `_TOTALS`). i18n strings added to `en.json` only — the other
      8 locale files fall back to showing the raw translation key (verified: `t()` returns
      `locale.get(key, key)`, so this degrades to an untranslated label rather than an error)
      until someone with those languages fills them in.
- [ ] Quality A/B (the real merge gate before flipping the default): scenario suite at
      temp=0, feature on vs off. Output will NOT be token-identical (recompute is
      approximate) — instead measure task-level fidelity: exact-match rate on extractive RAG
      answers, plus manual diff review. Define the acceptance bar up front: ≥95%
      answer-level match at recompute_pct=0.15, else raise pct until met and report the TTFT
      cost. **Not done** — needs a downloaded model and a live server run
      (`scripts/perf_bench.py --setting-key chunk_kv_reuse_enabled --stats-path
      admin/api/kv-reuse/stats`), which is a benchmarking session, not something achievable
      inside this coding session. This is the one item standing between "code complete" and
      "safe to enable by default."
      **Attempted 2026-07-07:** started a live server (`omlx serve --paged-ssd-cache-dir
      ...`) against the two models already present in the local HF cache —
      `mlx-community/Qwen3.5-0.8B-MLX-4bit` and `mlx-community/gemma-4-e2b-it-4bit` — and
      both are ineligible before any A/B is meaningful, though **not because they're
      VLM-tagged** (that was an initial misdiagnosis, corrected same day): the
      `Scheduler._do_prefill_with_chunk_reuse` VLM early-out only fires when a request
      actually carries image/video embeds (`request.vlm_inputs_embeds is not None`) — a
      text-only chat request to either model would sail past that check. The real
      blocker is `is_chunk_reuse_eligible()` (`omlx/cache/kv_reuse.py:456`), which
      requires every layer's cache object to be a plain `"KVCache"`:
      - Qwen3.5's `config.json` shows hybrid `layer_types` (periodic `linear_attention`
        layers, `full_attention_interval: 4`) → non-`KVCache` cache class for those
        layers.
      - Gemma-4's `text_config.layer_types` is mostly `sliding_attention` with occasional
        `full_attention` → `RotatingKVCache`-family layers.
      Both fail the plain-KVCache check by design (this is the documented "Rotating/
      window caches ... excluded in v1" non-goal, working as intended, not a new gap).
      Broader implication: this isn't a multimodality problem, it's an attention-
      architecture problem — sliding-window and hybrid-linear-attention models (which
      skew towards the same small/efficient models people run locally, and often
      happen to be multimodal too, hence the original mix-up) are out of scope for v1
      regardless of the VLM flag. Plain dense-attention causal LMs (Llama, Mistral, most
      Qwen2/3-dense, Phi) remain eligible. Conclusion: the quality gate still needs a
      plain text-only, pure-RoPE, dense-attention causal LM (e.g. a
      Llama-3.2-Instruct-4bit-class model) downloaded specifically for this measurement —
      neither model already on disk works. Still blocked; no fidelity numbers exist yet.
      **Measured 2026-07-07** on `mlx-community/Llama-3.2-1B-Instruct-4bit` (dense
      attention, plain `KVCache` all 16 layers — eligible, confirming the model-class
      analysis above). Two-part result:
      1. **`perf_bench.py --ab` measures nothing for this feature: 0 hits across all 4
         scenarios, empty stats, TTFT unchanged.** Two independent root causes, both
         real findings, not harness bugs:
         - **`content_hash` is never persisted.** `save_block` computes it and sets it
           on the in-RAM `PagedSSDBlockMetadata`, but the safetensors file-metadata dict
           (`paged_ssd_cache.py`, the `metadata = {...}` block in `save_block`) omits
           `content_hash`/`content_position`, so `_scan_existing_files` →
           `_read_file_metadata` rebuilds the index without them after every model
           reload. Since flipping the setting auto-unloads/reloads the model, the `--ab`
           flow structurally cannot see cross-pass hits. (Verified: blocks on disk from
           the OFF pass have no `content_hash` key in their metadata.) Fix: add the two
           fields to the file metadata dict — `to_dict`/`from_dict` on the metadata
           class already handle them.
           **Fixed 2026-07-07:** both fields now written in `save_block`'s file
           metadata and parsed back in `_read_file_metadata` (hex round-trip,
           tolerant of missing/garbled values). Regression test
           `tests/test_paged_ssd_cache.py::TestPagedSSDCacheManagerWithMLX::
           test_content_hash_survives_index_rescan` (save → fresh manager over the
           same dir → `find_content_hash_hit` still hits; verified to fail without
           the fix).
         - **Block-alignment sensitivity.** Lookup chunks the new prompt at fixed
           `paged_cache_block_size` (256) boundaries from token 0; a stored block only
           hits if the shared content sits at the *same* 256-token phase in both
           prompts. Natural-text permutations (the bench scenarios, and real RAG
           prompts) essentially never satisfy this, so hits require either
           block-aligned chunk padding or the planned message-boundary chunking
           (`message_token_offsets`, still unplumbed).
           **Fixed 2026-07-07 — message-boundary chunking plumbed end to end:**
           - *Offsets computed at the chat layer:*
             `BatchedEngine._compute_message_token_offsets` renders each
             `messages[:i]` prefix with the same template kwargs (minus the
             generation prompt) and double-validates every boundary — the rendered
             prefix must be a **string** prefix of the full prompt AND its token
             ids a **token** prefix of the full prompt's ids (BPE merges across a
             boundary would otherwise misalign the offset silently). Invalid
             boundaries are dropped, not guessed. Only computed when
             `chunk_kv_reuse_enabled` (it costs one full-prompt encode plus a
             render+encode per message); wired in both `chat()` and
             `stream_chat()`.
           - *Plumbing:* new `Request.message_token_offsets` field, carried through
             `BatchedEngine.generate`/`stream_generate` →
             `AsyncEngineCore.add_request` → `Request`.
           - *Store side:* `BlockAwarePrefixCache._store_message_content_chunks`
             (called from `store_cache` when offsets are passed, which the
             scheduler's `_async_store_cache_worker` only does when the feature is
             on) saves each message-aligned chunk as its OWN content-addressed SSD
             entry keyed by `kv_reuse.content_chunk_block_hash` (domain-separated
             from prefix-chain hashes; identical content dedupes via the existing
             has-block short-circuit). Guards: plain-KVCache stacks only, chunk KV
             fully present in the extracted cache (`_get_cache_seq_len`), and
             chunks shorter than `chunk_kv_min_chunk_tokens` skipped (propagated to
             the scheduler as `_chunk_kv_min_chunk_tokens`). The per-physical-block
             content hashes from Phase 1 still get written — aligned prompts can
             still hit them — but message chunks are what real prompts hit.
           - *Lookup side:* `_do_prefill_with_chunk_reuse` shifts the request's
             absolute offsets onto the prefix-cache-trimmed remainder's coordinates
             (dropping boundaries inside the cached prefix) and passes them to
             `chunks_with_manager_hits`, which now chunks at message boundaries
             when offsets are given, fixed block size otherwise.
           - *Bug found & fixed while wiring:* `_splice_reuse_step` applied the
             plan-relative `rope_delta` directly, ignoring the restored prefix
             length when `existing_cache` came from a prefix-cache hit — every
             spliced K under-shifted by exactly the cached-prefix token count.
             (Invisible in the earlier fidelity probe because those requests were
             built to defeat the prefix cache, so base offset was always 0.)
             `execute_chunk_prefill_plan` now captures the cache's entry offset and
             folds it into the splice delta; covered by
             `test_reuse_splice_folds_restored_prefix_into_rope_delta`.
           - *Tests:* `tests/test_kv_reuse.py` (`TestContentChunkBlockHash`,
             message-offset cases in `TestChunksWithManagerHits`,
             `TestStoreMessageContentChunks` — store→lookup round-trip at a
             different position, min-tokens filter, non-KVCache refusal, seq-len
             guard, dedupe), `tests/test_batched_engine.py::
             TestComputeMessageTokenOffsets` (offset correctness, non-prefix
             template render dropped, tokenizer-error safety),
             `tests/test_scheduler.py::TestChunkKVReusePrefillDispatch::
             test_message_offsets_translated_to_remainder_coordinates`. Full suite
             green (6225 passed).
      2. **With block-aligned prompts (controlled test, same session, no reload),
         the full pipeline works end-to-end.** Synthetic 1281-token prompts = 255-token
         head + 4×256-token content chunks, permuted between requests with differing
         heads (prefix cache defeated). Reuse fired: 4/5 chunks hit per request,
         ~868 tokens spliced vs ~170 forwarded per request at recompute_pct=0.15.
         **Fidelity: 4/4 extractive-QA outputs byte-identical to feature-off at
         temp=0** (small probe, but the ≥95% answer-match bar is met at 100% on it).
         **Perf on this 1B model: reuse is a net loss** — ~830–1080 ms wall vs
         ~420–490 ms plain warm prefill for the same ~1280-token prompts. SSD load +
         RoPE shift + several small per-span forwards lose to one fast dense prefill
         at this model size, exactly the "Load+shift slower than recompute on
         fast-prefill models / short chunks" risk. The Phase 4 min-prompt/min-model
         auto-disable threshold is not optional; benefit needs re-measuring on a
         larger model and longer prompts.
- [ ] Concurrency: recompute forwards go through the same chunked-prefill scheduling as
      normal prefill so decode of other requests is not starved. **Not done** — v1's scoping
      decision (single-shot external-prefill path only, see above) means chunk reuse never
      runs long enough in one synchronous burst to starve concurrent decode any more than
      `_do_external_prefill` already can for prompts of the same size; a dedicated
      chunked-scheduling integration is deferred to whenever the multi-step state machine
      gets chunk-reuse support (see "Later").

**What's NOT done yet, concretely:**
1. Live-model quality/perf measurement (the item above) — everything else is unblocked by
   this, but the feature should stay off by default until it's measured. With the
   2026-07-07 fixes (content-hash persistence + message-boundary chunking) the
   `perf_bench.py --ab` flow should now actually produce hits on the natural-text
   scenarios; re-run it on a dense-attention model (Llama-3.2-class) as the next
   session's first step.
2. Chunk reuse for very long prompts that go through the multi-step chunked-prefill state
   machine (only single-shot external-prefill-eligible prompts benefit today).
3. Non-English i18n strings for the new admin toggle.
4. Message-chunk store writes are per-request extra SSD volume (each qualifying
   message chunk becomes its own block file, deduped by content). No budget
   pressure expected under the shared LRU, but worth watching in the live run.

### Phase 4 — tuning & results

- [ ] Sweep `recompute_pct` ∈ {0.10, 0.15, 0.25} × chunker granularity; document the
      TTFT-vs-fidelity curve here.
- [ ] Reference machine (M1 8 GB): SSD load bandwidth may dominate — measure load+shift+
      recompute vs cold prefill per prompt length; auto-disable below a min prompt length
      (analog of `specprefill_threshold`).

## Risks

| Risk | Mitigation |
|---|---|
| Static recompute selection degrades quality on cross-chunk-reasoning prompts | Explicit fidelity bar in Phase 3; `recompute_pct` knob; dynamic selection deferred but designed for |
| SSD metadata schema change breaks existing caches | Versioned `to_dict`/`from_dict`; unknown-field-tolerant load; compat test against a pre-change cache dir |
| RoPE-shift subtleties (scaling variants: yarn, long-rope pre_scale — see `_get_pre_scale`) | Reuse specprefill's tested rope code; per-architecture unit test matrix (llama, qwen, gemma extracttarget list already in specprefill) |
| Prefill-path changes destabilize the untouched-flag-off path | All logic in new `omlx/cache/kv_reuse.py` + guarded call sites; off-path token-identity test in CI |
| Load+shift slower than recompute on fast-prefill models / short chunks | Phase 0 baseline + per-length measurement; min-chunk and min-prompt thresholds |

## Later (explicitly deferred)

- Dynamic recompute selection via layer-1 attention deviation (paper-faithful).
- TurboQuant-stored chunks (dequant→shift→requant on load, or shift in compressed domain
  once the A3 kernel exists).
- Combine with B1 sparse prefill: sparse-attend only within the recompute pass.
- Cross-request shared chunk store with admission control (privacy: same-user scoping).
- Extend chunk reuse into the multi-step chunked-prefill state machine
  (`_begin_prefill`/`_step_prefill_chunk`/`_advance_chunked_prefills`) so very long prompts
  benefit too, not just ones that fit the single-shot external-prefill path (Phase 3's
  "minimal isolated hook" scoping decision deliberately left this out).

## File touchpoints

| File | Change |
|---|---|
| `omlx/cache/kv_reuse.py` | chunker, content-hash index glue, RoPE shift, recompute-set selection; Phase 3: `execute_chunk_prefill_plan`/`_splice_reuse_step`, `is_chunk_reuse_eligible`, `chunks_with_manager_hits` (+ message-offset mode), stats accumulator; `rope_delta` bug fix; `content_chunk_block_hash`; base-offset fold into splice delta |
| `omlx/cache/paged_ssd_cache.py` | `content_hash` metadata field + secondary index; Phase 3: `find_content_hash_hit`; persistence fix: `content_hash`/`content_position` written to + read from safetensors file metadata |
| `omlx/cache/prefix_cache.py` | store-path hook (write chunk entries) — fetch-path arbitration lives in `scheduler.py`, not here (see below); `_store_message_content_chunks` + `store_cache(message_token_offsets=…)` |
| `omlx/request.py` | `Request.message_token_offsets` field |
| `omlx/engine_core.py` | `add_request(message_token_offsets=…)` → `Request` |
| `omlx/patches/specprefill.py` | none — imported (`manual_rope_with_freqs`, `_find_attention_layers`, `_get_attn_module`, position-mapped RoPE helpers) |
| `omlx/model_settings.py` | `chunk_kv_*` settings + compatibility validation |
| `omlx/scheduler.py` | Phase 3: `_do_prefill_with_chunk_reuse` (new method, after `_do_external_prefill`), dispatch branch at the "Normal (non-chunked) full prefill path" call site, `_chunk_kv_reuse_enabled`/`_chunk_kv_recompute_pct` scheduler attrs |
| `omlx/engine/batched.py` | Phase 3: propagate `chunk_kv_reuse_enabled`/`chunk_kv_recompute_pct`/`chunk_kv_min_chunk_tokens` from model_settings to the scheduler at engine start; `_compute_message_token_offsets` + wiring in `chat()`/`stream_chat()`/`generate()`/`stream_generate()` |
| `omlx/admin/routes.py` | Phase 3: `ModelSettingsRequest` fields, validation/conflict glue, requires-reload entries, diffusion/VLM reset paths, `GET /api/kv-reuse/stats` |
| `omlx/admin/templates/dashboard/_modal_model_settings.html`, `omlx/admin/static/js/dashboard.js`, `omlx/admin/i18n/en.json` | Phase 3: toggle UI + field wiring + strings |
| `scripts/perf_bench.py` | RAG/agent/multi-turn/prefix-control scenarios already added (Phase 0); not yet pointed at `chunk_kv_reuse_enabled`/`/api/kv-reuse/stats` for a real run |
| `tests/test_kv_reuse.py` | chunker, shift correctness, planner, compat refusal; Phase 3: `TestExecuteChunkPrefillPlan`, `TestChunksWithManagerHits`, corrected `rope_delta` assertions |
| `tests/test_scheduler.py` | Phase 3: `TestChunkKVReusePrefillDispatch` — the off-path/no-hit/partial-failure regression suite |
