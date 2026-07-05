# SPDX-License-Identifier: Apache-2.0
"""N-gram / prompt-lookup speculative decoding inside ``GenerationBatch``.

Draft-model-free speculation: an :class:`~omlx.speculative.ngram.NgramProposer`
matches the most recent n-gram of the request's token stream against earlier
occurrences (prompt lookup) and proposes the continuation as draft tokens.
The target model verifies all K drafts in one (K+1)-token forward; the longest
prefix whose per-position samples equal the drafts is accepted.

Correctness: drafts are *deterministic* proposals, so no acceptance-ratio /
residual-sampling math is needed. Each verify position is sampled from the
target distribution conditioned on the accepted prefix; a draft is accepted
iff it equals that sample. Emitted tokens are therefore exact target-model
samples — greedy output is bit-identical to standard decoding, and temp>0
output follows the exact target distribution.

Integration mirrors ``omlx.patches.mlx_lm_mtp.batch_generator`` (and reuses
its generic helpers): singleton-batch only, lazy activation from the standard
post-``__init__`` state, state dropped/reconciled on batch reshapes.

Invariant maintained at the end of every ``next()`` call (matching stock
mlx-lm: "``tokens`` always represents the tokens contained in the KV cache"):

  - ``gen_batch.tokens[0]`` + ``state.queue`` == prompt + all forwarded tokens
    (== cache contents); queue entries are in-cache but not yet emitted.
  - ``state.next_main`` (== ``gen_batch._next_tokens``) is the newest sampled
    token: not in cache, not emitted.

Rollback of rejected draft positions (cache holds K+1 new positions, we keep
1 + accepted):

  - **trim mode** — every layer cache is trimmable: ``cache.trim(K - a)``.
    Trimmability is checked *before* the speculative forward (a rotated
    RotatingKVCache is not trimmable and the MTP undo stash only arms for
    2-token updates); when the check fails the step falls back to a plain
    1-token decode, never to an unrecoverable state.
  - **gdn mode** — hybrid GDN models (Qwen3.5 via mlx-vlm) expose
    ``rollback_speculative_cache(caches, gdn_states, accepted, block_size)``
    which replays the accepted prefix through the pre-block SSM state and
    trims KV layers. ``gdn_states`` is captured by passing
    ``capture_layer_ids=[]``; capture support is verified on a 1-token probe
    step before the first speculative forward so a failed capture can never
    strand the cache.

Concurrency: like the MTP path, an active ngram state makes the enclosing
``BatchGenerator`` report its completion batch as full, so late-joining
requests wait instead of forcing a mid-queue reconcile. On an actual reshape
(extend/filter) the state is reconciled back to the stock invariant — free
when the queue is empty, ``trim(len(queue))`` in trim mode, or an MTP-style
re-prefill as the last resort.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..speculative.ngram import NgramProposer, NgramSpecConfig
from .mlx_lm_mtp.batch_generator import (
    _apply_processors,
    _batch_generator_allows_mtp_activation,
    _ensure_uint32,
    _has_grammar_processors,
    _logprobs,
    _model_mtp_decode_enabled,
    _proc_list,
    _reconcile_mtp_to_standard,
    _resolve_sampler,
    _set_singleton_mrope_delta,
    _trim_token_buffer,
)

logger = logging.getLogger(__name__)

_NGRAM_CONFIG_ATTR = "_omlx_ngram_spec"
_NGRAM_UNSUPPORTED_ATTR = "_omlx_ngram_spec_unsupported"


class _NgramStepFallback(RuntimeError):
    """Signal a clean fallback to the standard step."""


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class _NgramStats:
    """Per-request counters, logged at INFO on finish and merged to totals."""

    cycles: int = 0  # speculative verify cycles (a proposal was made)
    plain_steps: int = 0  # decode steps without a proposal
    proposed_tokens: int = 0  # total draft tokens proposed
    accepted_tokens: int = 0  # total draft tokens accepted
    init_emits: int = 0
    draft_emits: int = 0
    plain_emits: int = 0
    backbone_ms: float = 0.0  # verify + plain forwards
    propose_ms: float = 0.0  # proposer extend/propose
    sample_ms: float = 0.0  # sampling + acceptance check
    cache_ops_ms: float = 0.0  # rollback / trims


_TOTALS_LOCK = threading.Lock()
_TOTALS: Dict[str, float] = {}


def _accumulate_totals(stats: _NgramStats) -> None:
    with _TOTALS_LOCK:
        _TOTALS["requests"] = _TOTALS.get("requests", 0) + 1
        for f in (
            "cycles",
            "plain_steps",
            "proposed_tokens",
            "accepted_tokens",
            "init_emits",
            "draft_emits",
            "plain_emits",
            "backbone_ms",
            "propose_ms",
            "sample_ms",
            "cache_ops_ms",
        ):
            _TOTALS[f] = _TOTALS.get(f, 0) + getattr(stats, f)


def get_ngram_spec_totals(reset: bool = False) -> Dict[str, float]:
    """Cumulative n-gram speculation counters (admin stats / benchmarks)."""
    with _TOTALS_LOCK:
        snapshot = dict(_TOTALS)
        if reset:
            _TOTALS.clear()
    return snapshot


def _log_stats(uid: Any, stats: _NgramStats, finish_reason: str) -> None:
    emits = stats.init_emits + stats.draft_emits + stats.plain_emits
    if stats.proposed_tokens > 0:
        accept_str = (
            f"{stats.accepted_tokens}/{stats.proposed_tokens} "
            f"({stats.accepted_tokens / stats.proposed_tokens * 100:.1f}%)"
        )
    else:
        accept_str = "n/a"
    logger.info(
        "NgramSpec[%s] finish=%s tokens=%d cycles=%d plain=%d accept=%s "
        "emits[init=%d,draft=%d,plain=%d] "
        "timing[backbone=%.1fms propose=%.1fms sample=%.1fms cache=%.1fms]",
        uid,
        finish_reason,
        emits,
        stats.cycles,
        stats.plain_steps,
        accept_str,
        stats.init_emits,
        stats.draft_emits,
        stats.plain_emits,
        stats.backbone_ms,
        stats.propose_ms,
        stats.sample_ms,
        stats.cache_ops_ms,
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class _NgramState:
    uid: Any = None
    # Pending (token_id, logprobs_1d, source) emits; all already in cache.
    queue: Deque[Tuple[int, Any, str]] = field(default_factory=deque)
    proposer: Optional[NgramProposer] = None
    # Newest sampled token: not in cache, not emitted, next forward input.
    next_main: Optional[Any] = None  # (1,) uint32
    next_main_id: int = -1
    next_main_lp: Optional[Any] = None  # (vocab,)
    rollback_mode: str = "trim"  # "trim" | "gdn"
    # gdn mode: capture support proven by a 1-token probe step.
    gdn_verified: bool = False
    stats: _NgramStats = field(default_factory=_NgramStats)
    _finished: bool = False

    def finish(self, uid: Any, reason: str) -> None:
        if self._finished:
            return
        self._finished = True
        _log_stats(uid, self.stats, reason)
        _accumulate_totals(self.stats)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def apply() -> bool:
    """Wrap ``GenerationBatch`` / ``BatchGenerator`` with the ngram hooks.

    One-shot: safe to call repeatedly, chains cleanly with the MTP patch
    (each wrapper dispatches only on its own state/eligibility and defers
    to the previously-installed ``next`` otherwise).
    """
    try:
        from mlx_lm.generate import BatchGenerator, GenerationBatch
    except ImportError:
        logger.debug("mlx_lm.generate not importable; ngram spec unavailable")
        return False

    if not hasattr(GenerationBatch, "_omlx_ngram_patched"):
        original_next = GenerationBatch.next
        original_filter = GenerationBatch.filter
        original_extend = GenerationBatch.extend

        def patched_next(self, *args, **kwargs):
            if _is_ngram_eligible(self):
                try:
                    state = _prepare_state(self)
                    if state is not None:
                        return _ngram_next(self, state)
                except _NgramStepFallback as exc:
                    logger.debug("ngram next() fallback to standard step: %s", exc)
                    _drop_state(self, "step-fallback")
            else:
                state = getattr(self, "_omlx_ngram_state", None)
                if state is not None:
                    _reconcile_to_standard(self, state)
                    _drop_state(self, "ineligible")
            return original_next(self, *args, **kwargs)

        def patched_extend(self, batch, *args, **kwargs):
            state = getattr(self, "_omlx_ngram_state", None)
            if state is not None:
                _reconcile_to_standard(self, state)
                _drop_state(self, "extend-reconciled")
            result = original_extend(self, batch, *args, **kwargs)
            _drop_state(batch, "donor-extended")
            _drop_invalid_state(self, "extend")
            return result

        def patched_filter(self, keep, *args, **kwargs):
            result = original_filter(self, keep, *args, **kwargs)
            _drop_invalid_state(self, "filter")
            return result

        GenerationBatch.next = patched_next
        GenerationBatch.extend = patched_extend
        GenerationBatch.filter = patched_filter
        GenerationBatch._omlx_ngram_patched = True

    if not hasattr(BatchGenerator, "_omlx_ngram_patched"):
        original_bg_next = BatchGenerator._next

        def patched_bg_next(self, *args, **kwargs):
            gen_batch = getattr(self, "_generation_batch", None)
            if gen_batch is not None:
                gen_batch._omlx_ngram_activation_safe = (
                    _batch_generator_allows_mtp_activation(self)
                )
            if (
                gen_batch is not None
                and getattr(gen_batch, "_omlx_ngram_state", None) is not None
                and (getattr(gen_batch, "uids", None) or [])
            ):
                had = hasattr(self, "completion_batch_size")
                old = getattr(self, "completion_batch_size", None)
                # Report the completion batch as full so late-join requests
                # wait instead of forcing a mid-queue reconcile (MTP does
                # the same for its active state).
                self.completion_batch_size = 0
                try:
                    return original_bg_next(self, *args, **kwargs)
                finally:
                    if had:
                        self.completion_batch_size = old
                    elif hasattr(self, "completion_batch_size"):
                        delattr(self, "completion_batch_size")
            return original_bg_next(self, *args, **kwargs)

        BatchGenerator._next = patched_bg_next
        BatchGenerator._omlx_ngram_patched = True
    return True


def activate_ngram_spec(model: Any, model_settings: Any) -> bool:
    """Apply the patch and attach the per-model config (engine load time)."""
    if not apply():
        return False
    from ..speculative import ngram as _ngram_mod

    cfg = NgramSpecConfig(
        min_n=int(
            getattr(model_settings, "ngram_spec_min_n", None)
            or _ngram_mod.DEFAULT_MIN_N
        ),
        max_n=int(
            getattr(model_settings, "ngram_spec_max_n", None)
            or _ngram_mod.DEFAULT_MAX_N
        ),
        max_draft=int(
            getattr(model_settings, "ngram_spec_max_draft", None)
            or _ngram_mod.DEFAULT_MAX_DRAFT
        ),
    )
    setattr(model, _NGRAM_CONFIG_ATTR, cfg)
    if hasattr(model, _NGRAM_UNSUPPORTED_ATTR):
        try:
            delattr(model, _NGRAM_UNSUPPORTED_ATTR)
        except AttributeError:
            pass
    logger.info(
        "NgramSpec enabled: min_n=%d max_n=%d max_draft=%d",
        cfg.min_n,
        cfg.max_n,
        cfg.max_draft,
    )
    return True


# ---------------------------------------------------------------------------
# Eligibility / state lifecycle
# ---------------------------------------------------------------------------


def _model_candidates(model: Any) -> List[Any]:
    candidates = [model]
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            candidates.append(inner)
    return candidates


def _get_ngram_config(model: Any) -> Optional[NgramSpecConfig]:
    for candidate in _model_candidates(model):
        cfg = getattr(candidate, _NGRAM_CONFIG_ATTR, None)
        if cfg is not None:
            return cfg
    return None


def _is_ngram_eligible(gen_batch: Any) -> bool:
    model = getattr(gen_batch, "model", None)
    if model is None:
        return False
    if _get_ngram_config(model) is None:
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) != 1:
        return False
    # An in-flight state always finishes its own request (even when the
    # model was marked unsupported mid-request by a failed gdn probe — the
    # state then plain-steps to completion without further speculation).
    state = getattr(gen_batch, "_omlx_ngram_state", None)
    if state is not None and uids[0] == state.uid:
        return True
    if any(
        getattr(candidate, _NGRAM_UNSUPPORTED_ATTR, False)
        for candidate in _model_candidates(model)
    ):
        return False
    # Arbitration: native MTP owns the batch when enabled (settings also
    # forbid the combination; this guards mixed/stale loads).
    if _model_mtp_decode_enabled(model):
        return False
    if _has_grammar_processors(gen_batch):
        return False
    return bool(getattr(gen_batch, "_omlx_ngram_activation_safe", True))


def _state_valid(gen_batch: Any, state: Optional[_NgramState]) -> bool:
    if state is None:
        return False
    uids = getattr(gen_batch, "uids", None)
    return bool(uids is not None and len(uids) == 1 and uids[0] == state.uid)


def _drop_state(gen_batch: Any, reason: str) -> Optional[_NgramState]:
    state = getattr(gen_batch, "_omlx_ngram_state", None)
    if state is None:
        return None
    state.finish(getattr(state, "uid", "?"), reason)
    try:
        delattr(gen_batch, "_omlx_ngram_state")
    except AttributeError:
        pass
    logger.debug("ngram state dropped: %s", reason)
    return state


def _drop_invalid_state(gen_batch: Any, reason: str) -> None:
    state = getattr(gen_batch, "_omlx_ngram_state", None)
    if state is None:
        return
    if _state_valid(gen_batch, state):
        return
    _drop_state(gen_batch, reason)


def _model_call_accepts_capture(model: Any) -> bool:
    """True when ``model(inputs, cache=..., capture_layer_ids=[])`` can work."""
    import inspect

    try:
        params = inspect.signature(model.__call__).parameters.values()
    except (TypeError, ValueError):
        return False
    for param in params:
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "capture_layer_ids":
            return True
    return False


def _all_trimmable(prompt_cache: List[Any]) -> bool:
    if not prompt_cache:
        return False
    for c in prompt_cache:
        try:
            if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                return False
        except Exception:
            return False
    return True


def _resolve_rollback_target(model: Any) -> Optional[Any]:
    for candidate in _model_candidates(model):
        if hasattr(candidate, "rollback_speculative_cache"):
            return candidate
    return None


def _resolve_rollback_mode(gen_batch: Any) -> Optional[str]:
    if _all_trimmable(getattr(gen_batch, "prompt_cache", None) or []):
        return "trim"
    model = gen_batch.model
    if _resolve_rollback_target(model) is not None and _model_call_accepts_capture(
        model
    ):
        return "gdn"
    return None


def _mark_unsupported(gen_batch: Any, reason: str) -> None:
    logger.info("NgramSpec disabled for this model: %s", reason)
    try:
        setattr(gen_batch.model, _NGRAM_UNSUPPORTED_ATTR, True)
    except Exception:
        pass


def _prepare_state(gen_batch: Any) -> Optional[_NgramState]:
    state = getattr(gen_batch, "_omlx_ngram_state", None)
    if _state_valid(gen_batch, state):
        return state
    if state is not None:
        _drop_state(gen_batch, "stale-owner")

    if getattr(gen_batch, "_next_tokens", None) is None or not gen_batch.uids:
        return None
    next_logprobs = getattr(gen_batch, "_next_logprobs", None)
    if not next_logprobs:
        return None

    mode = _resolve_rollback_mode(gen_batch)
    if mode is None:
        _mark_unsupported(
            gen_batch,
            "no rollback path (caches not trimmable, no rollback_speculative_cache)",
        )
        return None

    import mlx.core as mx

    cfg = _get_ngram_config(gen_batch.model)
    next_main = _ensure_uint32(gen_batch._next_tokens)
    mx.eval(next_main)
    next_main_id = int(next_main.tolist()[0])

    state = _NgramState(uid=gen_batch.uids[0])
    state.rollback_mode = mode
    state.next_main = next_main
    state.next_main_id = next_main_id
    state.next_main_lp = next_logprobs[0]
    state.proposer = NgramProposer(cfg)
    t0 = time.perf_counter()
    # Committed stream = cache contents (tokens[0]) + the pending sample.
    state.proposer.extend(list(gen_batch.tokens[0]) + [next_main_id])
    state.stats.propose_ms += (time.perf_counter() - t0) * 1000

    gen_batch._omlx_ngram_state = state
    logger.info(
        "NgramSpec active for uid=%s (rollback=%s, prompt=%d tokens)",
        state.uid,
        mode,
        len(gen_batch.tokens[0]),
    )
    return state


# ---------------------------------------------------------------------------
# next() dispatch
# ---------------------------------------------------------------------------


def _ngram_next(gen_batch: Any, state: _NgramState) -> Any:
    if not state.queue:
        _set_singleton_mrope_delta(gen_batch)
        _run_cycle(gen_batch, state)
        if not state.queue:
            raise _NgramStepFallback("cycle produced no emit tokens")
    token_id, logprobs_1d, source = state.queue.popleft()
    if source == "init":
        state.stats.init_emits += 1
    elif source == "draft":
        state.stats.draft_emits += 1
    else:
        state.stats.plain_emits += 1
    return _emit_response(gen_batch, state, token_id, logprobs_1d)


def _forward(
    gen_batch: Any, inputs_2d: Any, *, capture: bool
) -> Tuple[Any, Optional[list]]:
    """Run the target forward; returns ``(logits, gdn_states_or_None)``."""
    model = gen_batch.model
    if capture:
        # return_hidden keeps wrappers (VLMModelAdapter) from unwrapping the
        # LanguageModelOutput down to bare logits — gdn_states rides on it.
        out = model(
            inputs_2d,
            cache=gen_batch.prompt_cache,
            capture_layer_ids=[],
            return_hidden=True,
        )
    else:
        out = model(inputs_2d, cache=gen_batch.prompt_cache)
    if hasattr(out, "logits"):
        return out.logits, getattr(out, "gdn_states", None)
    if isinstance(out, tuple):
        return out[0], None
    return out, None


def _gdn_states_usable(gdn_states: Optional[list]) -> bool:
    return bool(gdn_states) and all(
        isinstance(s, tuple) and len(s) >= 12 for s in gdn_states
    )


def _run_cycle(gen_batch: Any, state: _NgramState) -> None:
    """Run one decode cycle: speculative verify on a proposal, else a plain
    1-token step. Populates ``state.queue`` with >= 1 emit."""
    import mlx.core as mx

    if state.next_main is None:
        raise _NgramStepFallback("cycle entered without next_main")

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    stats = state.stats

    # ---- proposal --------------------------------------------------------
    t0 = time.perf_counter()
    draft_ids: Optional[List[int]] = None
    remaining = gen_batch.max_tokens[0] - gen_batch._num_tokens[0]
    gdn_ready = state.rollback_mode != "gdn" or state.gdn_verified
    if remaining > 1 and gdn_ready:
        if state.rollback_mode == "trim" and not _all_trimmable(
            gen_batch.prompt_cache
        ):
            # e.g. RotatingKVCache after rotation: skip speculation, the
            # plain step below keeps decoding safely.
            draft_ids = None
        else:
            draft_ids = state.proposer.propose(max_draft=remaining - 1)
    stats.propose_ms += (time.perf_counter() - t0) * 1000

    capture = state.rollback_mode == "gdn"

    if not draft_ids:
        # ---- plain 1-token step (with gdn capture probe on first use) ----
        stats.plain_steps += 1

        want_probe = capture and not state.gdn_verified
        t0 = time.perf_counter()
        try:
            logits, gdn_states = _forward(
                gen_batch, state.next_main[None, :], capture=want_probe
            )
        except TypeError as exc:
            # Call-time signature rejection: nothing was forwarded, the
            # cache is untouched — safe to hand the step back to the
            # standard path and disable ngram for this model.
            if want_probe:
                _mark_unsupported(gen_batch, f"capture_layer_ids rejected: {exc}")
                raise _NgramStepFallback("gdn capture probe failed")
            raise
        mx.eval(logits)
        stats.backbone_ms += (time.perf_counter() - t0) * 1000

        if want_probe:
            if _gdn_states_usable(gdn_states):
                state.gdn_verified = True
            else:
                # The forward already ran, so finish this step normally;
                # this request keeps plain-stepping (gdn_verified stays
                # False) and future requests skip ngram entirely.
                _mark_unsupported(
                    gen_batch, "gdn capture probe returned no usable states"
                )

        prev_buf = None
        if procs is not None:
            prev_buf = gen_batch._token_context[0].update_and_fetch(state.next_main)

        t0 = time.perf_counter()
        step_logits = logits[:, -1, :]
        step_logits = _apply_processors(procs, prev_buf, step_logits)
        lp = _logprobs(step_logits)  # (1, vocab)
        sampled = sampler(lp)
        mx.eval(sampled)
        new_id = int(sampled.tolist()[0])
        stats.sample_ms += (time.perf_counter() - t0) * 1000

        # next_main is now in cache -> emit it; new sample becomes pending.
        state.queue.append((state.next_main_id, state.next_main_lp, "plain"))
        state.next_main = _ensure_uint32(sampled)
        state.next_main_id = new_id
        state.next_main_lp = lp.squeeze(0)

        t0 = time.perf_counter()
        state.proposer.extend([new_id])
        stats.propose_ms += (time.perf_counter() - t0) * 1000
    else:
        # ---- speculative verify ------------------------------------------
        k = len(draft_ids)
        stats.cycles += 1
        stats.proposed_tokens += k

        draft_arr = mx.array(draft_ids, dtype=mx.uint32)
        inputs = mx.concatenate([state.next_main, draft_arr])  # (K+1,)

        t0 = time.perf_counter()
        logits, gdn_states = _forward(gen_batch, inputs[None, :], capture=capture)
        mx.eval(logits)
        stats.backbone_ms += (time.perf_counter() - t0) * 1000

        prev_bufs: List[Any] = []
        if procs is not None:
            buf = gen_batch._token_context[0]
            for i in range(k + 1):
                prev_bufs.append(buf.update_and_fetch(inputs[i : i + 1]))

        t0 = time.perf_counter()
        pos_logits = logits[0]  # (K+1, vocab)
        if procs is not None:
            rows = [
                _apply_processors(procs, prev_bufs[i], pos_logits[i : i + 1])
                for i in range(k + 1)
            ]
            pos_logits = mx.concatenate(rows, axis=0)
        lp_all = _logprobs(pos_logits)  # (K+1, vocab)
        sampled = sampler(lp_all)  # (K+1,)
        mx.eval(sampled)
        sampled_ids = [int(x) for x in sampled.tolist()]

        accepted = 0
        while accepted < k and sampled_ids[accepted] == draft_ids[accepted]:
            accepted += 1
        stats.accepted_tokens += accepted
        stats.sample_ms += (time.perf_counter() - t0) * 1000

        # ---- rollback rejected positions ---------------------------------
        n_trim = k - accepted
        t0 = time.perf_counter()
        if n_trim > 0:
            if state.rollback_mode == "gdn":
                if not _gdn_states_usable(gdn_states):
                    # Probe verified capture works, so this is unexpected —
                    # and the cache now holds unremovable rejected tokens.
                    # Surface loudly instead of continuing corrupted.
                    raise RuntimeError(
                        "ngram verify forward returned no gdn states; "
                        "cache cannot be rolled back"
                    )
                target = _resolve_rollback_target(gen_batch.model)
                target.rollback_speculative_cache(
                    gen_batch.prompt_cache,
                    gdn_states,
                    accepted,
                    k + 1,
                )
            else:
                for c in gen_batch.prompt_cache:
                    c.trim(n_trim)
            if procs is not None:
                _trim_token_buffer(gen_batch, n_trim)
        stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

        # ---- queue emits: next_main + accepted drafts (all in cache) -----
        state.queue.append((state.next_main_id, state.next_main_lp, "plain"))
        for i in range(accepted):
            state.queue.append((draft_ids[i], lp_all[i], "draft"))

        # The sample at the first mismatch (or the bonus position when all
        # drafts were accepted) becomes the new pending token.
        bonus_id = sampled_ids[accepted]
        state.next_main = _ensure_uint32(sampled[accepted : accepted + 1])
        state.next_main_id = bonus_id
        state.next_main_lp = lp_all[accepted]

        t0 = time.perf_counter()
        state.proposer.extend(list(draft_ids[:accepted]) + [bonus_id])
        stats.propose_ms += (time.perf_counter() - t0) * 1000

    # Keep the stock fields pointing at the pending sample so a queue-empty
    # reshape needs no reconcile work at all.
    gen_batch._next_tokens = state.next_main
    gen_batch._next_logprobs = [state.next_main_lp]


# ---------------------------------------------------------------------------
# Reconcile / emit
# ---------------------------------------------------------------------------


def _reconcile_to_standard(gen_batch: Any, state: _NgramState) -> bool:
    """Rewind to the stock invariant so standard decode can resume.

    Queue empty: nothing to do (``_next_tokens`` is maintained every cycle
    and ``tokens[0]`` matches the cache). Queue non-empty: the cache holds
    ``len(queue)`` not-yet-emitted tokens — drop them so standard decode
    re-derives from ``queue[0]``.
    """
    if not state.queue:
        return True
    import mlx.core as mx

    n = len(state.queue)
    first_id, first_lp, _src = state.queue[0]
    if _all_trimmable(gen_batch.prompt_cache):
        for c in gen_batch.prompt_cache:
            c.trim(n)
        if _proc_list(gen_batch) is not None:
            _trim_token_buffer(gen_batch, n)
        gen_batch._next_tokens = mx.array([first_id], dtype=mx.uint32)
        gen_batch._next_logprobs = [first_lp]
        state.queue.clear()
        logger.debug(
            "ngram reconciled to standard via trim(%d) (uid=%s)", n, state.uid
        )
        return True
    # GDN caches cannot rewind: rebuild deterministically by re-prefilling
    # the emitted tokens (MTP's reconcile handles queue[0] identically).
    ok = _reconcile_mtp_to_standard(gen_batch, state)
    if ok:
        state.queue.clear()
    else:
        logger.warning(
            "ngram reconcile failed for uid=%s; %d queued tokens dropped",
            state.uid,
            n,
        )
    return ok


def _emit_response(
    gen_batch: Any, state: _NgramState, token_id: int, logprobs_1d: Any
) -> List[Any]:
    """Standard next() epilogue for one emitted token (mirrors the MTP path)."""
    Response = type(gen_batch).Response

    finish_reason: Optional[str] = None
    gen_batch.tokens[0].append(token_id)
    gen_batch._num_tokens[0] += 1
    if gen_batch._num_tokens[0] >= gen_batch.max_tokens[0]:
        finish_reason = "length"

    new_state, match_sequence, current_state = gen_batch.state_machines[0].match(
        gen_batch._matcher_states[0], token_id
    )
    gen_batch._matcher_states[0] = new_state
    if match_sequence is not None and current_state is None:
        finish_reason = "stop"

    if finish_reason is not None:
        prompt_cache = gen_batch.extract_cache(0)
        all_tokens = gen_batch.tokens[0]
        response = Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=finish_reason,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=prompt_cache,
            all_tokens=all_tokens,
        )
        state.finish(gen_batch.uids[0], finish_reason)
        if hasattr(gen_batch, "_omlx_ngram_state"):
            try:
                delattr(gen_batch, "_omlx_ngram_state")
            except AttributeError:
                pass
        gen_batch.filter([])
        return [response]

    return [
        Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=None,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=None,
            all_tokens=None,
        )
    ]
