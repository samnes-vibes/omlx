# SPDX-License-Identifier: Apache-2.0
"""Tests for n-gram / prompt-lookup speculative decoding.

The core contract is the greedy-identity property: with temperature 0 the
patched decode must produce bit-identical tokens to standard decoding, for
any mix of accepted and rejected drafts. A tiny random-weight llama model
with a small vocabulary gives natural accept/reject mixes.
"""

import pytest

import mlx.core as mx

from omlx.model_settings import ModelSettings
from omlx.speculative.ngram import NgramSpecConfig


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


class TestModelSettingsValidation:
    def test_defaults_off(self):
        s = ModelSettings()
        assert s.ngram_spec_enabled is False
        assert s.ngram_spec_min_n is None
        assert s.ngram_spec_max_n is None
        assert s.ngram_spec_max_draft is None

    def test_roundtrip(self):
        s = ModelSettings(
            ngram_spec_enabled=True,
            ngram_spec_min_n=3,
            ngram_spec_max_n=5,
            ngram_spec_max_draft=16,
        )
        restored = ModelSettings.from_dict(s.to_dict())
        assert restored.ngram_spec_enabled is True
        assert restored.ngram_spec_min_n == 3
        assert restored.ngram_spec_max_n == 5
        assert restored.ngram_spec_max_draft == 16

    @pytest.mark.parametrize(
        "other", ["mtp_enabled", "dflash_enabled", "vlm_mtp_enabled"]
    )
    def test_mutually_exclusive_with_other_speculative_paths(self, other):
        kwargs = {"ngram_spec_enabled": True, other: True}
        if other == "vlm_mtp_enabled":
            kwargs["vlm_mtp_draft_model"] = "x"
        with pytest.raises(ValueError, match="ngram_spec_enabled"):
            ModelSettings(**kwargs)

    def test_allowed_with_turboquant(self):
        s = ModelSettings(ngram_spec_enabled=True, turboquant_kv_enabled=True)
        assert s.ngram_spec_enabled and s.turboquant_kv_enabled


# ---------------------------------------------------------------------------
# Decode-path tests against a tiny random-weight model
# ---------------------------------------------------------------------------


def _make_tiny_model(vocab_size=64):
    from mlx_lm.models import llama

    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    mx.random.seed(1234)
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


def _decode(model, prompt, max_tokens=48, stop_tokens=None):
    """Run one request through BatchGenerator; return the emitted token list."""
    from mlx_lm.generate import BatchGenerator

    bg = BatchGenerator(
        model=model,
        max_tokens=max_tokens,
        stop_tokens=stop_tokens,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=64,
    )
    bg.insert([list(prompt)])
    out = []
    for _ in range(max_tokens * 4 + 16):
        result = bg.next()
        responses = result[1] if isinstance(result, tuple) else result
        done = False
        for r in responses or []:
            out.append(r.token)
            if r.finish_reason is not None:
                done = True
        if done:
            break
    return out


def _enable_ngram(model, **kwargs):
    from omlx.patches.ngram_spec import _NGRAM_CONFIG_ATTR, apply

    assert apply()
    setattr(model, _NGRAM_CONFIG_ATTR, NgramSpecConfig(**kwargs))


def _disable_ngram(model):
    from omlx.patches.ngram_spec import _NGRAM_CONFIG_ATTR

    if hasattr(model, _NGRAM_CONFIG_ATTR):
        delattr(model, _NGRAM_CONFIG_ATTR)


# A prompt with heavy internal repetition so the proposer fires often; the
# random model's continuations won't reliably match, exercising both the
# accept and the reject/trim paths.
_REPEATED = [5, 9, 12, 7, 5, 9, 12, 7, 5, 9, 12, 7, 3, 5, 9, 12]


class TestGreedyIdentity:
    def test_identity_repetitive_prompt(self):
        model = _make_tiny_model()
        baseline = _decode(model, _REPEATED)
        _enable_ngram(model, min_n=1, max_n=4, max_draft=8)
        try:
            speculative = _decode(model, _REPEATED)
        finally:
            _disable_ngram(model)
        assert speculative == baseline
        assert len(baseline) > 0

    def test_identity_across_draft_lengths(self):
        model = _make_tiny_model()
        prompt = list(range(8)) + list(range(8)) + [2, 4]
        baseline = _decode(model, prompt)
        for max_draft in (1, 2, 5, 16):
            _enable_ngram(model, min_n=1, max_n=3, max_draft=max_draft)
            try:
                assert _decode(model, prompt) == baseline, f"K={max_draft}"
            finally:
                _disable_ngram(model)

    def test_identity_with_stop_tokens(self):
        model = _make_tiny_model()
        prompt = _REPEATED
        baseline = _decode(model, prompt)
        # Stop on a token the baseline actually emits mid-stream, so the
        # speculative run must cut its queue at the same position.
        stop_candidates = [t for t in baseline[2:-1]]
        stop = [[stop_candidates[len(stop_candidates) // 2]]]
        expected = _decode(model, prompt, stop_tokens=stop)
        _enable_ngram(model, min_n=1, max_n=4, max_draft=8)
        try:
            got = _decode(model, prompt, stop_tokens=stop)
        finally:
            _disable_ngram(model)
        assert got == expected

    def test_identity_max_tokens_boundary(self):
        model = _make_tiny_model()
        for cap in (1, 2, 3, 7):
            baseline = _decode(model, _REPEATED, max_tokens=cap)
            assert len(baseline) == cap
            _enable_ngram(model, min_n=1, max_n=4, max_draft=8)
            try:
                got = _decode(model, _REPEATED, max_tokens=cap)
            finally:
                _disable_ngram(model)
            assert got == baseline, f"max_tokens={cap}"


class TestSpeculationActuallyRuns:
    def test_stats_show_cycles_and_accepts(self):
        from omlx.patches.ngram_spec import get_ngram_spec_totals

        model = _make_tiny_model(vocab_size=16)
        # Tiny vocab: generated text collides with prompt n-grams constantly.
        prompt = ([3, 5, 3, 5, 3, 5, 7] * 3)[:20]
        get_ngram_spec_totals(reset=True)
        _enable_ngram(model, min_n=1, max_n=4, max_draft=8)
        try:
            _decode(model, prompt, max_tokens=64)
        finally:
            _disable_ngram(model)
        totals = get_ngram_spec_totals()
        assert totals.get("requests", 0) >= 1
        assert totals.get("cycles", 0) + totals.get("plain_steps", 0) > 0
        # The repetitive stream must produce at least one speculative cycle.
        assert totals.get("cycles", 0) > 0
        emits = (
            totals.get("init_emits", 0)
            + totals.get("draft_emits", 0)
            + totals.get("plain_emits", 0)
        )
        assert emits == 64

    def test_unsupported_model_marks_and_falls_back(self):
        from omlx.patches import ngram_spec

        class _NoRollback:
            pass

        gen_batch = type("GB", (), {})()
        gen_batch.model = _NoRollback()
        gen_batch.prompt_cache = [object()]  # not trimmable, no gdn
        assert ngram_spec._resolve_rollback_mode(gen_batch) is None


class TestEligibility:
    def test_no_config_no_dispatch(self):
        from omlx.patches import ngram_spec

        model = _make_tiny_model()
        gb = type("GB", (), {})()
        gb.model = model
        gb.uids = [1]
        assert ngram_spec._is_ngram_eligible(gb) is False

    def test_multi_uid_not_eligible(self):
        from omlx.patches import ngram_spec

        model = _make_tiny_model()
        _enable_ngram(model)
        try:
            gb = type("GB", (), {})()
            gb.model = model
            gb.uids = [1, 2]
            assert ngram_spec._is_ngram_eligible(gb) is False
            gb.uids = [1]
            gb.samplers = []
            gb.logits_processors = []
            gb.fallback_sampler = None
            assert ngram_spec._is_ngram_eligible(gb) is True
        finally:
            _disable_ngram(model)
