# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the n-gram draft proposer."""

import pytest

from omlx.speculative.ngram import NgramProposer, NgramSpecConfig


def make(min_n=2, max_n=4, max_draft=8, tokens=None):
    p = NgramProposer(NgramSpecConfig(min_n=min_n, max_n=max_n, max_draft=max_draft))
    if tokens is not None:
        p.extend(tokens)
    return p


class TestNgramSpecConfig:
    def test_defaults(self):
        cfg = NgramSpecConfig()
        assert cfg.min_n == 2
        assert cfg.max_n == 4
        assert cfg.max_draft == 8

    def test_invalid_min_n(self):
        with pytest.raises(ValueError):
            NgramSpecConfig(min_n=0)

    def test_invalid_max_n(self):
        with pytest.raises(ValueError):
            NgramSpecConfig(min_n=3, max_n=2)

    def test_invalid_max_draft(self):
        with pytest.raises(ValueError):
            NgramSpecConfig(max_draft=0)


class TestPropose:
    def test_miss_on_fresh_stream(self):
        p = make(tokens=[1, 2, 3, 4, 5])
        assert p.propose() is None

    def test_basic_hit(self):
        # ... 10 20 30 40 ... then suffix ends with 10 20 -> propose 30 40 ...
        p = make(tokens=[10, 20, 30, 40, 50, 99, 10, 20])
        assert p.propose() == [30, 40, 50, 99, 10, 20]

    def test_max_draft_cap(self):
        p = make(max_draft=3, tokens=[10, 20, 30, 40, 50, 99, 10, 20])
        assert p.propose() == [30, 40, 50]

    def test_per_call_cap_overrides_down(self):
        p = make(max_draft=8, tokens=[10, 20, 30, 40, 50, 99, 10, 20])
        assert p.propose(max_draft=2) == [30, 40]
        assert p.propose(max_draft=0) is None

    def test_longest_n_wins(self):
        # 4-gram (1 2 3 4) -> 100..., but 2-gram (3 4) also occurs later -> 200...
        p = make(min_n=2, max_n=4, tokens=[1, 2, 3, 4, 100, 7, 3, 4, 200, 8, 1, 2, 3, 4])
        # Suffix ...1 2 3 4 matches the 4-gram; its continuation starts at 100.
        assert p.propose()[0] == 100

    def test_latest_occurrence_wins(self):
        # (5 6) occurs twice with different continuations; latest wins.
        p = make(min_n=2, max_n=2, tokens=[5, 6, 111, 0, 5, 6, 222, 0, 5, 6])
        assert p.propose()[0] == 222

    def test_current_suffix_does_not_shadow(self):
        # The suffix's own occurrence has no continuation and must not be
        # indexed over the earlier (useful) occurrence.
        p = make(min_n=2, max_n=2, tokens=[7, 8, 42, 7, 8])
        assert p.propose() == [42, 7, 8]

    def test_self_overlapping_repetition(self):
        # "a a a a" proposes more of the same run.
        p = make(min_n=2, max_n=2, tokens=[9, 9, 9, 9])
        out = p.propose()
        assert out is not None and out[0] == 9

    def test_incremental_matches_bulk(self):
        tokens = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 1, 4, 1, 5]
        bulk = make(tokens=tokens)
        inc = make()
        for t in tokens:
            inc.extend([t])
        assert inc.propose() == bulk.propose()
        assert inc._index == bulk._index

    def test_repeating_tail_falls_back_to_first_occurrence(self):
        # In a constant run the latest suffix match sits at the stream's
        # end (1-token continuation); the first occurrence must provide a
        # full-length draft so the check-token + drafts split still works.
        p = make(min_n=1, max_n=4, max_draft=8)
        p.extend([9] * 30)
        out = p.propose()
        assert out is not None and len(out) >= 2
        assert all(t == 9 for t in out)

    def test_extend_in_chunks_matches_bulk(self):
        tokens = list(range(10)) + [3, 4, 5, 6] + list(range(10))
        bulk = make(tokens=tokens)
        chunked = make()
        chunked.extend(tokens[:7])
        chunked.extend(tokens[7:12])
        chunked.extend(tokens[12:])
        assert chunked._index == bulk._index
        assert len(chunked) == len(tokens)

    def test_short_stream_below_min_n(self):
        p = make(min_n=2, max_n=4, tokens=[1])
        assert p.propose() is None

    def test_continuation_shorter_than_cap(self):
        # Match near the end of the stream: continuation truncated, not padded.
        p = make(max_draft=8, tokens=[1, 2, 3, 0, 1, 2])
        assert p.propose() == [3, 0, 1, 2]
