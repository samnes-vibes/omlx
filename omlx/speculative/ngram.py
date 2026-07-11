# SPDX-License-Identifier: Apache-2.0
"""N-gram / prompt-lookup draft proposer for speculative decoding.

Draft-model-free speculation source: match the most recent n-gram of the
token stream (prompt + generated tokens) against earlier occurrences in the
same stream, and propose the tokens that followed the previous occurrence as
draft tokens. The target model verifies the drafts in a single forward pass
(see ``omlx.patches.ngram_spec``).

Effective on workloads where the output echoes the input — summarization,
code editing, RAG answers with quotes, agent/tool loops. Near-neutral on
freeform generation (few matches → plain decode steps).

Design notes:
  - The proposer owns its token stream (``extend()``); the caller feeds it
    every committed token exactly once, so indexing cost is O(1) amortized
    per token regardless of how often ``propose()`` is called.
  - An n-gram ending at position ``e`` (exclusive) is only indexed once at
    least one continuation token exists (``e <= len(tokens) - 1``), so the
    current suffix never shadows the prior occurrence it should match.
  - Latest occurrence wins: repeated n-grams map to their most recent
    continuation, which adapts to the local text (PLD/vLLM ngram behavior).
  - Longest n first: ``propose`` tries ``max_n`` down to ``min_n`` and takes
    the first hit, preferring more specific (higher-precision) matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# Defaults chosen per docs/experimental/ngram_speculation_plan.md; tuned on
# the reference workloads in scripts/perf_bench.py.
DEFAULT_MIN_N = 2
DEFAULT_MAX_N = 4
DEFAULT_MAX_DRAFT = 8


@dataclass(frozen=True)
class NgramSpecConfig:
    """Load-time configuration attached to a model instance."""

    min_n: int = DEFAULT_MIN_N
    max_n: int = DEFAULT_MAX_N
    max_draft: int = DEFAULT_MAX_DRAFT

    def __post_init__(self) -> None:
        if self.min_n < 1:
            raise ValueError(f"min_n must be >= 1, got {self.min_n}")
        if self.max_n < self.min_n:
            raise ValueError(
                f"max_n ({self.max_n}) must be >= min_n ({self.min_n})"
            )
        if self.max_draft < 1:
            raise ValueError(f"max_draft must be >= 1, got {self.max_draft}")


class NgramProposer:
    """Incremental n-gram index over one request's token stream.

    One instance per request. Feed committed tokens via ``extend()``; call
    ``propose()`` for draft tokens continuing the current suffix.
    """

    def __init__(self, config: NgramSpecConfig):
        self._config = config
        self._tokens: List[int] = []
        # n-gram tuple -> [first_end, latest_end] (end positions, exclusive)
        # of occurrences that have at least one continuation token. The
        # latest occurrence is preferred (adapts to local text), but when
        # its continuation is too short — typical inside a repeating tail,
        # where the latest match sits at the very end of the stream — the
        # first occurrence provides the long continuation instead.
        self._index: Dict[Tuple[int, ...], List[int]] = {}
        # Highest end position already indexed (exclusive position value).
        self._indexed_end = 0

    @property
    def config(self) -> NgramSpecConfig:
        return self._config

    def __len__(self) -> int:
        return len(self._tokens)

    def extend(self, new_tokens: Iterable[int]) -> None:
        """Append committed tokens and index the newly-completed n-grams."""
        self._tokens.extend(int(t) for t in new_tokens)
        tokens = self._tokens
        min_n = self._config.min_n
        max_n = self._config.max_n
        # End positions must leave >= 1 continuation token: e in [.., len-1].
        limit = len(tokens) - 1
        start = max(self._indexed_end + 1, min_n)
        for e in range(start, limit + 1):
            for n in range(min_n, max_n + 1):
                if n > e:
                    break
                key = tuple(tokens[e - n : e])
                entry = self._index.get(key)
                if entry is None:
                    self._index[key] = [e, e]
                else:
                    entry[1] = e
        if limit > self._indexed_end:
            self._indexed_end = limit

    def propose(self, max_draft: Optional[int] = None) -> Optional[List[int]]:
        """Return draft tokens continuing the current suffix, or None on miss.

        ``max_draft`` caps the draft length below the configured maximum
        (used near the request's max_tokens limit).
        """
        cap = self._config.max_draft
        if max_draft is not None:
            cap = min(cap, max_draft)
        if cap < 1:
            return None
        tokens = self._tokens
        length = len(tokens)
        for n in range(self._config.max_n, self._config.min_n - 1, -1):
            if length < n:
                continue
            entry = self._index.get(tuple(tokens[length - n : length]))
            if entry is None:
                continue
            first_end, latest_end = entry
            draft = tokens[latest_end : latest_end + cap]
            if len(draft) < min(2, cap) and first_end != latest_end:
                # Latest match sits at the stream's tail (short continuation);
                # the first occurrence carries the long continuation.
                draft = tokens[first_end : first_end + cap]
            if draft:
                return list(draft)
        return None
