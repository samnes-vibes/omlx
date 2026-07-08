# SPDX-License-Identifier: Apache-2.0
"""Offline calibration for draft-free sparse prefill.

Runs long calibration prompts through the model with attention capture at
the SDPA seam, measures per-head attention-mass recall of candidate sparse
patterns (a_shape, vertical_slash) at a fixed FLOP budget, and writes a
per-head pattern JSON consumed by omlx.patches.sparse_prefill.

Usage:
    uv run python -m omlx.sparse_calibration --model <model-id> \
        [--budget 0.1] [--target-tokens 16384] [--output <path>]

Recall gate (plan Phase 1): if mean recall at the budget is < 0.90 the
tool exits non-zero — enabling sparse prefill would hurt quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .patches import sparse_prefill as sp

logger = logging.getLogger(__name__)

_SAMPLE_ROWS = 64  # trailing query rows evaluated per chunk (matches runtime)
_MIN_CAPTURE_QLEN = 256


# ---------------------------------------------------------------------------
# Candidate grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    kind: str  # "a_shape" | "vertical_slash"
    sink: int
    window: int  # for a_shape this is derived at eval time from the budget

    def key(self) -> str:
        return f"{self.kind}:s{self.sink}:w{self.window}"


# a_shape: sink S, window = budget_keys - S
_A_SHAPE_SINKS = (64, 256, 1024)
# vertical_slash: sink 64, window W, columns = budget_keys - sink - window
_VSLASH_WINDOWS = (256, 1024)
_VSLASH_SINK = 64


def _candidates() -> List[Candidate]:
    cands = [Candidate("a_shape", s, 0) for s in _A_SHAPE_SINKS]
    cands += [Candidate("vertical_slash", _VSLASH_SINK, w) for w in _VSLASH_WINDOWS]
    return cands


# ---------------------------------------------------------------------------
# Calibration corpus
# ---------------------------------------------------------------------------

_TOPICS = [
    "unified memory bandwidth", "the scheduler admission queue",
    "paged KV cache spill", "speculative decoding acceptance",
    "quantization group sizes", "attention sink stability",
    "prefill chunk boundaries", "the admin dashboard metrics",
    "tokenizer merge tables", "rotary position embeddings",
    "thermal throttling on laptops", "benchmark reproducibility",
]

_VERBS = ["improves", "constrains", "dominates", "amortizes", "invalidates",
          "stabilizes", "regresses", "saturates"]

_CODE_NAMES = ["cache", "batch", "chunk", "layer", "head", "token", "block",
               "queue", "budget", "offset"]


def build_calibration_text(seed: int = 0, approx_words: int = 30000) -> str:
    """Deterministic mixed-domain text: prose + code + Q/A transcript.

    Aperiodic on purpose — periodic text would bias the vertical-column
    statistics toward the repetition period.
    """
    rng = random.Random(seed)
    parts: List[str] = []
    words = 0
    section = 0
    while words < approx_words:
        section += 1
        kind = rng.choice(["prose", "code", "qa"])
        if kind == "prose":
            para = []
            for _ in range(rng.randint(4, 9)):
                t1, t2 = rng.sample(_TOPICS, 2)
                v = rng.choice(_VERBS)
                para.append(
                    f"In section {section}, {t1} {v} {t2} when the workload "
                    f"exceeds {rng.randint(2, 64)} concurrent requests."
                )
            parts.append(" ".join(para))
        elif kind == "code":
            n1, n2 = rng.sample(_CODE_NAMES, 2)
            lines = [f"def process_{n1}_{section}({n2}, limit={rng.randint(8, 512)}):"]
            for i in range(rng.randint(3, 8)):
                a, b = rng.sample(_CODE_NAMES, 2)
                lines.append(f"    {a}_{i} = {b}.get({rng.randint(0, 99)}, None)")
            lines.append(f"    return sum(x or 0 for x in locals().values() if isinstance(x, int))")
            parts.append("\n".join(lines))
        else:
            t = rng.choice(_TOPICS)
            parts.append(
                f"Q{section}: What limits {t} in practice?\n"
                f"A{section}: Mostly {rng.choice(_TOPICS)}; see section "
                f"{rng.randint(1, max(1, section))} for the measurement setup."
            )
        words += len(parts[-1].split())
    return "\n\n".join(parts)


def build_calibration_tokens(tokenizer, target_tokens: int, seed: int = 0) -> List[int]:
    text = build_calibration_text(seed=seed, approx_words=int(target_tokens * 1.2))
    toks = tokenizer.encode(text)
    while len(toks) < target_tokens:
        text = build_calibration_text(seed=seed + 1, approx_words=target_tokens)
        toks = toks + tokenizer.encode(text)
        seed += 1
    return toks[:target_tokens]


# ---------------------------------------------------------------------------
# Recall evaluation
# ---------------------------------------------------------------------------


class _RecallAccumulator:
    """Per-layer, per-head, per-candidate running mean of recall."""

    def __init__(self) -> None:
        # layer -> head -> candidate_key -> [sum, count]
        self.data: Dict[int, Dict[int, Dict[str, List[float]]]] = {}

    def add(self, layer: int, head: int, cand_key: str, recall: float) -> None:
        self.data.setdefault(layer, {}).setdefault(head, {}).setdefault(
            cand_key, [0.0, 0]
        )
        cell = self.data[layer][head][cand_key]
        cell[0] += recall
        cell[1] += 1

    def mean(self, layer: int, head: int, cand_key: str) -> float:
        s, n = self.data[layer][head][cand_key]
        return s / n if n else 0.0


def evaluate_chunk(
    acc: _RecallAccumulator,
    layer: int,
    queries: mx.array,
    keys: mx.array,
    scale: float,
    budget: float,
) -> None:
    """Compute per-head recall of every candidate on this chunk's tail rows."""
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = keys.shape[1]
    K = keys.shape[-2]
    group = n_q_heads // n_kv_heads
    rows = min(_SAMPLE_ROWS, L)

    q_s = queries[..., L - rows :, :]
    if group > 1:
        keys_e = mx.repeat(keys, group, axis=1)
    else:
        keys_e = keys
    scores = (q_s @ keys_e.transpose(0, 1, 3, 2)) * scale
    pos = np.arange(K - rows, K)  # absolute positions of sampled rows
    col = mx.arange(K)
    causal = col[None, :] <= mx.array(pos)[:, None]
    scores = mx.where(causal[None, None], scores, mx.array(-mx.inf))
    A = np.array(mx.softmax(scores.astype(mx.float32), axis=-1))[0]  # (Hq, rows, K)

    budget_keys = int(budget * K)
    cum = np.cumsum(A, axis=-1)  # (Hq, rows, K)

    def window_mass(W: int) -> np.ndarray:
        # per (head, row): mass in (p-W, p]
        lo = np.maximum(pos - W, -1)
        hi_v = cum[:, np.arange(rows), pos]  # == total causal mass ≈ 1
        lo_v = np.where(lo[None, :] >= 0, np.take_along_axis(
            cum, np.maximum(lo, 0)[None, :, None].repeat(A.shape[0], 0), axis=2
        )[:, :, 0], 0.0)
        return hi_v - lo_v

    # kv-group mean column mass (self-consistent with the runtime estimate)
    col_mean_group = A.reshape(n_kv_heads, group, rows, K).mean(axis=(1, 2))

    for cand in _candidates():
        if cand.kind == "a_shape":
            S = cand.sink
            W = budget_keys - S
            if W <= 64:
                continue
            recall = cum[:, :, S - 1] + window_mass(W)  # (Hq, rows)
            rec_h = recall.mean(axis=1)
            for h in range(n_q_heads):
                acc.add(layer, h, cand.key(), float(rec_h[h]))
        else:
            S, W = cand.sink, cand.window
            C = budget_keys - S - W
            if C <= 0:
                continue
            C = min(C, K)
            base = cum[:, :, S - 1] + window_mass(W)
            for g in range(n_kv_heads):
                top = np.argpartition(col_mean_group[g], K - C)[K - C:]
                colsel = np.zeros(K, dtype=bool)
                colsel[top] = True
                colsel[:S] = False
                for gi in range(group):
                    h = g * group + gi
                    # exclude columns already inside each row's window
                    contrib = np.where(
                        colsel[None, :] & (np.arange(K)[None, :] <= (pos - W)[:, None]),
                        A[h],
                        0.0,
                    ).sum(axis=-1)
                    rec = base[h] + contrib
                    acc.add(layer, h, cand.key(), float(rec.mean()))


# ---------------------------------------------------------------------------
# Capture-mode prefill
# ---------------------------------------------------------------------------


def calibrate_model(
    model,
    tokenizer,
    budget: float = sp.DEFAULT_BUDGET,
    target_tokens: int = 16384,
    chunk_size: int = 2048,
    min_context: int = 4096,
    seeds: Tuple[int, ...] = (0, 1),
    progress: bool = True,
) -> Dict:
    """Run capture prefills and fit per-head patterns. Returns the JSON dict."""
    from mlx_lm.models import base as mlx_base
    from mlx_lm.models.cache import make_prompt_cache

    acc = _RecallAccumulator()
    sp._install_layer_taggers(model)
    original_sdpa = mlx_base.scaled_dot_product_attention

    def capture_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        layer = sp._STATE.current_layer
        L = queries.shape[-2]
        K = keys.shape[-2]
        if (
            layer is not None
            and sinks is None
            and L >= _MIN_CAPTURE_QLEN
            and K >= min_context
            and not hasattr(keys, "_state")
        ):
            try:
                evaluate_chunk(acc, layer, queries, keys, scale, budget)
            except Exception:
                logger.warning("calibration capture failed on layer %s", layer,
                               exc_info=True)
        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    import sys as _sys

    patched_mods = []
    mlx_base.scaled_dot_product_attention = capture_sdpa
    for mod_name, mod in list(_sys.modules.items()):
        if mod is None or not mod_name.startswith("mlx_lm.models."):
            continue
        if getattr(mod, "scaled_dot_product_attention", None) is original_sdpa:
            setattr(mod, "scaled_dot_product_attention", capture_sdpa)
            patched_mods.append(mod)

    try:
        for seed in seeds:
            tokens = build_calibration_tokens(tokenizer, target_tokens, seed=seed)
            cache = make_prompt_cache(model)
            prompt = mx.array(tokens)
            n = len(tokens)
            t0 = time.time()
            for start in range(0, n, chunk_size):
                chunk = prompt[start : start + chunk_size]
                model(chunk[None], cache=cache)
                mx.eval([c.state for c in cache])
                mx.clear_cache()
                if progress:
                    print(
                        f"\r  prompt seed={seed}: {min(start + chunk_size, n)}/{n} "
                        f"tokens ({time.time() - t0:.0f}s)",
                        end="",
                        flush=True,
                    )
            if progress:
                print()
            del cache
            mx.clear_cache()
    finally:
        mlx_base.scaled_dot_product_attention = original_sdpa
        for mod in patched_mods:
            setattr(mod, "scaled_dot_product_attention", original_sdpa)
        sp._remove_layer_taggers(model)

    # Fit: per head argmax candidate
    layers_out: Dict[str, List[Dict]] = {}
    all_recalls: List[float] = []
    kind_counts: Dict[str, int] = {}
    for layer in sorted(acc.data):
        heads_out = []
        for head in sorted(acc.data[layer]):
            best_key, best_recall, best_cand = None, -1.0, None
            for cand in _candidates():
                ck = cand.key()
                if ck not in acc.data[layer][head]:
                    continue
                r = acc.mean(layer, head, ck)
                if r > best_recall:
                    best_key, best_recall, best_cand = ck, r, cand
            if best_cand is None:
                continue
            window = (
                int(budget * target_tokens) - best_cand.sink
                if best_cand.kind == "a_shape"
                else best_cand.window
            )
            heads_out.append(
                {
                    "kind": best_cand.kind,
                    "sink": best_cand.sink,
                    "window": window,
                    "recall": round(best_recall, 4),
                }
            )
            all_recalls.append(best_recall)
            kind_counts[best_cand.kind] = kind_counts.get(best_cand.kind, 0) + 1
        if heads_out:
            layers_out[str(layer)] = heads_out

    mean_recall = float(np.mean(all_recalls)) if all_recalls else 0.0
    return {
        "version": 1,
        "budget": budget,
        "target_tokens": target_tokens,
        "mean_recall": round(mean_recall, 4),
        "pattern_counts": kind_counts,
        "layers": layers_out,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model path or HF id")
    parser.add_argument("--budget", type=float, default=sp.DEFAULT_BUDGET)
    parser.add_argument("--target-tokens", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--recall-gate", type=float, default=0.90)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    from mlx_lm import load

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)

    result = calibrate_model(
        model,
        tokenizer,
        budget=args.budget,
        target_tokens=args.target_tokens,
        chunk_size=args.chunk_size,
    )
    result["model"] = args.model

    out = args.output or sp.default_calibration_path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    print(f"\nPattern distribution: {result['pattern_counts']}")
    print(f"Mean recall @ budget {args.budget}: {result['mean_recall']:.3f}")
    print(f"Wrote {out}")

    if result["mean_recall"] < args.recall_gate:
        print(
            f"FAIL: mean recall {result['mean_recall']:.3f} < gate "
            f"{args.recall_gate} — sparse prefill would hurt quality at this "
            "budget; raise --budget and re-run.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
