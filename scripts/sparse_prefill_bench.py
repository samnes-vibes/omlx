#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Direct (no-server) prefill benchmark for draft-free sparse prefill.

Measures chunked-prefill wall time dense vs sparse at several prompt
lengths on the same model, and runs a needle-in-haystack QA check on both
paths (sparse prefill approximates attention, so token identity is not
expected — the needle answer must survive).

Usage:
    uv run python scripts/sparse_prefill_bench.py \
        --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
        [--lengths 4096,8192,16384,32768] [--budget 0.1] [--threshold 8192]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from omlx.patches import sparse_prefill as sp
from omlx.sparse_calibration import build_calibration_text


def make_needle_tokens(tokenizer, n_tokens: int, needle_pos: float = 0.5):
    """Chat-templated long-doc QA prompt with a mid-document needle,
    tokenized to ~n_tokens."""
    text = build_calibration_text(seed=99, approx_words=int(n_tokens * 1.1))
    paras = text.split("\n\n")
    needle = "NOTE: the magic checkpoint code is TANGERINE-42."
    paras.insert(int(len(paras) * needle_pos), needle)
    doc = "\n\n".join(paras)
    question = (
        "\n\nWhat is the magic checkpoint code mentioned in the NOTE in the "
        "document above? Reply with just the code. /no_think"
    )
    # Trim the document (not the question) to hit the target token count
    overhead = 64  # template + question tokens, approximate
    doc_toks = tokenizer.encode(doc)[: n_tokens - overhead]
    doc = tokenizer.decode(doc_toks)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": doc + question}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(prompt)


def chunked_prefill(model, tokens, chunk_size=2048):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    prompt = mx.array(tokens)
    n = len(tokens)
    t0 = time.perf_counter()
    logits = None
    for start in range(0, n, chunk_size):
        logits = model(prompt[start : start + chunk_size][None], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
    mx.eval(logits)
    dt = time.perf_counter() - t0
    return cache, logits, dt


def greedy_decode(model, cache, logits, tokenizer, n_steps=48):
    out = []
    y = mx.argmax(logits[:, -1, :], axis=-1)
    for _ in range(n_steps):
        out.append(int(y.item()))
        logits = model(y.reshape(1, 1), cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)
    return tokenizer.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="4096,8192,16384,32768")
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--threshold", type=int, default=8192)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--calibration", type=Path, default=None)
    args = ap.parse_args()

    from mlx_lm import load

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)

    calib_path = args.calibration or sp.default_calibration_path(args.model)
    data = sp.load_calibration(calib_path)
    print(
        f"Calibration: {calib_path.name}, mean recall "
        f"{data.get('mean_recall')}, patterns {data.get('pattern_counts')}"
    )

    lengths = [int(x) for x in args.lengths.split(",")]
    results = []
    for n in lengths:
        tokens = make_needle_tokens(tokenizer, n)
        row = {"tokens": len(tokens)}
        for mode in ("dense", "sparse"):
            if mode == "sparse":
                sp._STATE.patterns = sp._patterns_from_calibration(data)
                sp._STATE.budget = args.budget
                sp._STATE.threshold = args.threshold
                sp._install_layer_taggers(model)
                sp.apply_sparse_prefill_patch()
                sp._STATE.enabled = True
                sp._STATE.sparse_calls = sp._STATE.dense_calls = 0
                sp._STATE.keys_attended = sp._STATE.keys_total = 0
            else:
                sp._STATE.enabled = False

            # Warm-up pass at the smallest length only (kernel compile)
            cache, logits, dt = chunked_prefill(
                model, tokens, chunk_size=args.chunk_size
            )
            answer = greedy_decode(model, cache, logits, tokenizer)
            needle_ok = "TANGERINE-42" in answer or "TANGERINE" in answer
            row[mode] = {
                "prefill_s": round(dt, 2),
                "tok_per_s": round(len(tokens) / dt, 1),
                "needle_ok": needle_ok,
                "answer": answer.strip()[:60],
            }
            if mode == "sparse":
                row[mode]["sparse_calls"] = sp._STATE.sparse_calls
                row[mode]["density"] = (
                    round(sp._STATE.keys_attended / sp._STATE.keys_total, 3)
                    if sp._STATE.keys_total
                    else None
                )
                sp.deactivate_sparse_prefill(model)
            del cache
            gc.collect()
            mx.clear_cache()

        row["speedup"] = round(
            row["dense"]["prefill_s"] / row["sparse"]["prefill_s"], 2
        )
        results.append(row)
        print(json.dumps(row))

    print("\n| tokens | dense s | sparse s | speedup | density | needle d/s |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['tokens']} | {r['dense']['prefill_s']} | "
            f"{r['sparse']['prefill_s']} | {r['speedup']}x | "
            f"{r['sparse'].get('density')} | "
            f"{'Y' if r['dense']['needle_ok'] else 'N'}/"
            f"{'Y' if r['sparse']['needle_ok'] else 'N'} |"
        )


if __name__ == "__main__":
    main()
