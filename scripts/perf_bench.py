#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""General A/B performance benchmark CLI for a running oMLX server.

Measures TTFT and decode throughput per workload scenario, plus feature stats
from the admin API. Run once with a feature off and once with it on (or use
--ab to flip a model setting automatically via the admin API and run both).

Usage:
    uv run python scripts/perf_bench.py --model <model-id>            # single pass
    uv run python scripts/perf_bench.py --model <model-id> --ab       # off vs on
    uv run python scripts/perf_bench.py --model <model-id> --scenario code_edit
    uv run python scripts/perf_bench.py --model <model-id> --ab \
        --setting-key chunk_kv_reuse_enabled --stats-path admin/api/kv-reuse/stats

Requires: a server started with `omlx serve` on --base-url (default
http://localhost:8000) with admin auth disabled or --admin-token set.

Scenarios (all temp=0 so runs are comparable and lossless):
    summarize      — summarize a repetitive document (echo-heavy)
    code_edit      — "rewrite this function with a small change" (echo-heavy)
    rag            — answer with quotes from provided context (echo-heavy)
    freeform       — open-ended prose (control; expects ~neutral result)
    rag_permuted   — same context chunks, reordered per run (non-prefix reuse)
    agent_loop     — stable head + varying tail (simulated tool-output loop)
    multi_turn_edit — conversation with an edited middle turn across runs
    prefix_control — identical prompt every run (strict-prefix; expect ~0 gain)
    long_context   — ≥16K-token document QA (prefill-bound; for
                     --setting-key sparse_prefill_enabled A/B)

Scenarios with multiple prompt variants (rag_permuted, agent_loop,
multi_turn_edit) cycle through their variants across warmup+measured runs
instead of repeating one fixed prompt, so the harness can exercise cache
paths that a byte-identical repeat would never touch.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Scenario prompts
# ---------------------------------------------------------------------------

_DOC = """The oMLX server is a local inference server for Apple Silicon.
The server implements continuous batching for concurrent requests. The
server implements a paged KV cache with an SSD spill tier. The server
implements a prefix cache so repeated prompts skip prefill. The server
implements speculative decoding so generation can emit several tokens per
forward pass. The scheduler admits requests while a memory guard watches
the unified memory ceiling. The memory guard throttles prefill when the
projected usage would exceed the ceiling. The admin dashboard exposes
model settings, benchmarks and download management for the server.
""" * 3

_CODE = '''def process_records(records, filters, transform=None):
    """Apply filters then an optional transform to each record."""
    results = []
    for record in records:
        keep = True
        for f in filters:
            if not f(record):
                keep = False
                break
        if not keep:
            continue
        if transform is not None:
            record = transform(record)
        results.append(record)
    return results
'''

# Independent content blocks for the chunk-reuse scenarios below: each is
# self-contained so it can be relocated to any position in a prompt without
# reading oddly, which is what lets rag_permuted reorder them per run.
_CHUNKS = {
    "memory": "The scheduler admits requests while a memory guard watches "
    "the unified memory ceiling. The memory guard throttles prefill when "
    "the projected usage would exceed the ceiling.\n",
    "cache": "The server implements a paged KV cache with an SSD spill "
    "tier. The server implements a prefix cache so repeated prompts skip "
    "prefill.\n",
    "spec": "The server implements speculative decoding so generation can "
    "emit several tokens per forward pass.\n",
    "admin": "The admin dashboard exposes model settings, benchmarks and "
    "download management for the server.\n",
}

_RAG_PERMUTED_ORDERS = [
    ("memory", "cache", "spec", "admin"),
    ("cache", "admin", "memory", "spec"),
    ("spec", "memory", "admin", "cache"),
]


def _rag_permuted_variant(order):
    context = "".join(_CHUNKS[name] for name in order)
    return [
        {
            "role": "user",
            "content": "Context:\n" + context + "\n\nUsing only the context "
            "above, explain what the memory guard does. Quote the relevant "
            "sentences verbatim in your answer.",
        }
    ]


_AGENT_LOOP_HEAD = (
    "You are an agent working through a multi-step task. Shared project "
    "context:\n\n" + _DOC
)
_AGENT_LOOP_TAILS = [
    "Tool output (step 1): file src/a.py changed, 12 lines added. "
    "Summarize the change and suggest the next tool call.",
    "Tool output (step 2): tests failed: 2 failures in test_cache.py. "
    "Summarize the failures and suggest the next tool call.",
    "Tool output (step 3): lint passed, 0 warnings. Summarize the status "
    "and suggest the next tool call.",
]


def _agent_loop_variant(tail):
    return [
        {"role": "user", "content": _AGENT_LOOP_HEAD},
        {"role": "assistant", "content": "Understood, I'll track progress across steps."},
        {"role": "user", "content": tail},
    ]


_MULTI_TURN_HEAD = "Context:\n" + _DOC + "\n\nWhat does the prefix cache do?"
_MULTI_TURN_MIDDLES = [
    "The prefix cache skips prefill for repeated prompts by reusing "
    "previously computed KV entries for a shared prefix.",
    "It matches the new request's token prefix against cached blocks and "
    "reuses their KV state instead of recomputing them.",
    "Repeated prompt prefixes hit a cache of already-computed key/value "
    "tensors, avoiding redundant prefill work.",
]
_MULTI_TURN_TAIL = "Now do the same for the memory guard, quoting verbatim."


def _multi_turn_edit_variant(middle):
    return [
        {"role": "user", "content": _MULTI_TURN_HEAD},
        {"role": "assistant", "content": middle},
        {"role": "user", "content": _MULTI_TURN_TAIL},
    ]


def _long_context_doc(approx_words: int = 11000) -> str:
    """Deterministic aperiodic long document (~16K+ tokens) for the
    long_context scenario — the length regime sparse prefill targets."""
    import random

    rng = random.Random(7)
    topics = [
        "unified memory bandwidth", "scheduler admission", "KV cache spill",
        "speculative acceptance", "quantization groups", "prefill chunking",
        "thermal throttling", "tokenizer merges", "rotary embeddings",
    ]
    verbs = ["improves", "constrains", "dominates", "amortizes", "regresses"]
    parts, words, section = [], 0, 0
    while words < approx_words:
        section += 1
        sents = []
        for _ in range(rng.randint(4, 8)):
            t1, t2 = rng.sample(topics, 2)
            sents.append(
                f"Section {section}: {t1} {rng.choice(verbs)} {t2} beyond "
                f"{rng.randint(2, 64)} concurrent requests."
            )
        parts.append(" ".join(sents))
        words += len(parts[-1].split())
    # Needle for a QA check roughly mid-document
    parts.insert(len(parts) // 2, "NOTE: the magic checkpoint code is TANGERINE-42.")
    return "\n\n".join(parts)


SCENARIOS = {
    "summarize": {
        "messages": [
            {
                "role": "user",
                "content": "Summarize the following document in about ten "
                "sentences, quoting key phrases verbatim where possible:\n\n"
                + _DOC,
            }
        ],
        "max_tokens": 300,
    },
    "code_edit": {
        "messages": [
            {
                "role": "user",
                "content": "Rewrite this function so that it counts how many "
                "records were filtered out and returns (results, dropped_count). "
                "Keep everything else identical and show the full function:\n\n"
                + _CODE,
            }
        ],
        "max_tokens": 300,
    },
    "rag": {
        "messages": [
            {
                "role": "user",
                "content": "Context:\n" + _DOC + "\n\nUsing only the context "
                "above, explain what the memory guard does. Quote the relevant "
                "sentences verbatim in your answer.",
            }
        ],
        "max_tokens": 250,
    },
    "long_context": {
        # ≥16K-token document QA: the prefill-bound regime that prefill-side
        # features (sparse_prefill_enabled, specprefill_enabled) target.
        "messages": [
            {
                "role": "user",
                "content": "Context:\n" + _long_context_doc() + "\n\nWhat is "
                "the magic checkpoint code mentioned in the NOTE? Answer with "
                "just the code.",
            }
        ],
        "max_tokens": 30,
    },
    "freeform": {
        "messages": [
            {
                "role": "user",
                "content": "Write a short story about a lighthouse keeper who "
                "discovers something unusual on the shore.",
            }
        ],
        "max_tokens": 300,
    },
    "rag_permuted": {
        "variants": [_rag_permuted_variant(order) for order in _RAG_PERMUTED_ORDERS],
        "max_tokens": 250,
    },
    "agent_loop": {
        "variants": [_agent_loop_variant(tail) for tail in _AGENT_LOOP_TAILS],
        "max_tokens": 200,
    },
    "multi_turn_edit": {
        "variants": [
            _multi_turn_edit_variant(middle) for middle in _MULTI_TURN_MIDDLES
        ],
        "max_tokens": 250,
    },
    "prefix_control": {
        "messages": [
            {
                "role": "user",
                "content": "Summarize the following document in about ten "
                "sentences, quoting key phrases verbatim where possible:\n\n"
                + _DOC,
            }
        ],
        "max_tokens": 300,
    },
}


def _variants(scenario):
    """Return the list of message-lists for a scenario (>=1 entries)."""
    if "variants" in scenario:
        return scenario["variants"]
    return [scenario["messages"]]


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------


_ADMIN_COOKIE = {"value": None}


def _request(url, payload=None, method=None, token=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if "/admin/" in url and _ADMIN_COOKIE["value"]:
        req.add_header("Cookie", f"omlx_admin_session={_ADMIN_COOKIE['value']}")
    return urllib.request.urlopen(req, timeout=timeout)


def admin_login(base_url, api_key):
    """Fetch an admin session cookie via /admin/auto-login?key=..."""
    import http.cookiejar

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    try:
        opener.open(f"{base_url}/admin/auto-login?key={api_key}", timeout=30)
    except urllib.error.HTTPError:
        pass
    for cookie in jar:
        if cookie.name == "omlx_admin_session":
            _ADMIN_COOKIE["value"] = cookie.value
            return True
    return False


def _get_json(url, token=None):
    with _request(url, token=token) as resp:
        return json.loads(resp.read())


def _post_json(url, payload, token=None):
    with _request(url, payload=payload, token=token) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def run_streaming_completion(
    base_url, model, messages, max_tokens, token=None, cache_bust=False
):
    """One streaming chat completion; returns (ttft_s, decode_s, n_tokens)."""
    if cache_bust:
        # Unique text at the *start* of the first message changes the token
        # prefix, so the server's prefix cache cannot skip any prefill work.
        import uuid

        messages = [dict(m) for m in messages]
        messages[0]["content"] = f"[bench {uuid.uuid4().hex}]\n{messages[0]['content']}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url}/v1/chat/completions"
    start = time.perf_counter()
    ttft = None
    last_chunk_t = None
    completion_tokens = None
    chunks = 0
    with _request(url, payload=payload, method="POST", token=token) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                completion_tokens = usage["completion_tokens"]
            for choice in chunk.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    chunks += 1
                    now = time.perf_counter() - start
                    if ttft is None:
                        ttft = now
                    last_chunk_t = now
    total = time.perf_counter() - start
    if ttft is None:
        ttft = total
    n_tokens = completion_tokens if completion_tokens else chunks
    # Some transports (e.g. urllib buffering a fast local response) deliver
    # every content chunk in one burst, making ttft ≈ total and collapsing
    # the decode window to ~0 — dividing tokens by that near-zero window
    # produces nonsense tok/s (seen: 400k+ on a real 27B model). Detect the
    # degenerate split (last chunk arrived within 1ms of the first, or only
    # one chunk total) and fall back to spreading the *whole* request time
    # over the decode calculation instead of a clamped near-zero window.
    decode_s = (last_chunk_t or total) - ttft
    if chunks <= 1 or decode_s < 1e-3:
        if not _WARNED_BURST["seen"]:
            print(
                "warning: chunk timing looks bursty (ttft ≈ total) — "
                "falling back to whole-request timing for tok/s; treat "
                "these numbers as approximate",
                file=sys.stderr,
            )
            _WARNED_BURST["seen"] = True
        decode_s = total
        ttft = 0.0
    return ttft, max(decode_s, 1e-9), n_tokens


_WARNED_BURST = {"seen": False}


def run_scenario(base_url, model, name, runs, warmup, token=None, cache_bust=False):
    scenario = SCENARIOS[name]
    max_tokens = scenario["max_tokens"]
    variant_cycle = itertools.cycle(_variants(scenario))
    for _ in range(warmup):
        run_streaming_completion(
            base_url, model, next(variant_cycle), max_tokens, token=token,
            cache_bust=cache_bust,
        )
    ttfts, rates, tokens = [], [], []
    for i in range(runs):
        ttft, decode_s, n = run_streaming_completion(
            base_url, model, next(variant_cycle), max_tokens, token=token,
            cache_bust=cache_bust,
        )
        ttfts.append(ttft)
        rates.append(n / decode_s)
        tokens.append(n)
        print(
            f"    run {i + 1}/{runs}: {n} tok, ttft {ttft * 1000:.0f} ms, "
            f"{n / decode_s:.1f} tok/s",
            flush=True,
        )
    return {
        "ttft_ms": statistics.median(ttfts) * 1000,
        "decode_tok_s": statistics.median(rates),
        "tokens": statistics.median(tokens),
    }


def _context_prompt(n_tokens):
    """Deterministic filler prompt of roughly ``n_tokens`` tokens.

    Reuses the aperiodic long-document generator (~0.75 words/token is a
    close enough English heuristic for context *points*, not exact counts)
    plus a fixed question that forces a real completion.
    """
    doc = _long_context_doc(approx_words=max(int(n_tokens * 0.72), 16))
    return [
        {
            "role": "user",
            "content": "Context:\n" + doc + "\n\nSummarize the recurring "
            "themes of the sections above in a few sentences.",
        }
    ]


_SPEC_STATS_PATH = "admin/api/spec-decode/stats"


def _spec_row(path_name, totals, tok_s):
    """Distill one path's cumulative counters into a table row dict."""
    cycles = totals.get("cycles", 0)
    row = {"path": path_name, "tok_s": tok_s, "cycles": cycles}
    if path_name == "mtp":
        drafts = totals.get("draft_tokens", 0)
        row["alpha"] = totals.get("accepted_drafts", 0) / cycles if cycles else None
        row["accept_rate"] = (
            totals.get("accepted_drafts", 0) / drafts if drafts else None
        )
        row["draft_ms"] = totals.get("draft_ms", 0.0) / cycles if cycles else None
        row["verify_ms"] = totals.get("verify_ms", 0.0) / cycles if cycles else None
    elif path_name == "ngram":
        proposed = totals.get("proposed_tokens", 0)
        row["alpha"] = totals.get("accepted_tokens", 0) / cycles if cycles else None
        row["accept_rate"] = (
            totals.get("accepted_tokens", 0) / proposed if proposed else None
        )
        row["draft_ms"] = totals.get("propose_ms", 0.0) / cycles if cycles else None
        row["verify_ms"] = totals.get("backbone_ms", 0.0) / cycles if cycles else None
    elif path_name == "dflash":
        row["alpha"] = None
        row["accept_rate"] = (
            totals.get("acceptance_weighted", 0.0) / cycles if cycles else None
        )
        row["draft_ms"] = None  # loop lives inside dflash_mlx; not visible
        row["verify_ms"] = None
    return row


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.2f}"


def run_spec_breakdown(args, token):
    """mirror-sd P0.1: per-context-point speculation breakdown table.

    For each context point: reset the cumulative spec counters, run the
    measured completions, then read back /admin/api/spec-decode/stats and
    report per-cycle draft/verify time and acceptance alongside tok/s.
    Whichever speculative path was active shows up with cycles > 0; rows
    for idle paths are skipped.
    """
    contexts = [int(c) for c in args.contexts.split(",")]
    results = {}
    for ctx_len in contexts:
        print(f"== context {ctx_len} ==", flush=True)
        messages = _context_prompt(ctx_len)
        for _ in range(args.warmup):
            run_streaming_completion(
                args.base_url, args.model, messages, args.max_tokens,
                token=token, cache_bust=args.cache_bust,
            )
        fetch_stats(args.base_url, _SPEC_STATS_PATH, token=token, reset=True)
        rates = []
        for i in range(args.runs):
            ttft, decode_s, n = run_streaming_completion(
                args.base_url, args.model, messages, args.max_tokens,
                token=token, cache_bust=args.cache_bust,
            )
            rates.append(n / decode_s)
            print(
                f"    run {i + 1}/{args.runs}: {n} tok, "
                f"ttft {ttft * 1000:.0f} ms, {n / decode_s:.1f} tok/s",
                flush=True,
            )
        tok_s = statistics.median(rates)
        totals = fetch_stats(args.base_url, _SPEC_STATS_PATH, token=token)
        rows = [
            _spec_row(name, path_totals, tok_s)
            for name, path_totals in (totals or {}).items()
            if isinstance(path_totals, dict) and path_totals.get("cycles")
        ]
        results[ctx_len] = rows or [
            {"path": "none", "tok_s": tok_s, "cycles": 0, "alpha": None,
             "accept_rate": None, "draft_ms": None, "verify_ms": None}
        ]

    print()
    print(
        f"{'context':>8} {'path':<7} {'tok/s':>8} {'alpha':>7} "
        f"{'accept':>8} {'draft ms/cyc':>13} {'verify ms/cyc':>14}"
    )
    for ctx_len, rows in results.items():
        for row in rows:
            print(
                f"{ctx_len:>8} {row['path']:<7} {row['tok_s']:>8.1f} "
                f"{_fmt(row.get('alpha')):>7} "
                f"{_fmt(row.get('accept_rate'), pct=True):>8} "
                f"{_fmt(row.get('draft_ms')):>13} "
                f"{_fmt(row.get('verify_ms')):>14}"
            )
    if args.json:
        print(json.dumps({str(k): v for k, v in results.items()}, indent=2))
    return 0


def fetch_stats(base_url, stats_path, token=None, reset=False):
    try:
        sep = "&" if "?" in stats_path else "?"
        suffix = f"{sep}reset=true" if reset else ""
        data = _get_json(f"{base_url}/{stats_path.lstrip('/')}{suffix}", token=token)
        return data.get("totals", data)
    except Exception:
        return {}


def set_setting_value(base_url, model, setting_key, value, token=None):
    """Set a model setting via the admin API (auto-reloads loaded models)."""
    with _request(
        f"{base_url}/admin/api/models/{model}/settings",
        payload={setting_key: value},
        method="PUT",
        token=token,
    ) as resp:
        result = json.loads(resp.read())
    return result


def set_setting_enabled(base_url, model, setting_key, enabled, token=None):
    """Flip a boolean model setting via the admin API (auto-reloads loaded models)."""
    return set_setting_value(base_url, model, setting_key, bool(enabled), token=token)


def summarize_stats(stats):
    """Best-effort human-readable summary; falls back to raw key=value pairs
    for feature stats shapes this script doesn't know about (e.g. a future
    chunk-KV-reuse stats endpoint)."""
    if not stats:
        return "(no stats)"
    if "proposed_tokens" in stats:
        proposed = stats.get("proposed_tokens", 0)
        accepted = stats.get("accepted_tokens", 0)
        cycles = stats.get("cycles", 0)
        plain = stats.get("plain_steps", 0)
        rate = f"{accepted / proposed * 100:.1f}%" if proposed else "n/a"
        total_steps = cycles + plain
        emits = (
            stats.get("init_emits", 0)
            + stats.get("draft_emits", 0)
            + stats.get("plain_emits", 0)
        )
        tpc = f"{emits / total_steps:.2f}" if total_steps else "n/a"
        return (
            f"cycles={cycles} plain={plain} accept={accepted}/{proposed} ({rate}) "
            f"tokens/step={tpc}"
        )
    return " ".join(f"{k}={v}" for k, v in stats.items())


def run_pass(base_url, model, scenarios, runs, warmup, stats_path, token=None,
             cache_bust=False):
    results = {}
    for name in scenarios:
        print(f"  scenario: {name}")
        fetch_stats(base_url, stats_path, token=token, reset=True)
        results[name] = run_scenario(
            base_url, model, name, runs, warmup, token=token, cache_bust=cache_bust
        )
        results[name]["stats"] = fetch_stats(base_url, stats_path, token=token)
        print(f"    stats: {summarize_stats(results[name]['stats'])}")
    return results


def print_comparison(off, on):
    print(f"\n{'scenario':<16} {'off tok/s':>10} {'on tok/s':>10} {'speedup':>8} "
          f"{'ttft off':>9} {'ttft on':>9}")
    for name in off:
        o, n = off[name], on[name]
        speedup = n["decode_tok_s"] / o["decode_tok_s"]
        print(
            f"{name:<16} {o['decode_tok_s']:>10.1f} {n['decode_tok_s']:>10.1f} "
            f"{speedup:>7.2f}x {o['ttft_ms']:>8.0f}ms {n['ttft_ms']:>8.0f}ms"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True, help="model id as served")
    ap.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="repeatable; default: all scenarios",
    )
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--api-key",
        default=None,
        help="oMLX API key; used as Bearer for /v1 and for admin auto-login",
    )
    ap.add_argument(
        "--ab",
        action="store_true",
        help="flip --setting-key off->on via admin API and compare",
    )
    ap.add_argument(
        "--setting-key",
        default="ngram_spec_enabled",
        help="boolean model-settings field to flip with --ab "
        "(default: ngram_spec_enabled)",
    )
    ap.add_argument(
        "--spec-breakdown",
        action="store_true",
        help="mirror-sd P0.1: per-context-point speculation breakdown "
        "(tok/s, alpha, accept rate, draft/verify ms per cycle) for the "
        "currently active speculative path",
    )
    ap.add_argument(
        "--contexts",
        default="64,512,2048,4096",
        help="comma-separated context points (approx prompt tokens) for "
        "--spec-breakdown (default: 64,512,2048,4096)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="completion length per --spec-breakdown run (default: 128)",
    )
    ap.add_argument(
        "--sweep-values",
        default=None,
        help=(
            "comma-separated JSON values to sweep --setting-key over "
            "(e.g. 1,2,3,4); one pass per value, compared against the first"
        ),
    )
    ap.add_argument(
        "--stats-path",
        default="admin/api/ngram-spec/stats",
        help="admin API path (relative to --base-url) returning feature stats "
        "(default: admin/api/ngram-spec/stats)",
    )
    ap.add_argument(
        "--cache-bust",
        action="store_true",
        help="prepend a unique nonce to each request so the server's prefix "
        "cache never skips prefill (required for TTFT/prefill A/Bs)",
    )
    ap.add_argument("--json", action="store_true", help="emit raw JSON results")
    args = ap.parse_args()

    scenarios = args.scenario or sorted(SCENARIOS)
    token = args.api_key
    if token and not admin_login(args.base_url, token):
        print("warning: admin auto-login failed; admin API calls may 401",
              file=sys.stderr)

    try:
        _get_json(f"{args.base_url}/v1/models", token=token)
    except urllib.error.URLError as e:
        print(f"error: server not reachable at {args.base_url}: {e}", file=sys.stderr)
        return 1

    if args.spec_breakdown:
        return run_spec_breakdown(args, token)

    if args.sweep_values:
        values = [json.loads(v) for v in args.sweep_values.split(",")]
        passes = {}
        for value in values:
            print(f"== pass: {args.setting_key} = {value!r} ==")
            set_setting_value(args.base_url, args.model, args.setting_key, value,
                              token=token)
            time.sleep(2)
            passes[str(value)] = run_pass(args.base_url, args.model, scenarios,
                                          args.runs, args.warmup, args.stats_path,
                                          token)
            print()
        base_key = str(values[0])
        print(f"{'scenario':<12} " + " ".join(
            f"{f'{v} tok/s':>12}" for v in passes) + f" {'best':>8}")
        for name in scenarios:
            rates = {v: passes[v][name]["decode_tok_s"] for v in passes}
            best = max(rates, key=rates.get)
            cells = " ".join(f"{rates[v]:>12.1f}" for v in passes)
            rel = rates[best] / rates[base_key]
            print(f"{name:<12} {cells} {best:>4} {rel:.2f}x")
        if args.json:
            print(json.dumps(passes, indent=2))
    elif args.ab:
        print(f"== pass 1: {args.setting_key} OFF ==")
        set_setting_enabled(args.base_url, args.model, args.setting_key, False, token=token)
        time.sleep(2)
        off = run_pass(
            args.base_url, args.model, scenarios, args.runs, args.warmup,
            args.stats_path, token, cache_bust=args.cache_bust,
        )
        print(f"\n== pass 2: {args.setting_key} ON ==")
        set_setting_enabled(args.base_url, args.model, args.setting_key, True, token=token)
        time.sleep(2)
        on = run_pass(
            args.base_url, args.model, scenarios, args.runs, args.warmup,
            args.stats_path, token, cache_bust=args.cache_bust,
        )
        print_comparison(off, on)
        if args.json:
            print(json.dumps({"off": off, "on": on}, indent=2))
    else:
        results = run_pass(
            args.base_url, args.model, scenarios, args.runs, args.warmup,
            args.stats_path, token, cache_bust=args.cache_bust,
        )
        for name, r in results.items():
            print(
                f"{name:<16} ttft {r['ttft_ms']:.0f} ms, "
                f"{r['decode_tok_s']:.1f} tok/s | {summarize_stats(r['stats'])}"
            )
        if args.json:
            print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
