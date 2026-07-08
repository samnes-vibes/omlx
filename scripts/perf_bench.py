#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generic A/B performance benchmark CLI for a running oMLX server.

Measures TTFT and decode throughput per workload scenario against a live
server, optionally flipping a single admin model-setting off/on between
passes so an optimization's real-world impact can be compared directly.

Usage:
    uv run python scripts/perf_bench.py --model <model-id>
    uv run python scripts/perf_bench.py --model <model-id> \
        --ab --setting-key ngram_spec_enabled
    uv run python scripts/perf_bench.py --model <model-id> --scenario code_edit

Requires: a server started with `omlx serve` on --base-url (default
http://localhost:8000) with admin auth disabled or --api-key set.

Scenarios (all temp=0 so runs are comparable and lossless):
    summarize  — summarize a repetitive document (echo-heavy)
    code_edit  — "rewrite this function with a small change" (echo-heavy)
    rag        — answer with quotes from provided context (echo-heavy)
    freeform   — open-ended prose (control; expects ~neutral result)
    long_context — ≥16K-token document QA (prefill-bound; for
                   --setting-key sparse_prefill_enabled A/B)

To adapt for a new optimization branch: pass --setting-key <your_flag> to
toggle the relevant model setting via the admin API, and optionally
--stats-path <admin/api/...> to fetch/reset feature-specific stats between
passes (printed as raw JSON alongside each scenario's results).
"""

from __future__ import annotations

import argparse
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

def _long_context_doc(approx_words: int = 20000) -> str:
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
}


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


def run_streaming_completion(base_url, model, scenario, token=None):
    """One streaming chat completion; returns (ttft_s, decode_s, n_tokens)."""
    payload = {
        "model": model,
        "messages": scenario["messages"],
        "max_tokens": scenario["max_tokens"],
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url}/v1/chat/completions"
    start = time.perf_counter()
    ttft = None
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
                    if ttft is None:
                        ttft = time.perf_counter() - start
    total = time.perf_counter() - start
    if ttft is None:
        ttft = total
    n_tokens = completion_tokens if completion_tokens else chunks
    return ttft, max(total - ttft, 1e-9), n_tokens


def run_scenario(base_url, model, name, runs, warmup, token=None):
    scenario = SCENARIOS[name]
    for _ in range(warmup):
        run_streaming_completion(base_url, model, scenario, token=token)
    ttfts, rates, tokens = [], [], []
    for i in range(runs):
        ttft, decode_s, n = run_streaming_completion(
            base_url, model, scenario, token=token
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


def fetch_admin_stats(base_url, stats_path, token=None, reset=False):
    """Fetch feature-specific stats from an admin endpoint, if configured."""
    if not stats_path:
        return {}
    try:
        sep = "&" if "?" in stats_path else "?"
        suffix = f"{sep}reset=true" if reset else ""
        data = _get_json(f"{base_url}/{stats_path.lstrip('/')}{suffix}", token=token)
        return data.get("totals", data)
    except Exception:
        return {}


def set_setting_enabled(base_url, model, setting_key, enabled, token=None):
    """Flip a boolean model setting via the admin API (auto-reloads loaded models)."""
    with _request(
        f"{base_url}/admin/api/models/{model}/settings",
        payload={setting_key: bool(enabled)},
        method="PUT",
        token=token,
    ) as resp:
        return json.loads(resp.read())


def summarize_stats(stats):
    if not stats:
        return "(no stats)"
    return ", ".join(f"{k}={v}" for k, v in stats.items())


def run_pass(base_url, model, scenarios, runs, warmup, stats_path, token=None):
    results = {}
    for name in scenarios:
        print(f"  scenario: {name}")
        fetch_admin_stats(base_url, stats_path, token=token, reset=True)
        results[name] = run_scenario(
            base_url, model, name, runs, warmup, token=token
        )
        results[name]["stats"] = fetch_admin_stats(base_url, stats_path, token=token)
        print(f"    stats: {summarize_stats(results[name]['stats'])}")
    return results


def print_comparison(off, on):
    print(f"\n{'scenario':<12} {'off tok/s':>10} {'on tok/s':>10} {'speedup':>8} "
          f"{'ttft off':>9} {'ttft on':>9}")
    for name in off:
        o, n = off[name], on[name]
        speedup = n["decode_tok_s"] / o["decode_tok_s"]
        print(
            f"{name:<12} {o['decode_tok_s']:>10.1f} {n['decode_tok_s']:>10.1f} "
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
        default=None,
        help="model settings key to toggle for --ab (e.g. ngram_spec_enabled)",
    )
    ap.add_argument(
        "--stats-path",
        default=None,
        help="admin API path for feature stats, e.g. admin/api/ngram-spec/stats",
    )
    ap.add_argument("--json", action="store_true", help="emit raw JSON results")
    args = ap.parse_args()

    if args.ab and not args.setting_key:
        print("error: --ab requires --setting-key", file=sys.stderr)
        return 1

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

    if args.ab:
        print(f"== pass 1: {args.setting_key} OFF ==")
        set_setting_enabled(args.base_url, args.model, args.setting_key, False, token=token)
        time.sleep(2)
        off = run_pass(args.base_url, args.model, scenarios, args.runs, args.warmup,
                        args.stats_path, token)
        print(f"\n== pass 2: {args.setting_key} ON ==")
        set_setting_enabled(args.base_url, args.model, args.setting_key, True, token=token)
        time.sleep(2)
        on = run_pass(args.base_url, args.model, scenarios, args.runs, args.warmup,
                       args.stats_path, token)
        print_comparison(off, on)
        if args.json:
            print(json.dumps({"off": off, "on": on}, indent=2))
    else:
        results = run_pass(args.base_url, args.model, scenarios, args.runs,
                            args.warmup, args.stats_path, token)
        for name, r in results.items():
            print(
                f"{name:<12} ttft {r['ttft_ms']:.0f} ms, "
                f"{r['decode_tok_s']:.1f} tok/s | {summarize_stats(r['stats'])}"
            )
        if args.json:
            print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
