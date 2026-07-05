#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding benchmark CLI for a running oMLX server.

Measures TTFT and decode throughput per workload scenario, plus n-gram
speculation acceptance stats from the admin API. Run once with the feature
off and once with it on (or use --ab to flip the setting automatically via
the admin API and run both).

Usage:
    uv run python scripts/spec_bench.py --model <model-id>            # single pass
    uv run python scripts/spec_bench.py --model <model-id> --ab       # off vs on
    uv run python scripts/spec_bench.py --model <model-id> --scenario code_edit

Requires: a server started with `omlx serve` on --base-url (default
http://localhost:8000) with admin auth disabled or --admin-token set.

Scenarios (all temp=0 so runs are comparable and lossless):
    summarize  — summarize a repetitive document (echo-heavy)
    code_edit  — "rewrite this function with a small change" (echo-heavy)
    rag        — answer with quotes from provided context (echo-heavy)
    freeform   — open-ended prose (control; expects ~neutral result)
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


def fetch_ngram_stats(base_url, token=None, reset=False):
    try:
        suffix = "?reset=true" if reset else ""
        return _get_json(
            f"{base_url}/admin/api/ngram-spec/stats{suffix}", token=token
        )["totals"]
    except Exception:
        return {}


def set_ngram_enabled(base_url, model, enabled, token=None):
    """Flip ngram_spec_enabled via the admin API (auto-reloads loaded models)."""
    with _request(
        f"{base_url}/admin/api/models/{model}/settings",
        payload={"ngram_spec_enabled": bool(enabled)},
        method="PUT",
        token=token,
    ) as resp:
        result = json.loads(resp.read())
    if result.get("requires_reload") and not (
        result.get("auto_reloaded") or not result.get("auto_unloaded")
    ):
        pass
    return result


def summarize_ngram_stats(stats):
    if not stats:
        return "(no ngram stats)"
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


def run_pass(base_url, model, scenarios, runs, warmup, token=None):
    results = {}
    for name in scenarios:
        print(f"  scenario: {name}")
        fetch_ngram_stats(base_url, token=token, reset=True)
        results[name] = run_scenario(
            base_url, model, name, runs, warmup, token=token
        )
        results[name]["ngram"] = fetch_ngram_stats(base_url, token=token)
        print(f"    ngram: {summarize_ngram_stats(results[name]['ngram'])}")
    return results


def print_comparison(off, on):
    print(f"\n{'scenario':<12} {'off tok/s':>10} {'on tok/s':>10} {'speedup':>8} "
          f"{'accept':>8} {'ttft off':>9} {'ttft on':>9}")
    for name in off:
        o, n = off[name], on[name]
        speedup = n["decode_tok_s"] / o["decode_tok_s"]
        ng = n.get("ngram") or {}
        proposed = ng.get("proposed_tokens", 0)
        acc = (
            f"{ng.get('accepted_tokens', 0) / proposed * 100:.0f}%"
            if proposed
            else "-"
        )
        print(
            f"{name:<12} {o['decode_tok_s']:>10.1f} {n['decode_tok_s']:>10.1f} "
            f"{speedup:>7.2f}x {acc:>8} {o['ttft_ms']:>8.0f}ms {n['ttft_ms']:>8.0f}ms"
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
        help="flip ngram_spec_enabled off->on via admin API and compare",
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

    if args.ab:
        print("== pass 1: ngram_spec OFF ==")
        set_ngram_enabled(args.base_url, args.model, False, token=token)
        time.sleep(2)
        off = run_pass(args.base_url, args.model, scenarios, args.runs, args.warmup, token)
        print("\n== pass 2: ngram_spec ON ==")
        set_ngram_enabled(args.base_url, args.model, True, token=token)
        time.sleep(2)
        on = run_pass(args.base_url, args.model, scenarios, args.runs, args.warmup, token)
        print_comparison(off, on)
        if args.json:
            print(json.dumps({"off": off, "on": on}, indent=2))
    else:
        results = run_pass(
            args.base_url, args.model, scenarios, args.runs, args.warmup, token
        )
        for name, r in results.items():
            print(
                f"{name:<12} ttft {r['ttft_ms']:.0f} ms, "
                f"{r['decode_tok_s']:.1f} tok/s | {summarize_ngram_stats(r['ngram'])}"
            )
        if args.json:
            print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
