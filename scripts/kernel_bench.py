#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: fused TurboQuant verify attention vs dequantize+SDPA.

Isolated kernel latency (not end-to-end — use scripts/perf_bench.py for that).
Reports µs/call for each (q_len, context) cell and the fused/dequant speedup;
the deliverable is the crossover point.

Usage:
    python scripts/kernel_bench.py [--contexts 2048 8192 32768] \
        [--q-lens 2 8 16 32] [--kv-heads 4] [--q-heads 32] [--dim 128] \
        [--iters 50]
"""

import argparse
import time

import mlx.core as mx


def _bench(fn, iters, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # µs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192, 32768])
    ap.add_argument("--q-lens", type=int, nargs="+", default=[2, 8, 16, 32])
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--q-heads", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    import mlx_vlm.turboquant as tq

    from omlx.custom_kernels.tq_attention import fused_verify_attention

    D = args.dim
    scale = D**-0.5
    print(
        f"q_heads={args.q_heads} kv_heads={args.kv_heads} D={D} "
        f"bits={args.bits} iters={args.iters}"
    )
    print(f"{'ctx':>8} {'q_len':>6} {'fused µs':>10} {'dequant µs':>11} {'speedup':>8}")

    for T in args.contexts:
        cache = tq.TurboQuantKVCache(bits=float(args.bits), seed=0)
        mx.random.seed(T)
        keys = mx.random.normal((1, args.kv_heads, T, D)).astype(mx.float16)
        values = mx.random.normal((1, args.kv_heads, T, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        ks, vs = cache.state

        for L in args.q_lens:
            q = mx.random.normal((1, args.q_heads, L, D)).astype(mx.float16)

            gated = (
                fused_verify_attention(cache, q, ks, vs, scale=scale, mask="causal")
                is None
            )

            def fused():
                out = fused_verify_attention(
                    cache, q, ks, vs, scale=scale, mask="causal", force=True
                )
                assert out is not None
                return out

            def dequant():
                k, v = cache.dequantize()
                cols = mx.arange(T)[None, :]
                rows = (T - L) + mx.arange(L)[:, None]
                mask = mx.where(cols <= rows, 0.0, mx.finfo(mx.float32).min)
                return mx.fast.scaled_dot_product_attention(
                    q,
                    k.astype(q.dtype),
                    v.astype(q.dtype),
                    scale=scale,
                    mask=mask.astype(q.dtype),
                )

            fus = _bench(fused, args.iters)
            deq = _bench(dequant, args.iters)
            note = "  (gated: falls back in prod)" if gated else ""
            print(
                f"{T:>8} {L:>6} {fus:>10.1f} {deq:>11.1f} {deq / fus:>7.2f}x{note}"
            )


if __name__ == "__main__":
    main()
