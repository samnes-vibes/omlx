# Low-RAM Perf Optimization Map

Date: 2026-07-04
Base commit: `eca4a7c5`
Companion to: [low_ram_qwen_perf_investigation.md](low_ram_qwen_perf_investigation.md) (hardware constraints, model inventory, measurement tooling — read that first)

## Scope and framing

This document maps performance optimization targets in oMLX for the memory-constrained reference machine (M1 Mac mini, 8 GB unified memory, effective serving budget ~3.7 GB covering weights + KV cache + runtime). Every item below is filtered through that constraint: **memory is the only scarce resource on this box; compute is not.** Targets are grouped into three buckets:

1. **Low-hanging fruit** — small code changes or pure configuration, measurable with the existing admin benchmark.
2. **Needs benchmark/test-suite improvements first** — the optimization is plausible but the measurement to validate it does not exist yet.
3. **Other libraries' territory** — root cause lives in MLX, mlx-lm, mlx-vlm, or transformers; oMLX can only work around it.

---

## 0. Prerequisites (before any optimization work)

1. **Commit the environment fixes** — `.python-version`, `omlx/_transformers_compat.py`, and the import line in `omlx/__init__.py`. Without them `uv run` does not work at all on a clean install (see the investigation doc, "Uncommitted repo changes"). A `git stash` or fresh checkout reintroduces both breakages.
2. **Capture a baseline** — one clean PP/TG run for `Qwen3.5-0.8B-MLX-4bit` via the admin benchmark (`omlx/admin/benchmark.py`, one-click from the dashboard). No optimization lands without a before/after pair.

---

## 1. Low-hanging fruit

### 1a. Small-system static reserve is over-conservative

`omlx/process_memory_enforcer.py:53` — any system under 24 GB total RAM gets a **flat 4 GB static reserve regardless of memory-guard tier**; on an 8 GB machine half the RAM is always withheld.

Two angles:

- **Zero-code lever that already exists:** the `custom` tier (`--memory-guard-gb`, see `_get_static_ceiling` at `omlx/process_memory_enforcer.py:495-496`) bypasses the small-system reserve and applies only a 2 GB static reserve. `--memory-guard-gb 5` raises the ceiling from ~3.7 GB to ~5–6 GB on this box today.
- **Code change:** make the small-system reserve tier-aware (e.g. `aggressive` → 3 GB) instead of flat. Small, easily testable diff; the reclaimed space is directly usable KV-cache room.

### 1b. `mx.set_cache_limit(total_mem)` lets the MLX buffer pool eat the ceiling

`omlx/cli.py:324` sets the MLX buffer-cache limit to full physical RAM. The pool is only trimmed every 512 steps (`mlx_cache_cleanup_interval`, `omlx/scheduler.py:1334`) plus the deferred clear after request completion. On 8 GB, pool bloat counts against the enforcer's dynamic ceiling and triggers prefill throttling for memory that is actually reclaimable.

Fix shape: clamp the cache limit to the enforcer's hard ceiling on small systems (`min(total_mem, ceiling)`).

**Caution:** the limit exists to keep freed Metal buffers pooled (issue #300 — `buf->release()` while the GPU still uses the buffer causes kernel panics). The clamped limit must stay above the peak working set; test carefully.

### 1c. `--max-concurrent-requests` default of 8 is wrong for this budget

Default `max_num_seqs: 8` (`omlx/config.py:89`, CLI help at `omlx/cli.py:846`). With ~3.7 GB total, admitting 8 concurrent requests means admission immediately collides with the prefill memory guard (throttle/retry churn instead of useful work).

Fix shape: derive the default from the memory ceiling and loaded model size (roughly `ceiling − weights` divided by expected per-request KV footprint) instead of a constant. Measurable now with the batch benchmark at sizes 2/4.

### 1d. TurboQuant KV is off by default — highest-leverage setting on a KV-starved box

`turboquant_kv_enabled: bool = False` (`omlx/model_settings.py:145`); code and tests (`tests/test_turboquant*.py`) already exist. 4-bit KV roughly quadruples usable context/concurrency in the same memory.

- **Zero-code:** enable per-model in model settings and benchmark quality + throughput on `Qwen3.5-0.8B-MLX-4bit`.
- **Small code change:** auto-enable (or surface a recommendation in the admin UI) when the memory ceiling is below a threshold.

Note: `mtp_enabled` and `turboquant_kv_enabled` are mutually exclusive (`omlx/model_settings.py:232`), so this choice interacts with item 2d.

### 1e. Prefill-guard safety margins are static and likely too fat (borderline — see bucket 2)

`omlx/scheduler.py:3203-3221` — `_PREFILL_TRANSIENT_SAFETY = 1.3`, `_PREFILL_HEADROOM_SAFETY = 0.90`, `_PREFILL_ABORT_MARGIN = 0.90`, step tiers (1024, 512). On a small machine every unnecessary margin percent comes straight out of chunk size and prefill throughput. The `PrefillTransientTracker` already measures realized per-chunk memory growth, so the data to tighten margins exists.

**Why borderline:** mis-tuning here does not degrade gracefully — an under-margined chunk can trip an uncatchable Metal command-buffer abort (SIGABRT). Wants the memory-pressure benchmark from bucket 2 before touching defaults.

---

## 2. Needs benchmark/test-suite improvements first

### 2a. Profile Python-side per-token overhead (do this before optimizing anything Python-side)

On a 0.8B model, decode steps are fast enough that the scheduler loop, per-request streaming detokenizers (`omlx/scheduler.py:2445-2467`), and output-collector plumbing may dominate wall time — the investigation doc's open question #5.

Missing tooling:

- A py-spy (or Instruments) profile of one generate request on the reference machine.
- A scheduler-step **micro**benchmark using mocked models — existing `tests/test_scheduler*.py` verify correctness, not time. This also becomes the perf-regression guard for any later scheduler changes.

Outcome decides whether items like 1b or any Python-path optimization matter at all at this model scale.

### 2b. TTFT / concurrency benchmark mode

The admin benchmark measures PP/TG throughput but not time-to-first-token under concurrent load — which is exactly what suffers when the prefill guard throttles admissions. Without it, the effects of 1c and 1e are invisible.

Also: `VALID_BATCH_SIZES = [2, 4, 8]` (`omlx/admin/benchmark.py:36`) — sizes 4/8 do not even fit on this machine with anything but the smallest models. The benchmark needs a memory-ceiling-aware mode (skip/flag configs that cannot fit instead of OOM-failing them).

### 2c. SSD cold-tier latency benchmark

With ~3.7 GB after weights, the paged SSD cache (`omlx/cache/paged_ssd_cache.py`) gets exercised far more than on large dev machines — the investigation doc calls this an underexplored angle. The benchmark covers partial-prefix hits but does not separate **hot-tier hit vs. SSD hit vs. miss** latency. Build that scenario first; only then tune hot/SSD tier sizing defaults (`hot_cache_max_size` default "0", `omlx/settings.py:298`).

### 2d. MTP (multi-token prediction) for Qwen3.5

Patches exist for both text and VLM variants (`omlx/patches/mlx_lm_mtp/qwen35_model.py`, `omlx/patches/mlx_vlm_mtp/`); `mtp_enabled` defaults to `False` (`omlx/model_settings.py:193`). Unlike DFlash (ruled out — needs two resident models), MTP needs no second model, making it **the only speculative-decoding path that fits this hardware**.

Missing tooling: an MTP on/off comparison mode in the benchmark plus an acceptance-rate metric — the speedup is entirely acceptance-dependent and may be small on a 0.8B model. Also requires the checkpoint to ship MTP weights, and conflicts with TurboQuant KV (see 1d).

### 2e. `chunked_prefill` and prefill step sizing under concurrency

`chunked_prefill: bool = False` (`omlx/scheduler.py:1307`), `prefill_step_size: 2048` with adaptive downshift tiers (1024, 512). Chunked prefill trades per-step overhead for concurrent-request TTFT — only measurable once 2b exists.

---

## 3. Other libraries' territory

| Where | Issue | oMLX status |
|---|---|---|
| **MLX core** | Fused SDPA supports head_dim {64, 80, 128} only; head_dim=256 prefill needs the `omlx/patches/sdpa256_attention.py` workaround (O(L) tiled kernel). Proper fused 256 kernel belongs upstream. (Does not affect Qwen3.5-0.8B.) | Workaround shipped |
| **MLX core** | Metal allocator buffer-release race (issue #300) forces `mx.set_cache_limit(total_mem)` in `omlx/cli.py:324`; `mx.clear_cache` / IOKit `completeMemory()` race (#435) forces `_DEFERRED_CLEAR_DELAY = 8`. Root causes are MLX/driver-side. | Workarounds shipped; 1b can only soften, not remove |
| **mlx-lm** | `tokenizer_utils.py` calls `AutoTokenizer.register("NewlineTokenizer", ...)` with a bare string → crashes every transformers 4.57.6–5.13.0. Still present on mlx-lm main. | Shimmed in `omlx/_transformers_compat.py`; **should be reported/PR'd upstream** |
| **mlx-lm** | BatchGenerator step internals require heavy monkey-patching (`omlx/scheduler.py:600-1070` — row realignment, cache merge/filter/extend passthroughs). If profiling (2a) shows per-token overhead there, the right fix is an mlx-lm PR, not another patch layer. | Patched; upstream candidates TBD after 2a |
| **transformers** | `_LazyAutoMapping.register()` unconditionally reads `key.__module__` → `AttributeError` on string keys. Same bug as above, other end. | Shimmed |
| **mlx-vlm** | Qwen3.5 loads via the VLM engine even for text-only serving; if the profile shows vision-branch overhead on text requests, that belongs upstream. | Unverified — check in 2a |
| **onnxruntime / markitdown** | No cp314 wheel for `onnxruntime==1.20.1` (transitive via markitdown → magika). Environment constraint, not perf. | Worked around via `.python-version` pin |

---

## Recommended order

1. Prerequisites (commit env fixes, capture baseline).
2. **1a + 1c** — pure memory-budget wins, measurable immediately with the existing benchmark.
3. **2a profile** — decides whether 1b and any Python-side work are worth doing at this model scale.
4. **2b benchmark extension (TTFT + ceiling-aware batch sizes)** — unlocks validating 1e, 2e, and re-validating 1c under load.
5. **1d TurboQuant experiment** in parallel (settings-only, needs no code).
6. 2c / 2d as follow-on tracks once the above land.
