# Low-RAM Qwen Performance Investigation — Handoff Notes

Date: 2026-07-04
Base commit: `eca4a7c5` (2026-07-02, "fix(server): align chat stream thinking detection with templates")

## Goal

Develop and validate performance optimizations for Qwen model serving in oMLX, on a machine that is memory-constrained enough to make every allocation matter. This doc exists so a fresh session/agent can pick up the investigation without re-deriving the hardware and dependency constraints below.

---

## Hardware

- **Machine**: Mac mini, Apple M1, **8 GB unified memory** (`sysctl hw.memsize` = 8 GB, confirmed via `system_profiler SPHardwareDataType`)
- macOS (Darwin 25.2.0), single chip, no swap tuning applied

This is the single most important constraint in this doc — most "obvious" performance ideas (bigger batches, bigger KV cache, speculative decoding with a second model resident) are memory-bound before they're compute-bound on this box.

---

## Memory ceiling mechanics (read this before choosing a model)

`omlx/process_memory_enforcer.py` derives a hard ceiling the whole server is bound by:

```
ceiling = min(static_ceiling, dynamic_ceiling, metal_cap)
static_ceiling  = total_ram - tier.static_reserve
dynamic_ceiling = omlx_phys + free + inactive + active * reclaim_ratio   (tier-dependent)
```

Critically, `omlx/process_memory_enforcer.py:48-51`:

```python
_SMALL_SYSTEM_RESERVE = 4 * 1024**3   # 4 GB
_SMALL_SYSTEM_THRESHOLD = 24 * 1024**3  # 24 GB
```

Any system under 24 GB total RAM gets a **flat 4 GB static reserve regardless of memory-guard tier** ("safe"/"balanced"/"aggressive" all collapse to the same static reserve on small systems — the tier only changes the *dynamic* reclaim ratio). On this 8 GB Mac mini that means:

- Static ceiling ≈ 8 GB − 4 GB = **4 GB**, further reduced by whatever else is using RAM at that instant (dynamic ceiling). Observed in practice: **3.65–3.9 GB**, moving with system load.
- This ~3.7-3.9 GB budget has to cover model weights **and** KV cache **and** oMLX/Python/MLX runtime overhead.

Confirmed empirically: loading `mlx-community/gemma-4-e2b-it-4bit` (3.50 GB weights) failed with:
```
Cannot load gemma-4-e2b-it-4bit: projected memory 3.89GB would exceed the memory ceiling 3.65GB
(current: 397.61MB, model: 3.50GB)
```

**Practical rule of thumb for this machine: model weights should stay well under ~2 GB** to leave meaningful headroom for KV cache and concurrent requests. Anything approaching 3+ GB weights is a one-shot, no-headroom load.

---

## Why DFlash (speculative decoding) was ruled out first

Originally considered DFlash (block-diffusion speculative decoding, see `docs/experimental/dflash_mlx_integration.md`) as the perf target, but ruled it out for this hardware:

1. DFlash requires **two models resident simultaneously** (target + draft), which is exactly the wrong shape for a ~3.7 GB budget.
2. DFlash only supports Qwen and Gemma4 architectures (`omlx/engine/dflash.py:is_dflash_compatible`), and needs a **specific published draft checkpoint** per target — checked the z-lab HF org and the smallest available target/draft pair is **Qwen3.5-4B + z-lab/Qwen3.5-4B-DFlash** (~4B target, ~1B draft). No draft checkpoint exists for anything in the 1B-or-under range, so a tiny model can't use DFlash at all even in principle.
3. Even ignoring memory, DFlash's speedup comes from amortizing a slow autoregressive decode step across a fast parallel draft — sub-1B models already decode fast, so the theoretical upside is small even on unconstrained hardware.

Conclusion: **DFlash is not a fit for this machine or for small Qwen models.** Regular (non-speculative) serving performance work is the right target here.

---

## Models currently installed on this machine

`~/.omlx/models/mlx-community/`:

| Model | Weights | Architecture | Notes |
|---|---|---|---|
| `gemma-4-e2b-it-4bit` | 3.50 GB | `gemma4` (VLM engine) | Too large for reliable loading alongside anything else on this box; kept for occasional reference/testing only |
| `Qwen3.5-0.8B-MLX-4bit` | 0.61 GB | `qwen3_5` (VLM engine — Qwen3.5 ships multimodal-capable) | **Default model**, fits comfortably, this is the one to iterate on |

Server is run via (from the repo, dev checkout, no global `omlx` binary installed):
```bash
uv run omlx serve --model-dir ~/.omlx/models --port 8000 --log-level info
```
Admin UI: `http://localhost:8000/admin`. Note: the model directory is only scanned at server startup / on admin-triggered downloads — if a model is dropped into `--model-dir` out-of-band (e.g. via `hf download` instead of the admin downloader), the server needs a restart to discover it.

There was no pre-existing `~/.omlx` directory on this machine before this session — this is a from-scratch setup, not a machine with prior oMLX history.

---

## Uncommitted repo changes from this session (unrelated to Qwen perf, but required for `uv run` to work at all here)

`git status --short` currently shows:
```
 M omlx/__init__.py
?? .python-version
?? omlx/_transformers_compat.py
```

These fix two environment bugs discovered while just trying to get `uv sync` + `pytest` working on this machine, **not related to Qwen performance**:

1. **`.python-version` → `3.13`**: `pyproject.toml`'s `requires-python = ">=3.11"` has no upper bound, so a bare `uv sync` picks the newest installed interpreter. Only Python 3.14 was installed via `uv python install` on this machine, and `onnxruntime==1.20.1` (pulled in transitively via `markitdown[pdf,docx,pptx]` → `magika`) has no `cp314` wheel. Pinning to 3.13 fixes this.
2. **`omlx/_transformers_compat.py`** (+ one import line in `omlx/__init__.py`): the pinned mlx-lm commit's `tokenizer_utils.py` calls `AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)` — a bare string instead of a config class. `transformers`' `_LazyAutoMapping.register()` unconditionally reads `key.__module__` on that argument and crashes with `AttributeError: 'str' object has no attribute '__module__'`. Verified this reproduces on every transformers version from 4.57.6 through 5.13.0, and is still present on mlx-lm's current main branch — it's an upstream bug, not something a version pin routes around. `uv.lock` is gitignored in this repo (fresh resolve every time), so **anyone doing a clean install today will hit this**, independent of the Qwen perf work.

These are currently **uncommitted**. Decide whether to commit them (they're needed for the dev workflow to function at all right now) before/alongside any perf work, otherwise a `git stash` or fresh checkout will reintroduce both breakages.

---

## Where to actually look for Qwen-specific perf work

Code relevant to Qwen model paths and general serving performance (not exhaustive, but a starting map):

- `omlx/patches/mlx_lm_mtp/qwen35_model.py`, `omlx/patches/mlx_vlm_mtp/qwen35_*` — Qwen3.5 MTP (multi-token prediction) patches for both text and VLM variants
- `omlx/patches/qwen3_6_nested_visual.py` — Qwen3.6 nested visual patch
- `omlx/patches/turboquant_attention.py` + `omlx/turboquant_kv.py` — quantized KV-cache attention path (TurboQuant); relevant since KV cache memory is scarce here
- `omlx/patches/sdpa256_attention.py` — attention kernel variant, check applicability/threshold for small models
- `omlx/patches/index_cache.py` — index-based cache patch (used by DeepSeek/GLM DSA indexer per pyproject comments, check if it touches Qwen paths too)
- `omlx/scheduler.py` — FCFS continuous-batching scheduler; on 8 GB, `--max-concurrent-requests` is a live lever worth profiling (default 8, likely too high for this box's headroom)
- `omlx/cache/` — `paged_cache.py` (GPU paged KV), `hybrid_cache.py`/`prefix_cache.py` (hot tier + prefix sharing), `paged_ssd_cache.py` (SSD cold tier) — with only ~3-3.5 GB free after weights, the SSD-offload path is going to get exercised far more heavily here than on a well-provisioned dev machine, which may itself be a useful/underexplored perf angle
- `omlx/memory_monitor.py` / `omlx/process_memory_enforcer.py` — the guard rails described above; tightening the prefill memory estimate has outsized value on small-RAM machines since the margin for error is so thin

Relevant existing tests to model new perf tests after (most use mocked models, so they don't require this machine's limited real RAM):
`tests/test_scheduler*.py`, `tests/test_turboquant*.py`, `tests/test_prefix_cache*.py`, `tests/test_sdpa256_attention.py`, `tests/test_mlx_lm_mtp_patch.py`, `tests/test_vlm_mtp*.py`

---

## Measurement tooling already in the project

Don't hand-roll benchmark scripts — `omlx/admin/benchmark.py` + the admin dashboard's one-click benchmark already measures prefill (PP) and generation (TG) tok/s, including partial-prefix-cache-hit scenarios (`VALID_PROMPT_LENGTHS` up to 200k tokens, `VALID_BATCH_SIZES` = [2, 4, 8] for continuous-batching runs — note batch sizes above 2 are unlikely to fit in the available headroom on this machine with anything but the smallest models). Use this for repeatable before/after numbers.

---

## Open questions / suggested starting points for the next session

1. Confirm whether the two uncommitted environment fixes above should be committed before starting perf work (recommended — they block `uv run` entirely otherwise).
2. Get one clean baseline PP/TG number for `Qwen3.5-0.8B-MLX-4bit` via the admin benchmark tool before touching any code — no optimization work should start without a baseline.
3. Investigate `--max-concurrent-requests` and hot/SSD cache tier sizing defaults specifically under the ~3.7 GB ceiling — the current defaults (max-concurrent-requests=8, hot-cache-max-size percentage-based) were likely tuned against larger-memory dev machines and may be actively counterproductive here (e.g. admission of concurrent requests that immediately trip the prefill memory guard).
4. Check whether a slightly larger Qwen3.5 model (e.g. 1.7B, if one exists in mlx-community 4-bit) still fits comfortably and gives a more representative target than 0.8B for "real" workloads, without repeating the gemma-4-e2b oversize mistake.
5. Profile (py-spy or Instruments) a single generate request on `Qwen3.5-0.8B-MLX-4bit` before assuming where time goes — the earlier assumption for DFlash (that decode speed is the bottleneck) does not necessarily hold for a model this small; prefill, tokenizer overhead, or Python-side scheduling loop overhead could easily dominate at this scale.
