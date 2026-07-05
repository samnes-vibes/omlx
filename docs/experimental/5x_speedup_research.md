# 5x–10x Speedup Research: Prefill & Token Generation

Date: 2026-07-05
Companion to: [dflash_mlx_integration.md](dflash_mlx_integration.md), [low_ram_perf_optimization_map.md](low_ram_perf_optimization_map.md)

## Framing

This document surveys techniques that could plausibly deliver **5x–10x** speedups for prefill (TTFT) or token generation (decode throughput) in oMLX on Apple Silicon. It builds on what already exists in the repo:

- **DFlash** (block-diffusion speculative decoding, `omlx/engine/dflash.py`) — 3–4x decode observed, 4.9–6x claimed in paper
- **SpecPrefill** (attention-guided sparse prefill, `omlx/patches/specprefill.py`) — draft model scores token importance, target prefills top-K% only
- **MTP** (multi-token prediction patches, `omlx/patches/mlx_lm_mtp/`, `mlx_vlm_mtp/`) — single-model speculation
- **TurboQuant KV** (`omlx/turboquant_kv.py`) — 4-bit KV cache
- Paged KV + SSD cache + prefix cache, chunked prefill

Key physical fact driving everything below: **decode on Apple Silicon is memory-bandwidth-bound, prefill is compute-bound (attention O(L²) at long context)**. A 5–10x win therefore comes from either (a) reading fewer bytes per emitted token, (b) emitting many tokens per target forward pass, (c) touching fewer tokens during prefill, or (d) using silicon that is currently idle (ANE, AMX/SME).

No single technique gets to 10x alone at realistic settings. The credible path is **stacking multiplicative, orthogonal wins** — the combination matrix at the end is the real deliverable.

---

## Track A — Token generation (decode)

### A1. DFlash 2.0 — remove the artificial ceilings on the existing engine

Current integration leaves large factors on the table:

1. **Context ceiling (4K) → sink+window verify.** The fallback at `DFLASH_MAX_CTX=4096` throws away the entire speedup exactly where decode is slowest (long context). The draft already runs sink=64/window=1024 KV; apply the same **attention-sink + sliding-window trick to the verify pass** so target verification cost stops growing with context. Combined with TurboQuant'd target KV (A3), DFlash could stay active to 32K+. This single change converts "3–4x on short prompts" into "3–4x always" — for long-context chat that is effectively a >5x improvement over today's fallback path.
2. **Prefix caching inside DFlash.** Every request currently does full prefill from scratch. Multi-turn chat re-prefills the whole conversation each turn. Exposing block-level KV save/restore from dflash-mlx (already noted as future work) makes turn-N TTFT near-zero — for a 3K-token history this is a 10x+ TTFT win on turns 2+.
3. **Tree/multi-block verification.** DFlash verifies one linear 16-token block per cycle. Verifying 2–3 candidate blocks as a tree (EAGLE-2/3-style tree attention, or [P-EAGLE](https://vllm.ai/blog/2026-03-13-p-eagle)'s parallel drafting, +30–69% over EAGLE-3) raises expected accepted-tokens-per-cycle. The verify pass is memory-bound, so verifying 32–48 tokens costs barely more than 16.
4. **Temperature-aware acceptance.** Current fork uses prefix-match acceptance under temperature, which needlessly rejects valid samples. Proper stochastic speculative sampling (accept with probability min(1, p_target/p_draft)) recovers most of the temp>0 acceptance loss — the paper's 4.1x at temp=1 vs our observed drop suggests ~1.3–1.5x recoverable.

**Estimated stack: 3–4x today → 5–7x, and available at all context lengths.**

### A2. Draft-free layered speculation — the zero-memory stack (fits the 8 GB box)

DFlash was ruled out on low-RAM machines because it needs two resident models. But three speculation techniques need **no extra model at all**, and they compose as a cascade:

1. **N-gram / prompt-lookup / suffix decoding.** Match the last generated n-gram against the prompt + generation history; propose the continuation as draft tokens. [SSSD](https://arxiv.org/html/2411.05894) reports up to 2.9x, vLLM's prompt lookup ~2.8x on summarization/code-editing — exactly the local-agent workloads (Claude-Code-style tool loops, RAG, code rewriting) where output echoes input heavily. Cost: a hash table. This is arguably the **highest ROI item in this whole document**: ~200 lines in the scheduler's speculation slot, works with every model, zero memory.
2. **MTP as verifier-drafter.** Where the checkpoint ships MTP weights (Qwen3.5, DeepSeek — patches already exist), MTP proposes 1–3 tokens per step. Cascade: n-gram proposes long drafts when text is repetitive; MTP fills in when it isn't.
3. **Self-speculative layer-skip** ([SWIFT](https://arxiv.org/pdf/2410.06916), [ConfLayers](https://arxiv.org/pdf/2604.14612)): draft = the same model with ~40–50% of layers skipped, verify = full model. No second model, no trained checkpoint, works on any architecture. 1.3–1.6x standalone, more when cascaded with n-gram.

The cascade shares one verification mechanism (the batch verify path MTP patches already implement in `batch_generator.py`). **Estimated: 2–4x decode on agentic/RAG workloads, ~1.5–2x on freeform chat — on any model, any RAM.**

### A3. Fused compressed-domain attention — custom Metal kernels (the bandwidth play)

Decode throughput ≈ bandwidth / bytes-touched-per-token. TurboQuant KV already stores KV at 4-bit, but if the kernel **dequantizes before attending**, the bandwidth win is lost. [Open-TQ-Metal](https://arxiv.org/pdf/2604.16957) demonstrates a fused `sdpa_int4` Metal kernel that attends **in the compressed domain**: 48x attention speedup at 128K context vs dequantize-then-attend, 3.2x KV memory compression, identical top-1 predictions. The empty `omlx/custom_kernels/` package is the natural landing spot.

Extensions in the same direction:

- **W4A4/W4A8 decode GEMMs**: MLX's 4-bit weights still compute in fp16 activations. On M5-class hardware with NVFP4 tensor units, quantized-activation GEMMs give a further ~1.5–2x (M5 Max: decode 58→112 tok/s reported for Qwen3.5-35B-A3B in NVFP4).
- **Fused verify kernel for DFlash/MTP**: the 16-token verify pass is a small-batch GEMM + attention — fusing it removes launch overhead that dominates at small models.

**Estimated: 1.5–3x decode at moderate context, far more at 32K+; multiplies with everything in A1/A2.**

### A4. ANE-pipelined speculation — draft on the Neural Engine, verify on the GPU (wild but real)

The ANE sits idle during all of oMLX's serving. 2026 measurements show it is no longer a toy: genuine ANE batch dispatch reaches [268 tok/s on a 0.8B model — 11.3x over sequential dispatch](https://github.com/AtomGradient/hybrid-ane-mlx-bench), i.e. GPU-class throughput at ~0.2 W.

Idea: **run the draft model (DFlash draft or a small AR draft) on the ANE via CoreML, while the GPU runs target verification via MLX — fully overlapped.** Classic speculative decoding serializes draft→verify; with two independent execution units the draft of block N+1 runs *during* verify of block N. Speculation then costs approximately zero GPU time, and the effective speedup approaches `accepted_tokens_per_cycle / 1` instead of being discounted by draft time. Pipelined draft-verify is an active research direction ([edge-cloud variant](https://arxiv.org/pdf/2603.19133)); on Apple Silicon both units share unified memory, so the handoff is a pointer, not a transfer.

Engineering cost is high (CoreML export of draft, static shapes, sync machinery) but this is the only idea on the decode side with a credible path to **8–10x**: DFlash acceptance ~12 tokens/cycle with draft time fully hidden and verify running on compressed KV (A3).

### A5. Diffusion-LM engine — make the *target* a dLLM (wildest)

DFlash uses diffusion for the draft only. The 2026 frontier makes the target itself a diffusion LM: [DiffusionGemma (26B) in vLLM](https://vllm.ai/blog/2026-06-10-diffusion-gemma) generates at ~6x an AR baseline (1,288 tok/s on H200); [dInfer](https://arxiv.org/pdf/2510.08666), [Fast-dLLM](https://nvlabs.github.io/Fast-dLLM/) and [LocalLeap](https://arxiv.org/pdf/2510.07081) (6.94x throughput) show the inference-side toolbox is maturing. Nobody has a dLLM runtime on MLX yet.

For oMLX: a new `DiffusionEngine` (BaseEngine impl, like DFlashEngine) running a masked-denoising loop with KV-cache tricks from Fast-dLLM. Apple Silicon suits dLLMs unusually well: the denoising step is a *batch* forward (compute-dense), which shifts decode from the bandwidth-bound regime into the compute regime where the GPU has headroom. Blocked on open-weight dLLMs in oMLX's size range (DiffusionGemma-26B needs conversion + quantization), but this is a genuine first-mover niche: **"first dLLM server on Apple Silicon"** with native ~5–6x generation throughput, no draft model, no acceptance-rate lottery.

---

## Track B — Prefill (TTFT)

### B1. SpecPrefill → draft-free dynamic sparse prefill (MInference-style)

SpecPrefill's cost structure has a flaw: the draft model must itself run **full O(L²) attention** over the prompt to score importance, and needs to be resident. [MInference](https://openreview.net/forum?id=fPBACAbqSN) shows the same sparsity is predictable **from the target model's own attention structure** — three static-per-head patterns (A-shape, vertical-slash, block-sparse) found by offline calibration, then applied training-free at runtime: **10x prefill at 1M context, 1.8–3x at 32–128K, no draft model, no quality regression** on the benchmarks tested. Follow-ups reduce indexing overhead further ([VSPrefill](https://arxiv.org/html/2603.04460v1), [IndexCache](https://arxiv.org/pdf/2603.12201) — cross-layer index reuse).

Concrete shape for oMLX: keep `specprefill.py`'s chunk-selection and RoPE machinery (it is the hard part and already written), replace the scoring stage — per-head pattern configs from a one-time calibration run (admin-panel job), a small Metal kernel for vertical-slash sparse attention. Draft model requirement disappears → works on the 8 GB box too.

**Estimated: 3–10x prefill at ≥16K context, scaling with length.**

### B2. CacheBlend-style non-prefix KV reuse — RAG/multi-turn killer feature

oMLX's prefix cache only helps when the reused text is a strict prefix. Local power users' dominant long-prompt workloads (RAG chunks, system-prompt + tool-schemas + files in varying order, agent loops that reorder context) break prefixing constantly. [CacheBlend](https://arxiv.org/pdf/2405.16444) (EuroSys'25 best paper) reuses **precomputed per-chunk KV regardless of position** and selectively recomputes ~10–15% of tokens to fix cross-attention: **2.2–3.3x TTFT, 3x throughput**, quality preserved; [follow-ups](https://arxiv.org/abs/2510.10129) refine the recompute selection.

oMLX already has the storage layer for this — the paged SSD cache is a per-chunk KV store waiting to happen. Missing pieces: content-hash chunk identity, positional re-encoding on load (the RoPE-shift machinery in `specprefill.py` is *exactly* the required primitive — reuse it), and the selective-recompute pass. Multiplies with B1: sparse-prefill only the recomputed subset.

**Estimated: 2–3x TTFT on RAG/agent prompts; effectively 10x+ on multi-turn with reordered context vs today's cold prefill.**

### B3. Disaggregated prefill: ANE prefill + GPU decode

Prefill is a large-batch matmul workload — the ANE's ideal shape. [SqueezeBits' 2026 write-up](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176) demonstrates NPU-prefill/GPU-decode disaggregation on Apple Silicon; [NPUMoE](https://arxiv.org/abs/2604.18788) reports **1.32–5.55x prefill latency reduction** for MoE models on Apple NPUs, with 1.8–7.4x energy savings. Even when ANE prefill is merely *equal* in speed, it frees the GPU to keep decoding other requests → under concurrency, TTFT and decode stop stealing from each other (today chunked prefill interleaves them on one GPU). This also pairs with A4 (same CoreML export infrastructure).

**Estimated: 1.3–5x prefill depending on model shape (MoE benefits most), plus decode isolation under load; big energy/thermal win for always-on local serving.**

### B4. Hardware-generation lever: NVFP4/M5 tensor-unit prefill

M5-class chips add tensor units MLX exploits via NVFP4: reported Qwen3.5-35B-A3B prefill 1,154 → 1,810 tok/s and headline "2.70x prefill" vs prior gen. Not a code change oMLX controls, but oMLX should **auto-select NVFP4 checkpoints/kernels when the silicon supports them** (model-profiles knows the chip; today it doesn't branch on it). Free multiplier for every M5+ user.

---

## Combination matrix — paths to 5–10x

Multipliers are rough geometric estimates on plausible workloads; they multiply because they attack different bottlenecks (tokens-per-pass × bytes-per-token × tokens-touched).

| Stack | Path | Est. combined | Hardware |
|---|---|---|---|
| **Decode, big box** | A1 (DFlash 2.0, 5x) × A3 (compressed verify, 1.5x) | **~7x** | 32 GB+ |
| **Decode, big box, max** | A1 × A3 × A4 (ANE-pipelined draft) | **~8–10x** | 32 GB+, high eng. cost |
| **Decode, 8 GB box** | A2 cascade (n-gram + MTP + layer-skip, 2.5x) × A3 (1.5x) | **~3–4x** | any |
| **Decode, greenfield** | A5 (dLLM engine) | **~5–6x native** | blocked on model availability |
| **Prefill, long ctx** | B1 (dynamic sparse, 4x @32K) × A3 kernel reuse | **~5–8x** | any |
| **Prefill, RAG/agent** | B2 (chunk KV reuse, 3x) × B1 on recomputed subset | **~6–10x** | any (SSD cache exists) |
| **Prefill, MoE + ANE** | B3 (3x) × B4 (NVFP4, 1.5–2x) | **~5x** | M5+, MoE models |

## Recommended order

1. **A2.1 n-gram/prompt-lookup speculation** — days of work, zero memory, 2–3x on the workloads local users actually run. Do this first.
2. **B2 CacheBlend on the SSD cache** — the storage and RoPE-shift primitives already exist in-repo; highest TTFT leverage per engineering hour.
3. **A1.1 + A1.2 DFlash context ceiling + prefix cache** — turns the existing 3–4x into an always-on 5x+.
4. **B1 draft-free sparse prefill** — replace SpecPrefill's scoring stage with calibrated static patterns; keep its selection/RoPE code.
5. **A3 fused int4 attention kernel** in `custom_kernels/` — multiplies with everything; reference implementation exists (Open-TQ-Metal).
6. **A4 / A5 / B3** — research spikes (ANE pipeline, dLLM engine, ANE prefill): high ceiling, high cost; prototype behind experimental flags like DFlash was.

## Sources

- [DFlash paper (arXiv:2602.06036)](https://arxiv.org/pdf/2602.06036) · [P-EAGLE in vLLM](https://vllm.ai/blog/2026-03-13-p-eagle) · [AdaSD](https://arxiv.org/pdf/2512.11280)
- [SSSD n-gram speculation](https://arxiv.org/html/2411.05894) · [vLLM/Aphrodite prompt lookup](https://aphrodite.pygmalion.chat/spec-decoding/ngram/) · [SWIFT self-speculative](https://arxiv.org/pdf/2410.06916) · [ConfLayers](https://arxiv.org/pdf/2604.14612)
- [MInference](https://openreview.net/forum?id=fPBACAbqSN) · [VSPrefill](https://arxiv.org/html/2603.04460v1) · [IndexCache](https://arxiv.org/pdf/2603.12201) · [NSA](https://arxiv.org/pdf/2502.11089) · [MiniCPM4/InfLLM-v2 on-device](https://arxiv.org/pdf/2506.07900)
- [CacheBlend](https://arxiv.org/pdf/2405.16444) · [CacheClip](https://arxiv.org/abs/2510.10129) · [LMCache RAG 4.5x](https://blog.lmcache.ai/en/2024/10/09/beyond-prefix-caching-how-lmcache-speeds-up-rag-by-4-5x-by-one-line-of-change/)
- [Open-TQ-Metal fused int4 attention on Apple Silicon](https://arxiv.org/pdf/2604.16957) · [mlx-metal-kernels](https://github.com/manishklach/mlx-metal-kernels)
- [ANE/MLX hybrid benchmarks](https://github.com/AtomGradient/hybrid-ane-mlx-bench) · [SqueezeBits: NPU prefill + GPU decode](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176) · [NPUMoE](https://arxiv.org/abs/2604.18788) · [Orion: programming the ANE](https://arxiv.org/pdf/2603.06728)
- [DiffusionGemma in vLLM](https://vllm.ai/blog/2026-06-10-diffusion-gemma) · [dInfer](https://arxiv.org/pdf/2510.08666) · [Fast-dLLM](https://nvlabs.github.io/Fast-dLLM/) · [LocalLeap](https://arxiv.org/pdf/2510.07081) · [PSD](https://arxiv.org/pdf/2605.15609)
