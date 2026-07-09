# SPDX-License-Identifier: Apache-2.0
"""MTP draft-depth auto-tuning — oMLX analog of ``mtplx tune`` (Phase 2 spike).

Benchmarks decode throughput at draft depth 0 (plain autoregressive
baseline) and 1..N on the *live* loaded model, picks ``argmax(tps)`` and
persists the winner per (model, hardware) so ``mtp_draft_depth: "auto"``
can resolve to it at load time. A depth-0 winner resolves to MTP-off,
which converts the documented compute-bound M1/M2 net-negative case into
automated behavior.

Two deliberate spike-level choices:

- Trials re-stamp the per-instance markers the MTP dispatch already reads
  (``_omlx_mtp_draft_depth`` / ``_omlx_mtp_decode_enabled``) instead of
  reloading the model per depth. The stamps are only read at chain-refill
  time between requests, so between-trial flips are safe, and it makes
  each trial cheap enough to interleave.
- Depth trials run round-robin (d0, d1, ..., dK, d0, d1, ...) rather than
  fan-pinned like MTPLX, so thermal drift biases every depth equally.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from ..settings import resolve_default_base_path

logger = logging.getLogger(__name__)

TUNE_FILE_NAME = "mtp_tune.json"

# A depth-1 MTP model without the chained-draft hook (mtp_forward_hidden)
# can only run depths {off, 1}; with the hook we sweep up to 4 by default
# (the plan's measurement grid — deeper rarely pays before p^k dies off).
DEFAULT_MAX_DEPTH = 4

_PROMPT = (
    "Explain, step by step and in plain language, how a local inference "
    "server schedules concurrent requests, manages its KV cache, and "
    "decides when to apply speculative decoding. Use concrete examples."
)


def hardware_id() -> str:
    """Stable per-machine key: platform machine + chip brand string."""
    chip = ""
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        chip = platform.processor() or ""
    raw = f"{platform.machine()}-{chip}" if chip else platform.machine()
    return re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").lower()


def tune_store_path(base_path: Path | None = None) -> Path:
    base = Path(base_path) if base_path is not None else resolve_default_base_path()
    return base / TUNE_FILE_NAME


def _load_store(base_path: Path | None = None) -> dict:
    path = tune_store_path(base_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return {}


def load_tuned_depth(model_key: str, base_path: Path | None = None) -> int | None:
    """Return the tuned depth for (model, this machine), or None if untuned.

    0 means "MTP off wins on this machine".
    """
    entry = _load_store(base_path).get(model_key, {}).get(hardware_id())
    if entry is None:
        return None
    try:
        return int(entry["depth"])
    except Exception:
        return None


def save_tune_result(
    model_key: str,
    depth: int,
    tps_by_depth: dict[int, float],
    base_path: Path | None = None,
) -> Path:
    path = tune_store_path(base_path)
    store = _load_store(base_path)
    store.setdefault(model_key, {})[hardware_id()] = {
        "depth": int(depth),
        "tps_by_depth": {str(k): round(v, 2) for k, v in tps_by_depth.items()},
        "tuned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=2))
    tmp.replace(path)
    return path


def _stamp_candidates(model: Any) -> list[Any]:
    """The objects the MTP dispatch inspects for per-instance markers."""
    candidates = [model]
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            candidates.append(inner)
    return candidates


def _set_trial_depth(model: Any, depth: int) -> None:
    """Apply a trial depth to the live model instance (0 = MTP off)."""
    for candidate in _stamp_candidates(model):
        if not hasattr(candidate, "_omlx_mtp_decode_enabled"):
            continue
        candidate._omlx_mtp_decode_enabled = depth > 0
        candidate._omlx_mtp_draft_depth = max(1, depth)


def _snapshot_stamps(model: Any) -> list[tuple[Any, Any, Any]]:
    snaps = []
    for candidate in _stamp_candidates(model):
        snaps.append(
            (
                candidate,
                getattr(candidate, "_omlx_mtp_decode_enabled", None),
                getattr(candidate, "_omlx_mtp_draft_depth", None),
            )
        )
    return snaps


def _restore_stamps(snaps: list[tuple[Any, Any, Any]]) -> None:
    for candidate, enabled, depth in snaps:
        if enabled is not None:
            candidate._omlx_mtp_decode_enabled = enabled
        if depth is not None:
            candidate._omlx_mtp_draft_depth = depth


async def _measure_decode_tps(engine: Any, max_tokens: int) -> float:
    """One decode trial: generation tps of a temp-0 run (engine-reported)."""
    start = time.perf_counter()
    last = None
    async for output in engine.stream_generate(
        prompt=_PROMPT,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    ):
        last = output
    if last is None:
        return 0.0
    tps = float(getattr(last, "generation_tps", 0.0) or 0.0)
    if tps > 0:
        return tps
    # Fallback: wall-clock rate (includes prefill; only hit when the
    # engine doesn't report generation_tps).
    elapsed = max(time.perf_counter() - start, 1e-9)
    return float(getattr(last, "completion_tokens", 0) or 0) / elapsed


async def run_mtp_tune(
    engine: Any,
    model_key: str,
    depths: Sequence[int] | None = None,
    repeats: int = 2,
    max_tokens: int = 128,
    base_path: Path | None = None,
) -> dict:
    """Round-robin depth sweep on the live engine; persist and return results.

    The model must already be loaded with ``mtp_enabled=True`` (the MTP
    head is attached at load time; the tuner only flips per-instance
    decode markers). Raises ``ValueError`` otherwise.
    """
    model = getattr(engine, "_model", None)
    if model is None:
        raise ValueError("engine has no loaded model")

    stamped = [
        c
        for c in _stamp_candidates(model)
        if getattr(c, "_omlx_mtp_decode_enabled", None)
    ]
    if not stamped:
        raise ValueError(
            "model is not running with MTP decode enabled; load it with "
            "mtp_enabled=true before tuning (the tuner needs the attached "
            "MTP head to trial depths > 0)"
        )

    if depths is None:
        max_depth = (
            DEFAULT_MAX_DEPTH if hasattr(model, "mtp_forward_hidden") else 1
        )
        depths = list(range(0, max_depth + 1))
    depths = [int(d) for d in depths]

    snaps = _snapshot_stamps(model)
    samples: dict[int, list[float]] = {d: [] for d in depths}
    try:
        # Warmup at the current settings (JIT / Metal shader compile).
        await _measure_decode_tps(engine, max_tokens=16)
        for r in range(repeats):
            for d in depths:
                _set_trial_depth(model, d)
                tps = await _measure_decode_tps(engine, max_tokens)
                samples[d].append(tps)
                logger.info(
                    "MTP tune %s: depth %d repeat %d/%d -> %.1f tok/s",
                    model_key,
                    d,
                    r + 1,
                    repeats,
                    tps,
                )
    finally:
        _restore_stamps(snaps)

    tps_by_depth = {d: median(v) for d, v in samples.items() if v}
    winner = max(tps_by_depth, key=tps_by_depth.get)
    path = save_tune_result(model_key, winner, tps_by_depth, base_path)
    logger.info(
        "MTP tune %s: winner depth %d (%s); persisted to %s",
        model_key,
        winner,
        ", ".join(f"d{d}={t:.1f}" for d, t in sorted(tps_by_depth.items())),
        path,
    )
    return {
        "model": model_key,
        "hardware_id": hardware_id(),
        "winner_depth": winner,
        "tps_by_depth": {str(d): tps_by_depth[d] for d in sorted(tps_by_depth)},
        "repeats": repeats,
        "max_tokens": max_tokens,
    }
