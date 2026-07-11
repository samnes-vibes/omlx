# SPDX-License-Identifier: Apache-2.0
"""Generic A/B settings trial engine (Model Optimization Advisor P2b).

Measures decode throughput of the *current* settings against a single
*candidate* settings change (an advisor recommendation payload), so a
"suggest" recommendation can be upgraded to a measured claim before the
operator applies it. Combines ``mtp_tune.py``'s measurement discipline
(round-robin variants against thermal drift, warmup trial, median of
repeats, restore original state in ``finally``) with ``benchmark.py``'s
delivery model (background task + append-only SSE event log guarded by
an ``asyncio.Condition``).

Variant application has two paths:

- ``mtp_draft_depth`` only: re-stamp the per-instance MTP markers on the
  live model (same trick as ``mtp_tune._set_trial_depth``) — cheap, no
  reload.
- anything else (``mtp_enabled``, ``turboquant_kv_enabled``, ...): these
  are read at model-load time, so the candidate variant is loaded through
  the engine pool's transient ``runtime_settings`` variant mechanism
  (reload per variant switch; persisted settings are never mutated).
  Much more expensive — the trial reports ``reload_required`` up front so
  the UI can warn before starting.

Results persist to ``ab_trials.json`` keyed by
(model, hardware, rec_id, settings_hash) so a stale trial result is not
reused after the operator changes something else.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Optional

from ..settings import resolve_default_base_path
from .mtp_tune import (
    _measure_decode_tps,
    _restore_stamps,
    _set_trial_depth,
    _snapshot_stamps,
    hardware_id,
    load_tuned_depth,
)

logger = logging.getLogger(__name__)

TRIAL_FILE_NAME = "ab_trials.json"

# Settings keys that can be flipped on the live model instance without a
# reload (per-instance stamps read at chain-refill time between requests).
_STAMPABLE_KEYS = frozenset({"mtp_draft_depth"})

_ab_trial_runs: dict[str, "ABTrialRun"] = {}


@dataclass
class ABTrialRun:
    """Tracks one running A/B trial. Same SSE shape as ``BenchmarkRun``:
    events are appended to `events` under `cond`; subscribers replay from
    offset 0 then wait on `cond`; `terminal` marks the final event."""

    trial_id: str
    model_id: str
    model_key: str
    rec_id: str
    variants: list[dict]  # [{"label": ..., "settings": {...}}, ...]
    settings_hash: str
    reload_required: bool
    repeats: int = 2
    max_tokens: int = 128
    status: str = "running"  # running, completed, error
    events: list[dict] = field(default_factory=list)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    terminal: bool = False
    task: Optional[asyncio.Task] = None
    result: dict | None = None
    error_message: str = ""
    # Store override for tests; None = the server's default base path.
    base_path: Path | None = None


def get_trial(trial_id: str) -> ABTrialRun | None:
    return _ab_trial_runs.get(trial_id)


def get_active_trial() -> ABTrialRun | None:
    for run in _ab_trial_runs.values():
        if run.status == "running":
            return run
    return None


def cleanup_old_trials(max_runs: int = 10) -> None:
    done = [
        tid
        for tid, r in _ab_trial_runs.items()
        if r.status in ("completed", "error")
    ]
    for tid in done[:-max_runs] if len(done) > max_runs else []:
        del _ab_trial_runs[tid]


def settings_hash(settings_dict: dict) -> str:
    """Stable short hash of a settings dict for staleness keying."""
    canonical = json.dumps(settings_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def changed_keys(variants: list[dict]) -> set[str]:
    """Keys whose value differs between the current and candidate variant."""
    if len(variants) < 2:
        return set()
    a, b = variants[0]["settings"], variants[1]["settings"]
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def requires_reload(keys: set[str], current_settings: dict) -> bool:
    """True when any changed key is read at model-load time.

    Only ``mtp_draft_depth`` is stampable, and only while the loaded model
    already has the MTP head attached (``mtp_enabled`` on) — otherwise a
    depth > 0 has nothing to run on.
    """
    if not keys <= _STAMPABLE_KEYS:
        return True
    return not bool(current_settings.get("mtp_enabled", False))


def create_trial_run(
    model_id: str,
    model_key: str,
    rec_id: str,
    variants: list[dict],
    current_hash: str,
    reload_needed: bool,
    repeats: int = 2,
    max_tokens: int = 128,
) -> ABTrialRun:
    run = ABTrialRun(
        trial_id=f"abtrial-{uuid.uuid4().hex[:12]}",
        model_id=model_id,
        model_key=model_key,
        rec_id=rec_id,
        variants=variants,
        settings_hash=current_hash,
        reload_required=reload_needed,
        repeats=repeats,
        max_tokens=max_tokens,
    )
    _ab_trial_runs[run.trial_id] = run
    return run


# --- result store (same read/write pattern as mtp_tune's store) ---


def trial_store_path(base_path: Path | None = None) -> Path:
    base = Path(base_path) if base_path is not None else resolve_default_base_path()
    return base / TRIAL_FILE_NAME


def _load_store(base_path: Path | None = None) -> dict:
    path = trial_store_path(base_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return {}


def save_trial_result(
    model_key: str,
    rec_id: str,
    entry: dict,
    base_path: Path | None = None,
) -> Path:
    path = trial_store_path(base_path)
    store = _load_store(base_path)
    store.setdefault(model_key, {}).setdefault(hardware_id(), {})[rec_id] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=2))
    tmp.replace(path)
    return path


def load_trial_results(
    model_key: str,
    current_hash: str,
    base_path: Path | None = None,
) -> dict[str, dict]:
    """Per-rec-id trial entries for (model, this machine), hash-filtered.

    Entries whose ``settings_hash`` no longer matches the model's current
    settings are dropped — the measurement was taken against a different
    configuration and must not be quoted for this one.
    """
    entries = _load_store(base_path).get(model_key, {}).get(hardware_id(), {})
    if not isinstance(entries, dict):
        return {}
    return {
        rec_id: e
        for rec_id, e in entries.items()
        if isinstance(e, dict) and e.get("settings_hash") == current_hash
    }


# --- runner ---


async def _send_event(run: ABTrialRun, event: dict) -> None:
    async with run.cond:
        run.events.append(event)
        if event.get("type") in ("result", "error"):
            run.terminal = True
        run.cond.notify_all()


def _stamp_variant_depth(model: Any, model_key: str, settings: dict) -> None:
    """Apply a stampable variant (mtp_draft_depth change) to the live model."""
    depth = settings.get("mtp_draft_depth", 1)
    if depth == "auto":
        tuned = load_tuned_depth(model_key)
        depth = tuned if tuned is not None else 1
    if not settings.get("mtp_enabled", False):
        depth = 0
    _set_trial_depth(model, int(depth))


async def run_ab_trial(run: ABTrialRun, engine_pool: Any, settings_manager: Any) -> None:
    """Round-robin the variants; emit SSE events; persist the result.

    Restores the original state in ``finally``: the stamp path restores the
    per-instance markers; the reload path re-acquires the engine with
    persisted settings (the transient ``runtime_settings`` variant never
    touches what's on disk).
    """
    start_ts = time.perf_counter()
    labels = [v["label"] for v in run.variants]
    samples: dict[str, list[float]] = {label: [] for label in labels}
    snaps = None
    try:
        await _send_event(
            run,
            {
                "type": "start",
                "trial_id": run.trial_id,
                "rec_id": run.rec_id,
                "variants": labels,
                "reload_required": run.reload_required,
                "repeats": run.repeats,
                "max_tokens": run.max_tokens,
            },
        )

        total = run.repeats * len(run.variants)
        completed = 0
        engine = await engine_pool.get_engine(run.model_id, force_lm=True)
        if not run.reload_required:
            model = getattr(engine, "_model", None)
            if model is None:
                raise ValueError("engine has no loaded model")
            snaps = _snapshot_stamps(model)
        # Warmup at current settings (JIT / Metal shader compile).
        await _measure_decode_tps(engine, max_tokens=16)

        for _ in range(run.repeats):
            for variant in run.variants:
                if run.reload_required:
                    if variant["label"] == "current":
                        engine = await engine_pool.get_engine(
                            run.model_id, force_lm=True
                        )
                    else:
                        candidate = settings_manager.get_settings(run.model_id)
                        for k, v in variant["settings"].items():
                            if hasattr(candidate, k):
                                setattr(candidate, k, v)
                        engine = await engine_pool.get_engine(
                            run.model_id,
                            force_lm=True,
                            runtime_settings=candidate,
                        )
                        # A fresh load needs its own warmup so shader
                        # compile doesn't bill the first candidate sample.
                        await _measure_decode_tps(engine, max_tokens=16)
                    model = getattr(engine, "_model", None)
                else:
                    model = getattr(engine, "_model", None)
                if model is not None and "mtp_draft_depth" in variant["settings"]:
                    _stamp_variant_depth(model, run.model_key, variant["settings"])

                tps = await _measure_decode_tps(engine, run.max_tokens)
                samples[variant["label"]].append(tps)
                completed += 1
                await _send_event(
                    run,
                    {
                        "type": "progress",
                        "variant": variant["label"],
                        "tps": round(tps, 1),
                        "completed": completed,
                        "total": total,
                    },
                )

        variants_out = {
            label: {
                "tps_median": round(median(vals), 1),
                "samples": [round(v, 1) for v in vals],
            }
            for label, vals in samples.items()
            if vals
        }
        gain_pct = None
        cur = variants_out.get("current", {}).get("tps_median")
        cand = variants_out.get("candidate", {}).get("tps_median")
        if cur and cand:
            gain_pct = round((cand - cur) / cur * 100, 1)

        result = {
            "type": "result",
            "rec_id": run.rec_id,
            "variants": variants_out,
            "gain_pct": gain_pct,
            "reload_required": run.reload_required,
            "elapsed_s": round(time.perf_counter() - start_ts, 1),
        }
        save_trial_result(
            run.model_key,
            run.rec_id,
            {
                "settings_hash": run.settings_hash,
                "gain_pct": gain_pct,
                "variants": variants_out,
                "reload_required": run.reload_required,
                "trial_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            base_path=run.base_path,
        )
        run.result = result
        run.status = "completed"
        await _send_event(run, result)
        logger.info(
            "A/B trial %s (%s / %s): gain %s%% in %.1fs",
            run.trial_id,
            run.model_key,
            run.rec_id,
            gain_pct,
            time.perf_counter() - start_ts,
        )
    except Exception as e:
        logger.exception("A/B trial %s failed", run.trial_id)
        run.status = "error"
        run.error_message = str(e)
        await _send_event(run, {"type": "error", "message": str(e)})
    finally:
        try:
            if snaps is not None:
                _restore_stamps(snaps)
            elif run.reload_required:
                # Leave the model on persisted settings (variant reload if
                # the candidate was the last one loaded).
                await engine_pool.get_engine(run.model_id, force_lm=True)
        except Exception:
            logger.warning(
                "A/B trial %s: failed to restore original engine state",
                run.trial_id,
                exc_info=True,
            )
