"""
Session runner — drives a full multi-trial session through a DoricLightSource.

Responsibilities:
  - Build the trial list (deterministic from RANDOM_SEED + REPS).
  - Per trial: build stim → upload → fire at the scheduled onset → log → stop.
  - Write a CSV session log with trial metadata + Python-clock onset timestamps.
    (Authoritative onsets come from the Doric TTL-out captured by the OneBox.)

The session schedule treats each trial as a block:
    block_period = STIM_DURATION + ISI       (jittered)

Uploads happen during the ISI, so the ~1 s DLL settle is absorbed as long as
ISI > ~1.5 s (which it is at 10 s base).
"""
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    REPS,
    ISI_BASE_SEC,
    ISI_JITTER_FRAC,
    RANDOM_SEED,
    LOG_ROOT,
    SETTLE_MS_AFTER_SEND_SETTINGS,
)
from stimuli import build_stim
from doric_client import DoricLightSource


# ---------------------------------------------------------------------------
# Trial order
# ---------------------------------------------------------------------------
def build_trial_list(reps: dict = REPS, seed: int = RANDOM_SEED) -> list[str]:
    """Deterministic shuffled trial list."""
    names: list[str] = []
    for name, n in reps.items():
        if n > 0:
            names.extend([name] * n)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    return names


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class SessionLogger:
    """Writes session metadata + per-trial rows to a timestamped folder."""

    FIELDS = [
        'trial_num', 'stim_name', 'kind',
        'duration_s', 'period_ms',
        't_upload_start_s', 't_upload_done_s',
        't_fire_s', 't_stop_s',
        'isi_planned_s',
    ]

    def __init__(self, log_root: Path = LOG_ROOT):
        stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.folder = Path(log_root) / stamp
        self.folder.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.folder / 'trials.csv'
        self._fh = open(self.csv_path, 'w', newline='')
        self._w = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._fh.flush()

    def write_trial(self, row: dict):
        self._w.writerow({k: row.get(k, '') for k in self.FIELDS})
        self._fh.flush()

    def write_meta(self, meta: dict):
        """Session-level metadata (one-time, at start)."""
        import json
        with open(self.folder / 'session_meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---------------------------------------------------------------------------
# Per-trial execution
# ---------------------------------------------------------------------------
def _upload(doric: DoricLightSource, spec: dict):
    if spec['kind'] == 'custom':
        doric.send_custom(
            samples_1000=spec['payload'],
            period_ms=spec['period_ms'],
            n_seq=spec['n_seq'],
        )
    elif spec['kind'] == 'complex':
        doric.send_complex(spec['payload'])
    else:
        raise ValueError(f"Unknown spec kind: {spec['kind']!r}")


def _sleep_until(target_t: float):
    while True:
        dt = target_t - time.perf_counter()
        if dt <= 0:
            return
        # Use perf_counter-driven sleep; Windows sleep granularity is ~15 ms,
        # good enough since our ISIs are multi-second.
        time.sleep(min(dt, 0.5))


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
def run_session(doric: DoricLightSource, logger: SessionLogger,
                dry_run_secs_per_trial: float | None = None) -> None:
    """Run the full session through the open DoricLightSource.

    If dry_run_secs_per_trial is a number, skip device calls entirely and
    just loop with that fake stim duration — useful for exercising the
    schedule/logger without hardware.
    """
    trials = build_trial_list()
    rng = np.random.default_rng(RANDOM_SEED + 1)  # separate stream for ISIs

    logger.write_meta(dict(
        n_trials=len(trials),
        trial_counts={k: trials.count(k) for k in sorted(set(trials))},
        random_seed=RANDOM_SEED,
        isi_base_s=ISI_BASE_SEC,
        isi_jitter_frac=ISI_JITTER_FRAC,
        settle_ms_after_send_settings=SETTLE_MS_AFTER_SEND_SETTINGS,
        started_at=datetime.now().isoformat(timespec='seconds'),
    ))

    # First fire is a bit in the future so we have time to upload trial 1.
    next_fire_t = time.perf_counter() + 2.0

    for i, name in enumerate(trials, start=1):
        spec = build_stim(name)
        duration_s = spec['duration_s']

        # ---- upload during the ISI ---------------------------------------
        t_upload_start = time.perf_counter()
        if dry_run_secs_per_trial is None:
            _upload(doric, spec)
        t_upload_done = time.perf_counter()

        # ---- wait for scheduled onset ------------------------------------
        _sleep_until(next_fire_t)

        # ---- fire --------------------------------------------------------
        t_fire = time.perf_counter()
        if dry_run_secs_per_trial is None:
            doric.start()
        print(f"[{i:>3}/{len(trials)}] {name:<12}  fire @ "
              f"{t_fire - 0:.3f}s   dur={duration_s:.1f}s")

        # ---- let it play -------------------------------------------------
        time.sleep(dry_run_secs_per_trial if dry_run_secs_per_trial is not None
                   else duration_s)

        # ---- stop (defensive) --------------------------------------------
        if dry_run_secs_per_trial is None:
            doric.stop()
        t_stop = time.perf_counter()

        # ---- schedule next onset -----------------------------------------
        isi = ISI_BASE_SEC * (1.0 + rng.uniform(-ISI_JITTER_FRAC, ISI_JITTER_FRAC))
        next_fire_t = t_fire + duration_s + isi

        logger.write_trial(dict(
            trial_num=i,
            stim_name=name,
            kind=spec['kind'],
            duration_s=duration_s,
            period_ms=spec['period_ms'],
            t_upload_start_s=f"{t_upload_start:.6f}",
            t_upload_done_s=f"{t_upload_done:.6f}",
            t_fire_s=f"{t_fire:.6f}",
            t_stop_s=f"{t_stop:.6f}",
            isi_planned_s=f"{isi:.3f}",
        ))

    print(f"Session done. Log: {logger.folder}")
