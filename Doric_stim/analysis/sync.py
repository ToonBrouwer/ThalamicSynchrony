"""Pulse extraction & stim-type decoding from ADC10/ADC11."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STIM_NAMES = [
    "ChirpFwd", "ChirpRev",
    "NoiseLp100", "NoiseLp500",
    "Sine10", "Sine25", "Sine60",
    "Ramp",
]
# From opto_stim_v8.ino: TTL_TYPE_MS[NUM_STIM_TYPES]
TTL_TYPE_MS = np.array([100, 150, 200, 250, 300, 350, 400, 450], dtype=float)
TTL_ONSET_MAX_MS = 50.0     # onset pulse is nominally 10 ms
TTL_TYPE_MIN_MS = 80.0      # type pulses are >= 100 ms


@dataclass
class Pulse:
    i_rise: int
    i_fall: int
    dur_s: float


@dataclass
class Trial:
    index: int
    stim_idx: int
    stim_name: str
    onset_sample: int
    onset_time_s: float
    type_pulse_ms: float
    stim_duration_s: float = 5.0  # measured from TTL onset-to-type gap


def find_pulses(x: np.ndarray, fs: float, thr: float = 1.5,
                min_dur_s: float = 0.002) -> list[Pulse]:
    """Return rising/falling pulse pairs on a clean TTL signal."""
    b = x > thr
    d = np.diff(b.astype(np.int8))
    rises = np.where(d == 1)[0] + 1
    falls = np.where(d == -1)[0] + 1
    if b[0]:
        rises = np.r_[0, rises]
    if b[-1]:
        falls = np.r_[falls, b.size - 1]
    n = min(rises.size, falls.size)
    rises, falls = rises[:n], falls[:n]
    pulses = []
    for r, f in zip(rises, falls):
        dur = (f - r) / fs
        if dur >= min_dur_s:
            pulses.append(Pulse(int(r), int(f), float(dur)))
    return pulses


def decode_trials(pulses: list[Pulse], fs: float,
                  stim_duration_s: float = 5.0,
                  gap_min_s: float = 1.0,
                  gap_max_s: float = 25.0) -> list[Trial]:
    """Pair a short onset pulse with the next long type pulse.

    Accepts any gap in [gap_min_s, gap_max_s] so trials with slower
    per-sample math (sinf/powf) still decode. The real stim duration
    (gap minus TTL_PRE_MS=200ms) is stored in Trial.stim_duration_s.
    """
    short = [p for p in pulses if p.dur_s * 1000 <= TTL_ONSET_MAX_MS]
    long_ = [p for p in pulses if p.dur_s * 1000 >= TTL_TYPE_MIN_MS]
    long_sorted = sorted(long_, key=lambda p: p.i_rise)

    trials: list[Trial] = []
    used: set[int] = set()
    for on in short:
        t_on = on.i_rise / fs
        tp = None
        for cand in long_sorted:
            if id(cand) in used:
                continue
            gap = cand.i_rise / fs - t_on
            if gap < gap_min_s:
                continue
            if gap > gap_max_s:
                break                        # list is sorted, no more candidates
            tp = cand
            break
        if tp is None:
            continue
        used.add(id(tp))
        gap_s = (tp.i_rise - on.i_rise) / fs
        dur_ms = tp.dur_s * 1000
        stim_idx = int(np.argmin(np.abs(TTL_TYPE_MS - dur_ms)))
        trials.append(Trial(
            index=len(trials),
            stim_idx=stim_idx,
            stim_name=STIM_NAMES[stim_idx],
            onset_sample=on.i_rise,
            onset_time_s=t_on,
            type_pulse_ms=dur_ms,
            stim_duration_s=gap_s - 0.200,    # subtract TTL_PRE_MS
        ))
    return trials


def frame_rising_edges(x: np.ndarray, fs: float, thr: float = 1.5) -> np.ndarray:
    """Return sample indices of rising edges on the camera-TTL channel."""
    b = x > thr
    d = np.diff(b.astype(np.int8))
    return (np.where(d == 1)[0] + 1).astype(np.int64)
