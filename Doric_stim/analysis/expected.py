"""Regenerate the intended stimulus waveform so we can overlay it on measured data.

Mirrors opto_stim_v8/opto_stim_v8.ino:
  - Sines are analytic (sin)
  - Chirps use the same phase accumulator + per-sample freq multiplier
  - Ramp: linear rise over RAMP_RISE_MS then flat
  - Noise: xorshift32 LFSR + 2nd-order Butterworth LPF at 100/500 Hz
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

# --- Arduino parameters (keep in sync with opto_stim_v8.ino) ---------------
FS = 10000.0
STIM_DURATION = 5.0
RAMP_RISE_MS = 250.0
CHIRP_F_START = 0.5
CHIRP_F_END = 200.0
SINE_FREQS = {"Sine10": 10.0, "Sine25": 25.0, "Sine60": 60.0}


def _chirp_forward(n: int, fs: float = FS,
                   f0: float = CHIRP_F_START, f1: float = CHIRP_F_END):
    f_mult = np.exp(np.log(f1 / f0) / n)
    inst_f = f0 * (f_mult ** np.arange(n))
    dphi = 2 * np.pi * inst_f / fs
    phase = np.cumsum(dphi)
    return 0.5 + 0.5 * np.sin(phase), inst_f


def _chirp_reverse(n: int, fs: float = FS,
                   f0: float = CHIRP_F_START, f1: float = CHIRP_F_END):
    # same multiplier but starting high, going low
    f_mult = np.exp(-np.log(f1 / f0) / n)
    inst_f = f1 * (f_mult ** np.arange(n))
    dphi = 2 * np.pi * inst_f / fs
    phase = np.cumsum(dphi)
    return 0.5 + 0.5 * np.sin(phase), inst_f


def _noise_lp(n: int, fc: float, fs: float = FS, seed: int = 0):
    rng = np.random.default_rng(seed)
    wn = rng.uniform(-1.0, 1.0, size=n)
    b, a = sps.butter(2, fc, "low", fs=fs)
    y = sps.filtfilt(b, a, wn)
    # normalise to 0..1 for overlay
    y = y / (3 * y.std() + 1e-12)
    return 0.5 + 0.5 * np.clip(y, -1, 1)


def expected_waveform(stim_name: str, fs: float = FS,
                      dur_s: float = STIM_DURATION,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (t, y[0..1], meta).  meta may include 'inst_f' (Hz)."""
    n = int(dur_s * fs)
    t = np.arange(n) / fs
    meta: dict = {}

    if stim_name in SINE_FREQS:
        f = SINE_FREQS[stim_name]
        y = 0.5 + 0.5 * np.sin(2 * np.pi * f * t)
        meta["inst_f"] = np.full(n, f)
    elif stim_name == "ChirpFwd":
        y, inst_f = _chirp_forward(n, fs)
        meta["inst_f"] = inst_f
    elif stim_name == "ChirpRev":
        y, inst_f = _chirp_reverse(n, fs)
        meta["inst_f"] = inst_f
    elif stim_name == "Ramp":
        y = np.ones(n)
        rise = int(RAMP_RISE_MS / 1000.0 * fs)
        y[:rise] = np.linspace(0, 1, rise, endpoint=False)
    elif stim_name == "NoiseLp100":
        y = _noise_lp(n, fc=100.0, fs=fs, seed=seed)
        meta["fc"] = 100.0
    elif stim_name == "NoiseLp500":
        y = _noise_lp(n, fc=500.0, fs=fs, seed=seed)
        meta["fc"] = 500.0
    else:
        raise ValueError(f"unknown stim {stim_name!r}")

    return t, y.astype(np.float32), meta


def expected_peak_freq_hz(stim_name: str) -> tuple[float, float] | None:
    """Return (fmin, fmax) expected for PSD/spectrogram sanity checks."""
    if stim_name in SINE_FREQS:
        f = SINE_FREQS[stim_name]
        return (f * 0.9, f * 1.1)
    if stim_name in ("ChirpFwd", "ChirpRev"):
        return (CHIRP_F_START, CHIRP_F_END)
    if stim_name == "NoiseLp100":
        return (0.0, 100.0)
    if stim_name == "NoiseLp500":
        return (0.0, 500.0)
    return None
