"""
Stimulus generators.

Two kinds of output:
  1. Custom waveform    — np.uint16 array of length 1000, ready to upload to
                          DoricLightSource.send_custom()
  2. Complex spec       — list of LightSourceComplexModulation structs, ready
                          to upload to DoricLightSource.send_complex()

Each build_* function returns a dict:
    {'kind': 'custom' | 'complex',
     'payload': array or list of ComplexModulation,
     'period_ms': float,           # trial duration
     'n_seq': int,                 # always 1 here but kept for clarity
     'name': str}

So session.py can dispatch uniformly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter

from doric_client import LightSourceComplexModulation, LightSourceComplexMode

from config import (
    MAX_CURRENT_MA,
    NOISE_AMP_FRAC,
    STIM_DURATION_CUSTOM_S,
    N_CUSTOM_SAMPLES,
    STIM_DURATION_COMPLEX_S,
    CHIRP_F0, CHIRP_F1,
    NOISE_LP_HZ, NOISE_SEED, NOISE_RAMP_MS,
    SINE_FREQS,
    RAMP_RISE_MS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_uint16_ma(sig_01: np.ndarray) -> np.ndarray:
    """Map [0,1] → [0, MAX_CURRENT_MA] as uint16, clipped."""
    sig = np.clip(sig_01, 0.0, 1.0) * MAX_CURRENT_MA
    return np.round(sig).astype(np.uint16)


def _effective_fs() -> float:
    return N_CUSTOM_SAMPLES / STIM_DURATION_CUSTOM_S    # 500 Hz by default


# ---------------------------------------------------------------------------
# Custom-mode waveforms (chirp, noise, ramp) — 1000 samples
# ---------------------------------------------------------------------------
def make_chirp(fwd: bool) -> np.ndarray:
    """Log sweep f0 → f1 over STIM_DURATION_CUSTOM_S. Bipolar around MAX/2."""
    T = STIM_DURATION_CUSTOM_S
    N = N_CUSTOM_SAMPLES
    t = np.linspace(0, T, N, endpoint=False)
    k = (CHIRP_F1 / CHIRP_F0) ** (1.0 / T)
    phase = 2 * np.pi * CHIRP_F0 * (k ** t - 1) / np.log(k)
    # Amplitude fraction scales the swing (matches noise convention)
    amp = NOISE_AMP_FRAC / 2.0
    sig_01 = 0.5 + amp * np.sin(phase)
    if not fwd:
        sig_01 = sig_01[::-1]
    return _to_uint16_ma(sig_01)


def make_noise_lp(cutoff_hz: float, seed: int) -> np.ndarray:
    """Lowpass FIR-filtered white noise, 250 ms onset ramp, centred at MAX/2."""
    T = STIM_DURATION_CUSTOM_S
    N = N_CUSTOM_SAMPLES
    fs = _effective_fs()
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} must be < Nyquist={nyq} "
            f"(effective sample rate {fs} Hz)"
        )

    # Oversample the noise generation, then filter. FIR taps chosen small
    # because effective fs is only 500 Hz (vs the notebook's 10 kHz).
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-1.0, 1.0, size=N + 400)
    taps = firwin(201, cutoff_hz / nyq, pass_zero=True, window='hamming')
    filt = lfilter(taps, 1.0, raw)[200:][:N]
    peak = np.max(np.abs(filt))
    if peak == 0:
        filt = np.zeros_like(filt)
    else:
        filt = filt / peak * (NOISE_AMP_FRAC / 2.0)

    sig_01 = 0.5 + filt

    # 250 ms onset ramp — linearly ramp the deviation-from-centre up from 0.
    ramp_n = int(NOISE_RAMP_MS / 1000.0 * fs)
    if ramp_n > 0:
        ramp = np.linspace(0.0, 1.0, ramp_n)
        sig_01[:ramp_n] = 0.5 + (sig_01[:ramp_n] - 0.5) * ramp

    return _to_uint16_ma(sig_01)


def make_ramp() -> np.ndarray:
    """Linear rise to max over RAMP_RISE_MS, then plateau at max."""
    T = STIM_DURATION_CUSTOM_S
    N = N_CUSTOM_SAMPLES
    fs = _effective_fs()
    rise_n = int(RAMP_RISE_MS / 1000.0 * fs)
    sig_01 = np.ones(N, dtype=np.float64)
    sig_01[:rise_n] = np.linspace(0.0, 1.0, rise_n)
    return _to_uint16_ma(sig_01)


# ---------------------------------------------------------------------------
# Complex-mode specs (sinusoids) — native primitives, no sample cap
# ---------------------------------------------------------------------------
def make_sine_complex(freq_hz: float, duration_s: float = STIM_DURATION_COMPLEX_S):
    """Single Complex.Sine primitive — pristine at any frequency."""
    period_ms = 1000.0 / freq_hz
    n_cycles  = int(round(duration_s * freq_hz))
    mod = LightSourceComplexModulation()
    mod.mode = LightSourceComplexMode.Sine
    mod.current = int(MAX_CURRENT_MA)   # amplitude cap of the sine
    mod.periodMs = period_ms
    mod.timeOnMs = period_ms            # full cycle active
    mod.nbOfPulsesPerSeq = n_cycles
    mod.nbOfSeq = 1
    mod.startingDelayMs = 0
    mod.delayBetweenSeqMs = 0
    return [mod]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def build_stim(name: str) -> dict:
    """Return a stimulus spec ready to upload.

    Returned dict keys:
        kind       : 'custom' | 'complex'
        payload    : np.uint16[1000] (custom) or list[ComplexModulation] (complex)
        period_ms  : float, duration the device should play for
        n_seq      : 1
        name       : str (echo of input)
        duration_s : float, convenience
    """
    if name == 'ChirpFwd':
        return dict(kind='custom', payload=make_chirp(fwd=True),
                    period_ms=STIM_DURATION_CUSTOM_S * 1000.0, n_seq=1,
                    name=name, duration_s=STIM_DURATION_CUSTOM_S)

    if name == 'ChirpRev':
        return dict(kind='custom', payload=make_chirp(fwd=False),
                    period_ms=STIM_DURATION_CUSTOM_S * 1000.0, n_seq=1,
                    name=name, duration_s=STIM_DURATION_CUSTOM_S)

    if name == 'NoiseLp100':
        return dict(kind='custom', payload=make_noise_lp(NOISE_LP_HZ, NOISE_SEED),
                    period_ms=STIM_DURATION_CUSTOM_S * 1000.0, n_seq=1,
                    name=name, duration_s=STIM_DURATION_CUSTOM_S)

    if name == 'Ramp':
        return dict(kind='custom', payload=make_ramp(),
                    period_ms=STIM_DURATION_CUSTOM_S * 1000.0, n_seq=1,
                    name=name, duration_s=STIM_DURATION_CUSTOM_S)

    if name.startswith('Sine') and name.endswith('Hz'):
        freq = float(name[4:-2])
        if int(freq) not in SINE_FREQS:
            raise ValueError(f"Sine freq {freq} Hz not in SINE_FREQS={SINE_FREQS}")
        return dict(kind='complex', payload=make_sine_complex(freq),
                    period_ms=STIM_DURATION_COMPLEX_S * 1000.0, n_seq=1,
                    name=name, duration_s=STIM_DURATION_COMPLEX_S)

    raise ValueError(f"Unknown stimulus name: {name!r}")
