"""
Stimulus generator for LED driver experiments.
Generates CSVs (1000 samples, values in [0,1]) and summary plots for:
  - Step (ramp to constant)
  - Sinusoidal at multiple frequencies
  - Chirp (forward and backward)
  - Band-limited white noise

All stimuli share a consistent mean power of 0.5.
"""

import numpy as np
from scipy.signal import butter, filtfilt, chirp as scipy_chirp, spectrogram
import matplotlib.pyplot as plt
from pathlib import Path

# ── Global parameters ────────────────────────────────────────────────
N_TOTAL = 1000          # number of output points (LED driver limit)
T_TOTAL = 1.0           # total duration in seconds  <-- CHANGE THIS to stretch time
FS = N_TOTAL / T_TOTAL  # effective sampling rate (Hz)
NYQUIST = FS / 2        # max representable frequency

N_RAMP = int(0.25 * N_TOTAL)  # 25% ramp for step & noise onset (250 samples)
MEAN = 0.5              # DC offset / average power level
AMPLITUDE = 0.5         # full swing: signal spans [0, 1]
SEED = 42

FREQS = [2, 10, 25, 60, 100, 200]  # Hz - target frequencies

OUT_DIR = Path("stimuli")
OUT_DIR.mkdir(exist_ok=True)

t = np.linspace(0, T_TOTAL, N_TOTAL, endpoint=False)  # time vector

print(f"Configuration: {N_TOTAL} points over {T_TOTAL} s")
print(f"  Effective sample rate: {FS:.0f} Hz, Nyquist: {NYQUIST:.0f} Hz")
print(f"  Ramp: {N_RAMP} samples ({N_RAMP / FS * 1000:.0f} ms)")
print()


# ── Helper functions ─────────────────────────────────────────────────
def onset_ramp(n_total, n_ramp):
    """Smooth onset envelope: 0->1 over n_ramp, then 1 for the rest."""
    env = np.ones(n_total)
    env[:n_ramp] = np.linspace(0, 1, n_ramp)
    return env


def save_csv(name, signal):
    path = OUT_DIR / f"{name}.csv"
    np.savetxt(path, signal, delimiter=",")
    return path


def clip01(signal):
    return np.clip(signal, 0.0, 1.0)


# ── 1.  Step stimulus (ramp -> constant) ─────────────────────────────
def make_step(n_ramp=N_RAMP, level=MEAN):
    """Linear ramp from 0 to `level`, then hold."""
    ramp = np.linspace(0.0, level, n_ramp)
    hold = np.full(N_TOTAL - n_ramp, level)
    return np.concatenate([ramp, hold])


# ── 2.  Sinusoidal stimuli ───────────────────────────────────────────
def make_sine(freq):
    """
    Cosine-phase sine wave: starts at 0, rises to 1, oscillates [0, 1].
    Uses -cos so first motion is always upward from 0.
    """
    if freq > NYQUIST:
        print(f"  WARNING: {freq} Hz exceeds Nyquist ({NYQUIST:.0f} Hz)")
    sig = MEAN - AMPLITUDE * np.cos(2 * np.pi * freq * t)
    return clip01(sig)


# ── 3.  Chirp stimuli ───────────────────────────────────────────────
def make_chirp(f_start, f_end, method="linear"):
    """
    Frequency sweep from f_start -> f_end.
    Phase = 180 deg so it starts at 0 (bottom), consistent with sinusoids.
    """
    sweep = scipy_chirp(t, f0=f_start, f1=f_end, t1=T_TOTAL,
                        method=method, phi=180)
    sig = MEAN + AMPLITUDE * sweep
    return clip01(sig)


# ── 4.  Band-limited white noise ─────────────────────────────────────
def make_noise(cutoff_hz, seed=SEED):
    """
    Gaussian noise, low-pass filtered, with step-like ramp onset.
    Ramp from 0 to MEAN over N_RAMP samples, then noise around MEAN.
    """
    rng = np.random.default_rng(seed)
    noise_raw = rng.standard_normal(N_TOTAL)

    # low-pass filter
    if cutoff_hz < NYQUIST:
        b, a = butter(4, cutoff_hz / NYQUIST, btype="low")
        noise_raw = filtfilt(b, a, noise_raw)

    # normalise so noise fills available range around MEAN
    peak = np.max(np.abs(noise_raw)) + 1e-12
    noise_norm = noise_raw / peak * AMPLITUDE * 0.9  # 90% of range to avoid constant clipping

    # build signal: linear ramp from 0 -> MEAN, then MEAN + noise
    ramp = np.linspace(0.0, MEAN, N_RAMP)
    tail = MEAN + noise_norm[N_RAMP:]
    sig = np.concatenate([ramp, clip01(tail)])
    return clip01(sig)


# ======================================================================
#  Generate all stimuli
# ======================================================================
stimuli = {}

# -- Step --
stimuli["step"] = make_step()

# -- Sinusoids --
for f in FREQS:
    stimuli[f"sine_{f}Hz"] = make_sine(f)

# -- Chirps --
stimuli["chirp_fwd_2-200Hz"] = make_chirp(2, 200)
stimuli["chirp_bwd_200-2Hz"] = make_chirp(200, 2)
stimuli["chirp_log_fwd_2-200Hz"] = make_chirp(2, 200, method="logarithmic")
stimuli["chirp_log_bwd_200-2Hz"] = make_chirp(200, 2, method="logarithmic")

# -- White noise at different bandwidths --
for bw in [10, 50, 100, 200, 500]:
    stimuli[f"noise_lp_{bw}Hz"] = make_noise(cutoff_hz=bw, seed=SEED)

# -- Save all CSVs --
print(f"Saving {len(stimuli)} stimuli to {OUT_DIR}/")
for name, sig in stimuli.items():
    save_csv(name, sig)
    print(f"  {name}.csv  (min={sig.min():.3f}, max={sig.max():.3f}, mean={sig.mean():.3f})")


# ======================================================================
#  Plotting
# ======================================================================
PLOT_DIR = Path("stimuli/plots")
PLOT_DIR.mkdir(exist_ok=True)
t_ms = t * 1000  # time in ms for plotting


def plot_group(title, names, filename, ncols=2):
    """Plot a group of stimuli in a tiled figure."""
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.5 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for idx, name in enumerate(names):
        ax = axes[idx // ncols, idx % ncols]
        ax.plot(t_ms, stimuli[name], linewidth=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Value")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Time (ms)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=150)
    print(f"Saved plot: {path}")
    plt.close(fig)


def plot_time_frequency(title, names, filename, ncols=2, nperseg=128):
    """Plot time-frequency spectrograms for a group of stimuli."""
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for idx, name in enumerate(names):
        ax = axes[idx // ncols, idx % ncols]
        sig = stimuli[name]
        f_spec, t_spec, Sxx = spectrogram(sig, fs=FS, nperseg=nperseg,
                                           noverlap=nperseg - 4, nfft=256)
        # limit frequency axis to relevant range
        f_max = min(300, NYQUIST)
        f_mask = f_spec <= f_max
        ax.pcolormesh(t_spec * 1000, f_spec[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-12),
                      shading="gouraud", cmap="inferno")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Freq (Hz)")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Time (ms)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=150)
    print(f"Saved plot: {path}")
    plt.close(fig)


def plot_combined(title, names, filename, ncols=2, nperseg=128):
    """Plot waveform + spectrogram side-by-side for each stimulus."""
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for idx, name in enumerate(names):
        sig = stimuli[name]
        # waveform
        ax_wave = axes[idx, 0]
        ax_wave.plot(t_ms, sig, linewidth=0.8)
        ax_wave.set_ylim(-0.02, 1.02)
        ax_wave.set_ylabel("Value")
        ax_wave.set_title(f"{name} - waveform", fontsize=9)
        # spectrogram
        ax_spec = axes[idx, 1]
        f_spec, t_spec, Sxx = spectrogram(sig, fs=FS, nperseg=nperseg,
                                           noverlap=nperseg - 4, nfft=256)
        f_max = min(300, NYQUIST)
        f_mask = f_spec <= f_max
        ax_spec.pcolormesh(t_spec * 1000, f_spec[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-12),
                           shading="gouraud", cmap="inferno")
        ax_spec.set_ylabel("Freq (Hz)")
        ax_spec.set_title(f"{name} - spectrogram", fontsize=9)
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=150)
    print(f"Saved plot: {path}")
    plt.close(fig)


# --- Waveform plots ---
plot_group("Step stimulus", ["step"], "step.png", ncols=1)

sine_names = [f"sine_{f}Hz" for f in FREQS]
plot_group("Sinusoidal stimuli", sine_names, "sinusoids.png", ncols=2)

chirp_names = [
    "chirp_fwd_2-200Hz", "chirp_bwd_200-2Hz",
    "chirp_log_fwd_2-200Hz", "chirp_log_bwd_200-2Hz",
]
plot_group("Chirp stimuli", chirp_names, "chirps.png", ncols=2)

noise_names = [f"noise_lp_{bw}Hz" for bw in [10, 50, 100, 200, 500]]
plot_group("Band-limited white noise", noise_names, "noise.png", ncols=2)

# --- Time-frequency spectrograms ---
plot_time_frequency("Sinusoid spectrograms", sine_names, "sinusoids_tf.png", ncols=2)
plot_time_frequency("Chirp spectrograms", chirp_names, "chirps_tf.png", ncols=2)
plot_time_frequency("Noise spectrograms", noise_names, "noise_tf.png", ncols=2)

# --- Combined overview: waveform + spectrogram ---
overview_names = [
    "step", "sine_10Hz", "sine_100Hz",
    "chirp_fwd_2-200Hz", "chirp_bwd_200-2Hz", "noise_lp_100Hz",
]
plot_combined("Stimulus overview", overview_names, "overview_combined.png")

print("\nDone! All stimuli and plots saved.")
print(f"\nNote: With T_TOTAL={T_TOTAL}s and {N_TOTAL} points, "
      f"Nyquist={NYQUIST:.0f} Hz.")
print("To make stimuli longer (e.g. 2s), change T_TOTAL at the top.")
print("This lowers the effective sample rate and Nyquist limit.")
