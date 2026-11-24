import numpy as np
from scipy.signal import butter, filtfilt

def uniform_noise_01_with_mean(n, mean=0.5, seed=None):
    if not (0.0 <= mean <= 1.0):
        raise ValueError("mean must be in [0,1].")

    rng = np.random.default_rng(seed)

    # Uniform(0,1) has mean 0.5 exactly
    if np.isclose(mean, 0.5):
        low, high = 0.0, 1.0
    else:
        delta = min(mean, 1.0 - mean)
        low, high = mean - delta, mean + delta

    return rng.uniform(low, high, size=n)

def lowpass_filter(x, fs, cutoff_hz, order=4):
    if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        raise ValueError("cutoff_hz must be between 0 and fs/2.")
    b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
    return filtfilt(b, a, x)

def make_signal(
    n_total=1000,
    n_ramp=250,
    mean=0.5,
    use_noise=True,          # <-- toggle noise on/off
    fs=1000.0,
    cutoff_hz=10.0,
    seed=42
):
    # Ramp from 0 to mean
    ramp = np.linspace(0.0, mean, n_ramp)

    n_tail = n_total - n_ramp

    if not use_noise:
        # No noise: flat line at the same mean after the ramp
        tail = np.full(n_tail, mean)
    else:
        # Uniform noise (bounded in [0,1]) with mean-centered support
        tail = uniform_noise_01_with_mean(n_tail, mean=mean, seed=seed)

        # Low-pass filter to set "frequency content"
        tail = lowpass_filter(tail, fs=fs, cutoff_hz=cutoff_hz, order=4)

        # Filtering can overshoot slightly; clip back to [0,1]
        
        tail = np.clip(tail, 0.0, 1.0)

        # Optional: recenter to keep the post-filter mean close to desired mean
        # (clipping can shift it slightly)
        tail = np.clip(tail + (mean - np.mean(tail)), 0.0, 1.0)

    return np.concatenate([ramp, tail])

# ---- Generate CSV's ----

N_TOTAL = 1000
N_RAMP  = 250
MEAN    = 0.5
FS      = 1000.0

cutoff_frequencies = [10, 50, 100, 200,500]  # Hz

ramp = np.linspace(0.0, MEAN, N_RAMP)

for fc in cutoff_frequencies:
    signal = make_signal(use_noise=True, mean=0.5, cutoff_hz=100.0)
    filename = f"signal_lp_{fc:.1f}Hz.csv"
    np.savetxt(filename, signal, delimiter=",")
    print(f"Saved {filename}")

signal_no_noise   = make_signal(use_noise=False, mean=0.5)
np.savetxt("signal_no_noise.csv", signal_no_noise, delimiter=",")

import matplotlib.pyplot as plt

# Time / sample index
t = np.arange(len(signal_no_noise))
plt.figure(figsize=(10, 4))
plt.plot(t, signal_with_noise, label="With noise", linewidth=1)
plt.plot(t, signal_no_noise, label="No noise", linewidth=2)
plt.xlabel("Sample index")
plt.ylabel("Signal value")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()
