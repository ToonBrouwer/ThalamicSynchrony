"""
Editable constants for the Doric-driven TRN stimulation session.

Edit only this file to change stimuli / timing / device paths.
Everything else derives from values here.
"""
from pathlib import Path

# =============================================================================
# Doric DLL / SDK paths
# =============================================================================
# Absolute path to DoricSystem.dll
DLL_PATH = (
    r"C:\Users\brouw\Documents\PhD\Projects\Thalamic_Synchrony"
    r"\TRN_Stimulation\Code\Doric_Stim\DoricSystemDLL\DoricSystemDLL-2026-April"
    r"\API\lib\x64\release\DoricSystem.dll"
)

# Folder containing Doric's authoritative Python struct definitions
# (lightsource_defs.py, doric_system_defs.py). We import them directly
# so we don't fall out of sync when Doric ships an SDK update.
EXAMPLES_PYTHON_DIR = (
    r"C:\Users\brouw\Documents\PhD\Projects\Thalamic_Synchrony"
    r"\TRN_Stimulation\Code\Doric_Stim\DoricSystemDLL\DoricSystemDLL-2026-April"
    r"\Examples\Python\LightSource"
)

# =============================================================================
# Device
# =============================================================================
# UID of the LED driver. Leave as None on first run — run_session.py will
# print available devices via available_devices_infos() and you paste the
# UID in here.
DORIC_UID = None          # e.g. b"c006855a20e326fe"
CHANNEL = 1               # 1-based for user sanity (converted to 0-based zero-indexed enum internally)

# =============================================================================
# Current / intensity
# =============================================================================
# Peak current in mA. Waveforms are scaled so the sample value in the
# 1000-point Custom buffer never exceeds this.
MAX_CURRENT_MA = 1000

# For bipolar stimuli (chirp, noise), the waveform is centred at
# MAX_CURRENT_MA / 2 and swings ± NOISE_AMP_FRAC * MAX_CURRENT_MA / 2.
# 0.9 matches the Arduino's ±88%-of-center behaviour.
NOISE_AMP_FRAC = 0.9

# =============================================================================
# Stimulus parameters
# =============================================================================
# Custom-mode stimuli (chirp, noise) are limited to 1000 samples. At 2 s
# per trial that gives 500 Hz update rate, safely above the 200 Hz chirp
# top end and the 100 Hz noise cutoff.
STIM_DURATION_CUSTOM_S   = 2.0
N_CUSTOM_SAMPLES         = 1000

# Complex-mode stimuli (sines, ramp) run at whatever duration we choose
# since they use the device's native primitives (no sample cap).
STIM_DURATION_COMPLEX_S  = 5.0

# Chirp parameters — log sweep
CHIRP_F0 = 0.5            # Hz
CHIRP_F1 = 200.0          # Hz

# Noise parameters
NOISE_LP_HZ = 100.0       # lowpass cutoff
NOISE_SEED  = 12345
NOISE_RAMP_MS = 250.0     # linear rise into noise at trial onset

# Sinusoid frequencies (native Complex.Sine — no aliasing)
SINE_FREQS = [2, 10, 25, 60, 100]

# Ramp parameters (also runs through Custom, 200 Hz update is fine for a ramp)
RAMP_RISE_MS = 250.0

# =============================================================================
# Session structure
# =============================================================================
# Edit per-stimulus reps. Set to 0 to exclude. Names must match those in
# stimuli.build_trial() below.
REPS = {
    'ChirpFwd':   20,
    'ChirpRev':   20,
    'NoiseLp100': 20,
    'Sine2Hz':    20,
    'Sine10Hz':   20,
    'Sine25Hz':   20,
    'Sine60Hz':   20,
    'Sine100Hz':  20,
    'Ramp':       20,
}

# Inter-stimulus interval (measured stim-onset to next-stim-onset is
# ISI_BASE_SEC + stim_duration, jittered).
ISI_BASE_SEC    = 10.0
ISI_JITTER_FRAC = 0.20    # ± 20 %
RANDOM_SEED     = 12345   # trial order — fixed → reproducible session

# =============================================================================
# DLL timing
# =============================================================================
# The SDK example waits 1000 ms after ls_send_settings_uid to let the
# settings propagate to the device. If empirical testing shows this is
# safe to reduce, edit here.
SETTLE_MS_AFTER_SEND_SETTINGS = 1000

# Wait after init() and after open_device_uid() (also from SDK example).
WAIT_AFTER_INIT_MS = 5000
WAIT_AFTER_OPEN_MS = 5000

# =============================================================================
# Logging
# =============================================================================
# One new subfolder per session, named by launch timestamp.
LOG_ROOT = Path(__file__).parent / "logs"
