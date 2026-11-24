"""
READ THE LIGHT STIMULI FROM THE IO BOARD
========================================

Three lines of the IO board (OneBox-ADC) describe every stimulus:

    gate       high for the 5 s of the stimulus                       (ADC10 in E29-E32)
    message    a line of text sent 160 to 56 ms before the gate       (ADC3)
    command    the analog command to the LED driver = the waveform    (ADC9)

The message looks like   T12 S5 D50   and means
    T   trial number within one run of the stimulus script, from 1
    S   stimulus type, 0 to 7, in the order of STIM_NAMES below
    D   light level: 20 = low, 50 = high

How the message is sent (worked out from E32_session1, checked against the waveform of all 160
stimuli): the line is low at rest; every character is a high start bit, 8 data bits with the
lowest bit first, and a low stop bit; one bit lasts 1 ms.

The functions here are small on purpose. They take plain arrays and know nothing about files.
"""
import re

import numpy as np

STIM_NAMES = ["ChirpFwd", "ChirpRev", "NoiseLp100", "NoiseLp500", "Sine10", "Sine25", "Sine60", "Ramp"]   # S = 0 .. 7
LEVEL_NAMES = {20: "low", 50: "high"}                                                                     # D
BIT_MS = 1.0                      # length of one bit of the message
MESSAGE_WINDOW_S = 0.4            # the message is looked for in this stretch before the gate
MIN_STIMULUS_S = 4.0              # a gate pulse shorter than this is not a stimulus (the script starts with a 15 ms blip)


def is_high(line):
    """A digital line as True/False: above the middle between its lowest and its highest value."""
    return line > (int(line.min()) + int(line.max())) / 2


def pulses(high):
    """First sample and first sample after, for every stretch in which a True/False line is True."""
    change = np.diff(high.astype(np.int8))
    start, stop = np.flatnonzero(change == 1) + 1, np.flatnonzero(change == -1) + 1
    if high[0]:
        start = np.r_[0, start]
    if high[-1]:
        stop = np.r_[stop, len(high)]
    return start, stop


def read_message(high, fs):
    """The text on the message line. `high` is the line as True/False, `fs` its sampling rate."""
    bit = BIT_MS / 1000 * fs
    text, sample = "", 0
    while sample < len(high):
        if not high[sample]:                                   # wait for a start bit
            sample += 1
            continue
        middles = (sample + (np.arange(10) + 0.5) * bit).astype(int)     # start bit, 8 data bits, stop bit
        if middles[-1] >= len(high):
            break
        bits = high[middles]
        text += "?" if bits[9] else chr(sum(int(value) << place for place, value in enumerate(bits[1:9])))
        sample = int(sample + 9.5 * bit)                       # into the stop bit; the next start bit is found from there
        while sample < len(high) and high[sample]:
            sample += 1
    return text


def parse_message(text):
    """'T12 S5 D50' -> (12, 5, 50). None if the text is not a stimulus message."""
    found = re.fullmatch(r"T(\d+) S(\d+) D(\d+)\s*", text)
    return tuple(int(number) for number in found.groups()) if found else None


def waveform_type(command, fs):
    """
    The stimulus type read from the shape of the light command of one stimulus (volts, gate on to gate off).
    This is the check on the message: it uses nothing but the recorded waveform.

        ramp     flat after the first half second
        sine     nearly all power at one frequency (10, 25 or 60 Hz)
        chirp    the frequency content of the second half is much higher (forward) or lower (reverse) than of the first
        noise    neither; the 500 Hz noise has power above 150 Hz, the 100 Hz noise has not
    Returns one of STIM_NAMES, or 'unknown'.
    """
    from scipy import signal

    if command[int(0.5 * fs):].std() / (np.ptp(command) + 1e-9) < 0.02:
        return "Ramp"
    centred = command - command.mean()
    frequency, power = signal.welch(centred, fs=fs, nperseg=32768)
    band = (frequency > 0.8) & (frequency < 800)
    frequency, power = frequency[band], power[band]
    peak = frequency[np.argmax(power)]
    if power[(frequency > 0.8 * peak) & (frequency < 1.2 * peak)].sum() / power.sum() > 0.6:
        sines = {"Sine10": 10.0, "Sine25": 25.0, "Sine60": 60.0}
        name = min(sines, key=lambda sine: abs(np.log(sines[sine] / peak)))
        return name if 0.8 < peak / sines[name] < 1.25 else "unknown"
    centre = []                                                # mean frequency of the first and of the second half
    for half in (centred[:len(centred) // 2], centred[len(centred) // 2:]):
        f, p = signal.welch(half, fs=fs, nperseg=16384)
        keep = (f > 0.8) & (f < 800)
        centre.append((f[keep] * p[keep]).sum() / p[keep].sum())
    drift = centre[1] / centre[0]
    if drift > 3:
        return "ChirpFwd"
    if drift < 1 / 3:
        return "ChirpRev"
    if 0.8 < drift < 1.25:
        return "NoiseLp500" if power[frequency > 150].sum() / power.sum() > 0.15 else "NoiseLp100"
    return "unknown"


def light_onset_sample(command, fraction=0.1):
    """First sample at which the command of one stimulus has risen by `fraction` of its range (99.9th percentile). -1 if never."""
    start = np.median(command[:6])
    above = np.flatnonzero(command > start + fraction * (np.percentile(command, 99.9) - start))
    return int(above[0]) if len(above) else -1
