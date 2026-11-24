r"""
STEP 1. WHAT IS ON THE IO BOARD OF EVERY SESSION
================================================

Before any NWB file is built: look at every channel of the IO board (OneBox-ADC) of every session,
and read the stimuli from it. Nothing is assumed from memory; the figure shows each channel.

    conda activate neuroline311
    python io_board_inventory.py                      every session in sessions.csv not done yet
    python io_board_inventory.py E32_session1 ...     only these (done again)

It only reads from Z:. For every session it writes, in OUT_FOLDER:

    io_board_<session>.png     every channel: the whole session, 25 s around one stimulus, 0.3 s at its onset
    io_board_<session>.csv     per channel: lowest, highest and middle value, number of pulses
    stimuli_<session>.csv      one row per stimulus: gate on and off, the message, type, level, light
    inventory.csv              one row per session: the numbers to check (see part 6)

Times here are seconds from the first sample of the IO-board file, NOT the synchronised clock of
the NWB file. This step is for checking the wiring and the stimuli, not for timing.

Which channel carries what is in sessions.csv (adc_command, adc_gate, adc_message, adc_camera).
If a column is wrong for a session, the figure shows it: correct sessions.csv and run again.
"""
# %% SETTINGS ------------------------------------------------------------------------------------
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]                     # the Code folder
PROJECT = CODE.parent                                          # TRN_Stimulation
RAW_ROOT = Path(r"Z:\toonbrouwer\TRNstimulation")              # the raw recordings (read only)
OUT_FOLDER = PROJECT / "Results" / "inventory"
EARLIER_NWB = PROJECT / "Data" / "nwb_v2"                      # the files of the earlier pipeline, only to compare with
PIECE_SAMPLES = 4_000_000                                      # the IO-board file is read this many samples at a time (about 100 MB)

sys.path.insert(0, str(CODE))
from trn import light

OUT_FOLDER.mkdir(parents=True, exist_ok=True)
table = pd.read_csv(CODE / "sessions.csv", dtype=str).set_index("session")
asked = [name for name in sys.argv[1:] if not name.startswith("-")]
sessions = asked or [name for name in table.index if not (OUT_FOLDER / f"stimuli_{name}.csv").exists()]
print(f"{len(sessions)} sessions to do: {', '.join(sessions)}")

summary = []
for session in sessions:
    row = table.loc[session]
    command_channel, gate_channel = int(row["adc_command"]), int(row["adc_gate"])
    message_channel, camera_channel = int(row["adc_message"]), int(row["adc_camera"])

    # %% 1. READ the whole IO-board file of this session (1 to 3 GB) --------------------------------
    recording = next(iter(sorted((RAW_ROOT / row["raw_folder"]).glob("Record Node*/experiment*/recording*"))), None)
    if recording is None:
        print(f"{session}: no recording found in {RAW_ROOT / row['raw_folder']}, skipped")
        continue
    info = next(c for c in json.loads((recording / "structure.oebin").read_text())["continuous"] if "OneBox-ADC" in c["folder_name"])
    n_channels, fs = info["num_channels"], info["sample_rate"]
    to_volts = np.array([channel["bit_volts"] for channel in info["channels"]])
    print(f"\n{session}: reading the IO board ...", flush=True)
    # The file is read in pieces. Reading a file of more than 2 GB from Z: in one go returns zeros
    # without any error (seen on 24 Nov 2026), so every piece is also checked for that.
    adc_file = recording / "continuous" / "OneBox-129.OneBox-ADC" / "continuous.dat"
    n_samples = adc_file.stat().st_size // 2 // n_channels
    adc = np.empty((n_samples, n_channels), np.int16)
    with open(adc_file, "rb") as file:
        for first in range(0, n_samples, PIECE_SAMPLES):
            piece = np.fromfile(file, dtype=np.int16, count=min(PIECE_SAMPLES, n_samples - first) * n_channels).reshape(-1, n_channels)
            if len(piece) != min(PIECE_SAMPLES, n_samples - first) or not piece.any():
                raise SystemExit(f"{session}: reading {adc_file} failed at sample {first} (too short, or only zeros). Nothing written.")
            adc[first:first + len(piece)] = piece
    print(f"  {n_samples / fs / 60:.1f} minutes, {n_channels} channels at {fs} Hz")

    # %% 2. EVERY CHANNEL in numbers ----------------------------------------------------------------
    channels = []
    for channel in range(n_channels):
        line = adc[:, channel]
        starts, _ = light.pulses(light.is_high(line))
        channels.append(dict(channel=f"ADC{channel}", lowest_v=line.min() * to_volts[channel], highest_v=line.max() * to_volts[channel],
                             middle_v=float(np.median(line[::100])) * to_volts[channel], pulses=len(starts), pulses_per_s=len(starts) / (n_samples / fs)))
    channels = pd.DataFrame(channels)
    channels.to_csv(OUT_FOLDER / f"io_board_{session}.csv", index=False)

    # %% 3. THE STIMULI: gate, message, light -------------------------------------------------------
    gate_on, gate_off = light.pulses(light.is_high(adc[:, gate_channel]))
    is_stimulus = (gate_off - gate_on) / fs >= light.MIN_STIMULUS_S
    message_line = light.is_high(adc[:, message_channel])
    window = int(light.MESSAGE_WINDOW_S * fs)
    stimuli = []
    for on, off in zip(gate_on[is_stimulus], gate_off[is_stimulus]):
        text = light.read_message(message_line[max(on - window, 0):on], fs)
        numbers = light.parse_message(text)
        command = adc[on:off, command_channel] * to_volts[command_channel]
        stimuli.append(dict(
            gate_on_s=on / fs, gate_off_s=off / fs, message=text.strip(),
            trial_in_block=numbers[0] if numbers else -1, stim_code=numbers[1] if numbers else -1, level_code=numbers[2] if numbers else -1,
            stimulus_type=light.STIM_NAMES[numbers[1]] if numbers and numbers[1] < len(light.STIM_NAMES) else "",
            light_level=light.LEVEL_NAMES.get(numbers[2], "") if numbers else "",
            light_peak_v=command.max(), light_mean_v=command.mean()))
    stimuli = pd.DataFrame(stimuli)
    if len(stimuli):
        stimuli["block"] = (stimuli["trial_in_block"] == 1).cumsum()
    stimuli.to_csv(OUT_FOLDER / f"stimuli_{session}.csv", index=False)

    # the camera: one pulse to low per frame
    frames_on, _ = light.pulses(~light.is_high(adc[:, camera_channel]))

    # %% 4. COMPARE with the trials table of the earlier pipeline (types were read from the waveform there)
    type_agrees = level_agrees = earlier_trials = np.nan
    earlier_file = EARLIER_NWB / f"{session}.nwb"
    if earlier_file.exists() and len(stimuli):
        import h5py
        with h5py.File(earlier_file, "r") as earlier:
            earlier_type = earlier["intervals/trials/stimulus_type"][:].astype(str)
            earlier_level = earlier["intervals/trials/light_level"][:].astype(str)
        earlier_trials = len(earlier_type)
        if earlier_trials == len(stimuli):
            type_agrees = float((earlier_type == stimuli["stimulus_type"].to_numpy()).mean())
            level_agrees = float((earlier_level == stimuli["light_level"].to_numpy()).mean())

    # %% 5. THE FIGURE: every channel -----------------------------------------------------------------
    per_second = int(round(fs))
    n_seconds = n_samples // per_second
    lowest, highest = np.zeros((n_seconds, n_channels), np.int16), np.zeros((n_seconds, n_channels), np.int16)
    for first in range(0, n_seconds, 200):
        piece = adc[first * per_second:min(first + 200, n_seconds) * per_second].reshape(-1, per_second, n_channels)
        lowest[first:first + len(piece)], highest[first:first + len(piece)] = piece.min(1), piece.max(1)
    onsets = gate_on[is_stimulus]
    centre = int(onsets[min(2, len(onsets) - 1)]) if len(onsets) else n_samples // 2          # the third stimulus
    views = [(centre - int(5 * fs), centre + int(20 * fs)), (centre - int(0.2 * fs), centre + int(0.1 * fs))]
    role = {command_channel: "light command", gate_channel: "gate", message_channel: "message", camera_channel: "camera"}

    fig, ax = plt.subplots(n_channels, 3, figsize=(16, 1.25 * n_channels + 1), gridspec_kw=dict(width_ratios=[2.2, 1.5, 1]))
    for channel in range(n_channels):
        ax[channel, 0].fill_between(np.arange(n_seconds), lowest[:, channel] * to_volts[channel], highest[:, channel] * to_volts[channel], color="#333333", lw=0)
        for column, (a, b) in enumerate(views, start=1):
            a, b = max(a, 0), min(b, n_samples)
            ax[channel, column].plot((np.arange(a, b) - centre) / fs, adc[a:b, channel] * to_volts[channel], color="#333333", lw=0.6)
            ax[channel, column].axvline(0, color="#B03A2E", lw=0.5, ls=":")
        label = f"ADC{channel}" + (f"\n{role[channel]}" if channel in role else "")
        ax[channel, 0].set_ylabel(label + "\n(V)", rotation=0, ha="right", va="center", fontsize=8)
        for panel in ax[channel]:
            panel.tick_params(labelsize=7)
            panel.spines[["top", "right"]].set_visible(False)
            if channel < n_channels - 1:
                panel.set_xticklabels([])
    ax[0, 0].set_title("whole session (lowest and highest value per second)", loc="left", fontsize=9)
    ax[0, 1].set_title("25 s around the third stimulus (red = gate goes high)", loc="left", fontsize=9)
    ax[0, 2].set_title("0.3 s around that onset", loc="left", fontsize=9)
    ax[-1, 0].set_xlabel("time from the start of the IO-board file (s)")
    ax[-1, 1].set_xlabel("time from the gate going high (s)")
    ax[-1, 2].set_xlabel("time from the gate going high (s)")
    fig.suptitle(f"{session}: every channel of the IO board. Named channels are those in sessions.csv; {len(stimuli)} stimuli found on the gate", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FOLDER / f"io_board_{session}.png", dpi=110)
    plt.close(fig)

    # %% 6. ONE ROW of numbers to check for this session -------------------------------------------
    read = stimuli["stim_code"] >= 0 if len(stimuli) else pd.Series(dtype=bool)
    numbers = dict(
        session=session, minutes=round(n_samples / fs / 60, 1),
        gate_pulses=len(gate_on), stimuli=int(is_stimulus.sum()), short_gate_pulses=int((~is_stimulus).sum()),
        stimulus_s=round(float(np.median((gate_off - gate_on)[is_stimulus] / fs)), 4) if is_stimulus.any() else np.nan,
        onset_to_onset_s=round(float(np.median(np.diff(onsets) / fs)), 3) if len(onsets) > 1 else np.nan,
        messages_read=int(read.sum()), blocks=int(stimuli["block"].max()) if len(stimuli) else 0,
        stimuli_per_block=" ".join(str(n) for n in stimuli.groupby("block").size()) if len(stimuli) else "",
        per_type=" ".join(f"{name}:{n}" for name, n in stimuli["stimulus_type"].value_counts().reindex(light.STIM_NAMES).fillna(0).astype(int).items()) if len(stimuli) else "",
        levels=" ".join(f"{name or 'none'}:{n}" for name, n in stimuli["light_level"].value_counts().items()) if len(stimuli) else "",
        earlier_trials=earlier_trials, type_agrees_with_earlier=type_agrees, level_agrees_with_earlier=level_agrees,
        camera_pulses=len(frames_on), camera_per_s=round(len(frames_on) / max((frames_on[-1] - frames_on[0]) / fs, 1), 2) if len(frames_on) > 1 else 0)
    summary.append(numbers)
    print("  " + ", ".join(f"{name} {value}" for name, value in numbers.items() if name != "session"), flush=True)
    del adc, message_line

# %% 7. THE TABLE of all sessions done so far ---------------------------------------------------------
inventory_file = OUT_FOLDER / "inventory.csv"
if summary:
    done = pd.DataFrame(summary).set_index("session")
    if inventory_file.exists():
        before = pd.read_csv(inventory_file, index_col="session")
        done = pd.concat([before[~before.index.isin(done.index)], done])
    done.reindex([name for name in table.index if name in done.index]).to_csv(inventory_file)
    print(f"\nDone. Table: {inventory_file}")
