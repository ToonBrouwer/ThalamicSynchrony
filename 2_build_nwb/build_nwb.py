r"""
STEP 2. BUILD THE NWB FILE OF ONE SESSION
=========================================

This script reads top to bottom. Each numbered part does one thing and says what and why.
There are no hidden steps: what you read is what runs. It is the CPO build (build_nwb.py of the
CPO ephys project) with the stimulus parts rewritten for light.

How to use it
-------------
1. Fill in THIS SESSION, the first block below.
2. Run it:
       conda activate neuroline311
       python build_nwb.py
   For a session listed in sessions.csv:   python build_nwb.py E29_session2
   For every session in sessions.csv:      python build_all.py
   In VS Code or Jupyter you can also run it part by part: every "# %%" line starts a cell.

What it makes
-------------
    <NWB_FOLDER>\<session>.nwb   the file. It is always built from scratch, under a temporary name,
                                 and only replaces the previous file when every check has passed.

    in the file                                 written by
    ------------------------------------------  -----------------------------------------------
    session, subject, stimulation site          this script (part 4)
    probe and electrodes                        Neuroline (part 5)
    units, one table per shank                  neuroconv, as Neuroline calls it (part 6)
    light command of the whole session          this script (parts 7 and 11)
    trials: one row per light stimulus          this script (parts 8 and 10)
    blocks, light steps, example waveforms      this script (parts 9 and 10)
    treadmill lines                             this script (parts 7 and 11)
    LFP                                         this script, from Neuroline's LFP in the job folder (part 12)

Not in the file yet: face motion.
The format is described in docs\DATA_FORMAT_SPEC_TRN.md. Nothing is written to Z:.

Where to change things
----------------------
    the session, the animal, its folders, its IO-board channels    THIS SESSION, just below
    a limit, a name                                                SETTINGS, below that
    which columns the trials table has                             part 10, and their descriptions in TRIAL_COLUMNS
    how a block is defined                                         part 9
    what is checked at the end                                     part 13
"""
# %% THIS SESSION: fill in everything here -----------------------------------------------------------
from pathlib import Path

SESSION = "E32_session1"              # <animal>_<session>; becomes the file name
TARGET_REGION = "thalamus"            # where the probe ends; it passes cortex and hippocampus on the way

# the animal ("?" = not known yet: it is written as unknown and the script says so)
SUBJECT = "E32"
SEX = "?"                             # M or F
DATE_OF_BIRTH = "?"                   # YYYY-MM-DD
GENOTYPE = "VGAT-ChR2"
STRAIN = "?"
SPECIES = "Mus musculus"

# the stimulation
STIMULATION_SITE = "thalamic reticular nucleus (TRN)"
LIGHT_SOURCE = "Doric LED driver, periscope"
EXCITATION_NM = "?"                   # wavelength of the light in nm

# where this session's data is
RAW_FOLDER = Path(r"Z:\toonbrouwer\TRNstimulation\E322026-08-04_17-54-17PeriscopeTest_session1")     # the Open Ephys folder
JOB_FOLDER = Path(r"Z:\toonbrouwer\TRNstimulation\SpikeSorted"
                  r"\E322026-08-04_17-54-17PeriscopeTest_session1_20261006_143822_job206")           # the Neuroline output

# where the NWB file goes
NWB_FOLDER = Path(r"C:\Users\brouw\Documents\PhD\Projects\Thalamic_Synchrony\TRN_Stimulation\nwb")

# the IO board (OneBox-ADC): which signal is on which channel in THIS session.
# Checked per session in step 1 (Results\inventory\io_board_<session>.png).
ADC_COMMAND = 9                       # the analog command to the LED driver = the light waveform
ADC_GATE = 10                         # high for the 5 s of every stimulus
ADC_MESSAGE = 3                       # the text "T<trial> S<type> D<level>" before every stimulus
ADC_CAMERA = 11                       # one pulse to low per video frame
ADC_TREADMILL = 4                     # treadmill encoder line
ADC_DIRECTION = 0                     # treadmill direction line

SHANKS_USABLE = [1, 2, 3, 4]          # shanks with a finished sort; all sorted units are stored anyway
ADD_LFP = True                        # the LFP belongs in the file (about 10 GB per shank-hour). It is written in
                                      # pieces, so it needs little memory. False = a quick test of the rest.


# %% MANY SESSIONS: `python build_nwb.py E29_session2` takes that session from sessions.csv -----------
# With a session name on the command line, the block above is replaced by that session's row in
# sessions.csv (in the Code folder). build_all.py uses this to build one session after another.
import sys

if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    import pandas as pd
    table = pd.read_csv(Path(__file__).resolve().parents[1] / "sessions.csv", dtype=str).fillna("?").set_index("session")
    if sys.argv[1] not in table.index:
        raise SystemExit(f"'{sys.argv[1]}' is not in sessions.csv. Sessions there: {', '.join(table.index)}")
    row = table.loc[sys.argv[1]]
    SESSION, TARGET_REGION, SUBJECT = sys.argv[1], row["target_region"], row["subject"]
    SEX, DATE_OF_BIRTH, GENOTYPE, STRAIN = row["sex"], row["date_of_birth"], row["genotype"], row["strain"]
    RAW_FOLDER = RAW_FOLDER.parent / row["raw_folder"]
    JOB_FOLDER = JOB_FOLDER.parent / row["job_folder"]
    SHANKS_USABLE = [int(shank) for shank in row["shanks_usable"].split()]
    ADC_COMMAND, ADC_GATE, ADC_MESSAGE = int(row["adc_command"]), int(row["adc_gate"]), int(row["adc_message"])
    ADC_CAMERA, ADC_TREADMILL, ADC_DIRECTION = int(row["adc_camera"]), int(row["adc_treadmill"]), int(row["adc_direction"])
ADD_LFP = ADD_LFP and "--no-lfp" not in sys.argv              # python build_nwb.py --no-lfp  = a quick test without LFP


# %% SETTINGS: the same for every session -------------------------------------------------------------
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

# who and where (goes into every file)
EXPERIMENTER = "Brouwer, Toon"
LAB = "Cognitive and Systems Neuroscience group"
INSTITUTION = "University of Amsterdam"
TIMEZONE = "Europe/Amsterdam"

# IO board
N_ADC_CHANNELS = 12
PIECE_SAMPLES = 4_000_000         # the IO-board file is read this many samples at a time. Reading a file of more
                                  # than 2 GB from Z: in one go returns zeros without an error (seen 24 Nov 2026).
TREADMILL_HZ = 1000               # the treadmill lines are averaged per millisecond
STEP_WINDOW_S = 0.001             # after every light step: the stretch in which spikes can be the electrical transient

# the trials table: every column and what it means (this text goes into the file)
TRIAL_COLUMNS = {
    "light_onset_time": "first moment the light command has risen by 10% of its range in this stimulus (s)",
    "message_time": "start of the stimulus message on the IO board (s)",
    "block": "block the stimulus belongs to (see the blocks table)",
    "protocol": "opto: 5 s light stimuli of eight types",
    "trial_in_block": "T of the stimulus message: counter within one run of the stimulus script, from 1",
    "stim_code": "S of the stimulus message: 0 ChirpFwd, 1 ChirpRev, 2 NoiseLp100, 3 NoiseLp500, 4 Sine10, 5 Sine25, 6 Sine60, 7 Ramp",
    "stimulus_type": "the name that belongs to stim_code",
    "level_code": "D of the stimulus message: 20 or 50",
    "light_level": "low (level_code 20) or high (level_code 50)",
    "message_found": "1 if the stimulus message was read; 0 if stimulus_type comes from the waveform and the level is unknown",
    "waveform_type": "stimulus type read from the shape of the recorded light command, as a check on the message",
    "type_agrees": "1 if waveform_type equals stimulus_type",
    "sine_hz": "frequency of the sine (Hz); empty for the other types",
    "chirp_direction": "up (ChirpFwd) or down (ChirpRev); empty text for the other types",
    "noise_cutoff_hz": "cut-off of the noise (Hz); empty for the other types",
    "light_peak_v": "99.9th percentile of the light command during the stimulus (V)",
    "light_mean_v": "mean of the light command during the stimulus (V)",
    "starts_with_step": "1 if the light jumps at onset (sines, chirps); 0 if it rises from zero (ramp, noises)",
    "dark_before_s": "time since the previous stimulus ended (s); empty for the first stimulus",
    "prev_stimulus_type": "type of the stimulus before this one in the block; empty text for the first of a block",
    "prev_light_level": "light level of the stimulus before this one in the block; empty text for the first of a block",
}

# the trn package (the trn folder in Code) reads the stimuli from the IO board
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trn import light


# %% 1. CHECK that the block at the top is filled in ------------------------------------------------
# Folders and the session name must be right: without them nothing can be built.
# Details of the animal and the light that are still "?" do not stop the build: they are written
# as unknown, listed here and in the build notes, and filled in by rebuilding later.
missing = [f"{name} is not filled in" for name, value in dict(SESSION=SESSION, SUBJECT=SUBJECT, TARGET_REGION=TARGET_REGION).items()
           if str(value).strip() in ("?", "")]
for name, folder in dict(RAW_FOLDER=RAW_FOLDER, JOB_FOLDER=JOB_FOLDER).items():
    if not folder.is_dir():
        missing.append(f"{name} does not exist: {folder}")
if SEX not in ("M", "F", "U", "?"):
    missing.append("SEX must be M, F or ?")
if missing:
    raise SystemExit("Fix this in the THIS SESSION block at the top of build_nwb.py, then run again:\n  " + "\n  ".join(missing))

unknown = [name for name, value in dict(SEX=SEX, DATE_OF_BIRTH=DATE_OF_BIRTH, GENOTYPE=GENOTYPE, STRAIN=STRAIN,
                                        EXCITATION_NM=EXCITATION_NM).items() if str(value).strip() in ("?", "")]
print(f"Session {SESSION}: {SUBJECT}, {TARGET_REGION}")
if unknown:
    print(f"NOTE: not filled in yet, written as unknown: {', '.join(unknown)}")


# %% 2. THE INPUTS: find every folder, and stop if one is missing ---------------------------------
recording_folder = next(iter(sorted(RAW_FOLDER.glob("Record Node*/experiment*/recording*"))), None)
if recording_folder is None:
    raise SystemExit(f"No recording found in {RAW_FOLDER}")
adc_file = recording_folder / "continuous" / "OneBox-129.OneBox-ADC" / "continuous.dat"

# the sorted shanks: {1: folder, 2: folder, ...}. A shank without metrics.csv did not finish and is left out.
analyser_folders = {int(f.name[-1]): f for f in sorted((JOB_FOLDER / "spikes" / "analyser").glob("OneBox-129.ProbeA-?"))
                    if (f / "metrics.csv").exists()}
lfp_folders = {int(f.name[-1]): f for f in sorted((JOB_FOLDER / "lfp").glob("OneBox-129.ProbeA-?")) if f.is_dir()}

print(f"  raw        {recording_folder}")
print(f"  job        {JOB_FOLDER}")
print(f"  shanks     sorted: {sorted(analyser_folders)}   LFP: {sorted(lfp_folders)}")
if not analyser_folders or not adc_file.exists():
    raise SystemExit("Sorted shanks or the IO-board file are missing: nothing was built.")

# where the file goes. It is built under a temporary name and renamed at the very end.
nwb_file = NWB_FOLDER / f"{SESSION}.nwb"
building = NWB_FOLDER / f"{SESSION}_building.nwb"
cache_folder = NWB_FOLDER / f"{SESSION}_build"               # small files that are slow to make, and tables to look at
nwb_file.parent.mkdir(parents=True, exist_ok=True)
cache_folder.mkdir(parents=True, exist_ok=True)
building.unlink(missing_ok=True)


# %% 3. NEUROLINE: open the recording ---------------------------------------------------------------
# Neuroline's EphysStream reads the Open Ephys settings, finds the four shanks ("probes" for
# Neuroline), and synchronises all streams. From it we get:
#   stream.probes      one per shank, with its electrode positions
#   stream.daq.time    the synchronised time of every IO-board sample   = the clock of this file
# It only reads from Z:. It takes a few minutes because it reads the sample numbers of every stream.
from neuroline.instances.streams import EphysStream

print("\nNeuroline reads the recording and synchronises the streams ...", flush=True)
stream = EphysStream(str(RAW_FOLDER))
for probe in stream.probes:
    probe.split_broadband_recording()        # Neuropixels 2.0: one stream holds both the spikes and the LFP


# %% 4. NEW FILE: session, subject and stimulation site --------------------------------------------
# Same content as Neuroline's Session.init_nwb(). We make the file ourselves because init_nwb()
# can only write inside the raw folder, and because we want session_id in it.
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject
from pynwb.ogen import OptogeneticStimulusSite

timezone = pytz.timezone(TIMEZONE)
start_time = timezone.localize(datetime.strptime(stream.recording_date, "%d %b %Y %H:%M:%S"))
date_of_birth = None if DATE_OF_BIRTH == "?" else timezone.localize(datetime.fromisoformat(DATE_OF_BIRTH))

nwb = NWBFile(
    session_id=SESSION,
    session_description=f"Light on the TRN, recording through cortex and hippocampus into the {TARGET_REGION}; recording {RAW_FOLDER.name}",
    identifier=str(uuid.uuid4()),                    # a new random ID for every build
    session_start_time=start_time,                   # all times in the file count from the start of acquisition
    experimenter=EXPERIMENTER, lab=LAB, institution=INSTITUTION,
    subject=Subject(subject_id=SUBJECT, species=SPECIES, strain="unknown" if STRAIN == "?" else STRAIN,
                    genotype="unknown" if GENOTYPE == "?" else GENOTYPE, sex="U" if SEX == "?" else SEX, date_of_birth=date_of_birth))
light_device = nwb.create_device(name="light source", description=LIGHT_SOURCE)
nwb.add_ogen_site(OptogeneticStimulusSite(
    name="TRN", device=light_device, location=STIMULATION_SITE,
    description=f"Light delivered with a periscope; the opsin is expressed in {GENOTYPE} mice. The TRN is stimulated, not recorded.",
    excitation_lambda=float("nan") if EXCITATION_NM == "?" else float(EXCITATION_NM)))
nwb.create_processing_module(name="behavior", description="Processed behavioural data")
build_notes = dict(spec="DATA_FORMAT_SPEC_TRN", built=datetime.now().isoformat(timespec="seconds"), session=SESSION,
                   target_region=TARGET_REGION, raw_folder=RAW_FOLDER.name, job_folder=JOB_FOLDER.name,
                   shanks_usable=" ".join(str(shank) for shank in SHANKS_USABLE), not_filled_in=" ".join(unknown),
                   io_board=dict(command=ADC_COMMAND, gate=ADC_GATE, message=ADC_MESSAGE, camera=ADC_CAMERA,
                                 treadmill=ADC_TREADMILL, direction=ADC_DIRECTION))
with NWBHDF5IO(str(building), mode="w") as io:
    io.write(nwb)

# tell Neuroline which file to write to
stream.set_nwb(str(building))
for probe in stream.probes:
    probe.ap_recording.set_nwb(str(building))
    probe.lfp_recording.set_nwb(str(building))


# %% 5. NEUROLINE: probe and electrodes -------------------------------------------------------------
stream.to_nwb()


# %% 6. UNITS, one table per shank -------------------------------------------------------------------
# Neuroline's own export (fork_add_sorting_analyser_to_nwb) cannot be used here, for a checked
# reason: it stores the location of every spike as a table of units x the largest spike count,
# which for these sessions is 2 to 19 GB per session (10 GB in memory for one shank of
# E32_session3). So this part calls the two functions Neuroline's export itself calls
# (neuroconv's units writer and Neuroline's electrode lookup) and then adds, per unit:
#   every quality and waveform metric          (Neuroline 1.1.0 collects these but never writes them)
#   shank, depth_um, x_um                      where the unit is
#   spike_amplitudes, spike_x_um, spike_depth_um   one value per spike, in the order of spike_times
# The per-spike columns are stored like spike_times (one long list plus where each unit ends),
# which takes 0.1 to 0.6 GB per session instead.
# The raw recording is opened, not loaded: it is only needed for the clock of the spike times.
import spikeinterface.core as si
from neuroconv.tools.spikeinterface.spikeinterface import _add_units_table_to_nwbfile
from neuroline.utils.forks import fork_get_electrode_group_indices

spikes_in_job = {}                                           # per shank: the number of spikes of every unit, for the check in part 13
for probe in stream.probes:
    shank = int(probe.ap_recording.stream_name[-1])
    if shank not in analyser_folders:
        print(f"shank {shank}: no finished sort in this job, skipped")
        continue
    folder = analyser_folders[shank]
    probe_name = probe.pri_obj.annotations["name"]

    # load_extensions=False: only what is asked for below is read. Loading everything fails on the
    # single-spike waveforms, which cannot be read from Z: in one go.
    analyser = si.load_sorting_analyzer(folder=str(folder), load_extensions=False)
    unit_ids = analyser.unit_ids
    spike_vector = analyser.sorting.to_spike_vector()        # every spike: its sample and its unit, in time order
    per_unit = np.bincount(spike_vector["unit_index"], minlength=len(unit_ids))
    spikes_in_job[shank] = per_unit
    print(f"shank {shank}: {len(unit_ids)} units, {len(spike_vector):,} spikes ...", flush=True)

    sorting = analyser.sorting.select_units(unit_ids=unit_ids)
    sorting.set_property("unit_name", np.array([f"{SESSION}_s{shank}_unit_{unit}" for unit in unit_ids]))   # E32_session1_s1_unit_12
    raw_recording = probe.ap_recording._load_raw_data()
    sorting.register_recording(raw_recording)                # spike samples become times on the synchronised clock
    templates = analyser.get_extension("templates")

    with NWBHDF5IO(str(building), mode="r+") as io:
        nwb = io.read()
        electrode_rows = fork_get_electrode_group_indices(raw_recording, nwbfile=nwb, probe_name=probe_name)
        _add_units_table_to_nwbfile(
            sorting=sorting, nwbfile=nwb, unit_ids=unit_ids, property_descriptions={}, skip_properties=None,
            write_in_processing_module=True, units_table_name=f"units_{probe_name}",
            unit_table_description=f"Units of shank {shank}, all of them, with their metrics (written as Neuroline's export does)",
            waveform_means=templates.get_templates(), waveform_sds=templates.get_templates(operator="std"),
            unit_electrode_indices=[electrode_rows] * len(unit_ids))
        units_table = nwb.processing["ecephys"][f"units_{probe_name}"]

        for extension in ("quality_metrics", "template_metrics"):
            metrics = analyser.get_extension(extension).get_data().loc[unit_ids]
            for column in metrics.columns:
                values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(float)
                if column not in units_table.colnames and not np.isnan(values).all():      # no empty columns
                    units_table.add_column(name=column, description=f"{column} (SpikeInterface {extension})", data=values)

        position = analyser.get_extension("unit_locations").get_data()
        units_table.add_column(name="unit_locations", description="x,y,z coordinates for each unit", data=position)
        units_table.add_column(name="shank", description="shank the unit is on (1-4)", data=np.full(len(position), shank))
        units_table.add_column(name="depth_um", description="position along the shank in µm (0 = the lowest site)",
                               data=position[:, 1].astype(float))
        units_table.add_column(name="x_um", description="position across the shank in µm", data=position[:, 0].astype(float))

        # one value per spike. The spikes are regrouped per unit (keeping their time order), which is
        # the order of spike_times; `ends` says where each unit's spikes end in the long list.
        by_unit = np.argsort(spike_vector["unit_index"], kind="stable")
        ends = np.cumsum(per_unit)
        amplitudes = analyser.get_extension("spike_amplitudes").get_data()
        locations = analyser.get_extension("spike_locations").get_data()
        if len(amplitudes) != len(spike_vector) or len(locations) != len(spike_vector):
            raise SystemExit(f"shank {shank}: {len(spike_vector)} spikes, {len(amplitudes)} amplitudes, {len(locations)} locations.")
        units_table.add_column(name="spike_amplitudes", description="amplitude of every spike (µV), in the order of spike_times",
                               data=np.asarray(amplitudes)[by_unit].astype(np.float32), index=ends)
        units_table.add_column(name="spike_x_um", description="position of every spike across the shank (µm), in the order of spike_times",
                               data=np.asarray(locations["x"])[by_unit].astype(np.float32), index=ends)
        units_table.add_column(name="spike_depth_um", description="position of every spike along the shank (µm), in the order of spike_times",
                               data=np.asarray(locations["y"])[by_unit].astype(np.float32), index=ends)

        for name in ("neuroline_parameters.json", "parameters.json"):       # Neuroline's sorting settings (the name differs by version)
            settings = folder / name
            if settings.exists():
                nwb.add_scratch(data=settings.read_text(), name=f"params_spikes_analyser_{folder.name}",
                                description=f"Neuroline parameters (analyser) for {folder.name}")
                break
        io.write(nwb)
    del analyser, sorting, templates, spike_vector, amplitudes, locations


# %% 7. IO BOARD: the light command, the gate, the message, camera pulses, treadmill ----------------
# One pass through the IO-board file (1 to 3 GB on Z:), in pieces. Kept from it:
#   command         the light command of the whole session, as recorded (raw counts)
#   gate_on/off     the sample at which the gate goes high and low, for every gate pulse
#   message_line    the message line as high/low, to read the text before every stimulus (part 8)
#   camera_pulses   the sample of every camera pulse = every video frame
#   treadmill, direction    the two treadmill lines averaged per millisecond, in volts
# Times: Neuroline gives the synchronised time of every sample. The IO board runs at a steady
# rate, so time = first time + sample / rate; the build stops if that is off by more than 0.05 ms.
# The result is saved in the cache folder; delete io_board.npz there to read the file again.
io_cache = cache_folder / "io_board.npz"
if io_cache.exists():
    saved = np.load(io_cache)
    command, message_line = saved["command"], saved["message_line"]
    gate_on, gate_off, camera_pulses = saved["gate_on"], saved["gate_off"], saved["camera_pulses"]
    treadmill, direction = saved["treadmill"], saved["direction"]
    adc_start, adc_rate, counts_to_volts = float(saved["adc_start"]), float(saved["adc_rate"]), float(saved["counts_to_volts"])
else:
    adc_time = np.asarray(stream.daq.time)                   # Neuroline: synchronised time of every sample
    n_samples = adc_file.stat().st_size // 2 // N_ADC_CHANNELS
    if len(adc_time) != n_samples:
        raise SystemExit(f"IO board: {n_samples} samples in the file, {len(adc_time)} time stamps from Neuroline.")
    adc_start, adc_rate = float(adc_time[0]), (n_samples - 1) / float(adc_time[-1] - adc_time[0])
    check = np.arange(0, n_samples, 30_000)
    off_ms = np.abs(adc_time[check] - (adc_start + check / adc_rate)).max() * 1e3
    if off_ms > 0.05:
        raise SystemExit(f"IO board: the clock is not steady (up to {off_ms:.3f} ms from a straight line). Do not continue, look into it.")
    channel_info = next(c for c in json.loads((recording_folder / "structure.oebin").read_text())["continuous"]
                        if "OneBox-ADC" in c["folder_name"])["channels"]
    counts_to_volts = float(channel_info[ADC_COMMAND]["bit_volts"])

    print("\nReading the IO board ...", flush=True)
    wanted = [ADC_COMMAND, ADC_GATE, ADC_MESSAGE, ADC_CAMERA, ADC_TREADMILL, ADC_DIRECTION]
    lines = np.empty((n_samples, len(wanted)), np.int16)
    with open(adc_file, "rb") as file:
        for first in range(0, n_samples, PIECE_SAMPLES):
            n_piece = min(PIECE_SAMPLES, n_samples - first)
            piece = np.fromfile(file, dtype=np.int16, count=n_piece * N_ADC_CHANNELS).reshape(-1, N_ADC_CHANNELS)
            if len(piece) != n_piece or not piece.any():
                raise SystemExit(f"IO board: reading failed at sample {first} (too short, or only zeros).")
            lines[first:first + n_piece] = piece[:, wanted]
    command = lines[:, 0].copy()
    gate_on, gate_off = light.pulses(light.is_high(lines[:, 1]))
    message_line = light.is_high(lines[:, 2])
    camera_pulses, _ = light.pulses(~light.is_high(lines[:, 3]))
    millisecond = (np.arange(n_samples) / adc_rate * TREADMILL_HZ).astype(np.int64)
    in_millisecond = np.maximum(np.bincount(millisecond), 1)
    treadmill = (np.bincount(millisecond, weights=lines[:, 4]) / in_millisecond * counts_to_volts).astype(np.float32)
    direction = (np.bincount(millisecond, weights=lines[:, 5]) / in_millisecond * counts_to_volts).astype(np.float32)
    del lines, millisecond
    np.savez_compressed(io_cache, command=command, message_line=message_line, gate_on=gate_on, gate_off=gate_off,
                        camera_pulses=camera_pulses, treadmill=treadmill, direction=direction,
                        adc_start=adc_start, adc_rate=adc_rate, counts_to_volts=counts_to_volts)


def time_of(sample):
    """The time on the session clock of an IO-board sample number."""
    return adc_start + np.asarray(sample) / adc_rate


# %% 8. STIMULI: every gate pulse of 5 s, its message and its light ---------------------------------
# A stimulus = a gate pulse of at least 4 s (the stimulus script starts every run with a 15 ms blip
# of the gate without light; those are left out and counted).
# Its type and level are in the message sent just before it; the type is also read from the shape
# of the recorded light, as a check. If a message cannot be read, the type from the waveform is used.
is_stimulus = (gate_off - gate_on) / adc_rate >= light.MIN_STIMULUS_S
window = int(light.MESSAGE_WINDOW_S * adc_rate)
rows = []
for on, off in zip(gate_on[is_stimulus], gate_off[is_stimulus]):
    before = message_line[max(on - window, 0):on]
    text = light.read_message(before, adc_rate)
    numbers = light.parse_message(text)
    volts = command[on:off] * counts_to_volts
    from_waveform = light.waveform_type(volts, adc_rate)
    onset = light.light_onset_sample(volts)
    stim_code = numbers[1] if numbers and numbers[1] < len(light.STIM_NAMES) else -1
    rows.append(dict(
        start_time=float(time_of(on)), stop_time=float(time_of(off)),
        light_onset_time=float(time_of(on + onset)) if onset >= 0 else np.nan,
        message_time=float(time_of(max(on - window, 0) + np.argmax(before))) if before.any() else np.nan,
        trial_in_block=numbers[0] if numbers else -1, stim_code=stim_code,
        stimulus_type=light.STIM_NAMES[stim_code] if stim_code >= 0 else from_waveform,
        level_code=numbers[2] if numbers else -1, light_level=light.LEVEL_NAMES.get(numbers[2], "unknown") if numbers else "unknown",
        message_found=int(stim_code >= 0), waveform_type=from_waveform,
        light_peak_v=float(np.percentile(volts, 99.9)), light_mean_v=float(volts.mean()),
        gate_on_sample=int(on), gate_off_sample=int(off)))
events = pd.DataFrame(rows)
if not len(events):
    raise SystemExit(f"No stimulus found on the gate (IO-board channel {ADC_GATE}). Check the channels in THIS SESSION against step 1.")
print(f"\n{len(gate_on)} gate pulses: {len(events)} stimuli, {int((~is_stimulus).sum())} shorter than {light.MIN_STIMULUS_S:g} s left out")
print(f"  message read for {events['message_found'].mean():.1%}; type from the waveform agrees for "
      f"{(events['waveform_type'] == events['stimulus_type']).mean():.1%}")


# %% 9. BLOCKS: one block per run of the stimulus script ---------------------------------------------
# The trial counter of the message starts again at 1 with every run of the script: that is a block.
# (Without messages, a pause of more than 30 s starts a new block.)
if events["message_found"].all():
    new_block = events["trial_in_block"] == 1
else:
    new_block = (events["start_time"] - events["stop_time"].shift()).fillna(np.inf) > 30
    events["trial_in_block"] = events.groupby(new_block.cumsum()).cumcount() + 1
new_block.iloc[0] = True
events["block"] = [f"{number:02d}_opto" for number in new_block.cumsum()]
events["protocol"] = "opto"
blocks = events.groupby("block").agg(start_time=("start_time", "first"), stop_time=("stop_time", "last"),
                                     n_trials=("start_time", "size")).reset_index()
blocks["protocol"] = "opto"
blocks["source_folder"] = RAW_FOLDER.name
blocks = blocks[["start_time", "stop_time", "block", "protocol", "n_trials", "source_folder"]]
print("\nBlocks:")
print(blocks[["block", "n_trials", "start_time", "stop_time"]].round(1).to_string(index=False))


# %% 10. TRIALS: the table, one row per light stimulus; light steps; example waveforms --------------
# To add a column: compute it here, and add its description to TRIAL_COLUMNS in the settings.
trials = events[["start_time", "stop_time", "light_onset_time", "message_time", "block", "protocol", "trial_in_block",
                 "stim_code", "stimulus_type", "level_code", "light_level", "message_found", "waveform_type"]].copy()
trials["type_agrees"] = (trials["waveform_type"] == trials["stimulus_type"]).astype(int)
trials["sine_hz"] = trials["stimulus_type"].map({"Sine10": 10.0, "Sine25": 25.0, "Sine60": 60.0})
trials["chirp_direction"] = trials["stimulus_type"].map({"ChirpFwd": "up", "ChirpRev": "down"}).fillna("")
trials["noise_cutoff_hz"] = trials["stimulus_type"].map({"NoiseLp100": 100.0, "NoiseLp500": 500.0})
trials["light_peak_v"] = events["light_peak_v"]
trials["light_mean_v"] = events["light_mean_v"]
trials["starts_with_step"] = trials["stimulus_type"].isin(["Sine10", "Sine25", "Sine60", "ChirpFwd", "ChirpRev"]).astype(int)
trials["dark_before_s"] = trials["start_time"] - trials["stop_time"].shift()
trials["prev_stimulus_type"] = trials.groupby("block")["stimulus_type"].shift().fillna("")
trials["prev_light_level"] = trials.groupby("block")["light_level"].shift().fillna("")
trials = trials.drop(columns=[c for c in trials.columns if trials[c].isna().all()])    # NWB wants no empty column
trials.to_csv(cache_folder / "trials.csv", index=False)                               # a copy to look at in Excel
blocks.to_csv(cache_folder / "blocks.csv", index=False)

# light steps: the light goes on and off abruptly; each step gives an electrical transient on every
# channel for less than a millisecond. One row per step, so analyses can leave that stretch out.
step_rows = []
for kind, samples in (("on", events["gate_on_sample"]), ("off", events["gate_off_sample"])):
    for trial, sample in enumerate(samples):
        around = command[max(sample - 6, 0):sample + 7] * counts_to_volts              # 0.2 ms either side
        step_rows.append(dict(start_time=float(time_of(sample)), stop_time=float(time_of(sample)) + STEP_WINDOW_S,
                              kind=kind, trial=trial, step_v=float(around.max() - around.min())))
light_steps = pd.DataFrame(step_rows).sort_values("start_time").reset_index(drop=True)

# example waveforms: the first stimulus of every type and level
examples = events.groupby(["stimulus_type", "light_level"]).first().reset_index()
print(f"\n{len(trials)} stimuli in {len(blocks)} blocks; {len(examples)} combinations of type and level")
print(trials.groupby(["stimulus_type", "light_level"]).size().unstack(fill_value=0).to_string())


# %% 11. WRITE: light command, trials, blocks, light steps, examples, treadmill ---------------------
from hdmf.backends.hdf5 import H5DataIO
from pynwb import TimeSeries
from pynwb.epoch import TimeIntervals

with NWBHDF5IO(str(building), mode="r+") as io:
    nwb = io.read()

    nwb.add_stimulus(TimeSeries(
        name="light_command", unit="V", rate=adc_rate, starting_time=adc_start, conversion=counts_to_volts,
        data=H5DataIO(command, compression="gzip", chunks=(1_000_000,)),
        description=f"Command to the LED driver as recorded on IO-board channel ADC{ADC_COMMAND}, whole session. "
                    "Stored as raw counts; multiply by `conversion` for volts.",
        comments="This is the command, not the light at the tissue."))
    for example in examples.itertuples():
        nwb.add_stimulus_template(TimeSeries(
            name=f"{example.stimulus_type}_{example.light_level}", unit="V", rate=adc_rate, starting_time=0.0, conversion=counts_to_volts,
            data=H5DataIO(command[example.gate_on_sample:example.gate_off_sample], compression="gzip"),
            description=f"Light command of the first {example.stimulus_type} stimulus at {example.light_level} level "
                        f"(the stimulus that starts at {example.start_time:.3f} s); time 0 is the gate going high."))

    table = TimeIntervals(name="trials", description="one row per light stimulus; start_time is the gate going high")
    for column in trials.columns[2:]:                        # the first two, start_time and stop_time, are built in
        table.add_column(name=column, description=TRIAL_COLUMNS[column])
    for row in trials.to_dict("records"):
        table.add_row(**row)
    nwb.trials = table

    table = TimeIntervals(name="blocks", description="one row per block (= one run of the stimulus script)")
    for column in blocks.columns[2:]:
        table.add_column(name=column, description=column.replace("_", " "))
    for row in blocks.to_dict("records"):
        table.add_row(**row)
    nwb.add_time_intervals(table)

    table = TimeIntervals(name="light_steps", description="every moment the light goes on or off, and the millisecond after it, "
                                                          "in which spikes can be the electrical transient of the step")
    table.add_column(name="kind", description="on or off")
    table.add_column(name="trial", description="row of the trials table this step belongs to")
    table.add_column(name="step_v", description="size of the jump of the light command within 0.2 ms of the step (V)")
    for row in light_steps.to_dict("records"):
        table.add_row(**row)
    nwb.add_time_intervals(table)

    behaviour = nwb.processing["behavior"]
    for name, values, channel in (("treadmill", treadmill, ADC_TREADMILL), ("treadmill_direction", direction, ADC_DIRECTION)):
        behaviour.add(TimeSeries(
            name=name, unit="V", rate=float(TREADMILL_HZ), starting_time=adc_start, data=H5DataIO(values, compression="gzip"),
            description=f"Treadmill line on IO-board channel ADC{channel}, averaged per millisecond",
            comments="In volts, as recorded: the conversion to speed is not worked out yet."))

    build_notes.update(gate_pulses=int(len(gate_on)), stimuli=int(len(trials)), camera_pulses=int(len(camera_pulses)),
                       first_camera_pulse_s=float(time_of(camera_pulses[0])) if len(camera_pulses) else None,
                       last_camera_pulse_s=float(time_of(camera_pulses[-1])) if len(camera_pulses) else None)
    nwb.add_scratch(data=json.dumps(build_notes), name="build_notes", description="how this file was built")
    nwb.add_scratch(data=adc_rate, name="adc_fs", description="sampling rate of the IO board on the session clock (Hz)")
    io.write(nwb)
np.save(cache_folder / "camera_pulse_times.npy", time_of(camera_pulses))            # for face motion, later


# %% 12. THE LFP, written in pieces (skipped when ADD_LFP is False) --------------------------------
# Neuroline's own  probe.lfp_recording.processed.to_nwb()  loads a whole shank into memory, which
# does not fit next to the rest. So this part writes the same thing to the same place in the file,
# half a gigabyte at a time:
#   processing/ecephys/"LFP <probe>"   one series per shank, samples x channels, 32-bit floats in microvolts
# The LFP itself is Neuroline's: it is read from the job folder exactly as Neuroline saved it
# (band-passed and resampled to 1 kHz; its settings go into the file).
if ADD_LFP:
    from neuroconv.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
        SpikeInterfaceRecordingDataChunkIterator)
    from pynwb.ecephys import LFP, ElectricalSeries

    for probe in stream.probes:
        shank = int(probe.lfp_recording.stream_name[-1])
        if shank not in lfp_folders:
            print(f"shank {shank}: no LFP in the job folder, skipped")
            continue
        probe_name = probe.lfp_recording.probe_info["name"]
        lfp_recording = si.load(str(lfp_folders[shank]))                  # opened, not loaded
        settings_file = next(f for f in (lfp_folders[shank] / "neuroline_parameters.json", lfp_folders[shank] / "parameters.json") if f.exists())
        lfp_settings = json.loads(settings_file.read_text())
        print(f"shank {shank}: adding the LFP ({lfp_recording.get_num_samples():,} samples x "
              f"{lfp_recording.get_num_channels()} channels) ...", flush=True)

        with NWBHDF5IO(str(building), mode="r+") as io:
            nwb = io.read()
            # the electrodes of this shank, found by their label as Neuroline does
            labels = nwb.electrodes.to_dataframe()["label"]
            rows = labels.index[labels.str.startswith(probe_name) | labels.str.startswith(probe.pri_obj.name)].to_list()
            if len(rows) != lfp_recording.get_num_channels():
                raise SystemExit(f"shank {shank}: {len(rows)} electrodes in the file, "
                                 f"{lfp_recording.get_num_channels()} LFP channels.")
            series = ElectricalSeries(
                name=f"LFP {probe_name}",
                description="LFP in microvolts (multiply by `conversion` for volts)",
                data=H5DataIO(SpikeInterfaceRecordingDataChunkIterator(recording=lfp_recording, return_scaled=True,
                                                                       buffer_gb=0.5, display_progress=True)),
                conversion=1e-6,
                timestamps=lfp_recording.get_times(),                     # Neuroline's synchronised time of every sample
                electrodes=nwb.create_electrode_table_region(region=rows, description=f"{probe_name} electrodes"),
                filtering=f"Bandpass {lfp_settings['freq_cutoff'][0]}-{lfp_settings['freq_cutoff'][1]} Hz",
                comments=f"Neuroline LFP preprocessing: pipeline {lfp_settings['pipeline']}, resampled to "
                         f"{lfp_settings['resample_rate']} Hz, reference setting {lfp_settings['reference']}. "
                         "Written in pieces by build_nwb.py.")
            nwb.processing["ecephys"].add(LFP(electrical_series=series, name=f"LFP {probe_name}"))
            nwb.add_scratch(data=json.dumps(lfp_settings), name=f"params_lfp_preprocessing_{lfp_folders[shank].name}",
                            description=f"Neuroline LFP preprocessing parameters for {lfp_folders[shank].name}")
            io.write(nwb)


# %% 13. CHECK the file, then give it its final name ------------------------------------------------
# Every line prints ok or FAIL. A file with a FAIL keeps its temporary name, so the previous good
# file (if any) is not replaced.
problems = []


def expect(ok, text):
    """Print one check; remember it if it failed."""
    print(f"  {'ok  ' if ok else 'FAIL'}  {text}")
    if not ok:
        problems.append(text)


print(f"\nChecking {building.name} ({building.stat().st_size / 1e9:.1f} GB)")
with NWBHDF5IO(str(building), mode="r") as io:
    nwb = io.read()
    expect(nwb.session_id == SESSION, f"session_id is {nwb.session_id}")
    expect(nwb.electrodes is not None and len(nwb.electrodes) == 384 * len(stream.probes),
           f"{len(nwb.electrodes) if nwb.electrodes is not None else 0} electrodes (384 per shank expected)")

    # units: the same units and the same number of spikes per unit as in the job folder
    unit_tables = {name: table for name, table in nwb.processing["ecephys"].data_interfaces.items() if name.startswith("units_")}
    expect(len(unit_tables) == len(analyser_folders), f"{len(unit_tables)} units tables for {len(analyser_folders)} sorted shanks")
    for (name, table), (shank, in_job) in zip(sorted(unit_tables.items()), sorted(spikes_in_job.items())):
        last_spike = table["spike_times"].data[:]            # where each unit's spikes end in the long list
        spike_times = table["spike_times"].target.data[:]
        in_file = np.diff(np.r_[0, last_spike])
        expect(np.array_equal(in_file, in_job), f"{name}: {len(in_file)} units, {len(spike_times):,} spikes, as in the job folder")
        expect(spike_times.min() > 0, f"{name}: spike times from {spike_times.min():.1f} to {spike_times.max():.1f} s")
        expect("snr" in table.colnames and "presence_ratio" in table.colnames, f"{name}: quality metrics are in the table")
        expect(np.array_equal(table["spike_amplitudes"].data[:], last_spike) and len(table["spike_amplitudes"].target.data) == len(spike_times),
               f"{name}: one amplitude and one position per spike")

    in_file = nwb.trials.to_dataframe()
    expect(len(in_file) == len(trials), f"{len(in_file)} trials")
    expect(in_file["start_time"].is_monotonic_increasing and (in_file["stop_time"] > in_file["start_time"]).all(),
           "trials in time order, each stop after its start")
    expect(((in_file["stop_time"] - in_file["start_time"]).between(4.9, 5.1)).all(), "every stimulus lasts 5 s")
    expect(in_file["message_found"].mean() > 0.99, f"stimulus message read for {in_file['message_found'].mean():.2%} of trials")
    expect(in_file["type_agrees"].mean() > 0.99, f"type in the message and type of the waveform agree for {in_file['type_agrees'].mean():.2%} of trials")
    expect(len(nwb.intervals["light_steps"]) == 2 * len(trials), "two light steps per trial")
    expect("light_command" in nwb.stimulus and len(nwb.stimulus["light_command"].data) == len(command), "the light command is in the file")
    expect(len(nwb.stimulus_template) == len(examples), f"{len(nwb.stimulus_template)} example waveforms")
    signals = nwb.processing["behavior"].data_interfaces
    expect("treadmill" in signals and len(signals["treadmill"].data) == len(treadmill), "treadmill lines are in the file")
    lfp_series = [name for name in nwb.processing["ecephys"].data_interfaces if name.startswith("LFP")]
    print(f"  note  LFP: {', '.join(lfp_series) if lfp_series else 'not added (ADD_LFP is False)'}")
    print("  note  face motion is not in the file yet")

try:                                                         # the file must open in pynapple
    import pynapple
    in_pynapple = pynapple.load_file(str(building))
    expect(len(in_pynapple.keys()) > 0, "opens in pynapple")
    in_pynapple.close()                                      # or Windows cannot rename the file below
except Exception as error:
    expect(False, f"opens in pynapple ({str(error)[:150]})")

try:                                                         # the official NWB check; its remarks are advice
    from nwbinspector import inspect_nwbfile
    remarks = list(inspect_nwbfile(nwbfile_path=str(building)))
    print(f"\nNWB Inspector: {len(remarks)} remarks")
    for remark in remarks:
        print(f"  [{remark.importance.name}] {remark.message}  ({remark.location})")
except ImportError:
    print("\nNWB Inspector not installed, skipped (pip install nwbinspector).")

if problems:
    raise SystemExit(f"\n{len(problems)} check(s) failed. The file is left as {building}")
building.replace(nwb_file)                                   # replaces the previous file, if there is one
print(f"\nDone: {nwb_file}")
