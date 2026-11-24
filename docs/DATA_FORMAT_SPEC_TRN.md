# TRN stimulation data format, draft 3

Draft of 24 Nov 2026, written to be corrected. Draft 2 has Toon's answers of 24 Nov (marked **DECIDED**) the decoded stimulus message, and the result of step 1 on all 13 sessions. Nothing has been built or moved yet. It applies
`DATA_FORMAT_PRINCIPLES.md` (CPO project) to the TRN stimulation recordings, following
`HANDOFF_TO_NEW_PROJECT.md`. It is based on an inventory of this project's folders and of
`Z:\toonbrouwer\TRNstimulation` (read only).

**DECIDE** = a choice for Toon. **VERIFY** = not checked yet; to check before or during the first build.

In one sentence: one NWB file per session with units, LFP, the light command, a trials table with
one row per light stimulus, blocks, treadmill and face motion on one clock; built by one script
(`build_nwb.py`, copied from CPO and adapted), never edited by hand.

## 1. What is the same as CPO and what is not

Same (checked on E32 session 1): OneBox, stream names `OneBox-129.ProbeA-1..4` and
`OneBox-129.OneBox-ADC` (12 channels, 30300.5 Hz), quad-shank Neuropixels 2.0, a Neuroline job
folder per recording with `spikes\analyser\<shank>\metrics.csv` and `lfp\<shank>`.

Different:

| Point | CPO | TRN | Consequence |
|---|---|---|---|
| Stimulus | sounds, codes on parallel TTL lines | 5 s light waveforms, 8 types; a text message on one line before every stimulus | parts 8 and 10 of the build are rewritten |
| Stimulus copy | speaker copy on ADC1 | LED command on ADC9, gate on ADC10 | measured times come from these two lines |
| Neuroline version of the jobs | 1.1.0 | 1.0.2 | LFP settings are in `parameters.json`, not `neuroline_parameters.json`; there is no `spikes\preprocessed`. **VERIFY** that Neuroline 1.1.0 opens these jobs. |
| Video | several mp4 per session, Facemap | one `.raw` video per session with `_frames.npz`; motion from our own script | parts 9 and 11 change: blocks cannot come from videos |
| Artefacts | none | light transient on every channel; clusters made of it | section 6 |
| IO-board wiring | fixed | moved between sessions (E16) | the channel map is filled in per session |
| Shanks sorted | all | some shanks have no metrics (section 8) | skipped with a note, as CPO does |

## 2. Sessions and blocks

| Level | Definition | Name | Example |
|---|---|---|---|
| Subject | | `E<NN>` | `E32` |
| Session | one probe position, one animal, one day; one NWB file | `<subject>_session<N>` | `E32_session1` |
| Block | one run of the stimulus script: its trial counter starts again at 1 | `<NN>_opto` | `01_opto` |
| Shank | Neuroline calls each shank a probe | `ProbeA-1` .. `ProbeA-4` | |
| Unit | named by Neuroline per shank | `<session>_s<shank>_unit_<n>` | `E32_session1_s2_unit_132` |

- **DECIDED:** sessions keep the names in use: `E32_session1`, `E32_session2Fixed`, ... The date is
  in the file (`session_start_time`) and in `sessions.csv`.
- A block starts where the trial number in the stimulus message goes back to 1 (section 4). In
  E32 session 1 that gives two blocks of 80 stimuli. **VERIFY** for the other sessions.
- Target region: the probe passes cortex, hippocampus and thalamus; `TARGET_REGION = "thalamus"`.
  The stimulation site is TRN, which is not recorded.
- A session must be one Open Ephys folder. E16 has days recorded in two folders (`Session2_1`,
  `Session2_2`): not supported.

Sessions on the archive (15 sorted jobs, 22 raw folders):

| Animal | Raw folders | Sorted | In the current notebooks |
|---|---|---|---|
| E29 | 3 | 3 | 3 |
| E30 | 3 | 3 | 3 |
| E31 | 4 | 4 (session 4: one shank, no metrics) | 3 |
| E32 | 4 | 4 | 4 |
| E16 | 7 (some days in two folders) | 1 (Session1) | 0 |

**DECIDED:** the 13 sessions in use now. E31 session 4 and E16 later.

## 3. What is in the file

| Place in the file | Contents | Written by |
|---|---|---|
| file details | `session_id`, start time (Europe/Amsterdam), experimenter, lab, institution | `build_nwb.py` |
| `subject` | subject ID, species, strain, genotype (VGAT-ChR2), sex, date of birth | `build_nwb.py` |
| `general/optogenetics` | the stimulation site: TRN, light source, wavelength, periscope | `build_nwb.py` |
| `electrodes` | one row per recording site, 384 per shank | Neuroline `EphysStream.to_nwb()` |
| `processing/ecephys/units_ProbeA-<n>` | one table per shank, one row per unit, all metrics | Neuroline, metrics and position columns added by `build_nwb.py` |
| `processing/ecephys/LFP ProbeA-<n>` | every contact at 1 kHz, in µV | `build_nwb.py`, from Neuroline's LFP in the job folder |
| `stimulus/presentation/light_command` | the LED command as recorded on the IO board, whole session, in volts | `build_nwb.py` |
| `intervals/trials` | one row per light stimulus (section 4) | `build_nwb.py` |
| `intervals/blocks` | one row per block | `build_nwb.py` |
| `intervals/light_steps` | every abrupt change of the light, with a 1 ms window after it (section 6) | `build_nwb.py` |
| `processing/behavior/treadmill` | where the treadmill is wired (section 7) | `build_nwb.py` |
| `processing/behavior/motion_energy`, `motion_components` | face motion, one video per session | `build_nwb.py` |
| `scratch` | `build_notes` (spec version, date, old session name, raw and job folder, usable shanks, IO-board channel map), Neuroline's settings | both |

All times are in seconds on Neuroline's synchronised clock.

The light command is in the file because it is the stimulus and several analyses need its
waveform (transfer functions, the artefact screens). It is mostly zero, so it compresses well.

Not in the file: the gate line, camera pulses and sync lines (they are only used to build
tables), raw traces, single-spike waveforms.

- **DECIDED: LFP as 32-bit floats in µV**, as in CPO. No rounding and no cut-off. About 135 GB
  for the 13 sessions. The series carries `conversion = 1e-6`, so tools that apply it get volts.
- **Disk:** C: has 305 GB free. With 32-bit LFP, spike amplitudes and no per-spike locations the 13
  files are roughly 150 to 170 GB together, so they fit next to the existing `Data` folder.
- **DECIDED: the amplitude of every spike is stored** (from the analyser's `spike_amplitudes`), so
  drift and amplitude cut-offs can be checked and Phy can show amplitudes from the file.
  **VERIFY** how Neuroline's export handles it; if it does not, the build adds it as a column.
- **The location of every spike is stored too**, as `spike_x_um` and `spike_depth_um`, in the same
  way as the amplitudes: one long list in the order of `spike_times`. Neuroline's own export
  stores it as a table of units x the largest spike count, which would be 2 to 19 GB per session
  here (100 GB for all); as a list it is 0.1 to 0.4 GB per session. This is the one place where the
  build does not call Neuroline's export function but the two functions underneath it.
- **Correction to my notebook 00b:** the job's LFP is band-passed 0.1 to 300 Hz (Butterworth,
  order 5) before it is resampled to 1 kHz. I described it as unfiltered. The settings also list
  `reference: average`, but the processing steps list only the band-pass. **VERIFY** whether an
  average reference was applied.

## 4. Trials

One row per light stimulus (5 s of light). All sessions use one protocol with eight stimulus
types.

**The stimulus message.** Before every stimulus the Arduino sends a line of text on ADC3, from
160 to 56 ms before the gate goes high: 1 ms per bit, a high start bit, 8 data bits (lowest
first), a low stop bit. Decoded for all 160 stimuli of E32 session 1:

```
T1 S2 D20      trial 1 of the block, stimulus type 2, level 20
T2 S5 D50      trial 2, type 5, level 50
```

| Field | Meaning | Check on E32 session 1 |
|---|---|---|
| `T` | trial number within the run of the script, from 1 | runs 1 to 80 twice: two blocks |
| `S` | stimulus type, the order of the Arduino script | agrees with the type read from the waveform in 160 of 160 |
| `D` | light level | 20 on every low trial, 50 on every high trial |

| `S` | Type | | `S` | Type |
|---|---|---|---|---|
| 0 | ChirpFwd | | 4 | Sine10 |
| 1 | ChirpRev | | 5 | Sine25 |
| 2 | NoiseLp100 | | 6 | Sine60 |
| 3 | NoiseLp500 | | 7 | Ramp |

**DECIDED:** the stimulus type of a trial is the one in this message. The type read from the
recorded waveform is kept next to it as a check.

Checked on all 13 sessions (step 1, `Results\inventory\inventory.csv`): the message was read for
every one of the 4011 stimuli; type and level agree with the earlier, waveform-based tables in
every trial; blocks are 80 stimuli each (2 to 5 per session), except the last block of E31
session 1, which stops after 11; every run of the script starts with one short gate blip.

The digital TTL events of E32 session 1 hold only the 1 Hz sync, so they play no role there.
Trials are found on the gate line of the IO board. In three other sessions an earlier TTL route
gave wrong labels or drifting times; the message and the gate replace it.

| Column | Meaning |
|---|---|
| `start_time` | **DECIDED:** start of the stimulus: the gate going high, measured on the IO board (ADC10) |
| `stop_time` | the gate going low |
| `light_onset_time` | first moment the light command (ADC9) has risen by 10% of its range in this trial |
| `message_time` | start of the stimulus message |
| `block`, `protocol` | |
| `trial_in_block` | `T` of the message |
| `stim_code` | `S` of the message, 0 to 7 |
| `stimulus_type` | the name that belongs to `stim_code` |
| `level_code` | `D` of the message (20 or 50) |
| `light_level` | `low` or `high`, from `level_code` |
| `message_found` | 1 if the message was read; 0 if type and level come from the waveform |
| `waveform_type` | the type read from the recorded waveform |
| `type_agrees` | 1 if `waveform_type` equals `stimulus_type` |
| `sine_hz` | 10, 25 or 60; empty for the other types |
| `chirp_direction` | `up` or `down`; empty for the other types |
| `noise_cutoff_hz` | 100 or 500; empty for the other types |
| `light_peak_v`, `light_mean_v` | peak and mean of the command during the trial, in volts |
| `starts_with_step` | 1 if the light jumps at onset (sines, chirps), 0 if it rises from zero (ramp, noises) |
| `dark_before_s` | time since the previous light went off |
| `prev_stimulus_type`, `prev_light_level` | the stimulus before this one in the block |

- `start_time` is the gate because the ramp and the noises rise slowly: "the light crosses a
  threshold" comes 23 to 40 ms after the start of the waveform and depends on the stimulus. With
  the gate, time zero is the start of the waveform for every type; `light_onset_time` holds the
  threshold crossing.
- A gate pulse counts as a trial only when it lasts about 5 s. E32 session 1 starts with a 15 ms
  gate blip without light (161 gate pulses, 160 stimuli); the build reports what it leaves out.
- **DECIDED: one example waveform per stimulus type and level in `stimulus/templates`** (the first
  stimulus of each), so the shape of each stimulus can be read without cutting it from the session
  trace.
- Missing details of the animal or the light do not stop the build: they are written as unknown
  and listed in the build notes (`not_filled_in`).
- **DECIDE: a `stim` yes/no column.** Every trial has light; there are no no-light trials in the
  protocol. Proposed: no such column, and no sham rows in the table. Dark control periods are
  taken in the analysis from the pauses and from before and after the protocol.
- **VERIFY:** what `D20` and `D50` stand for (duty, percent of the driver range?), and the light
  at the tissue: the command in volts is the input of the LED driver. The calibration
  (`Calibration_info\doric_calibration.xlsx`) could give mW; not used yet.

## 5. Blocks

Columns: `start_time`, `stop_time`, `block`, `protocol`, `n_trials`, `source_folder`.
A block runs from its first to its last stimulus.
The face video is one per session, so it is not a block property: its name, frame count and pulse
count go into the build notes, and the build stops when frames and pulses differ by more than 2.

## 6. Light artefacts

What is known (notebooks 03 and 08e):

- Every abrupt change of the light gives a transient of under 1 ms on all channels.
- Some clusters consist of these transients: edge-locked clusters (fire at light on or off) and
  clusters that fire within 0.5 ms of fast light changes during the 500 Hz noise (0 to 16% of the
  clusters of a session).
- The sorting was not blanked. Under the 500 Hz noise the sorter both adds false spikes and
  loses real ones.

Proposed, following "the file does not judge":

- `intervals/light_steps` in the file: the time of every abrupt light change and a 1 ms window
  after it, so any analysis can leave those spikes out the same way.
- The artefact measures per unit (share of spikes in light, edge locking, step locking) are not
  in the units table. `unit_quality.ipynb` computes them from the file (spike times and the light
  command are both in it) and writes them as columns of `unit_labels.csv` next to the rules.
- **DECIDE:** re-sort with the artefact windows blanked? Not proposed now: it means new cluster
  jobs for every session. The 500 Hz noise stays unusable for population measures until then.

## 7. IO board, treadmill, face motion

Channel map from the older code and rig notes. **VERIFY per session by plotting every channel**
(the first step of the work, before anything is built).

| Channel | E29 to E32 | E16 |
|---|---|---|
| light command (LED driver, analog) | ADC9 | ADC9 or ADC0 |
| gate (high during the 5 s stimulus) | ADC10 | ADC10 or ADC4 |
| stimulus message (text before every stimulus) | ADC3 | ? |
| camera exposure (one edge per frame) | ADC11 | ? |
| treadmill ticks | ADC4 | ADC4, unless that is the gate |
| treadmill direction | ADC0 | ADC0, unless that is the command |

Checked for E32 session 1 by plotting every channel (`Results\inventory\io_board_E32_session1.png`):

| Channel | What it shows in E32 session 1 |
|---|---|
| ADC9 | the light command: the eight waveforms at two levels |
| ADC10 | the gate, high for 5.0006 s per stimulus, 8.18 s from onset to onset |
| ADC11 | camera exposure, about 29.4 pulses per second, from 250 s to 1840 s |
| ADC8 | the same camera pulses at a tenth of the size |
| ADC6 | 1 Hz sync |
| ADC3 | the stimulus message (section 4), 160 to 56 ms before every stimulus; ADC1, ADC2 and ADC5 show a smaller copy of it |
| ADC5, ADC7 | 1 Hz pulses and a trace of the ADC3 burst |
| ADC4 | square steps between 0 and 3.3 V, 39 in the session: treadmill |
| ADC0 | 0 to 0.7 V, slow steps and a few bursts: treadmill direction |

The map is filled in per session in the THIS SESSION block (and as columns of `sessions.csv` for
the batch), not detected: detection picked the wrong line before.

- Treadmill: a usable signal is known for one session only (E32 session 1). Stored where present;
  the build notes say when it is absent.
- Face motion: 13 videos have motion computed by our own script (`face_motion_all.py`), stored
  locally. **DECIDE:** keep that, or recompute with Facemap as CPO does (`extract_motion.py`) so
  both projects have the same signal. Proposed: Facemap, later; the first build uses what exists.

## 8. Known gaps in the sorting jobs

| Session | Shanks with metrics | Without |
|---|---|---|
| E29 session 1, E30 session 1 | 1 | 2 (analyser, no metrics), 3 and 4 not sorted |
| E31 session 2 | 1, 2 | 3 (no metrics), 4 not sorted |
| E32 session 4 | 1, 2, 3 | 4 (no metrics) |
| E31 session 4 | none | 1 (no metrics); not buildable |
| all others | 1 to 4 | |

In E29 and E30 the earlier LFP read had 96 contacts per shank instead of 384. **VERIFY** what the
job's LFP folder holds for those sessions; the CPO build expects one LFP channel per electrode.

## 9. Folders

Proposed, mirroring the CPO project. **DECIDE** before anything is moved.

```
TRN_Stimulation\
    Code\                        the git repository (ThalamicSynchrony)
        README.md                general
        nwb\                     build_nwb.py, build_all.py, sessions.csv, this spec, HOW_TO_BUILD_NWB.md,
                                 nwb_walkthrough.ipynb, nwb_examples.ipynb, unit_quality.ipynb,
                                 environment.yml, trn\ (decodes the light from the IO board)
        analysis\                the analysis notebooks, reading only NWB files
        rig\                     Arduino_Sync_Stim, Doric_Stim, Sensory_Stimuli, Synch_stim
        old\                     Opto_Response, analysis (old), getting_started, trnstim, pipeline
    nwb\                         <session>.nwb, <session>_build\, <session>_unit_quality\, logs\
    motion\                      face motion per video
    derived\                     tables the analysis notebooks write
    Results\                     figures
    Data\                        what is there now, untouched until the new files replace it
```

- Raw data and sorting jobs stay on Z: and are never written to. Files are built on C:.
- Nothing in `Data` is deleted by me. Once the new files exist and the notebooks run on them, the
  parts that can be regenerated (`nwb_v2`, `lfp`, `cache`, `screens`: about 190 GB) could go; that
  is Toon's call, as is the local copy of the E32 session 1 recording (170 GB).
- The repository has untracked folders and its `main` has diverged from GitHub. Nothing is
  committed or pushed without being asked.

## 10. Order of work

1. **Inventory:** one figure per session with every IO-board channel, and a table of what each
   session has. Reads about 1 to 3 GB per session from Z:.
2. **Timing:** gate against light command against TTL, per session.
3. **This spec agreed**, column by column.
4. **Build one session** (E32, 4 Aug 2026) end to end, open it in the walkthrough notebook, fix.
   Then a second session with the unchanged script. Then the batch.
5. **Unit quality** with the artefact measures; Phy.
6. **Port the analysis notebooks** (00 to 08e) to the new files, one at a time. The units table
   changes shape (one table per shank, `depth_um` instead of `depth_from_tip_um`), so every
   notebook goes through one loader.

## 11. Subject details still needed

The build stops on an empty field. Not found anywhere in this project: sex and date of birth of
E29, E30, E31, E32 (and E16); strain; the light source and wavelength; where the periscope sits.
