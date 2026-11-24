# Data format principles (any Neuroline + NWB dataset)

Version 1.0 (2026-11-23). This file holds the choices that do not depend on the experiment.
A dataset gets its own short spec that fills in the blanks listed in section 9; for the CPO
ephys data that is `DATA_FORMAT_SPEC_v1.md`.

## 1. Goal

One NWB file per session that holds everything an analysis needs, built by a script, so the
same analysis code runs over any selection of sessions.

## 2. Three layers

| Layer | What | Rule |
|---|---|---|
| Raw | acquisition folders, videos | never changed |
| Processed | Neuroline output (sorting, metrics, LFP, face motion, pupil) | left exactly as Neuroline wrote it; can be regenerated |
| NWB | one file per session | built from the two layers above by a script; never edited by hand |

When something is wrong in an NWB file, fix the source (labels, decoding, metadata) and
rebuild. That only works if a rebuild is quick and needs no manual steps.

## 3. Who writes what

Neuroline's own functions are used wherever they work. Own code only fills what Neuroline
cannot do for the dataset, and adds to the same file.

| Part | Written by | Function |
|---|---|---|
| File, session and subject details | own script, with the same content as Neuroline's `init_nwb()` | then `session.set_nwb(path)` |
| Probe and electrodes | Neuroline | `ephys_stream.to_nwb()` |
| Units, quality metrics, waveforms | Neuroline | `fork_add_sorting_analyser_to_nwb()` (see below) |
| LFP | Neuroline | `lfp_recording.processed.to_nwb()` |
| Trials, blocks, analog behaviour signals, anything with several videos per session | own script, PyNWB | |

Two reasons for the detours in this table:

- `init_nwb()` always writes the file inside the session folder. Creating the file ourselves
  with the same content lets it live in the project folder, and lets us set `session_id`.
- Neuroline's `processed.to_nwb()` for units loads the preprocessed traces, which are deleted
  after sorting. The function underneath it accepts the raw recording instead (opened
  without loading it), which is only used for the clock of the spike times. This is also the
  route for re-exporting after manual curation.

Use Neuroline's event export (`add_events_to_nwb`) only when the stimulus codes are on the
digital TTL lines and alignment to the TTL is good enough. Otherwise write the trials table
yourself.

## 4. What a session is

- **Session:** one probe position in one animal on one day. One NWB file.
- **Block:** one stimulus protocol within a session. A row in the blocks table, with start
  and stop time. Analyses select trials by block, never by file.
- Acquisition folders are not a level. When a session spans several folders they are sorted
  together so a unit keeps one ID; the blocks table records which folder a block came from.

## 5. Rules for the content

1. **One clock.** Every time in the file is in seconds on the synchronised clock of the
   acquisition system, starting at `session_start_time`.
2. **Stimulus times are measured** where a copy of the stimulus is recorded (speaker copy,
   photodiode). The trigger time is kept as a second column.
3. **Nothing is dropped at build time.** All units and all trials go in; quality is a set of
   columns and selection happens in the analysis.
4. **Derived signals in, raw side channels out.** Signals used as data (running, face
   motion, pupil) are stored. Channels that are only used to build tables (sync pulses,
   stimulus codes, camera pulses) stay in the raw data. NWB allows storing them too
   (`acquisition`); leaving them out is only about file size.
5. **No path is written in analysis code.** Paths follow from the project table.

### Trials table

One row per stimulus, one table for all protocols. Three groups of columns:

| Group | Examples |
|---|---|
| Timing and identity | `start_time` (measured onset), `stop_time`, `ttl_time`, `onset_measured`, `block`, `protocol`, `trial_in_block`, stimulus parameters |
| State | the hidden state of the task at this trial, a switch marker, a counter since the last switch, the previous state |
| Recent history | the previous stimulus, how often the current one was just repeated, how long since it last occurred |

Store the few history columns every analysis uses. Longer histories are computed by a
function from the identity and state columns.

### Units

- Unit ID: `<session>_s<shank>_<cluster number>`, unique across the project.
- One label column per source (sorter, manual curation, automatic classifier). Quality
  threshold sets are named in one file and applied by the loader, not stored.
- Region: `target_region` per session; `depth` per unit; `anatomical_location` empty until
  histology.

### Behaviour signals

- Constant rate (treadmill): rate and start time. With gaps (video-based): one series with a
  timestamp per sample, from the camera pulses.
- Anything computed per video that cannot be compared across videos (SVD components) is
  stored per block; anything that can (motion energy, pupil area) as one series.

## 6. NWB rules followed

From the NWB best-practice guide; `nwbinspector <file>` checks them.

- Times in seconds from `session_start_time`, never negative, increasing.
- Time is the first dimension of every array; units are SI.
- Trials: `stop_time` after `start_time`; extra time columns end in `_time`; no column that
  is empty for the whole table.
- Spike times increasing and above zero.
- Non-neural signals are plain `TimeSeries` in `processing/behavior`; LFP in
  `processing/ecephys`.
- Subject: `subject_id`, `species` as Latin name, `sex` as one letter, `age` or
  `date_of_birth`.
- `session_id` is the session ID of section 9; `identifier` is a new random ID per build.

Known differences, on purpose:

- Neuroline writes one units table per probe stream (`units_<probe>` in
  `processing/ecephys`) instead of the single standard `units` table. The loader joins them.
- True/false columns are stored as 0/1 when pynapple would otherwise drop them (**check this
  against the pynapple version in use**).

## 7. Around the files

- **Project table** (`sessions.csv`): one row per session. Hand-written columns (subject,
  date, target region, source folders, exclusions) and columns a script fills in by looking
  at the disk (sorted, metrics, LFP, behaviour, curated, NWB built, NWB checked).
- **Blocks table** (`blocks.csv`): one row per block, with its protocol, source folder and
  video files.
- **Loader:** one function that opens a session by ID and returns units, trials and signals.
  Analyses never open files themselves.
- **Curation:** on a local copy of the sorter output; labels are copied to a dated folder;
  the NWB is rebuilt.

## 8. Build and check

Build order per session: create file, electrodes, units, LFP, trials and blocks, behaviour
signals, notes (versions, parameters, source folders, spec version).

Checks after every build:

- `nwbinspector` reports no errors
- the file opens in pynapple
- unit count and spike count per unit match the sorter output
- trial count per block matches the protocol; share of measured onsets above 99 %
- every video's frame count matches its pulse run
- every behaviour signal covers every block

## 9. To fill in for a new dataset

- Session ID format, subject IDs, target regions
- Protocols, and how a block is recognised
- Acquisition channels: what is on which line (stimulus copy, codes, cameras, running, sync)
- Stimulus columns, the task state, and which history columns to store
- Which parts Neuroline can write for this data, checked function by function
- Videos per session and how frames are timed
- Quality threshold sets
- Where raw, processed and project folders live
