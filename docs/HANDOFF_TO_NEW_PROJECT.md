# Handoff: the same organisation for another project (TRN stimulation)

Written 24 Nov 2026 from the CPO ephys project. It is for whoever sets up the TRN stimulation
project, person or assistant, and has not seen the CPO work. It says what we built, what to copy
as it is, what to rewrite, in which order, and what went wrong on the way so it does not go wrong
again.

Everything named here is in
`C:\Users\brouw\Documents\PhD\Projects\CPO_ephys\Code\CPO_Ephys_SharedCode\CodeToon`
(git: `https://github.com/ToonBrouwer/CPO_Ephys.git`).

## 1. The idea in five lines

- **One NWB file per session** holds everything needed for analysis: units with all metrics, LFP,
  a trials table with one row per stimulus, the blocks, treadmill, face motion. Nobody needs the
  raw data or the sorting output to analyse.
- **One clock.** Everything in the file is in seconds on Neuroline's synchronised clock.
- **Stimulus times are measured, not commanded.** In CPO: the onset of the sound on the analog
  copy of the speaker signal, not the TTL.
- **The file does not judge.** All units are in it with their metrics. Which units are "good" is
  decided afterwards by rules in a notebook, saved as a table next to the file.
- **One script, read top to bottom, builds the file.** One block at the top holds everything about
  the session. No hidden configuration.

## 2. Read these first, in this order

| File | Why |
|---|---|
| `DATA_FORMAT_PRINCIPLES.md` | The project-independent version of the format. Section 9 lists what to fill in for a new dataset. Start here. |
| `DATA_FORMAT_SPEC_v1.md` | The CPO format in full. The template for the TRN spec. |
| `build_nwb.py` | The build, 14 numbered parts. |
| `HOW_TO_BUILD_NWB.md` | How a person runs it. |
| `nwb_walkthrough.ipynb`, `nwb_examples.ipynb` | What the file looks like and how to analyse it with pynapple. |
| `unit_quality.ipynb` | Quality rules, `unit_labels.csv`, Phy from the NWB file. |

## 3. What to copy and what to rewrite

`build_nwb.py` part by part:

| Part | What it does | For the TRN project |
|---|---|---|
| THIS SESSION block | session, subject, folders, usable shanks | keep the shape, new fields where needed (see section 5) |
| MANY SESSIONS + `sessions.csv` + `build_all.py` | batch over sessions | copy as is, new table |
| 1, 2 | checks that the block is filled in and every input exists | copy, adjust the list of inputs |
| 3, 4, 5 | open the recording with Neuroline, new file, probe and electrodes | copy as is, if the probe and Neuroline setup are the same |
| 6 | units per shank + metrics + `shank`, `depth_um`, `x_um` | copy as is |
| 7 | one pass over the IO board, cached | copy the method; **the channel list is CPO's** |
| 8 | decode the sounds (`cpo.data.build_events`) | **rewrite**: this is the stimulus |
| 9 | blocks = runs of face-camera pulses, matched to videos | copy if the cameras are recorded the same way |
| 10 | trials table incl. CPO state columns | **rewrite**: one row per stimulus, new columns |
| 11, 12 | face motion and treadmill into the file | copy if recorded the same way |
| 13 | LFP written in pieces | copy as is |
| 14 | checks, open with pynapple, rename to the final name | copy, new expectations |

Also copy as is: `unit_quality.ipynb`, `environment.yml`, `.gitignore`, the README split (general
one at the top of the repo, a specific one per person's folder).

Rewrite: the spec, the walkthrough and example notebooks (keep their structure: anatomy of the
file, then one question per example), and the `cpo` package. Only `cpo/data.py` (decoding the
events) and `cpo/config.py` are used by the build; the new project needs its own decoder.

## 4. Order of work that worked

1. Inventory: which sessions exist, what is on every IO-board channel, which videos. Do not trust
   memory; plot each channel.
2. Check the stimulus timing: measured onset against the TTL, per protocol.
3. Agree on the format in words first (the spec), column by column. Then build.
4. Build one session end to end. Open it in the walkthrough notebook. Fix. Only then a second
   session, unchanged script. Only then the batch.
5. Unit quality and Phy last.

## 5. To decide for the TRN project before writing code

These are open; I do not know this project. Fill them in with Toon.

- **What is a trial?** One row per what: a sound, a light pulse, a pulse train, a sound with or
  without stimulation? In CPO it is one row per sound.
- **How is the stimulation recorded?** Is there a copy of the laser/LED command or a photodiode on
  an analog channel? That is the measured onset. A TTL alone is the commanded time.
- **Stimulation columns:** power or current, pulse width, frequency, train duration, number of
  pulses, site, and a plain `stim` yes/no so trials can be compared with their controls.
- **Stimulation artefacts.** New compared with CPO, and it touches three things: the sorting
  (artefacts become "units"), the quality metrics, and the LFP. Decide whether artefact windows
  are blanked before sorting, and store those windows in the file as intervals either way.
- **Subject:** opsin, virus, injection site and date, fibre position. NWB has places for these
  (`Subject.genotype`, `virus`, `surgery`, `OptogeneticStimulusSite`). Use them.
- **Target region(s)** and which shanks are usable per session.
- **The IO-board channel map.** CPO: ADC0 treadmill, ADC1 speaker copy, ADC2 pupil camera,
  ADC3/5/7/8/10 code lines, ADC6 1 Hz sync, ADC11 face camera. Check every one again.
- **Is a session one Open Ephys folder?** Several folders per session is not supported.

## 6. Things that cost us time (do not repeat)

Neuroline:
- Use Neuroline's own functions wherever they work; write our own only for a checked reason.
- `fork_add_sorting_analyser_to_nwb` collects the quality and template metrics and never writes
  them. Part 6 adds them as columns.
- Neuroline's LFP export loads a whole shank (about 19 GB). Part 13 writes it in pieces instead.
- `si.load_sorting_analyzer(folder, load_extensions=False)`: loading the extensions from the NAS
  fails on the big waveform file.
- `waveform_mean` is in µV although the file labels it "volts".
- The electrode `location` is the placeholder "brain area". The target region is in the session
  description and build notes, not per electrode.
- In the cluster script, `"n_jobs"` must be the number `8`, not `"8"`. With the text version the
  sorting finishes but the analysis fails on every shank, so there are no metrics and no units to
  build from.
- Thalamus needed dead channels removed before sorting (GPU memory). Then the analyser has fewer
  channels than the 384 electrodes. **The four thalamus sessions have not been built yet**, so
  the build is only proven on cortex.

The NAS (Z:, SSHFS over Wi-Fi):
- Build on a local disk. Writing a large HDF5 file straight to Z: froze the mount.
- The mount drops during the day. `build_all.py` stops a build that shows no progress for 25
  minutes and goes on with the next session.
- Never test or "repair" the connection from a script or an assistant session: those calls hang.
- Explorer refuses to copy big files to Z: (it reports 7.8 GB total). Unsolved.

The file:
- About 42 to 51 GB and 20 to 30 minutes per cortex session, most of it LFP (30 GB) and per-spike
  locations (8 to 16 GB).
- Read LFP by time first, then pick channels: `lfp.get(start, end)`. Channel by channel is slow.
- Close every reader (pynapple too) before renaming or replacing the file on Windows.
- The build writes `<session>_building.nwb` and only gives it its final name when all checks pass.

Phy:
- `unit_quality.ipynb` makes the Phy folder from the NWB file. That gives mean waveforms,
  positions, correlograms and the rules as columns; no single spikes, spike amplitudes or
  features. If the new project needs real curation in Phy, store spike amplitudes and a sample of
  waveforms in the file from the start (about 3 GB per shank), or curate in the job folder.

## 7. How Toon wants the code and the work

- Simple, clear, changeable. One linear script with numbered parts; little hidden in functions;
  everything for one session in one block at the top; documentation in the file.
- The essential information is forced: the build stops when a field in the top block is empty.
- No configuration spread over yaml and csv. `sessions.csv` exists only for the batch.
- Notebooks show how to get each plot from the NWB file in a few visible lines.
- Diagnose and propose first; ask before editing code, copying large files or starting long runs.
- The raw data and sorts on Z: are never written to.
- Nothing is committed or pushed without being asked.

## 8. State of the CPO project on 24 Nov 2026

- Built: E34 Day3, E33 Day4, E37 Day2, E36 Day1 (all A1), in `CPO_ephys\nwb`.
- Not built: the four thalamus sessions (jobs 448 and 449 have no metrics because of `n_jobs`;
  451 and 452 failed at the input check, cause not yet looked at).
- Open: subject details for E33, E36, E37; the dB level outside the frequency sweep; which
  quality rule is the standard; a fresh install from `environment.yml` has not been tested;
  pupil is extracted for some sessions but not in the file.
