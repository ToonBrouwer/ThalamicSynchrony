"""Main driver: align video <-> OpenEphys, decode trials, write plots.

Usage:
    python run_analysis.py                    # auto-pair all sessions with their videos
    python run_analysis.py --session 2026-04-21_15-50-28
    python run_analysis.py --session <dir> --video <mp4>

Outputs go to analysis_output/<session_stem>/.
Video mask+trace is cached under analysis_output/_cache/ so re-runs are fast.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy import signal as sps

from io_data import find_recording, load_continuous
from plots import plot_mask_overview, plot_stim_summary, plot_trial
from sync import (STIM_NAMES, decode_trials, find_pulses, frame_rising_edges)
from video import process_video

STIM_DURATION_S = 5.0
PRE_S = 1.0
POST_S = 1.5
DORIC_DS = 30  # decimate ADC9 (~30300 -> ~1010 Hz) for plots & PSD


def _find_video_for_session(session_dir: Path, root: Path) -> Path | None:
    """Pair session folder '2026-04-21_15-50-28' with video whose timestamp is closest."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", session_dir.name)
    if not m:
        return None
    date_s = m.group(1).replace("-", "")
    time_s = m.group(2).replace("-", "")
    session_ts = int(date_s + time_s)

    best = None
    best_dt = 10**9
    for mp4 in sorted(root.glob("*.mp4")):
        mm = re.search(r"(\d{8})_(\d{6})", mp4.name)
        if not mm:
            continue
        vid_ts = int(mm.group(1) + mm.group(2))
        dt = abs(vid_ts - session_ts)
        if dt < best_dt:
            best_dt = dt
            best = mp4
    # accept within a few minutes
    return best if best is not None and best_dt < 600 else None


def epoch_doric(adc9: np.ndarray, fs: float, onset_sample: int,
                stim_dur_s: float = STIM_DURATION_S):
    a = onset_sample - int(PRE_S * fs)
    b = onset_sample + int((stim_dur_s + POST_S) * fs)
    a0 = max(a, 0); b0 = min(b, adc9.size)
    seg = adc9[a0:b0].astype(np.float32)
    seg_ds = sps.decimate(seg, DORIC_DS, ftype="iir", zero_phase=True)
    fs_ds = fs / DORIC_DS
    t0 = (a0 - onset_sample) / fs
    t = t0 + np.arange(seg_ds.size) / fs_ds
    return t, seg_ds, fs_ds


def epoch_camera(cam_trace: np.ndarray, frame_times_s: np.ndarray,
                 onset_time_s: float, stim_dur_s: float = STIM_DURATION_S):
    n = min(cam_trace.size, frame_times_s.size)
    t = frame_times_s[:n]
    m = (t >= onset_time_s - PRE_S) & (t <= onset_time_s + stim_dur_s + POST_S)
    return t[m] - onset_time_s, cam_trace[:n][m]


def process_session(session_dir: Path, video_path: Path, out_root: Path,
                    cache_dir: Path):
    print(f"\n=== {session_dir.name} ===")
    print(f"video: {video_path.name}")
    rec = find_recording(session_dir)
    data, fs, bv = load_continuous(rec)
    print(f"OE: {data.shape[0]} samples x {data.shape[1]} ch @ {fs:.1f} Hz  "
          f"({data.shape[0]/fs:.1f} s)")

    adc9 = data[:, 9].astype(np.float32) * bv
    adc10 = data[:, 10].astype(np.float32) * bv
    adc11 = data[:, 11].astype(np.float32) * bv

    # Trials
    pulses = find_pulses(adc10, fs)
    trials = decode_trials(pulses, fs, STIM_DURATION_S)
    print(f"pulses={len(pulses)}  trials={len(trials)}")
    if not trials:
        print("  [skip] no trials")
        return

    # Camera frame samples
    frame_samples = frame_rising_edges(adc11, fs)
    print(f"camera TTL edges: {frame_samples.size}")

    # Video processing (cached)
    mask, cam_trace, info = process_video(video_path, cache_dir=cache_dir)
    meta = info["meta"]
    tag = " (cached)" if info.get("cached") else ""
    print(f"video{tag}: {meta['n_frames']} frames @ {meta['fps']:.2f} fps  "
          f"{meta['width']}x{meta['height']}  mask={int(mask.sum())} px")

    # Reconcile TTL edge count with frame count
    n = min(frame_samples.size, cam_trace.size)
    if frame_samples.size != cam_trace.size:
        print(f"  [warn] {frame_samples.size} TTL edges vs {cam_trace.size} frames "
              f"-> using first {n}")
    cam_trace = cam_trace[:n]
    frame_times_s = frame_samples[:n] / fs

    if frame_times_s.size >= 2:
        cam_fs = 1.0 / float(np.median(np.diff(frame_times_s)))
    else:
        cam_fs = float(meta["fps"])
    print(f"camera fs (from TTLs): {cam_fs:.2f} Hz")

    out_dir = out_root / session_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_mask_overview(info["mean"], info["std"], mask, out_dir / "video_mask.png")

    # Per-trial plots + collect epochs for summaries
    by_stim: dict[str, list[dict]] = {n: [] for n in STIM_NAMES}
    measurements: list[dict] = []
    for tr in trials:
        stim_dur_s = float(tr.stim_duration_s)
        t_d, y_d, fs_d = epoch_doric(adc9, fs, tr.onset_sample, stim_dur_s)
        t_c, y_c = epoch_camera(cam_trace, frame_times_s, tr.onset_time_s,
                                stim_dur_s)

        title = (f"{tr.stim_name}  (trial {tr.index}, "
                 f"onset={tr.onset_time_s:.2f}s, dur={stim_dur_s:.2f}s, "
                 f"typeMs={tr.type_pulse_ms:.0f})")
        fname = f"trial{tr.index:02d}_{tr.stim_name}.png"
        meas = plot_trial(
            t_doric=t_d, doric_trace=y_d, doric_fs=fs_d,
            t_cam=t_c, cam_trace=y_c, cam_fs=cam_fs,
            stim_name=tr.stim_name, stim_dur_s=stim_dur_s,
            title=title, out_path=out_dir / tr.stim_name / fname,
            doric_fmax=300.0,
        )
        meas["stim_duration_s"] = stim_dur_s
        meas["trial"] = tr.index
        meas["onset_s"] = tr.onset_time_s
        measurements.append(meas)
        by_stim[tr.stim_name].append(dict(
            t_doric=t_d, doric=y_d, doric_fs=fs_d,
            t_cam=t_c, cam=y_c, cam_fs=cam_fs,
            trial=tr.index,
        ))

    # Per-stim-type summary overlays
    for name, epochs in by_stim.items():
        if not epochs:
            continue
        plot_stim_summary(
            name, epochs, STIM_DURATION_S,
            out_dir / name / f"_summary_{name}.png",
            doric_fmax=300.0,
        )

    # Trials CSV
    with open(out_dir / "trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "stim_idx", "stim_name", "onset_s", "type_pulse_ms"])
        for tr in trials:
            w.writerow([tr.index, tr.stim_idx, tr.stim_name,
                        f"{tr.onset_time_s:.4f}", f"{tr.type_pulse_ms:.1f}"])

    # Measurements CSV (measured vs expected freq per trial)
    with open(out_dir / "measurements.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "stim", "target_hz",
                    "doric_peak_hz", "doric_ratio",
                    "chirp_fstart_hz", "chirp_fend_hz",
                    "cam_peak_hz", "cam_aliased_expected_hz"])
        for m in measurements:
            tgt = m.get("target_hz")
            dpk = m.get("doric_peak_hz")
            ratio = (dpk / tgt) if (tgt and dpk) else ""
            w.writerow([
                m["trial"], m["stim"],
                f"{tgt:.2f}" if tgt else "",
                f"{dpk:.3f}" if dpk else "",
                f"{ratio:.4f}" if ratio != "" else "",
                f"{m.get('chirp_fstart_hz', ''):.3f}" if m.get("chirp_fstart_hz") else "",
                f"{m.get('chirp_fend_hz', ''):.3f}" if m.get("chirp_fend_hz") else "",
                f"{m.get('cam_peak_hz', ''):.3f}" if m.get("cam_peak_hz") else "",
                f"{m.get('cam_aliased_expected_hz', ''):.3f}" if m.get("cam_aliased_expected_hz") else "",
            ])

    # Print per-stim-type summary of measured frequencies
    print("\nmeasured frequency summary (Doric ADC9):")
    sine_ratios: list[float] = []
    for name in ("Sine10", "Sine25", "Sine60"):
        vals = [m["doric_peak_hz"] for m in measurements
                if m["stim"] == name and "doric_peak_hz" in m]
        tgts = [m["target_hz"] for m in measurements
                if m["stim"] == name and "target_hz" in m]
        if vals:
            import numpy as _np
            r = _np.array(vals) / _np.array(tgts)
            sine_ratios.extend(r.tolist())
            print(f"  {name:>8s}: measured {_np.mean(vals):6.3f} +- {_np.std(vals):.3f} Hz "
                  f"(target {tgts[0]:.0f}), ratio {r.mean():.4f}")
    for name in ("ChirpFwd", "ChirpRev"):
        starts = [m.get("chirp_fstart_hz") for m in measurements if m["stim"] == name]
        ends = [m.get("chirp_fend_hz") for m in measurements if m["stim"] == name]
        starts = [x for x in starts if x is not None]
        ends = [x for x in ends if x is not None]
        if starts:
            import numpy as _np
            print(f"  {name:>8s}: start~{_np.mean(starts):6.2f} Hz, "
                  f"end~{_np.mean(ends):6.2f} Hz")
    if sine_ratios:
        import numpy as _np
        print(f"  -> mean sine ratio (measured/target): {_np.mean(sine_ratios):.4f}")

    counts = {k: len(v) for k, v in by_stim.items()}
    print("per-stim counts:", counts)
    print(f"outputs -> {out_dir}")


def main():
    here = Path(__file__).resolve().parent
    root = here.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=str, default=None,
                    help="session folder name or path (default: all sessions in root)")
    ap.add_argument("--video", type=Path, default=None,
                    help="override video path for the selected session")
    ap.add_argument("--out", type=Path, default=root / "analysis_output")
    ap.add_argument("--force", action="store_true",
                    help="ignore video cache and rebuild")
    args = ap.parse_args()

    cache_dir = args.out / "_cache"

    if args.session is None:
        sessions = sorted(p for p in root.iterdir()
                          if p.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}",
                                                    p.name))
    else:
        p = Path(args.session)
        sessions = [p if p.exists() else (root / args.session)]

    if not sessions:
        print("no sessions found")
        sys.exit(1)

    for s in sessions:
        vid = args.video if (args.video and len(sessions) == 1) \
              else _find_video_for_session(s, root)
        if vid is None or not vid.exists():
            print(f"[skip] {s.name}: no matching video")
            continue
        process_session(s, vid, args.out, cache_dir)


if __name__ == "__main__":
    main()
