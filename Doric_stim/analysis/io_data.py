"""Minimal OpenEphys + video loaders for the Doric opto-stim session."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def find_recording(session_root: Path) -> Path:
    """Return the first .../recordingN directory inside a session."""
    hits = list(session_root.glob("Record Node */experiment*/recording*"))
    if not hits:
        raise FileNotFoundError(f"no recording under {session_root}")
    hits.sort()
    return hits[0]


def load_oebin(rec_dir: Path) -> dict:
    with open(rec_dir / "structure.oebin") as f:
        return json.load(f)


def load_continuous(rec_dir: Path):
    """Memory-map the ADC continuous.dat. Returns (data[n, nch], fs, bit_volts)."""
    oebin = load_oebin(rec_dir)
    cont = oebin["continuous"][0]
    nch = cont["num_channels"]
    fs = float(cont["sample_rate"])
    bv = float(cont["channels"][0]["bit_volts"])
    dat_path = rec_dir / "continuous" / cont["folder_name"] / "continuous.dat"
    raw = np.memmap(dat_path, dtype="int16", mode="r")
    n = raw.size // nch
    data = raw[: n * nch].reshape(n, nch)
    return data, fs, bv


def open_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    return cap


def video_meta(cap) -> dict:
    return dict(
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
