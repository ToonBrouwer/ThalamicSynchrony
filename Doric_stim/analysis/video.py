"""Video -> 1D pixel-intensity trace.

Single-pass pipeline:
  * ffmpeg (bundled by imageio_ffmpeg) decodes + downscales + converts to gray
    in one subprocess, writing raw bytes to stdout.
  * We stream frames into a uint8 RAM buffer at spatial_down resolution.
  * Compute per-pixel std in chunks, threshold -> mask (low-res).
  * Extract masked-mean trace from the same buffer.
  * Upsample mask/mean/std to full res for overview display.

The result (mask, trace, mean, std, meta) is cached to npz keyed on the
source mtime, so subsequent runs skip video decoding entirely.
"""
from __future__ import annotations

import subprocess
import sys
import time
from hashlib import md5
from pathlib import Path

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

from io_data import open_video, video_meta


def _cache_path(video_path: Path, cache_dir: Path) -> Path:
    key = md5(str(video_path.resolve()).encode()).hexdigest()[:10]
    return cache_dir / f"videocache_{video_path.stem}_{key}.npz"


def _decode_stream(video_path: Path, W: int, H: int, n_expected: int):
    """Yield (i, frame_uint8[H,W]) streamed from ffmpeg.

    Uses -vf scale for high-speed native downscaling. One decode pass.
    """
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-nostdin",
        "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-vf", f"scale={W}:{H}:flags=area",
        "-loglevel", "error",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=W * H * 8
    )
    try:
        fbytes = W * H
        i = 0
        while True:
            buf = proc.stdout.read(fbytes)
            if len(buf) < fbytes:
                break
            yield i, np.frombuffer(buf, dtype=np.uint8).reshape(H, W)
            i += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()


def _choose_downscale(width: int, height: int,
                      n_frames_est: int, ram_budget_gb: float) -> int:
    """Pick an integer spatial_down so that n_frames * (W/d) * (H/d) <= ram_budget."""
    budget_bytes = int(ram_budget_gb * (1024 ** 3))
    for d in (1, 2, 3, 4, 6, 8, 10, 12, 16):
        if max(1, n_frames_est) * (width // d) * (height // d) <= budget_bytes:
            return d
    return 16


def process_video(video_path: Path, cache_dir: Path | None = None,
                  spatial_down: int | None = None, percentile: float = 99.0,
                  ram_budget_gb: float = 2.0, force: bool = False,
                  verbose: bool = True):
    """Return (mask_full[H,W], trace[n_frames], info dict)."""
    video_path = Path(video_path)
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = _cache_path(video_path, cache_dir)
        if not force and cache_file.exists() and \
                cache_file.stat().st_mtime >= video_path.stat().st_mtime:
            d = np.load(cache_file, allow_pickle=False)
            info = {
                "meta": dict(n_frames=int(d["n_frames"]),
                             fps=float(d["fps"]),
                             width=int(d["width"]),
                             height=int(d["height"])),
                "mean": d["mean_img"],
                "std": d["std_img"],
                "thr": float(d["thr"]),
                "cached": True,
                "spatial_down": int(d["spatial_down"]),
            }
            return d["mask"].astype(bool), d["trace"].astype(np.float32), info

    # ── probe metadata via opencv (cheap) ────────────────────────────────
    cap = open_video(video_path)
    meta = video_meta(cap)
    cap.release()
    n_est = meta["n_frames"] if meta["n_frames"] > 0 else 10000

    if spatial_down is None:
        spatial_down = _choose_downscale(meta["width"], meta["height"],
                                         n_est, ram_budget_gb)
    W = max(8, meta["width"] // spatial_down)
    H = max(8, meta["height"] // spatial_down)
    if verbose:
        est_mb = n_est * W * H / (1024 ** 2)
        print(f"  decode @ {W}x{H} (down={spatial_down}), "
              f"est {n_est} frames, ~{est_mb:.0f} MB")

    # ── single-pass decode into RAM ──────────────────────────────────────
    t0 = time.time()
    frames: list[np.ndarray] = []
    last_print = t0
    for i, fr in _decode_stream(video_path, W, H, n_est):
        frames.append(fr.copy())  # buffer is reused by the stream iterator
        if verbose and time.time() - last_print > 2.0:
            n = i + 1
            dt = time.time() - t0
            print(f"    decoded {n} frames  ({n/dt:.0f} fps)  {dt:.1f}s")
            last_print = time.time()
    dt = time.time() - t0
    if verbose:
        print(f"  decoded {len(frames)} frames in {dt:.1f}s "
              f"({len(frames)/max(dt,1e-3):.0f} fps)")
    if not frames:
        raise RuntimeError("ffmpeg produced no frames")
    stack = np.stack(frames, axis=0)  # (n, H, W) uint8
    del frames
    n = stack.shape[0]
    # update meta in case ffmpeg saw a different count than header
    meta["n_frames"] = n

    # ── std in chunks (avoid 4x RAM spike) ──────────────────────────────
    sum_img = np.zeros((H, W), dtype=np.float64)
    sq_img = np.zeros((H, W), dtype=np.float64)
    for s in range(0, n, 1000):
        chunk = stack[s:s + 1000].astype(np.float32)
        sum_img += chunk.sum(axis=0, dtype=np.float64)
        sq_img += (chunk * chunk).sum(axis=0, dtype=np.float64)
    mean_small = sum_img / n
    var_small = np.clip(sq_img / n - mean_small * mean_small, 0, None)
    std_small = np.sqrt(var_small).astype(np.float32)
    thr = float(np.percentile(std_small, percentile))
    mask_small = (std_small >= thr)
    n_mask = int(mask_small.sum())
    if n_mask == 0:
        raise RuntimeError("empty mask (no pixels above std threshold)")

    # ── trace via (n, hw) @ mask (vectorised, no per-frame Python loop) ─
    mflat = mask_small.reshape(-1)
    # (n, h*w) uint8 -> pick mask cols -> mean. Converting chunk-wise to float32.
    trace = np.empty(n, dtype=np.float32)
    for s in range(0, n, 2000):
        e = min(s + 2000, n)
        v = stack[s:e].reshape(e - s, -1)[:, mflat]
        trace[s:e] = v.mean(axis=1, dtype=np.float64).astype(np.float32)

    # ── upsample small products to full frame for overview ──────────────
    fullW, fullH = meta["width"], meta["height"]
    mask_full = cv2.resize(mask_small.astype(np.uint8), (fullW, fullH),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    mean_full = cv2.resize(mean_small.astype(np.float32), (fullW, fullH),
                           interpolation=cv2.INTER_LINEAR)
    std_full = cv2.resize(std_small, (fullW, fullH),
                          interpolation=cv2.INTER_LINEAR)

    info = {
        "meta": meta,
        "mean": mean_full,
        "std": std_full,
        "thr": thr,
        "cached": False,
        "spatial_down": int(spatial_down),
    }

    if cache_file is not None:
        np.savez_compressed(
            cache_file,
            mask=mask_full,
            trace=trace,
            mean_img=mean_full,
            std_img=std_full,
            thr=np.float32(thr),
            n_frames=np.int64(meta["n_frames"]),
            fps=np.float64(meta["fps"]),
            width=np.int64(meta["width"]),
            height=np.int64(meta["height"]),
            spatial_down=np.int64(spatial_down),
        )
        if verbose:
            print(f"  cached -> {cache_file.name}")

    return mask_full, trace, info
