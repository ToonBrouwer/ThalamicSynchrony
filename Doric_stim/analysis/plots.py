"""Per-trial + per-stim-type characterisation plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from expected import expected_peak_freq_hz, expected_waveform


# ── measurement helpers ─────────────────────────────────────────────────────
def _measure_sine_peak(x: np.ndarray, fs: float, f_target: float,
                       bw_mult: float = 2.0) -> float | None:
    """Peak freq near f_target using zero-padded Welch (fine resolution)."""
    if x.size < 32:
        return None
    nperseg = min(x.size, int(fs * 4))
    nfft = max(nperseg, 1 << int(np.ceil(np.log2(nperseg * 4))))
    f, P = signal.welch(x - x.mean(), fs=fs, nperseg=nperseg, nfft=nfft)
    lo = f_target / bw_mult
    hi = f_target * bw_mult
    m = (f >= lo) & (f <= hi) & (f > 0)
    if not m.any():
        return None
    return float(f[m][np.argmax(P[m])])


def _measure_chirp_endpoints(x: np.ndarray, fs: float,
                             stim_dur_s: float, win_s: float = 0.4):
    """Estimate dominant freq in a short window at stim-start and stim-end."""
    n_win = int(win_s * fs)
    if x.size < 2 * n_win:
        return None, None

    def peak(seg):
        nperseg = min(seg.size, int(fs * win_s))
        nfft = 1 << int(np.ceil(np.log2(nperseg * 8)))
        f, P = signal.welch(seg - seg.mean(), fs=fs, nperseg=nperseg, nfft=nfft)
        if f.size < 2:
            return None
        return float(f[1:][np.argmax(P[1:])])

    return peak(x[:n_win]), peak(x[-n_win:])


def _tfr(x: np.ndarray, fs: float, fmax: float):
    nperseg = int(min(len(x), max(256, fs)))
    nperseg = max(32, min(nperseg, max(32, len(x) // 4)))
    noverlap = int(nperseg * 0.9)
    f, t, Sxx = signal.spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="density"
    )
    keep = f <= fmax
    return f[keep], t, Sxx[keep]


def _psd(x: np.ndarray, fs: float, fmax: float):
    nperseg = int(min(len(x), max(256, fs)))
    f, P = signal.welch(x, fs=fs, nperseg=nperseg)
    keep = f <= fmax
    return f[keep], P[keep]


def _norm_unit(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _psd_window(stim_name: str, stim_dur_s: float,
                edge_s: float = 0.5) -> tuple[float, float]:
    """Time window used for PSD. Noise skips the onset/offset ramps & edges."""
    if stim_name.startswith("Noise"):
        return (edge_s, stim_dur_s - edge_s)
    return (0.0, stim_dur_s)


# ── per-trial figure ────────────────────────────────────────────────────────
def plot_trial(t_doric: np.ndarray, doric_trace: np.ndarray, doric_fs: float,
               t_cam: np.ndarray, cam_trace: np.ndarray, cam_fs: float,
               stim_name: str, stim_dur_s: float,
               title: str, out_path: Path,
               doric_fmax: float = 300.0) -> dict:
    """Return a dict of measured values for bookkeeping."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="col")
    fig.suptitle(title, fontsize=12)

    expected_band = expected_peak_freq_hz(stim_name)
    tex, yex, exmeta = expected_waveform(stim_name, dur_s=stim_dur_s)

    psd_win = _psd_window(stim_name, stim_dur_s)
    m_in_d = (t_doric >= psd_win[0]) & (t_doric <= psd_win[1])
    m_in_c = (t_cam >= psd_win[0]) & (t_cam <= psd_win[1])

    # shared PSD freq range (x-axis linked between rows via sharex='col')
    psd_x_hi = doric_fmax

    measurements: dict = {"stim": stim_name}

    # ═══════════════ row 0: Doric DAC ═══════════════
    # time course
    ax = axes[0, 0]
    ax.plot(t_doric, doric_trace, lw=0.5, label="measured")
    if m_in_d.sum() > 10:
        lo, hi = float(np.min(doric_trace[m_in_d])), float(np.max(doric_trace[m_in_d]))
        ax.plot(tex, lo + yex * (hi - lo), lw=0.5, color="C3", alpha=0.6,
                label="expected")
    ax.axvspan(0, stim_dur_s, color="orange", alpha=0.08)
    ax.axvspan(*psd_win, color="yellow", alpha=0.04)  # PSD window highlight
    ax.set_title("Doric DAC — time course")
    ax.set_ylabel("V")
    ax.legend(loc="upper right", fontsize=8)

    # spectrogram
    ax = axes[0, 1]
    f, tt, Sxx = _tfr(doric_trace - doric_trace.mean(), doric_fs, doric_fmax)
    im = ax.pcolormesh(
        tt + t_doric[0], f, 10 * np.log10(Sxx + 1e-20),
        shading="auto", cmap="magma",
    )
    ax.set_title("Doric — spectrogram (dB)")
    ax.set_ylabel("frequency (Hz)")
    ax.axvline(0, color="w", lw=0.5, ls="--")
    ax.axvline(stim_dur_s, color="w", lw=0.5, ls="--")
    # camera Nyquist line (helps see aliasing limits)
    ax.axhline(cam_fs / 2.0, color="cyan", lw=0.6, ls=":", alpha=0.7)
    if stim_name.startswith("Chirp") and "inst_f" in exmeta:
        ax.plot(tex, exmeta["inst_f"], color="cyan", lw=1.0, alpha=0.8,
                label="expected f(t)")
        ax.set_yscale("log")
        ax.set_ylim(0.3, doric_fmax)
        ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, pad=0.01)

    # PSD
    ax = axes[0, 2]
    if m_in_d.sum() > 64:
        fd, Pd = _psd(doric_trace[m_in_d] - doric_trace[m_in_d].mean(),
                      doric_fs, doric_fmax)
        ax.semilogy(fd, Pd, label="measured", color="C0")
    if expected_band is not None:
        lo, hi = expected_band
        ax.axvspan(lo, hi, color="C3", alpha=0.08, label="expected band")
    # peak annotation
    if stim_name.startswith("Sine"):
        f_target = {"Sine10": 10.0, "Sine25": 25.0, "Sine60": 60.0}[stim_name]
        pk = _measure_sine_peak(doric_trace[m_in_d] - doric_trace[m_in_d].mean(),
                                doric_fs, f_target)
        if pk is not None:
            ax.axvline(pk, color="k", ls="--", lw=0.6)
            ax.axvline(f_target, color="C3", ls="--", lw=0.6)
            ax.set_title(f"Doric PSD — measured {pk:.2f} Hz (want {f_target:.0f})")
            measurements["doric_peak_hz"] = pk
            measurements["target_hz"] = f_target
        else:
            ax.set_title("Doric — PSD (stim window)")
    elif stim_name.startswith("Chirp"):
        fstart, fend = _measure_chirp_endpoints(
            doric_trace[m_in_d] - doric_trace[m_in_d].mean(),
            doric_fs, stim_dur_s)
        if fstart is not None and fend is not None:
            ax.set_title(f"Doric PSD — chirp start≈{fstart:.2f} Hz, "
                         f"end≈{fend:.2f} Hz (want 0.5→200)")
            measurements["chirp_fstart_hz"] = fstart
            measurements["chirp_fend_hz"] = fend
        else:
            ax.set_title("Doric — PSD (stim window)")
    else:
        ax.set_title(f"Doric — PSD ({psd_win[0]:.1f}–{psd_win[1]:.1f}s)")
    ax.set_ylabel("V²/Hz")
    ax.set_xscale("log")
    ax.set_xlim(0.1, psd_x_hi)
    ax.legend(loc="lower left", fontsize=8)

    # ═══════════════ row 1: Camera ═══════════════
    cam_fmax = cam_fs / 2.0
    ax = axes[1, 0]
    ax.plot(t_cam, cam_trace, lw=0.7, color="C2", label="measured")
    if m_in_c.sum() > 10:
        lo, hi = float(np.min(cam_trace[m_in_c])), float(np.max(cam_trace[m_in_c]))
        ax.plot(tex, lo + yex * (hi - lo), lw=0.5, color="C3", alpha=0.5,
                label="expected")
    ax.axvspan(0, stim_dur_s, color="orange", alpha=0.08)
    ax.axvspan(*psd_win, color="yellow", alpha=0.04)
    ax.set_title("Camera (masked pixels) — time course")
    ax.set_xlabel("time (s, 0 = stim onset)")
    ax.set_ylabel("mean intensity (a.u.)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    if cam_trace.size > 32:
        f, tt, Sxx = _tfr(cam_trace - cam_trace.mean(), cam_fs, cam_fmax)
        if Sxx.size:
            im = ax.pcolormesh(
                tt + t_cam[0], f, 10 * np.log10(Sxx + 1e-20),
                shading="auto", cmap="magma",
            )
            ax.axvline(0, color="w", lw=0.5, ls="--")
            ax.axvline(stim_dur_s, color="w", lw=0.5, ls="--")
            if stim_name.startswith("Chirp") and "inst_f" in exmeta:
                ax.plot(tex, np.clip(exmeta["inst_f"], 0, cam_fmax),
                        color="cyan", lw=1.0, alpha=0.8)
                ax.set_yscale("log")
                ax.set_ylim(0.3, cam_fmax)
            plt.colorbar(im, ax=ax, pad=0.01)
    ax.set_title("Camera — spectrogram (dB)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")

    ax = axes[1, 2]
    if m_in_c.sum() > 32:
        fc, Pc = _psd(cam_trace[m_in_c] - cam_trace[m_in_c].mean(),
                      cam_fs, cam_fmax)
        ax.semilogy(fc, Pc, label="measured", color="C2")
    if expected_band is not None:
        lo, hi = expected_band
        hi = min(hi, cam_fmax)
        if hi > lo:
            ax.axvspan(lo, hi, color="C3", alpha=0.08, label="expected band")
    # annotate camera peak for sines (if within Nyquist)
    if stim_name.startswith("Sine"):
        f_target = {"Sine10": 10.0, "Sine25": 25.0, "Sine60": 60.0}[stim_name]
        # camera aliased image: expected appears at |f - round(f/fs)*fs|
        aliased = abs(f_target - round(f_target / cam_fs) * cam_fs)
        pk_cam = _measure_sine_peak(cam_trace[m_in_c] - cam_trace[m_in_c].mean(),
                                    cam_fs, aliased, bw_mult=2.5)
        if pk_cam is not None:
            ax.axvline(pk_cam, color="k", ls="--", lw=0.6)
            ax.set_title(f"Camera PSD — peak {pk_cam:.2f} Hz "
                         f"(cam Nyquist {cam_fmax:.0f})")
            measurements["cam_peak_hz"] = pk_cam
            measurements["cam_aliased_expected_hz"] = float(aliased)
        else:
            ax.set_title("Camera — PSD")
    else:
        ax.set_title(f"Camera — PSD ({psd_win[0]:.1f}–{psd_win[1]:.1f}s)")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("a.u.²/Hz")
    ax.set_xscale("log")
    ax.set_xlim(0.1, psd_x_hi)  # shared with Doric PSD via sharex='col'
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return measurements


def plot_mask_overview(mean_img: np.ndarray, std_img: np.ndarray,
                       mask: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(mean_img, cmap="gray"); ax[0].set_title("mean frame")
    ax[1].imshow(std_img, cmap="magma"); ax[1].set_title("per-pixel std")
    ax[2].imshow(mean_img, cmap="gray")
    ax[2].imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.5)
    ax[2].set_title(f"active mask ({int(mask.sum())} px)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_stim_summary(stim_name: str, epochs: list[dict],
                      stim_dur_s: float, out_path: Path,
                      doric_fmax: float = 300.0):
    if not epochs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    fig.suptitle(f"{stim_name} — {len(epochs)} trial(s) overlay", fontsize=12)

    psd_win = _psd_window(stim_name, stim_dur_s)
    cam_fs = float(epochs[0]["cam_fs"])

    # Doric TC overlay (normalised)
    ax = axes[0, 0]
    for ep in epochs:
        td, yd = ep["t_doric"], ep["doric"]
        ax.plot(td, _norm_unit(yd), lw=0.4, alpha=0.7)
    tex, yex, _ = expected_waveform(stim_name)
    ax.plot(tex, yex, lw=0.8, color="k", alpha=0.7, label="expected")
    ax.axvspan(0, stim_dur_s, color="orange", alpha=0.08)
    ax.axvspan(*psd_win, color="yellow", alpha=0.04)
    ax.set_title("Doric DAC — normalised time course")
    ax.set_ylabel("norm. amp")
    ax.legend(loc="upper right", fontsize=8)

    # Doric PSD overlay
    ax = axes[0, 1]
    for ep in epochs:
        td, yd, fsd = ep["t_doric"], ep["doric"], ep["doric_fs"]
        m = (td >= psd_win[0]) & (td <= psd_win[1])
        if m.sum() < 64:
            continue
        fp, P = _psd(yd[m] - yd[m].mean(), fsd, doric_fmax)
        ax.semilogy(fp, P, lw=0.6, alpha=0.7)
    band = expected_peak_freq_hz(stim_name)
    if band is not None:
        ax.axvspan(*band, color="C3", alpha=0.08, label="expected band")
    ax.set_xscale("log")
    ax.set_xlim(0.1, doric_fmax)
    ax.set_title(f"Doric — PSD ({psd_win[0]:.1f}–{psd_win[1]:.1f}s)")
    ax.set_ylabel("V²/Hz")
    ax.legend(loc="lower left", fontsize=8)

    # Camera TC overlay
    ax = axes[1, 0]
    for ep in epochs:
        tc, yc = ep["t_cam"], ep["cam"]
        if yc.size == 0:
            continue
        ax.plot(tc, _norm_unit(yc), lw=0.5, alpha=0.7, color="C2")
    ax.plot(tex, yex, lw=0.8, color="k", alpha=0.6, label="expected")
    ax.axvspan(0, stim_dur_s, color="orange", alpha=0.08)
    ax.axvspan(*psd_win, color="yellow", alpha=0.04)
    ax.set_title("Camera — normalised time course")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("norm. intensity")
    ax.legend(loc="upper right", fontsize=8)

    # Camera PSD overlay (shares x with Doric PSD)
    ax = axes[1, 1]
    for ep in epochs:
        tc, yc, fsc = ep["t_cam"], ep["cam"], ep["cam_fs"]
        m = (tc >= psd_win[0]) & (tc <= psd_win[1])
        if m.sum() < 32 or yc.size == 0:
            continue
        fp, P = _psd(yc[m] - yc[m].mean(), fsc, fsc / 2.0)
        ax.semilogy(fp, P, lw=0.6, alpha=0.7, color="C2")
    if band is not None:
        lo, hi = band
        hi = min(hi, cam_fs / 2.0)
        if hi > lo:
            ax.axvspan(lo, hi, color="C3", alpha=0.08, label="expected band")
    ax.set_xscale("log")
    ax.set_xlim(0.1, doric_fmax)
    ax.set_title("Camera — PSD")
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel("a.u.²/Hz")
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
