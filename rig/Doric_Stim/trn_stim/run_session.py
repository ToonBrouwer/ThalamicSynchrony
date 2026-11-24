"""
Top-level CLI entry point.

Usage:
    python run_session.py                   # run the full session
    python run_session.py --list-devices    # print available UIDs and exit
    python run_session.py --dry-run         # no hardware, exercises scheduler
"""
from __future__ import annotations

import argparse
import sys
import time

from config import DORIC_UID, DLL_PATH, CHANNEL, REPS
from doric_client import DoricLightSource
from session import SessionLogger, build_trial_list, run_session


def list_devices():
    """Initialize DLL, call available_devices_infos, print, exit."""
    print(f"Loading DLL: {DLL_PATH}")
    d = DoricLightSource(dll_path=DLL_PATH, uid=None, channel=1)
    d.init(debug=True)
    print("Devices visible to the Doric DLL:")
    d.list_devices()
    print("\nCopy the UID string you need into config.py → DORIC_UID.")
    d.quit()


def run_dry():
    """Run the session schedule/logger without touching the Doric."""
    trials = build_trial_list()
    n_est_min = len(trials) * 7.0 / 60.0   # rough; stims are 2 or 5 s, ISI 10 s
    print(f"Dry run — {len(trials)} trials, ~{n_est_min:.1f} min if for real.")
    print(f"Per-stim counts: "
          f"{ {k: trials.count(k) for k in sorted(set(trials))} }")
    print("Using 0.2 s fake stimulus duration, 0.3 s ISI between trials.")
    print()
    # fake doric object — run_session only uses it when dry_run_secs_per_trial is None
    class _NoDoric:
        def __getattr__(self, _): return lambda *a, **k: None
    with SessionLogger() as logger:
        run_session(_NoDoric(), logger, dry_run_secs_per_trial=0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--list-devices', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    if args.dry_run:
        run_dry()
        return 0

    # ---- live session ------------------------------------------------------
    if DORIC_UID is None:
        print("ERROR: DORIC_UID is None in config.py.")
        print("Run `python run_session.py --list-devices` first, then paste the "
              "UID of your driver into config.py.")
        return 2

    # Print what's about to happen so user can hit Ctrl-C in time.
    trials = build_trial_list()
    counts = {k: trials.count(k) for k in sorted(set(trials))}
    print(f"About to run a {len(trials)}-trial session on channel {CHANNEL}.")
    print(f"Per-stim counts: {counts}")
    print()
    print("Check:")
    print(f"  - Doric LED driver powered on and USB-connected")
    print(f"  - DORIC_UID = {DORIC_UID!r} matches the unit you want to drive")
    print(f"  - Doric TTL-out → OneBox digital input, OneBox recording armed")
    print(f"  - LED / fibre in place on the animal")
    input("Press ENTER to start, or Ctrl-C to abort… ")

    with DoricLightSource(dll_path=DLL_PATH, uid=DORIC_UID, channel=CHANNEL) as doric:
        with SessionLogger() as logger:
            try:
                run_session(doric, logger)
            except KeyboardInterrupt:
                print("\nInterrupted — stopping channel.")
                doric.stop()
                raise

    return 0


if __name__ == '__main__':
    sys.exit(main())
