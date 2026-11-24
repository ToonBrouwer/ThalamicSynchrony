"""
Preflight — fires each unique stimulus once, in a fixed order, with a long
fixed ISI so you can scope-check the output and verify the OneBox receives
the sync TTL correctly.

Usage:
    python preflight.py

Prints what it's about to do and waits for you to confirm.
"""
from __future__ import annotations

import sys
import time

from config import DORIC_UID, DLL_PATH, CHANNEL, REPS, MAX_CURRENT_MA
from doric_client import DoricLightSource
from stimuli import build_stim
from session import _upload


PREFLIGHT_ISI_S = 5.0  # fixed ISI — long enough to see each stim clearly on scope


def main() -> int:
    # Order: all custom first (so scope can be set up for 0..MAX_CURRENT_MA),
    # then all complex. Only stims with REPS > 0 are included.
    names = [n for n, r in REPS.items() if r > 0]
    if not names:
        print("No stimuli enabled in REPS.")
        return 1

    print("Preflight plan:")
    print(f"  channel      : {CHANNEL}")
    print(f"  max current  : {MAX_CURRENT_MA} mA")
    print(f"  ISI          : {PREFLIGHT_ISI_S} s")
    print(f"  stimuli      : {names}")
    print()
    input("Press ENTER when scope + OneBox are armed… ")

    with DoricLightSource(dll_path=DLL_PATH, uid=DORIC_UID, channel=CHANNEL) as doric:
        for i, name in enumerate(names, start=1):
            spec = build_stim(name)
            print(f"[{i}/{len(names)}] {name:<12}  dur={spec['duration_s']:.1f}s  "
                  f"kind={spec['kind']}", end='  ', flush=True)
            _upload(doric, spec)
            doric.start()
            t0 = time.perf_counter()
            time.sleep(spec['duration_s'])
            doric.stop()
            print(f"(fired at perf_counter={t0:.3f})")
            time.sleep(PREFLIGHT_ISI_S)

    print("Preflight done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
