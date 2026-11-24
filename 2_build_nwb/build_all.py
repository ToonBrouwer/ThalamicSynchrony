r"""
BUILD THE NWB FILE OF EVERY SESSION IN sessions.csv, ONE AFTER ANOTHER
======================================================================

    conda activate neuroline311
    python build_all.py                          every session in sessions.csv that has no file yet
    python build_all.py E29_session2 E30_session2    only these

For each session it runs   python build_nwb.py <session>   as its own program, so one session
that fails does not stop the others. Everything a build prints goes to a log file.

- A session that already has its .nwb file is skipped. Delete the file to build it again.
- A build that gets stuck is stopped: if it prints nothing and its file does not grow for
  STUCK_AFTER_MINUTES (the network drive hanging, usually), it is ended and the next one starts.
- A failed or stopped build leaves no half file behind as the real one: build_nwb.py only gives
  the file its final name when every check has passed.

Afterwards, look at   <NWB_FOLDER>\build_all_log.csv   (one line per session) and, for a session
that failed, at   <NWB_FOLDER>\logs\<session>.log   (the last lines say why).
"""
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---- settings ----
NWB_FOLDER = Path(r"C:\Users\brouw\Documents\PhD\Projects\Thalamic_Synchrony\TRN_Stimulation\nwb")      # the same as in build_nwb.py
STUCK_AFTER_MINUTES = 25          # no output and no growth of the file for this long = stuck
NEEDED_FREE_GB = 60               # do not start a session with less free disk space than this
# ------------------

HERE = Path(__file__).parent
sessions = pd.read_csv(HERE.parent / "sessions.csv", dtype=str).fillna("?")["session"].tolist()
wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or sessions
logs = NWB_FOLDER / "logs"
logs.mkdir(parents=True, exist_ok=True)
summary = NWB_FOLDER / "build_all_log.csv"
if not summary.exists():
    summary.write_text("time,session,result,minutes,last_line\n")

print(f"{len(wanted)} sessions: {', '.join(wanted)}")
for number, session in enumerate(wanted, start=1):
    final = NWB_FOLDER / f"{session}.nwb"
    building = NWB_FOLDER / f"{session}_building.nwb"
    log = logs / f"{session}.log"
    print(f"\n=== {number}/{len(wanted)}  {session}  ({datetime.now():%a %H:%M}) ===", flush=True)
    if final.exists():
        print("already built, skipped")
        continue
    import shutil
    free_gb = shutil.disk_usage(NWB_FOLDER).free / 1e9
    if free_gb < NEEDED_FREE_GB:
        print(f"only {free_gb:.0f} GB free on {NWB_FOLDER.anchor}: stopping here.")
        break

    started = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as out:
        build = subprocess.Popen([sys.executable, "-u", str(HERE / "build_nwb.py"), session],
                                 stdout=out, stderr=subprocess.STDOUT, cwd=HERE)
        last_sign, last_size, result = time.time(), (0, 0), None
        while build.poll() is None:
            time.sleep(20)
            size = (log.stat().st_size, building.stat().st_size if building.exists() else 0)
            if size != last_size:                                  # it printed something or the file grew
                last_sign, last_size = time.time(), size
            elif time.time() - last_sign > STUCK_AFTER_MINUTES * 60:
                build.kill()
                result = f"stopped: nothing happened for {STUCK_AFTER_MINUTES} minutes"
        build.wait()
    if result is None:
        result = "built" if build.returncode == 0 and final.exists() else "failed"
    lines = [line.strip() for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    last_line = lines[-1][:200] if lines else ""
    minutes = (time.time() - started) / 60
    with open(summary, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), session, result, f"{minutes:.0f}", last_line])
    print(f"--> {result} after {minutes:.0f} min.  {last_line}", flush=True)

print(f"\nDone. Summary: {summary}")
