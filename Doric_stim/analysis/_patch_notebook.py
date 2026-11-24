"""One-off: patch inspect_stimuli_v3(1).ipynb to extend noise tables."""
import json
from pathlib import Path

p = Path(r'C:\Users\brouw\Documents\ThalamicSynchrony-main\Doric_stim\inspect_stimuli_v3(1).ipynb')
with open(p, encoding='utf-8') as f:
    nb = json.load(f)


def patch_cell(cell, old, new):
    src = cell['source']
    if isinstance(src, list):
        src = ''.join(src)
    assert old in src, f"NOT FOUND:\n{old!r}\n\nin:\n{src[:500]}"
    src2 = src.replace(old, new)
    cell['source'] = src2.splitlines(keepends=True)


# 1. Cell 4 — make NOISE_TBL_LEN cover a full trial + margin
patch_cell(
    nb['cells'][4],
    'NOISE_TBL_LEN  = SAMPLE_RATE',
    'NOISE_TBL_LEN  = int(STIM_DURATION * SAMPLE_RATE * 1.15)  # >= one full trial, no looping',
)

# 2. Cell 8 — fix the auto-comment + emit #define
patch_cell(
    nb['cells'][8],
    "noise_h += f'// {NOISE_TBL_LEN} samples each (1s at {SAMPLE_RATE}Hz), looped\\n'",
    "noise_h += f'// {NOISE_TBL_LEN} samples each ({NOISE_TBL_LEN/SAMPLE_RATE:.2f}s @ {SAMPLE_RATE}Hz), one-shot per trial\\n'",
)
patch_cell(
    nb['cells'][8],
    "noise_h += '#pragma once\\n#include <avr/pgmspace.h>\\n\\n'",
    "noise_h += '#pragma once\\n#include <avr/pgmspace.h>\\n\\n'\n"
    "noise_h += f'#define NOISE_TBL_LEN {NOISE_TBL_LEN}\\n\\n'",
)

with open(p, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("notebook patched OK")
