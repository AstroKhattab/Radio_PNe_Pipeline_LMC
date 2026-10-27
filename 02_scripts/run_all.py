"""Run the whole pipeline in order.

    python run_all.py            every step
    python run_all.py 4 5 6      only these steps, in this order

Each step can also be run on its own, e.g. `python step4_spectral_index.py`,
as long as the steps before it have been run once.  Step 2 has no script: its
measurements are the file 01_data/manual_fluxes.csv, read by Step 1.
"""

import os
import subprocess
import sys

STEPS = {
    "0": "step0_data.py",
    "1": "step1_crossmatch.py",
    "3": "step3_flux_gate.py",
    "4": "step4_spectral_index.py",
    "5": "step5_cohen_ratio.py",
    "6": "step6_score.py",
    "7": "step7_pnlf.py",
}

here = os.path.dirname(os.path.abspath(__file__))
chosen = sys.argv[1:] or list(STEPS)
for key in chosen:
    if key not in STEPS:
        raise SystemExit(f"unknown step {key}; choose from {', '.join(STEPS)}")
    result = subprocess.run([sys.executable, os.path.join(here, STEPS[key])], cwd=here)
    if result.returncode:
        raise SystemExit(f"{STEPS[key]} failed")
print("done: 03_figures, 04_catalogues, 05_tables, 07_notes")
