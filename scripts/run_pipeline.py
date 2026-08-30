"""
run_pipeline.py
===============
Runs the analysis end to end, in order, and stops at the first failure.

Every step reads only the outputs of the steps before it, so running this from
a clean 03_Outputs directory reproduces every number and every figure in the
paper from the raw catalogues and images in 01_Data.

    Step 1   step01b_build_parent_hash      parent sample from HASH V/163
    Step 2   step02c_crossmatch_surveys     MeerKAT and ASKAP cross-match
    Step 3a  step03d_forced_photometry      forced photometry on the misses
    Step 3b  step03e_vet_candidates         vetting those candidates
    Step 4a  step04a_detection_catalogue    one flux scale for everything
    Step 4b  step04b_spectral_index         criterion 1, thermal spectrum
    Step 4c  step04c_mir_radio_ratio        criterion 2, MIR/radio ratio
    Step 4d  step04d_flux_ceiling           criterion 3, flux ceiling
    Step 4e  step04e_classify               the criteria combined
    Step 4f  step04f_compare_optical        radio verdict against optical
    Step 5   step05_pnlf                    the luminosity function

Usage
-----
    python3 run_pipeline.py            run everything
    python3 run_pipeline.py --from 4a  restart part way through

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("1", "step01b_build_parent_hash.py", "parent sample from HASH V/163"),
    ("2", "step02c_crossmatch_surveys.py", "MeerKAT and ASKAP cross-match"),
    ("3a", "step03d_forced_photometry.py", "forced photometry on the misses"),
    ("3b", "step03e_vet_candidates.py", "vetting the candidates"),
    ("4a", "step04a_detection_catalogue.py", "detection catalogue"),
    ("4b", "step04b_spectral_index.py", "criterion 1: thermal spectrum"),
    ("4c", "step04c_mir_radio_ratio.py", "criterion 2: MIR/radio ratio"),
    ("4d", "step04d_flux_ceiling.py", "criterion 3: flux ceiling"),
    ("4e", "step04e_classify.py", "criteria combined"),
    ("4f", "step04f_compare_optical.py", "radio against optical"),
    ("5", "step05_pnlf.py", "luminosity function"),
]


def main():
    start_at = None
    if "--from" in sys.argv:
        start_at = sys.argv[sys.argv.index("--from") + 1]

    running = start_at is None
    t0 = time.time()
    for key, script, description in STEPS:
        if not running:
            if key != start_at:
                continue
            running = True
        print(f"\n{'#' * 70}")
        print(f"#  Step {key}  -  {description}")
        print(f"#  {script}")
        print(f"{'#' * 70}")
        t = time.time()
        result = subprocess.run([sys.executable, os.path.join(HERE, script)],
                                cwd=HERE)
        if result.returncode != 0:
            print(f"\nStep {key} failed ({script}); stopping.")
            sys.exit(result.returncode)
        print(f"[Step {key} finished in {time.time() - t:.0f} s]")

    print(f"\n{'#' * 70}")
    print(f"#  Pipeline complete in {time.time() - t0:.0f} s")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
