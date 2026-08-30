"""
run_pipeline.py
===============
Runs the analysis end to end, in order, and stops at the first failure.

Six steps.  Each reads only the outputs of the steps before it, so running this
against an empty 03_Outputs reproduces every number and every figure in the
paper from the catalogues and images in 01_Data.

Tables are written to 03_Outputs, paper figures to 05_Figures, and the
per-source inspection pages to 04_Inspect/figures.  The inspection scripts in
04_Inspect/scripts are quality checks and are not part of the chain; run them
separately when needed.

    Step 1   parent catalogue      HASH V/163 LMC PNe, plus the Reid columns
    Step 2   radio cross-match     MeerKAT 1.295 GHz and ASKAP-EMU 888 MHz
    Step 3   detections            visual recovery and the detection catalogue
    Step 4   the three criteria    4a spectral index, 4b MIR/radio, 4c ceiling
    Step 5   classification        evidence grade, spectral recut, and the
                                   comparison of the radio verdict against
                                   the optical classification
    Step 6   luminosity function   the radio PNLF

Usage
-----
    python3 run_pipeline.py             run everything
    python3 run_pipeline.py --from 4a   restart part way through
    python3 run_pipeline.py --list      show the steps and exit

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

STEPS = [
    ("1",  "step1_parent_catalogue.py",        "parent catalogue from HASH V/163"),
    ("2a", "step2a_crossmatch_meerkat.py",     "cross-match with MeerKAT 1.295 GHz"),
    ("2b", "step2b_crossmatch_askap.py",       "cross-match with ASKAP-EMU 888 MHz"),
    ("3a", "step3a_inspect_meerkat.py",        "inspection queue for the unmatched"),
    ("3b", "step3b_extract_visual.py",         "photometry of the visual detections"),
    ("3b2","step3b2_reextract_visual.py",      "aperture-corrected visual photometry"),
    ("3c", "step3c_detection_catalogue.py",    "detection catalogue and sky maps"),
    ("3d", "step3d_multisurvey_union.py",      "multi-survey union"),
    ("4a", "step4a_spectral_index.py",         "criterion 4a: spectral index"),
    ("4b", "step4b_mir_radio_ratio.py",        "criterion 4b: MIR/radio ratio"),
    ("4c", "step4c_flux_ceiling.py",           "criterion 4c: flux ceiling"),
    ("5a", "step5a_classify_confidence.py",    "combined evidence grade"),
    ("5b", "step5b_spectral_recut.py",         "spectral reliability recut"),
    ("5c", "step5c_detection_and_evidence.py", "detection and evidence summary"),
    ("5d", "step5d_criteria_tally.py",         "criteria tally, and radio against optical"),
    ("6a", "step6a_fit_pnlf_empirical.py",     "PNLF: ten empirical models"),
    ("6b", "step6b_fit_pnlf_ciardullo.py",     "PNLF: Ciardullo function"),
    ("6c", "step6c_pnlf_paper_figures.py",     "PNLF: paper figures"),
    ("6d", "step6d_pnlf_empirical_grid.py",    "PNLF: model grid"),
    ("6e", "step6e_pnlf_mask_vs_clean.py",     "PNLF: masking comparison"),
    ("6g", "step6a_fit_pnlf_empirical.py",     "PNLF: ten models, high-confidence sample"),
    ("6h", "step6d_pnlf_empirical_grid.py",    "PNLF: model grid, high-confidence sample"),
    ("7",  "step7_final_catalogue.py",      "the final catalogue"),
    ("fig","fig0_pipeline_diagram.py",         "the pipeline diagram figure"),
]

# A few of the older scripts resolve their paths relative to the working
# directory rather than to their own location, so they are run from the
# project root instead of from 02_Scripts.
# Steps that re-run an earlier script with different arguments.
EXTRA_ARGS = {"6g": ["--sample", "high"],
              "6h": ["--sample", "high"]}

RUN_FROM_BASE = {"step5b_spectral_recut.py",
                 "step5c_detection_and_evidence.py",
                 "step5d_criteria_tally.py",
                 "step6e_pnlf_mask_vs_clean.py",
                 "fig0_pipeline_diagram.py"}


def main():
    if "--list" in sys.argv:
        for key, script, description in STEPS:
            print(f"  {key:<4} {script:<34} {description}")
        return

    start_at = None
    if "--from" in sys.argv:
        start_at = sys.argv[sys.argv.index("--from") + 1]
        if start_at not in {k for k, _, _ in STEPS}:
            sys.exit(f"unknown step '{start_at}'; run --list to see them")

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
        print(f"{'#' * 70}", flush=True)
        t = time.time()
        cwd = BASE if script in RUN_FROM_BASE else HERE
        target = script if cwd == HERE else os.path.join(HERE, script)
        cmd = [sys.executable, "-u", target] + EXTRA_ARGS.get(key, [])
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            print(f"\nStep {key} failed ({script}); stopping.")
            sys.exit(result.returncode)
        print(f"[Step {key} finished in {time.time() - t:.0f} s]", flush=True)

    print(f"\n{'#' * 70}")
    print(f"#  Pipeline complete in {time.time() - t0:.0f} s")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
