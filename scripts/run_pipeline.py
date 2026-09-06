"""
run_pipeline.py
===============
Runs the analysis end to end, in order, and stops at the first failure.

Six steps.  Each reads only the outputs of the steps before it, so running this
against an empty outputs directory reproduces every number and every figure in
the paper from the catalogues and images named in config.yaml.

Tables go to the outputs directory, paper figures to figures/, machine-written
LaTeX tables to tables/, the numbers report to results/summary.md, and the
per-source inspection pages to the inspect directory.

    Step 1   parent catalogue      HASH V/163 LMC PNe, plus the R14 columns
    Step 2   radio cross-match     MeerKAT 1.295 GHz and ASKAP-EMU 888 MHz
    Step 3   detections            visual recovery and the detection catalogue
    Step 4   the three criteria    4a spectral index, 4b MIR/radio, 4c ceiling
    Step 5   classification        the criteria score, and the comparison of
                                   the radio verdict against the optical class
    Step 6   luminosity function   the radio PNLF
    Step 7   final catalogue       the published table

Usage
-----
    python3 run_pipeline.py              run everything
    python3 run_pipeline.py --list       show the steps and exit
    python3 run_pipeline.py --from 4a    restart part way through
    python3 run_pipeline.py --only 6a    run one step
    python3 run_pipeline.py --fast       skip the slow inspection cutouts

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("1",   "step1_parent_catalogue.py",        "parent catalogue from HASH V/163"),
    ("2a",  "step2a_crossmatch_meerkat.py",     "cross-match with MeerKAT 1.295 GHz"),
    ("2b",  "step2b_crossmatch_askap.py",       "cross-match with ASKAP-EMU 888 MHz"),
    ("3a",  "step3a_inspect_meerkat.py",        "inspection queue for the unmatched"),
    ("3b",  "step3b_extract_visual.py",         "photometry of the visual detections"),
    ("3b2", "step3b2_reextract_visual.py",      "wide-aperture check on that photometry"),
    ("3c",  "step3c_detection_catalogue.py",    "detection catalogue and sky maps"),
    ("3d",  "step3d_multisurvey_union.py",      "multi-survey union"),
    ("4a",  "step4a_spectral_index.py",         "criterion 4a: spectral index"),
    ("4b",  "step4b_mir_radio_ratio.py",        "criterion 4b: MIR/radio ratio"),
    ("4c",  "step4c_flux_ceiling.py",           "criterion 4c: flux ceiling"),
    ("5d",  "step5d_criteria_tally.py",         "the criteria score, and radio against optical"),
    ("5c",  "step5c_detection_and_evidence.py", "detection and evidence summary figure"),
    ("6a",  "step6a_fit_pnlf_empirical.py",     "PNLF: ten models"),
    ("6b",  "step6b_fit_pnlf_ciardullo.py",     "PNLF: Ciardullo function"),
    ("6c",  "step6c_pnlf_paper_figures.py",     "PNLF: paper figures"),
    ("6d",  "step6d_pnlf_empirical_grid.py",    "PNLF: model grid"),
    ("6e",  "step6e_pnlf_mask_vs_clean.py",     "PNLF: masking comparison"),
    ("6g",  "step6a_fit_pnlf_empirical.py",     "PNLF: ten models, high-confidence sample"),
    ("6h",  "step6d_pnlf_empirical_grid.py",    "PNLF: model grid, high-confidence sample"),
    ("7",   "step7_final_catalogue.py",         "the final catalogue"),
    ("fig", "fig0_pipeline_diagram.py",         "the pipeline diagram, Figure 1"),
    ("sum", "make_summary.py",                  "results/summary.md"),
]

# Steps that re-run an earlier script with different arguments.
EXTRA_ARGS = {"6g": ["--sample", "high"], "6h": ["--sample", "high"]}

# What --fast drops: the per-source cutout pages, which take longer than the
# rest of the pipeline together and are quality control rather than results.
FAST_ARGS = {"3a": ["--no-cutouts"]}


def main():
    keys = [k for k, _, _ in STEPS]

    if "--list" in sys.argv:
        for key, script, description in STEPS:
            print(f"  {key:<4} {script:<34} {description}")
        return

    fast = "--fast" in sys.argv
    only = None
    start_at = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    if "--from" in sys.argv:
        start_at = sys.argv[sys.argv.index("--from") + 1]
    for requested in (only, start_at):
        if requested is not None and requested not in keys:
            sys.exit(f"unknown step '{requested}'; run --list to see them")

    running = start_at is None
    t0 = time.time()
    for key, script, description in STEPS:
        if only is not None and key != only:
            continue
        if only is None and not running:
            if key != start_at:
                continue
            running = True
        print(f"\n{'#' * 70}")
        print(f"#  Step {key}  -  {description}")
        print(f"#  {script}")
        print(f"{'#' * 70}", flush=True)
        t = time.time()
        cmd = [sys.executable, "-u", os.path.join(HERE, script)]
        cmd += EXTRA_ARGS.get(key, [])
        if fast:
            cmd += FAST_ARGS.get(key, [])
        result = subprocess.run(cmd, cwd=HERE)
        if result.returncode != 0:
            print(f"\nStep {key} failed ({script}); stopping.")
            sys.exit(result.returncode)
        print(f"[Step {key} finished in {time.time() - t:.0f} s]", flush=True)

    print(f"\n{'#' * 70}")
    print(f"#  Pipeline complete in {time.time() - t0:.0f} s")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
