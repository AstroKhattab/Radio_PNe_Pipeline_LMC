#!/usr/bin/env python3
"""Run the PN LMC analysis workflow in its numbered scientific order.

The runner uses the current Python interpreter, stops immediately if a step
fails, checks the principal product of every step, and records a readable log
under 00_Context/Logs.  VOTable products are checked against the row count or
the column the next step depends on, so a step that runs to completion but
writes the wrong table is caught here rather than three steps later.

Set PNBASE to point at the data tree (the directory holding 01_Data,
03_Outputs and 04_Figures).  It defaults to the parent of this scripts
directory.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


BASE = Path(os.environ.get(
    "PNBASE", Path(__file__).resolve().parent.parent))
SCRIPTS = Path(__file__).resolve().parent
INSPECT = SCRIPTS / "inspect" if (SCRIPTS / "inspect").is_dir() else SCRIPTS / "Inspect"
LOGS = BASE / "00_Context" / "Logs"
OUTS = BASE / "03_Outputs"
FIGS = BASE / "04_Figures"

# label, script, principal product, is_inspection, expected rows, required column
STEPS = [
    ("00a", SCRIPTS / "step00a_make_methodology_workflow.py",
     FIGS / "step00a_methodology_workflow.pdf", False, None, None),
    ("00b1", SCRIPTS / "step00b_methodology_workflow_horizontal.py",
     FIGS / "step00b_methodology_workflow_horizontal.pdf", False, None, None),
    ("00b2", SCRIPTS / "step00b_methodology_workflow_2row.py",
     FIGS / "step00b_methodology_workflow_2row.pdf", False, None, None),
    ("01a", SCRIPTS / "step01a_build_reid.py",
     OUTS / "step01a_reid_master.vot", False, 679, "RP_ID"),
    ("02a", SCRIPTS / "step02a_crossmatch_meerkat.py",
     OUTS / "step02a_meerkat_matched.vot", False, 188, "RP_ID"),
    ("02b", SCRIPTS / "step02b_crossmatch_askap.py",
     OUTS / "step02b_askap_matched.vot", False, 61, "RP_ID"),
    ("03a", INSPECT / "step03a_inspect_meerkat.py",
     OUTS / "step03a_meerkat_to_check.vot", True, 468, "RP_ID"),
    ("03b", SCRIPTS / "step03b_extract_meerkat_visual.py",
     OUTS / "step03b_meerkat_visual.vot", False, 32, "RP_ID"),
    ("03b2", SCRIPTS / "step03b2_reextract_visual_photometry.py",
     OUTS / "step03b2_meerkat_visual_reextracted.vot", False, 32, "RP_ID"),
    ("03c", INSPECT / "step03c_inspect_askap.py",
     OUTS / "step03c_askap_inspection_queue.vot", True, None, "RP_ID"),
    ("04a", SCRIPTS / "step04a_build_meerkat_catalogue_and_multisurvey_figures.py",
     OUTS / "step04a_meerkat_detected.vot", False, 220, "RP_ID"),
    ("04b", SCRIPTS / "step04b_build_multisurvey_catalogue.py",
     OUTS / "step04b_multisurvey_master.vot", False, 679, "RP_ID"),
    ("04c", SCRIPTS / "step04c_crossmatch_hash.py",
     OUTS / "step04c_hash_status_all.vot", False, 225, "hash_status_label"),
    ("05a", SCRIPTS / "step05a_fit_spectral_index.py",
     OUTS / "step05a_spectral_index.vot", False, 225, "alpha_combined"),
    ("05b", INSPECT / "step05b_inspect_spectral_consistency.py",
     FIGS / "Inspect" / "step05b_spectral_consistency.pdf", True, None, None),
    ("06a", SCRIPTS / "step06a_mir_radio_ratio.py",
     OUTS / "step06a_mir_radio_ratio.vot", False, 225, "RP_ID"),
    ("06b", SCRIPTS / "step06b_classify_confidence.py",
     OUTS / "step06b_confidence.vot", False, 225, "our_classification"),
    # 05c needs the confidence columns, so it runs after 06b despite its number.
    ("05c", SCRIPTS / "step05c_recut_spectral_classes.py",
     OUTS / "step05c_spectral_index_recut.vot", False, 225, "sp_class_recut"),
    ("07a", SCRIPTS / "step07a_fit_pnlf_empirical.py",
     OUTS / "step07a_pnlf.vot", False, 225, "M_radio"),
    ("07b", SCRIPTS / "step07b_fit_pnlf_ciardullo.py",
     FIGS / "step07b_pnlf_ciardullo_fits.pdf", False, None, None),
    ("07c", SCRIPTS / "step07c_make_pnlf_paper_figures.py",
     FIGS / "step07c_pnlf_model_overlay.pdf", False, None, None),
    # Two scripts are numbered 07d.  The runner labels the grid figure 07d-grid
    # and the inspection notebook 07d-inspect; neither file is renamed, because
    # the paper's figure references use the current names.
    ("07d-grid", SCRIPTS / "step07d_pnlf_empirical_grid.py",
     FIGS / "step07d_pnlf_empirical_grid_10up.pdf", False, None, None),
    ("07d-inspect", INSPECT / "step07d_inspect_pnlf_consistency.py",
     FIGS / "Inspect" / "step07d_pnlf_consistency.pdf", True, None, None),
    ("07e", SCRIPTS / "step07e_fit_radio_pnlf_selection_aware.py",
     OUTS / "step07e_radio_pnlf_selection_catalogue.vot", False, 679, "log_radio_luminosity_W_Hz"),
    ("07f", SCRIPTS / "step07f_pnlf_mask_vs_clean.py",
     FIGS / "step07f_pnlf_mask_vs_clean.pdf", False, None, None),
    ("07g", SCRIPTS / "step07g_pnlf_grids_by_sample.py",
     FIGS / "step07g_pnlf_grid_A4_S1_ALL_published.pdf", False, None, None),
    ("08a", SCRIPTS / "step08a_detection_and_evidence.py",
     FIGS / "step08a_detection_and_evidence.pdf", False, None, None),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-step", choices=[row[0] for row in STEPS], default=STEPS[0][0])
    parser.add_argument("--to-step", choices=[row[0] for row in STEPS], default=STEPS[-1][0])
    parser.add_argument("--skip-inspection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def check_product(path, rows, column):
    """Return an error string if the product is missing or not what we expect."""
    if not path.exists() or path.stat().st_size == 0:
        return f"expected product is missing or empty: {path}"
    if path.suffix != ".vot" or (rows is None and column is None):
        return None
    try:
        from astropy.table import Table
        table = Table.read(path, format="votable")
    except Exception as error:
        return f"could not read {path.name} as a VOTable: {error}"
    if rows is not None and len(table) != rows:
        return f"{path.name} has {len(table)} rows, expected {rows}"
    if column is not None and column not in table.colnames:
        return f"{path.name} is missing the column {column}"
    return None


def main():
    args = parse_args()
    labels = [row[0] for row in STEPS]
    first = labels.index(args.from_step)
    last = labels.index(args.to_step)
    if first > last:
        raise SystemExit("--from-step must occur before --to-step")

    selected = STEPS[first : last + 1]
    if args.skip_inspection:
        selected = [row for row in selected if not row[3]]

    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = LOGS / f"pipeline_{stamp}.log"
    latest = LOGS / "pipeline_latest.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SCRIPTS), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env.setdefault("MPLBACKEND", "Agg")
    env["PNBASE"] = str(BASE)

    # Both log files are kept open and appended to; rewriting them on every
    # line made the runner quadratic in the length of the log.
    handles = [timestamped.open("w", encoding="utf-8"),
               latest.open("w", encoding="utf-8")]

    def emit(message):
        print(message, flush=True)
        for handle in handles:
            handle.write(message + "\n")
            handle.flush()

    try:
        emit(f"PN LMC pipeline started {datetime.now().isoformat(timespec='seconds')}")
        emit(f"Python: {sys.executable}")
        emit(f"Base:   {BASE}")
        emit("Steps:  " + ", ".join(row[0] for row in selected))

        for label, script, expected, _inspection, rows, column in selected:
            emit(f"\n===== STEP {label}: {script.name} =====")
            command = [sys.executable, str(script)]
            if args.dry_run:
                emit("DRY RUN: " + " ".join(command))
                continue

            process = subprocess.Popen(
                command,
                cwd=BASE,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert process.stdout is not None
            for output_line in process.stdout:
                emit(output_line.rstrip())
            return_code = process.wait()
            if return_code != 0:
                emit(f"FAILED: step {label} returned code {return_code}")
                return return_code

            problem = check_product(expected, rows, column)
            if problem:
                emit(f"FAILED: {problem}")
                return 2
            detail = f" ({rows} rows)" if rows is not None else ""
            emit(f"VERIFIED: {expected.relative_to(BASE)}{detail}")

        emit(f"\nPN LMC pipeline completed {datetime.now().isoformat(timespec='seconds')}")
        emit(f"Log: {timestamped}")
        return 0
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
