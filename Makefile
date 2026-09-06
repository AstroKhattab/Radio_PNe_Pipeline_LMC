# Entry points for the LMC radio PN pipeline.
#
#   make all        run the whole chain, then the checks and the summary
#   make fast       the same, without the per-source inspection cutouts
#   make step-4a    run one step (any key from `make list`)
#   make fig-pnlf   regenerate one figure
#   make check      run the bookkeeping checks against the current outputs
#   make summary    rewrite results/summary.md from the current outputs
#   make list       show the steps
#
# PYTHON can be overridden: make all PYTHON=/path/to/python3

PYTHON ?= python3
RUN     = $(PYTHON) scripts/run_pipeline.py

.PHONY: all fast list check summary clean-outputs

all:
	$(RUN)
	$(PYTHON) tests/check_bookkeeping.py

fast:
	$(RUN) --fast
	$(PYTHON) tests/check_bookkeeping.py

list:
	$(RUN) --list

check:
	$(PYTHON) tests/check_bookkeeping.py

summary:
	$(PYTHON) scripts/make_summary.py

# One target per step.  `make step-4a` is `run_pipeline.py --only 4a`.
STEPS = 1 2a 2b 3a 3b 3b2 3c 3d 4a 4b 4c 5d 5c 6a 6b 6c 6d 6e 6g 6h 7 fig sum
$(addprefix step-,$(STEPS)):
	$(RUN) --only $(patsubst step-%,%,$@)

# One target per paper figure, naming the figure rather than the step that
# happens to draw it.  Each depends on the tables already being present.
.PHONY: fig-pipeline fig-offsets fig-skymap fig-detections fig-snr fig-alpha \
        fig-mir fig-ceiling fig-evidence fig-pnlf fig-ciardullo fig-grid \
        fig-grid-high fig-mask

fig-pipeline:   ; $(RUN) --only fig
fig-offsets:    ; $(RUN) --only 2a && $(RUN) --only 2b
fig-skymap fig-detections fig-snr: ; $(RUN) --only 3c
fig-alpha:      ; $(RUN) --only 4a
fig-mir:        ; $(RUN) --only 4b
fig-ceiling:    ; $(RUN) --only 4c
fig-evidence:   ; $(RUN) --only 5c
fig-pnlf fig-ciardullo: ; $(RUN) --only 6b && $(RUN) --only 6c
fig-grid:       ; $(RUN) --only 6d
fig-grid-high:  ; $(RUN) --only 6h
fig-mask:       ; $(RUN) --only 6e

# Removes the derived tables so the next run starts from the raw catalogues.
# It does not touch 01_Data.
clean-outputs:
	@$(PYTHON) -c "import sys, shutil, os; sys.path.insert(0,'scripts'); \
from _support import config; d = config.load().out(); \
print('removing', d); shutil.rmtree(d, ignore_errors=True)"
