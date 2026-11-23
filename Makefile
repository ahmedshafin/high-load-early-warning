PY ?= python

.PHONY: install all audit clean-data features validate eda onset models event-level test clean help

help:
	@echo "make install      - pip install requirements"
	@echo "make all          - audit -> clean -> features -> validate -> eda -> onset models -> event-level"
	@echo "make test         - run unit tests"
	@echo "make clean        - remove generated data (keeps reports)"

install:
	$(PY) -m pip install -r requirements.txt pytest

all: audit clean-data features validate eda onset models event-level

audit:
	$(PY) src/00_dataset_audit.py

clean-data:
	$(PY) src/01_clean.py

features:
	$(PY) src/02_build_features.py

validate:
	$(PY) src/04_validate.py
	$(PY) src/07_validate_onset_targets.py

eda:
	$(PY) src/03_eda.py

onset:
	$(PY) src/08_rule_based_onset_baseline.py

models:
	$(PY) src/09_logistic_regression_onset.py
	$(PY) src/10_random_forest_onset.py

event-level: models
	$(PY) src/11_event_level_evaluation.py

test:
	$(PY) -m pytest -q

clean:
	rm -f data/clean/*.parquet data/model_ready/*.parquet data/model_ready/*.csv.gz