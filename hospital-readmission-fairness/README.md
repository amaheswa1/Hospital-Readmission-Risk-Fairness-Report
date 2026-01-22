# Hospital Readmission Risk + Fairness Report

End-to-end ML project that:
- Loads a hospital dataset (CSV)
- Cleans + validates data
- Trains a classifier to predict 30-day readmission
- Evaluates performance (ROC, precision/recall)
- Evaluates fairness by race and gender
- Exports a markdown report

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
make run
```

## Commands
- `make validate` — data checks
- `make train` — train model
- `make report` — generate report
- `make test` — run tests

## Dataset
Sample synthetic dataset included in:
`data/raw/readmissions.csv`

Columns:
- age, gender, race
- num_procedures, num_medications, time_in_hospital
- has_diabetes (0/1)
- readmitted_30 (label 0/1)
