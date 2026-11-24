# Coronary heart disease benchmark: experimental reproduction pipeline

Companion source code for **Auditing the Construction and Evaluation of a Historical Coronary Heart Disease Benchmark**.

This repository contains Python implementations of the experimental procedures described in the R4 manuscript. It does not distribute the clinical datasets. The three source links and input preparation instructions are in [data/README.md](data/README.md).

## Installation

Use Python 3.12 or 3.13 and a fresh virtual environment. Direct dependency versions are pinned for this implementation; the original experiment environment must be confirmed against its authenticated execution artifacts.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Data sources

| Resource | Rows | Access |
|---|---:|---|
| UCI Heart Disease, four processed cohorts | 920 | https://doi.org/10.24432/C52P4X |
| UCI Statlog Heart | 270 | https://doi.org/10.24432/C57303 |
| Comprehensive Heart Disease compilation | 1,190 | https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final |

The primary benchmark uses the four original UCI cohorts after exact deduplication. Statlog participates only in the provenance audit. Original measurements and missing tokens are preserved until the corresponding processing step. The published 2019 ESC pre-test probability values in `analysis.py` are from Table 5 of https://doi.org/10.1093/eurheartj/ehz425 (age, sex and three symptom categories; no asymptomatic probabilities assigned).

## Execution

1. Obtain the compilation CSV directly from Kaggle. Preserve the original category codes and row order.
2. Retrieve the UCI files and copy the local compilation:

```bash
python fetch_inputs.py --compilation /path/to/downloaded.csv
```

If the six input files already exist in `data/`, execute `python fetch_inputs.py --verify-only`. Supply `--expected-hashes /path/to/trusted.json` to compare against authenticated original checksums. Input hashes are also recorded in the analysis manifest.

3. Run the complete analysis:

```bash
python run.py --data data --output results --jobs 1
```

The output directory must be empty or new. Use a larger `--jobs` value for parallel grid searches if memory permits. `--skip-explanations` executes the non-explanation analyses only and does not satisfy complete reproduction verification.

4. Generate figures and compare the numerical outputs against the R4 manuscript:

```bash
python generate_figures.py --results results --output figures
python verify_results.py --results results
```

`verify_results.py` writes `results/verification.csv` and exits with status 1 if any checked value differs beyond the manuscript's rounding precision. It exits with a nonzero status if required files are absent. Reference values in the verifier are never used as model inputs or to adjust the experimental configuration.

## Experimental implementation

- Predictor-only linkage against all five source files, minimum five observed anchors, candidate-row mapping, outcome concordance, missing-value replacement audit and three-to-six-anchor sensitivity.
- Exact labelled duplicates, predictor-only duplicate and conflicting-label counts, near-duplicate tolerance analysis, and cross-partition predictor-overlap checks.
- Six supervised classifiers, eight candidate configurations each, common AUROC tuning, five outer folds, three inner folds, and training-only preprocessing.
- Twenty matched 80/20 contamination contrasts, with equal additional training counts and class composition.
- Independent LR, RF and XGBoost optimisation for each diagnostic-stage predictor subset.
- Four cholesterol representations on both datasets, source-held-out and within-source XGBoost evaluation, and the four-class source-classification probe.
- Class-stratified bootstrap intervals, sex and cholesterol-availability subgroups, three-classifier calibration, training-only XGBoost logistic recalibration and four operating rules.
- Raw-feature LIME with the entire fitted prediction pipeline, path-dependent SHAP across outer folds, first-fold interventional SHAP, absolute attribution ranks and top-three overlap.
- ESC age/sex/chest-pain comparison using independently tuned logistic regression and eligible out-of-fold assessment records.

## File and output map

| Script | Function |
|---|---|
| `fetch_inputs.py` | Local input retrieval and SHA-256 verification |
| `common.py` | Input validation, feature definitions, model grids, training pipeline and intervals |
| `linkage.py` | Record linkage, duplicate and near-duplicate audits |
| `analysis.py` / `run.py` | Model training, testing and all primary experimental analyses |
| `recalibrate.py` | Logistic probability calibration and recalibration utilities |
| `descriptives.py` | Cohort, missingness, fold and operating-rule tables |
| `generate_figures.py` | Figures 1, 2 and A1–A5 in PNG and SVG |
| `verify_results.py` | Numerical consistency checks against R4 |
| `config/experiments.json` | Explicit experimental settings and reconstruction choices |
| `tests/test_pipeline.py` | Software tests using temporary synthetic fixtures |

Main outputs: `audit.json`, `row_linkage.csv`, `fills.csv`, `anchor_sensitivity.csv`, `near_duplicates.csv`, `primary.json`, `ablation.csv`, `sources.csv`, `cholesterol.csv`, `subgroups.csv`, `calibration.json`, `thresholds.csv`, `threshold_folds.csv`, `esc.json`, `duplicates.csv`, `shap_comparison.csv`, `shap_stability.csv`, `lime_agreement.csv`, `manifest.json`, and appendix CSV/JSON tables.

The manifest records input hashes, software versions and experimental settings. Per-fold predictions and selected hyperparameters are saved locally. The threshold output includes per-fold thresholds, confusion counts and pooled operating characteristics. Aggregate appendix tables are generated from local input data and predictions, rather than copied from the manuscript. Table 1 is defined by the model grids; Tables A1 and A11 describe the feature schema and execution sequence. Appendix A documents earlier exploratory composite experiments and is outside the primary reproduction workflow.

## Numerical verification and provenance

This implementation is reconstructed from R4's methodological description, not an authenticated copy of the original source code. Some execution details, including estimator seeds, control-sampling implementation, bootstrap seed, and original dependency versions, require confirmation from the original execution artifacts. The configuration records explicit reconstruction choices. The primary split follows the manuscript's stratified design; predictor-identical cross-partition records are counted separately so any conflict can be investigated before interpreting performance. Matching hashes, fold assignments, and numerical outputs are necessary before claiming exact reproduction.

Software tests use synthetic fixtures solely to check execution. They do not constitute clinical experiments. See `TEST_STATUS.md` for the actual test environment and execution status. Full clinical-data verification requires the original input files. Neither the expected R4 metrics nor synthetic fixtures substitute for those inputs.

## Data protection

Raw data, local predictions, row-level linkage outputs, and generated figures/results are ignored by Git. Do not force-add third-party clinical datasets. After numerical verification, aggregate tables, environment manifests and checksums may be published separately as appropriate. Dataset licensing remains with the original providers.
