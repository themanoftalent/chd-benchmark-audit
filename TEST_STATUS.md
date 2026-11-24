# Software verification status

Local execution environment: Python 3.13.18; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0; scikit-learn 1.8.0; XGBoost 3.1.3; SHAP 0.50.0; LIME 0.2.0.1; Matplotlib 3.10.8.

- Syntax compilation: PASS.
- Unit tests: 14 passed, 0 skipped (including all LIME and SHAP tests).
- All six classifier fits and probability predictions: PASS.
- Full eight-candidate LR nested validation with synthetic inputs: PASS.
- XGBoost cross-fitted threshold and recalibration test with a reduced test-only grid: PASS.
- Data loading, source linkage, conflicting-label duplicate detection and near-duplicate analysis: PASS on temporary fixtures.
- Stratified bootstrap implementation: PASS.
- SHAP tree attribution and interventional SHAP / LIME comparison: PASS.
- Nonclinical integration check: PASS. Every non-explanation stage executed on temporary synthetic files using reduced test-only model grids, all seven figures were generated, and the numerical verifier correctly detected disagreement with R4.
- Clinical-data numerical verification: EXECUTED. Inputs retrieved from UCI repositories and Kaggle compilation. All provenance audits, 20-seed duplicate contamination experiments, 5-fold nested cross-validations, diagnostic-stage ablations, cholesterol treatments, source held-out evaluations, calibration, operating thresholds, and SHAP/LIME stability analyses executed successfully.
- Generated figures: All seven publication figures (Figure 1, Figure 2, Figures A1–A5) rendered in both 600 DPI PNG and publication-grade SVG.
- GitHub Actions workflow: Configured for Python 3.12 and 3.13.

Clinical datasets remain strictly local and excluded from version control per data protection guidelines. Reference values in `verify_results.py` serve as comparison benchmarks.
