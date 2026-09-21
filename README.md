# OECT temperature regression reproduction

Local code candidate. No repository has been created or uploaded. Experimental data, manuscript files, fitted model weights and experimental outputs are deliberately absent from this directory. Publication requires the authors' authorization.

## Scope

`reproduce.py` reconstructs the historical three-zone RBF support-vector regression and global RBF SVR evaluations from locally supplied features or raw curves. `extract_features.py` preserves the relevant extraction functions from the original analysis scripts. This is a portable reconstruction, not an unmodified copy of the historical entry point.

**The three-zone evaluation uses the true test temperature to select a zone. It is an oracle-zone benchmark, not an end-to-end temperature estimator.** No trained feature-based zone classifier was present in the identified original pipeline.

Features are selected by absolute Pearson correlation with temperature on the training set (separately within each zone). Historical SHAP analyses explain the fitted models after selection; they did not perform the top-k selection in this pipeline.

## Run locally

Use Python 3.9 and the tested dependencies listed in `requirements.txt`. Supply the private files separately; do not put them into this repository. The original saved estimators identify scikit-learn 0.24.2; the pinned environment here is the independently tested reproduction environment.

```bash
python reproduce.py --features /path/to/real_environment_valid61_features.csv --feature-names /path/to/selected30_feature_names.csv --out /path/to/local-output --all-variants --reference-predictions /path/to/model_predictions_selected30.csv
```

To re-extract features, replace `--features ...` with `--raw-dir /path/to/raw-curves`. The raw directory must contain paired `Device_<id>-<repeat>_transfer.csv` and `Device_<id>-<repeat>_transient.csv`. Their first columns hold gate voltage and time, respectively; subsequent column headers give temperature. Matching temperature columns form samples. Repeat 1 is the real-environment test set; other repeats are the temperature-stage training set. Retain the same input ordering for exact comparisons.

The included `feature_names.json` is the default 30-feature configuration. An optional 30-feature-name CSV contains one feature name per row with no header. The cached feature table requires `sample_id`, `temperature`, `repeat_type` and those 30 columns. `repeat_type` must be `temperature_stage` or `real_environment`.

Outputs are `predictions.csv`, `metrics.csv`, `parameters.json`, `environment.json`, and `top3_exact_shap.csv`, plus extracted features in raw mode. For the top-3 three-zone models, all eight feature coalitions are enumerated to verify exact empirical-background Shapley values; the historical full pipeline used SHAP KernelExplainer. This reconstruction does not regenerate t-SNE or higher-dimensional SHAP plots. Use an output directory outside this code directory. When reference predictions are supplied, comparison requires identical sample IDs and true temperatures, and reports a maximum prediction difference and a 1e-8 °C tolerance check.

## Model details

- Zones: T ≤ -6 °C; -6 < T ≤ 76 °C; T > 76 °C. Historical reported domain: -26 to 106 °C. The original code does not reject out-of-domain values.
- Pipeline: training median imputation, StandardScaler, RBF SVR.
- Grid: C ∈ {10, 100, 1000}; gamma ∈ {scale, 0.003, 0.01, 0.03, 0.1}; epsilon ∈ {0.25, 0.5, 1, 2} °C.
- Inner KFold: min(5, max(2, n_train // 12)) folds, shuffle=True, random_state=42; objective: negative MAE.
- Feature ranking precedes inner cross-validation, matching the historical implementation. This is not nested feature-selection validation, nor a held-out-device evaluation.
- `--all-variants` evaluates k=1,3,5,8,10,15,20 and all 30, for both model families. Default: top 3 and all 30.
- Each zone uses its own top 3 features. The union need not contain only three distinct features.

The code performs no network communication. No data-access promise or software license is assigned on behalf of the authors. Choose and approve a software license before public release; without one, others may view public code but do not receive a general right to reuse it. Public availability cannot be claimed until an approved repository actually exists.
