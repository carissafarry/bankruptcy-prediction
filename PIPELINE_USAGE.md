# Bankruptcy Prediction Pipeline – Usage Guide

Complete production pipeline for training 4 models (NGBoost, Random Forest, XGBoost, LightGBM) on 5 prediction horizons (1–5 years).

## Quick Start

```bash
cd src
python run_pipeline.py
```

## Full Usage

```bash
python run_pipeline.py \
  --data ./data/processed/financial_report_bank_zscore_clean.csv \
  --output ./output \
  --skip-shap  # (optional) skip SHAP analysis for faster testing
```

## Pipeline Stages

**See CODE_FLOW.md for detailed architecture & execution flows.**

### Stage 1: Model Training
- Trains NGBoost, Random Forest, XGBoost, LightGBM on 5 horizons (1–5 years)
- Time-based splits: 2014–2019 train, 2020–2021 val, 2022–2023 test
- Outputs: predictions CSV + search logs (iterations for XGBoost)

### Stage 2: SHAP Explainability (optional)
- Summary plots, feature importance, dependence plots
- Supported via `--skip-shap` flag in run_pipeline.py

## Modules

**See CODE_FLOW.md § "Model Configuration" for module details.**

- **`training_utils.py`** – Shared config, data loading, all 4 training functions, evaluation/export
- **`train_single_model.py`** – CLI: train single model across all horizons
- **`train_all_models.py`** – CLI: train all 4 models sequentially
- **`shap_analysis.py`** – SHAP explainability (unified interface for all models)
- **`run_pipeline.py`** – Orchestrator: training + optional SHAP analysis

## Configuration

**See CONFIGURATION.md for full reference.**

Key constants in `training_utils.py`:
```python
YEAR_START = 2014, YEAR_END = 2023
TRAIN_END_YEAR = 2019, VAL_END_YEAR = 2021  # Time-based splits
RECALL_TARGET = 0.75  # For threshold tuning (if not hardcoded)
RANDOM_STATE = 42     # Reproducibility seed
HARDCODED_THRESHOLD = 0.4  # Fixed for all models due to class imbalance
```

## Output Format: Predictions CSV

Each `<model>_predictions_<horizon>y.csv` contains:

```
symbol, calendar_year, period, time, tanggal, horizon,
distress_actual, prob_distress, pred_label, distance_to_threshold,
model_name, threshold_used, risk_bucket, confusion_type
```

**Columns:**
- `tanggal` – Quarter-end date (DD/MM/YYYY) for dashboard filtering
- `distress_actual` – True label (0=safe, 1=distressed)
- `prob_distress` – Model probability [0, 1]
- `pred_label` – Binary prediction based on threshold
- `risk_bucket` – Categorical risk: "Low Risk" (0–0.3), "Medium Risk" (0.3–0.6), "High Risk" (0.6–1.0)
- `confusion_type` – TP/TN/FP/FN
- `threshold_used` – Threshold used for pred_label

## Metrics Interpretation

**Type I Error:** FN / (TP + FN) – Missed distress (high regulatory risk)  
**Type II Error:** FP / (TN + FP) – False alarm (cost of unnecessary intervention)  
**Recall:** TP / (TP + FN) – % of actual distress caught  
**ROC-AUC:** Discrimination across all thresholds  
**PR-AUC:** Precision-recall trade-off (focus on distressed class)

## Tips for Publication

1. **Table 1: Model Comparison** – Accuracy, ROC-AUC, PR-AUC across models & horizons
2. **Table 2: Feature Importance** – Top 10 features from SHAP bar plots
3. **Figure 1: SHAP Summary** – Dot plots showing feature impact direction
4. **Figure 2: SHAP Dependence** – Key features (e.g., log_sales, operating_income_ratio)
5. **Figure 3: ROC/PR Curves** – Model discrimination over thresholds
6. **Appendix: Predictions** – Sample predictions CSV with risk classifications

## Troubleshooting

**"Data file not found"**
- Ensure `./data/processed/financial_report_bank_zscore_clean.csv` exists

**Memory error during SHAP**
- Reduce `SHAP_SAMPLE_N` or `SHAP_BACKGROUND_N` in `train_all_models.py`

**Threshold "No threshold meets recall target"**
- Reduce `RECALL_TARGET` or examine class imbalance in that horizon

**SHAP computation slow (NGBoost)**
- Uses KernelExplainer (model-agnostic) – slower than TreeExplainer. Reduce sample size or skip SHAP with `--skip-shap`.

## Citation

For publications, cite the model versions used:
- XGBoost: version in requirements.txt
- LightGBM: version in requirements.txt
- NGBoost: version in requirements.txt
- SHAP: version in requirements.txt
