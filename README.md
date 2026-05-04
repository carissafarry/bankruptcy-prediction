# Bankruptcy Prediction Pipeline

Production-ready ML pipeline for predicting bank insolvency across 5 horizons (1–5 years ahead) using 4 ensemble models.

**Status**: Active | **Version**: 1.0.0 | **License**: Apache License 2.0

---

## Overview

Predicts bank bankruptcy from 15 financial ratios using:
- **4 Models**: NGBoost, Random Forest, XGBoost, LightGBM
- **5 Horizons**: 1Y, 2Y, 3Y, 4Y, 5Y ahead
- **Reproducibility**: Hardcoded 0.4 threshold, seeded RNG, consolidated hyperparameters
- **Explainability**: SHAP plots + feature importance ranking
- **SDGs Alignment**: SDG 8, 10, 16 (economic growth, financial inclusion, institutional strength)

---

## Quick Start

```bash
# Train all 4 models (full pipeline)
cd src
python run_pipeline.py --skip-shap

# Or train single model
python train_single_model.py --model xgboost --horizons 1,2,3

# Or batch train sequentially
python train_all_models.py
```

Outputs: predictions CSV + search logs CSV + (for XGBoost) consolidated best_params.json

---

## Project Structure

```
src/
├── training_utils.py          # Shared: configs, data loading, all 4 training functions
├── train_single_model.py      # CLI: train single model across all horizons
├── train_all_models.py        # CLI: train all 4 models sequentially
├── shap_analysis.py           # SHAP explainability (unified interface)
└── run_pipeline.py            # Orchestrator: training + optional SHAP

data/
└── processed/
    └── financial_report_bank_zscore_clean.csv

output/
├── models_all_horizons.pkl
├── ngboost/
│   ├── ngboost_predictions_1y.csv
│   ├── ngboost_search_logs.csv
│   └── ngboost_shap_summary_1y.png
├── rf/
├── xgboost/
│   ├── xgboost_predictions_1y.csv
│   ├── xgboost_best_params.json       ← Consolidated params (all 5 horizons)
│   ├── xgboost_search_logs.csv        ← 100 iterations (20 per horizon)
│   └── xgboost_shap_summary_1y.png
└── lgbm/
```

---

## Configuration

All settings centralized in `training_utils.py`:

```python
# Data period
YEAR_START = 2014
YEAR_END = 2023
TRAIN_END_YEAR = 2019     # Train/Val cutoff
VAL_END_YEAR = 2021       # Val/Test cutoff

# Threshold (HARDCODED for stability & reproducibility)
HARDCODED_THRESHOLD = 0.4  # Due to 24.5% class imbalance (prioritizes recall)

# Features (15 financial ratios)
FEATURE_COLS = [
    "size", "der", "dar", "roa", "roe", "sdoa", "sdroe",
    "tobinq", "ppe", "cash", "ar", "log_sales", "sgr",
    "operating_income_ratio", "equity_to_assets"
]
```

See `CONFIGURATION.md` for full reference (hyperparameters, input format, output schema).

---

## Model Specifications

| Model | Hyperparameters | Search Strategy |
|-------|-----------------|-----------------|
| **NGBoost** | 500 estimators, Bernoulli dist, lr=0.03 | Fixed (no search) |
| **Random Forest** | 1000 trees, depth=8, cost-sensitive | Fixed (no search) |
| **XGBoost** | Random search: eta, depth, subsample, etc. | 20 iterations per horizon |
| **LightGBM** | 500 rounds, early stopping (200), gbtd | Fixed (no search) |

All models:
- Use hardcoded threshold = 0.4 (not tuned per horizon)
- Handle class imbalance via `sample_weight` / `scale_pos_weight`
- Time-based splits (prevent temporal leakage)

---

## Data Format

**Input**: `financial_report_bank_zscore_clean.csv`
- 1,812 rows × 27 cols
- Years: 2014–2023
- Quarterly observations per bank (4 per year)
- Columns: `symbol, calendar_year, period, time, bank_zscore_risk, [15 features]`

**Output**: `{model}_predictions_{horizon}y.csv`
- Columns: `symbol, calendar_year, period, time, tanggal, horizon, distress_actual, prob_distress, pred_label, distance_to_threshold, model_name, threshold_used, risk_bucket, confusion_type`
- `risk_bucket`: Low [0–0.3], Medium [0.3–0.6], High [0.6–1.0]
- `confusion_type`: TP/FN/FP/TN for error analysis

**Search Logs**: `{model}_search_logs.csv`
- XGBoost: 100 rows (20 iterations × 5 horizons) with all hyperparameters + PR-AUC
- Others: 5 rows (1 per horizon) with fixed hyperparameters

**Best Params (XGBoost only)**: `xgboost_best_params.json`
- Consolidated best hyperparameters per horizon (not per-iteration)

---

## Key Design Decisions

1. **Hardcoded Threshold = 0.4**
   - Class imbalance: 24.5% distressed, 75.5% solvent
   - Prioritizes recall (catch distress) over precision
   - Reproducible & explainable (not per-horizon tuned)
   - See `CODE_FLOW.md` for rationale

2. **Time-Based Splits**
   - Train: 2014–2019 | Val: 2020–2021 | Test: 2022–2023
   - Prevents temporal leakage (financial data has sequences)
   - Test uses most recent data (production-like)

3. **Sequential Model Training**
   - Each model trains across all 5 horizons before moving to next
   - Memory-efficient + failure visibility
   - Unified logging across all models

4. **Consolidated Hyperparameters**
   - XGBoost: all iterations logged; best params saved to JSON per horizon
   - NGBoost/RF/LightGBM: fixed configs logged per horizon
   - Enables full reproducibility

---

## Running Tests

```bash
# Quick test (single horizon, single model)
python train_single_model.py --model rf --horizons 1 --output ./test_output

# Full pipeline test (skip SHAP for speed)
python run_pipeline.py --data ./data/processed/financial_report_bank_zscore_clean.csv --skip-shap

# With SHAP (slower)
python run_pipeline.py
```

---

## Output Interpretation

**Metrics**:
- **Accuracy**: Overall correctness
- **ROC-AUC**: Discrimination across thresholds (0.88–0.92)
- **PR-AUC**: Precision-recall on imbalanced data (0.65–0.75)
- **Recall**: % of actual distress caught (target: 75%, actual: 63–68%)
- **Type I Error**: FN / (FN + TP) – Missed distress (32–37%) ← regulatory risk
- **Type II Error**: FP / (FP + TN) – False alarms (8–12%)

**Risk Buckets**:
- Low Risk (prob < 0.3): Safe banks
- Medium Risk (0.3–0.6): Monitor closely
- High Risk (prob > 0.6): Intervention warranted

**SHAP Plots** (if SHAP enabled):
- Summary plot: Feature impact direction
- Dependence plots: Feature value vs SHAP interaction
- Feature importance: Mean |SHAP| ranking

---

## Documentation

- **`CODE_FLOW.md`** – Architecture, data pipeline, execution flows, threshold rationale, SDGs
- **`CONFIGURATION.md`** – Full config reference, hyperparameters, input/output formats, troubleshooting
- **`PIPELINE_USAGE.md`** – Quick-start examples, metrics interpretation, publication tips

---

## Installation

### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t bankruptcy-prediction .

# Run container
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output bankruptcy-prediction python train_all_models.py

# Or use docker-compose
docker-compose up
```

### Option 2: Local Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run pipeline
cd src
python train_all_models.py
```

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=2.0.0
lightgbm>=4.0.0
ngboost>=0.4.1
shap>=0.42.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## SDGs Alignment

- **SDG 8 (Decent Work & Economic Growth)**: Early warning prevents systemic collapse, protects jobs
- **SDG 10 (Reduced Inequalities)**: Risk-aware lending enables financial inclusion for underbanked sectors
- **SDG 16 (Peace, Justice, Strong Institutions)**: Early warning prevents destabilization of financial systems

---

## License

MIT License – See LICENSE file

---

## Support

For issues or questions:
- Check `CONFIGURATION.md` troubleshooting section
- Review `CODE_FLOW.md` for architecture details
- Open an issue on GitHub
