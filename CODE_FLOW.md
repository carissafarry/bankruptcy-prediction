# Code Flow: Bankruptcy Prediction Pipeline

**Version**: 1.0.0 | **Updated**: May 2026

Production pipeline for training 4 ML models (NGBoost, Random Forest, XGBoost, LightGBM) predicting bank insolvency across 5 horizons (1-5 years ahead).

## 🌱 SDGs Integration

**UN Sustainable Development Goals Addressed:**
- **SDG 8 (Decent Work & Economic Growth):** Early bankruptcy prediction prevents systemic economic collapse, protects jobs
- **SDG 10 (Reduced Inequalities):** Risk-aware lending enables financial inclusion for underbanked sectors
- **SDG 16 (Peace, Justice, Strong Institutions):** Early warning prevents destabilization of financial systems

**Implementation:**
- Equitable risk assessment reducing lending discrimination
- Transparent model decisions (SHAP explainability) improve stakeholder trust
- Predictive capacity strengthens financial regulators' oversight

---

## Architecture

```
input/
  └── financial_report_bank_zscore_clean.csv (1812 rows × 27 cols, 2014–2023)
      ↓
src/
  ├── training_utils.py (shared utilities: configs, data, training, export)
  ├── train_single_model.py (single model CLI, all horizons)
  ├── train_all_models.py (all 4 models sequentially)
  ├── run_pipeline.py (orchestrator: training + optional SHAP)
  └── shap_analysis.py (explainability for all model types)
      ↓
output/
  ├── models_all_horizons.pkl (saved models dict)
  ├── ngboost/ ├── ngboost_predictions_1y.csv
  │           ├── ngboost_shap_summary_1y.png
  │           └── ngboost_feature_importance_1y.csv
  ├── rf/
  ├── xgboost/
  └── lgbm/
```

---

## Data Pipeline

### Stage 1: Data Preparation
**Input:** `financial_report_bank_zscore_clean.csv`
- Rows: 1,812 bank-quarter observations
- Columns: 27 (symbol, period, financial ratios, target)
- Years: 2014–2023

**Filtering:**
- Keep only complete 4-quarter bank-years (prevents partial data bias)
- Drop rows with missing features or target

**Horizon Labels (Forward-Shifted Targets):**
```python
for horizon in 1..5:
    df[f"distress_{horizon}y"] = df.groupby("symbol")["bank_zscore_risk"].shift(-horizon)
```
- `distress_1y`: Is bank in distress 1 year from now?
- `distress_2y`: Is bank in distress 2 years from now?
- ... up to 5 years ahead

**Features (15 Financial Ratios):**
```
size, der, dar, roa, roe, sdoa, sdroe, tobinq, ppe, cash, ar,
log_sales, sgr, operating_income_ratio, equity_to_assets
```

### Stage 2: Time-Based Splitting
**No temporal leakage; realistic chronological split:**
```
Train:      2014–2019 (6 years)
Validation: 2020–2021 (2 years) ← threshold tuning
Test:       2022–2023 (2 years) ← production-like data
```

**Why?** Test set uses most recent data (2022–2023), closest to deployment scenario.

### Stage 3: Class Imbalance Handling

**Problem:** Data is severely imbalanced
```
Positive (distress):   24.5%
Negative (solvent):    75.5%
Imbalance ratio:       ~3:1
```

**Impact without handling:**
- Model biases toward "solvent" predictions
- Missing actual bankruptcies (high False Negatives)
- Type I error = regulatory risk (bad)

**Solutions Applied:**
- **NGBoost & Random Forest:** `sample_weight` / `class_weight`
  - Penalize positive class errors by ratio: `neg_count / pos_count`
- **XGBoost & LightGBM:** `scale_pos_weight` parameter
  - Each positive sample weighted as ~3 negative samples

---

## Training Strategy

### Four Models Trained Sequentially
**Order:** NGBoost → Random Forest → XGBoost → LightGBM

Each model trains across all 5 horizons before moving to next model (memory-efficient, monitorable).

**Configuration per Model:**

| Model | Key Parameters | Rationale |
|-------|---|---|
| **NGBoost** | 500 estimators, Bernoulli dist, lr=0.03 | Probabilistic; handles class weight naturally |
| **Random Forest** | 1000 trees, depth=8, cost-sensitive weights | Robust ensemble; interpretable |
| **XGBoost** | 100–200 rounds, random hyperopt (20 iters), scale_pos_weight | Gradient boosting with class balance |
| **LightGBM** | 500 rounds, early stop (patience=200), scale_pos_weight | Fast, memory-efficient, handles imbalance |

All use `hardcode_threshold=0.4` (see next section).

---

## Threshold Rationale

### Why 0.4 (not 0.5)?

**1. Data Imbalance Reality**
- Default 0.5 threshold biases toward majority class (solvent)
- Would classify most banks as "safe" even when risky
- Unacceptable in risk management

**2. Type I vs Type II Error Trade-off**
```
Type I Error (FN):  Miss actual bankruptcy → Regulatory/financial loss
Type II Error (FP): False alarm → Unnecessary intervention cost

Priority: Recall (catch distress) > Precision (avoid false alarms)
```

**3. Validation Evidence**
- Across 5 notebooks: threshold ≈ 0.4–0.45 achieves ~75% recall
- Achieves 0.67–0.73 precision (acceptable trade-off)
- Consistent across all 4 models and horizons

**4. Production Stability**
- Fixed 0.4 ensures:
  - Reproducible predictions (no per-horizon tuning)
  - Explainable decisions (not a black-box optimization)
  - Consistent stakeholder expectations

**Validation Metrics at Threshold=0.4:**
```
Accuracy:    0.80–0.85 (balanced)
ROC-AUC:     0.88–0.92 (strong discrimination)
PR-AUC:      0.65–0.75 (good on imbalanced data)
Recall:      0.63–0.68 (catch most actual distress)
Type I Err:  0.32–0.37 (miss ~35% of bankruptcies → unacceptable)
             → Could lower threshold to 0.35 for higher recall if needed
```

---

## Model Configuration

### Shared Utilities (`training_utils.py`)

**Centralized module imported by all training scripts:**
- Configuration constants (dates, features, model names)
- Data loading & splitting
- All 4 training functions
- Threshold tuning logic
- CSV export with `tanggal` (quarter-end dates)
- Error metrics calculation

**Key exports:**
```python
load_and_prepare_data(path)         # → df_h, df_model
split_by_horizon(df, horizon)       # → (X_train, y_train), (X_val, y_val), (X_test, y_test, meta)
train_ngboost/rf/xgboost/lgbm(...)  # → model, threshold
evaluate_and_export(...)            # → CSV + metrics print
```

### Data Export Format

**CSV Columns (ordered):**
```
symbol, calendar_year, period, time, tanggal, horizon,
distress_actual, prob_distress, pred_label, distance_to_threshold,
model_name, threshold_used, risk_bucket, confusion_type
```

**Key Columns:**
- `tanggal`: Quarter-end date (DD/MM/YYYY) for dashboard filtering
- `prob_distress`: Raw probability [0, 1]
- `pred_label`: Binary prediction (1 if prob ≥ 0.4)
- `distance_to_threshold`: `prob - 0.4` (identify borderline cases)
- `risk_bucket`: Categorical ("Low Risk" [0–0.3], "Medium Risk" [0.3–0.6], "High Risk" [0.6–1.0])
- `confusion_type`: TP/FN/FP/TN (error analysis)

---

## Execution Flows

### Flow 1: Single Model Training
```bash
python train_single_model.py --model ngboost [--horizons 1,2,3] [--output ./output]
```

**Steps:**
```
Load data once
├── for horizon in [1, 2, 3, 4, 5]:
│   ├── split_by_horizon(df, horizon)
│   ├── train_fn(X_train, y_train, X_val, y_val, hardcode_threshold=0.4)
│   └── evaluate_and_export(...) → CSV file
└── Optionally save models dict to pickle
```

**Output:**
- `output/ngboost/ngboost_predictions_{1..5}y.csv` – Predictions + metadata
- `output/ngboost/ngboost_search_logs.csv` – Hyperparameters per horizon
- Optional: `output/ngboost/ngboost_models_all_horizons.pkl` (if --save-model)

### Flow 2: All Models (Batch)
```bash
python train_all_models.py [--data ./path] [--output ./output]
```

**Steps:**
```
Load data once
├── for model_name in ["ngboost", "rf", "xgboost", "lgbm"]:
│   ├── Create model_output_dir
│   └── for horizon in [1..5]:
│       ├── split_by_horizon(df, horizon)
│       ├── train_fn(..., hardcode_threshold=0.4)
│       └── evaluate_and_export(...)
└── Save models_all_horizons.pkl
```

**Output:**
- Per-model predictions: `output/{model}/{model}_predictions_{horizon}y.csv`
- Per-model search logs: `output/{model}/{model}_search_logs.csv` (hyperparameters + iterations)
- XGBoost consolidated params: `output/xgboost/xgboost_best_params.json` (all horizons)
- Pickle: `output/models_all_horizons.pkl`

### Flow 3: Full Pipeline with SHAP
```bash
python run_pipeline.py [--skip-shap]
```

**Steps:**
```
train_all_models()  ← Phase 1: Training
│   ├── Output: predictions CSVs
│   ├── Output: search_logs.csv per model
│   └── Output: xgboost_best_params.json (consolidated)
│
├─ [optional] Phase 2: SHAP Analysis
│  ├── for model in [ngboost, rf, xgboost, lgbm]:
│  │   └── analyze_all_horizons()
│  │       ├── Summary plot (dot)
│  │       ├── Feature importance bar chart
│  │       ├── Dependence plots (top 6 features)
│  │       └── Feature importance CSV
│  └── Output: `output/{model}/{model}_shap_*.png` + CSV
└─ Report complete
```

---

## Key Design Decisions

1. **Hardcoded Threshold (0.4):** Reflects imbalance; prioritizes recall
2. **Time-Based Splits:** Prevents temporal leakage; realistic test data
3. **Sequential Training:** Memory-efficient; failure visibility
4. **Shared `training_utils.py`:** Single source of truth; reduces duplication
5. **Tanggal Column:** Human-readable dates for dashboards

---

## Dependencies

```
pandas>=1.5.0, numpy>=1.23.0, scikit-learn>=1.2.0,
xgboost>=2.0.0, lightgbm>=4.0.0, ngboost>=0.4.1,
shap>=0.42.0, matplotlib>=3.6.0, seaborn>=0.12.0
```

---

**Document Version**: 1.0.0 | **Last Updated**: May 2026
