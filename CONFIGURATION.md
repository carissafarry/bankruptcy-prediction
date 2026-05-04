# Configuration Guide

**Version**: 1.0.0 | **Updated**: May 2026

Complete reference for configuring the Bankruptcy Prediction training pipeline.

---

## Overview

All configuration centralized in `src/training_utils.py`:

```python
# Data period
YEAR_START = 2014          # Earliest year in dataset
YEAR_END = 2023            # Latest year in dataset
TRAIN_END_YEAR = 2019      # Train/Validation cutoff
VAL_END_YEAR = 2021        # Validation/Test cutoff

# Features (15 financial ratios)
FEATURE_COLS = [
    "size", "der", "dar", "roa", "roe", "sdoa", "sdroe",
    "tobinq", "ppe", "cash", "ar", "log_sales", "sgr",
    "operating_income_ratio", "equity_to_assets"
]

# Training
RECALL_TARGET = 0.75       # Target recall for threshold tuning (if not hardcoded)
RANDOM_STATE = 42          # Reproducibility seed

# Threshold (HARDCODED)
HARDCODED_THRESHOLD = 0.4  # Do NOT tune per horizon; fixed for stability
```

---

## Threshold Configuration

### Hardcoded Threshold = 0.4

**See CODE_FLOW.md § "Threshold Rationale" for detailed explanation of why 0.4.**

Quick reference:
- Fixed at 0.4 due to class imbalance (24.5% positive)
- Prioritizes recall over precision (catch distress)
- Reproducible + explainable (not per-horizon tuned)

**To adjust threshold, modify `training_utils.py`:**
```python
model, threshold = train_fn(X_train, y_train, X_val, y_val, hardcode_threshold=0.4)
#                                                                           ↑ Change here
```

**Trade-offs if changed:**
- Lower to 0.35 → recall ≈ 78%, more false alarms
- Higher to 0.45 → precision improves, recall drops
- Regulatory requirements determine acceptable threshold

---

## Training Configuration

### Split Strategy

**Time-based splits (no temporal leakage):**
```
Train:       2014–2019 (fit model)
Validation:  2020–2021 (threshold tuning, if needed)
Test:        2022–2023 (final evaluation, production-like)
```

**See CODE_FLOW.md § "Data Pipeline" for rationale on temporal leakage prevention.**

Quick: Financial data has sequences → random shuffle leaks future info into training.

### Horizon Configuration

**Prediction horizons: 1, 2, 3, 4, 5 years**

- Horizon 1y: Is bank in distress next year?
- Horizon 2y: Is bank in distress 2 years from now?
- ... (models learn different patterns per horizon)

**All horizons trained for each model** (sequential batch training).

---

## CLI Arguments

### `train_single_model.py`

```bash
python train_single_model.py \
  --model ngboost \
  --data ./data/processed/financial_report_bank_zscore_clean.csv \
  --output ./output \
  --horizons 1,2,3,4,5 \
  --save-model
```

**Options:**
- `--model` (required): `ngboost`, `rf`, `xgboost`, or `lgbm`
- `--data`: Path to input CSV (default: `./data/processed/financial_report_bank_zscore_clean.csv`)
- `--output`: Output directory (default: `./output`)
- `--horizons`: Comma-separated list, e.g., `1,2,3` (default: `1,2,3,4,5`)
- `--save-model`: Save trained models to pickle (optional flag)

### `train_all_models.py`

```bash
python train_all_models.py \
  --data ./data/processed/financial_report_bank_zscore_clean.csv \
  --output ./output
```

**Options:**
- `--data`: Input CSV path (same default)
- `--output`: Output directory (default: `./output`)

Trains all 4 models sequentially.

### `run_pipeline.py`

```bash
python run_pipeline.py \
  --data ./data/processed/financial_report_bank_zscore_clean.csv \
  --output ./output \
  --skip-shap
```

**Options:**
- `--data`: Input CSV path
- `--output`: Output directory
- `--skip-shap`: Skip SHAP analysis (faster for testing)

---

## Input Data Format

### Required CSV Structure

**File:** `financial_report_bank_zscore_clean.csv`

**Essential Columns (must exist):**
```
symbol, calendar_year, period, time, bank_zscore_risk,
size, der, dar, roa, roe, sdoa, sdroe,
tobinq, ppe, cash, ar, log_sales, sgr,
operating_income_ratio, equity_to_assets
```

**Data Types:**
- `symbol`: str (bank identifier)
- `calendar_year`: int (2014–2023)
- `period`: str ("Q1", "Q2", "Q3", "Q4")
- `time`: str ("2022Q3" format)
- `bank_zscore_risk`: int (0=solvent, 1=distressed)
- Financial features: float (can be positive or negative)

**Validation:**
- No missing values in required columns (drop rows with NaN)
- Each bank must have complete 4-quarter observations per year
- 2014–2023 coverage

### Preprocessing Checklist

Before running pipeline:
```python
df = pd.read_csv('financial_report_bank_zscore_clean.csv')

# 1. Check required columns
required = ['symbol', 'calendar_year', 'period', 'time', 'bank_zscore_risk',
            'size', 'der', 'dar', 'roa', 'roe', ...]
assert all(col in df.columns for col in required)

# 2. Check no NaN in critical columns
assert not df[required].isnull().any().any()

# 3. Check year range
assert df['calendar_year'].between(2014, 2023).all()

# 4. Check period values
assert df['period'].isin(['Q1', 'Q2', 'Q3', 'Q4']).all()

# 5. Check target is binary
assert df['bank_zscore_risk'].isin([0, 1]).all()

# 6. Check complete 4Q per bank-year
year_counts = df.groupby(['symbol', 'calendar_year']).size()
assert (year_counts == 4).all()  # Each bank-year has 4 quarters
```

---

## Output Configuration

### Directory Structure

```
output/
├── models_all_horizons.pkl              ← Pickled model dict (if --save-model used)
├── ngboost/
│   ├── ngboost_predictions_1y.csv       ← Test predictions + metadata
│   ├── ngboost_predictions_2y.csv
│   ├── ngboost_predictions_3y.csv
│   ├── ngboost_predictions_4y.csv
│   ├── ngboost_predictions_5y.csv
│   ├── ngboost_search_logs.csv          ← Hyperparameters per horizon
│   ├── ngboost_shap_summary_1y.png      ← Only if --skip-shap not set
│   ├── ngboost_shap_importance_1y.png
│   ├── ngboost_dependence_size_1y.png
│   └── ngboost_feature_importance_1y.csv
├── rf/
│   ├── rf_predictions_1y.csv
│   ├── rf_search_logs.csv               ← Fixed hyperparameters per horizon
│   └── ... (SHAP plots if enabled)
├── xgboost/
│   ├── xgboost_predictions_1y.csv
│   ├── xgboost_best_params.json         ← Consolidated best params (all horizons)
│   ├── xgboost_search_logs.csv          ← All 100 iterations (20 per horizon × 5)
│   └── ... (SHAP plots if enabled)
└── lgbm/
    ├── lgbm_predictions_1y.csv
    ├── lgbm_search_logs.csv             ← Fixed hyperparameters per horizon
    └── ... (SHAP plots if enabled)
```

### CSV Output Columns (Ordered)

```
symbol, calendar_year, period, time, tanggal, horizon,
distress_actual, prob_distress, pred_label, distance_to_threshold,
model_name, threshold_used, risk_bucket, confusion_type
```

**Column Meanings (see CODE_FLOW.md § "Data Export Format" for details):**
- `tanggal`: Quarter-end date (DD/MM/YYYY) for dashboard filtering
- `prob_distress`: Model probability [0, 1]
- `pred_label`: Binary prediction (1 if prob ≥ 0.4, else 0)
- `distance_to_threshold`: prob − 0.4 (margin from decision boundary)
- `risk_bucket`: Categorical risk level (Low [0–0.3], Medium [0.3–0.6], High [0.6–1.0])
- `confusion_type`: TP/FN/FP/TN for error analysis

---

## Model Hyperparameters

### NGBoost
```python
NGBClassifier(
    Dist=Bernoulli,
    n_estimators=500,
    learning_rate=0.03,
    random_state=42
)
```

### Random Forest
```python
RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_leaf=15,
    min_samples_split=30,
    max_features='sqrt',
    class_weight={0: 1.0, 1: neg_count/pos_count},  # Dynamic
    random_state=42,
    n_jobs=-1
)
```

### XGBoost
```python
# Seeded RNG for reproducible hyperparameter selection
rng = np.random.default_rng(RANDOM_STATE)

xgb.train({
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'max_depth': rng.choice([3, 4, 5, 6, 7]),           # Random search
    'eta': rng.choice([0.01, 0.03, 0.05, 0.1]),        # Random search
    'min_child_weight': rng.choice([1, 3, 5]),          # Random search
    'subsample': rng.choice([0.6, 0.8, 1.0]),          # Random search
    'colsample_bytree': rng.choice([0.6, 0.8, 1.0]),   # Random search
    'lambda': rng.choice([0.5, 1.0, 1.5]),             # Random search
    'gamma': rng.choice([0.0, 0.5, 1.0]),              # Random search
    'scale_pos_weight': neg_count/pos_count,            # Class imbalance
    'random_state': 42, 'seed': 42
}, num_boost_round=3000, early_stopping_rounds=200, verbose_eval=False)
```

### LightGBM
```python
lgb.train({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'scale_pos_weight': neg_count/pos_count,
    'verbose': -1
}, train_set, num_boost_round=500,
   valid_sets=[val_set],
   callbacks=[lgb.early_stopping(200)])
```

---

## SHAP Configuration

### Sample Size
```python
SHAP_SAMPLE_N = 200         # Instances to explain
SHAP_BACKGROUND_N = 100     # Background for KernelExplainer (NGBoost)
```

**If memory errors:**
- Reduce both values (e.g., 100, 50)
- Smaller samples = faster but less representative

### Explainer Types (Auto-Selected)
- **TreeExplainer:** RF, XGBoost, LightGBM (fast)
- **KernelExplainer:** NGBoost (slower, model-agnostic)

---

## Dependencies

### Install
```bash
pip install -r requirements.txt
```

### Versions (Pinned)
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

**Version**: 1.0.0 | **Updated**: May 2026
