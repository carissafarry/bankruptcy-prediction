"""
Shared training utilities for bankruptcy prediction models.
Used by both train_single_model.py and train_all_models.py
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from sklearn.ensemble import RandomForestClassifier


# ====================
# CONFIGURATION
# ====================

YEAR_START = 2014
YEAR_END = 2023
TRAIN_END_YEAR = 2019
VAL_END_YEAR = 2021
RECALL_TARGET = 0.75
RANDOM_STATE = 42

FEATURE_COLS = [
    "size", "der", "dar", "roa", "roe", "sdoa", "sdroe",
    "tobinq", "ppe", "cash", "ar", "log_sales", "sgr",
    "operating_income_ratio", "equity_to_assets"
]

TARGET_COL = "bank_zscore_risk"

MODEL_TITLE_MAP = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "ngboost": "NGBoost"
}


# ====================
# UTILITY FUNCTIONS
# ====================

def quarter_end_date(year: int, period: str):
    """Convert year and quarter (Q1-Q4) to quarter-end date."""
    q_end = {
        "Q1": "03-31",
        "Q2": "06-30",
        "Q3": "09-30",
        "Q4": "12-31"
    }
    return pd.to_datetime(f"{year}-{q_end[period]}")


def build_threshold_table(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Build threshold tuning table."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    rows = []
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

        rows.append({
            "threshold": thr,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
            "type_i_error": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
            "type_ii_error": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def calc_type_errors(y_true: np.ndarray, y_proba: np.ndarray, thr: float = 0.5) -> Dict:
    """Calculate Type I/II errors."""
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Type_I_error": fn / (tp + fn) if (tp + fn) > 0 else np.nan,
        "Type_II_error": fp / (tn + fp) if (tn + fp) > 0 else np.nan,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "Recall": tp / (tp + fn) if (tp + fn) > 0 else np.nan
    }


def get_probabilities(model, X, model_type: str) -> np.ndarray:
    """Get probability predictions."""
    if model_type in ["xgb", "xgboost"]:
        if not isinstance(X, xgb.DMatrix):
            X = xgb.DMatrix(X)
        return model.predict(X)
    elif model_type == "lgbm":
        return model.predict(X)
    elif model_type in ["ngboost", "rf"]:
        return model.predict_proba(X)[:, 1]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ====================
# DATA LOADING & PREP
# ====================

def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned data and prepare horizon labels."""
    df = pd.read_csv(data_path)

    # Filter years
    df = df[df["calendar_year"].between(YEAR_START, YEAR_END)].copy()

    # Keep only complete 4Q bank-years
    ok_pairs = (
        df.groupby(["symbol", "calendar_year"])["period"]
        .nunique()
        .reset_index(name="n_quarters")
        .query("n_quarters == 4")[["symbol", "calendar_year"]]
    )
    df = df.merge(ok_pairs, on=["symbol", "calendar_year"], how="inner")

    # Sort for horizon shift
    df_h = df.sort_values(["symbol", "calendar_year"]).copy()

    # Create horizon labels (1-5 years ahead)
    for h in range(1, 6):
        df_h[f"distress_{h}y"] = (
            df_h.groupby("symbol")[TARGET_COL].shift(-h)
        )

    # Keep only rows with complete features
    df_model = df_h.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    return df_h, df_model


def split_by_horizon(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based train/val/test split."""
    label_col = f"distress_{horizon}y"
    df_hh = df.dropna(subset=[label_col]).copy()

    train_mask = df_hh["calendar_year"] <= TRAIN_END_YEAR
    val_mask = (df_hh["calendar_year"] > TRAIN_END_YEAR) & (df_hh["calendar_year"] <= VAL_END_YEAR)
    test_mask = df_hh["calendar_year"] > VAL_END_YEAR

    X_train = df_hh.loc[train_mask, FEATURE_COLS]
    y_train = df_hh.loc[train_mask, label_col].astype(int)

    X_val = df_hh.loc[val_mask, FEATURE_COLS]
    y_val = df_hh.loc[val_mask, label_col].astype(int)

    X_test = df_hh.loc[test_mask, FEATURE_COLS]
    y_test = df_hh.loc[test_mask, label_col].astype(int)

    df_test_meta = df_hh.loc[test_mask, ["symbol", "calendar_year", "period", "time"]]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test, df_test_meta)


# ====================
# TRAINING FUNCTIONS
# ====================

def train_ngboost(X_train, y_train, X_val, y_val, hardcode_threshold: float = None,
                  horizon: int = None, search_logs: list = None) -> Tuple:
    """Train NGBoost. Optionally use hardcoded threshold."""
    print("  Training NGBoost...")
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    sample_weight = np.where(y_train == 1, neg / pos, 1.0)

    ngb = NGBClassifier(
        Dist=Bernoulli,
        n_estimators=500,
        learning_rate=0.03,
        random_state=RANDOM_STATE
    )

    ngb.fit(X_train, y_train, sample_weight=sample_weight)
    proba_val = ngb.predict_proba(X_val)[:, 1]

    # Log hyperparameters (NGBoost uses fixed config)
    log_entry = {
        "horizon": horizon,
        "model": "ngboost",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "random_state": RANDOM_STATE
    }
    if search_logs is not None:
        search_logs.append(log_entry)

    if hardcode_threshold is not None:
        chosen_thr = hardcode_threshold
        print(f"    Using hardcoded threshold: {chosen_thr:.4f}")
    else:
        thr_df = build_threshold_table(y_val, proba_val)
        feasible = thr_df[thr_df["recall"] >= RECALL_TARGET]
        chosen_thr = feasible.sort_values("threshold").iloc[0]["threshold"] if not feasible.empty else 0.5
        print(f"    Chosen threshold: {chosen_thr:.4f}")

    return ngb, chosen_thr


def train_random_forest(X_train, y_train, X_val, y_val, hardcode_threshold: float = None,
                        horizon: int = None, search_logs: list = None) -> Tuple:
    """Train Random Forest. Optionally use hardcoded threshold."""
    print("  Training Random Forest...")
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    class_weight = {0: 1.0, 1: neg / pos}

    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=8,
        min_samples_leaf=15,
        min_samples_split=30,
        max_features="sqrt",
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    proba_val = rf.predict_proba(X_val)[:, 1]

    # Log hyperparameters (Random Forest uses fixed config)
    log_entry = {
        "horizon": horizon,
        "model": "rf",
        "n_estimators": 1000,
        "max_depth": 8,
        "min_samples_leaf": 15,
        "min_samples_split": 30,
        "max_features": "sqrt"
    }
    if search_logs is not None:
        search_logs.append(log_entry)

    if hardcode_threshold is not None:
        chosen_thr = hardcode_threshold
        print(f"    Using hardcoded threshold: {chosen_thr:.4f}")
    else:
        thr_df = build_threshold_table(y_val, proba_val)
        feasible = thr_df[thr_df["recall"] >= RECALL_TARGET]
        chosen_thr = feasible.sort_values("threshold").iloc[0]["threshold"] if not feasible.empty else 0.5
        print(f"    Chosen threshold: {chosen_thr:.4f}")

    return rf, chosen_thr


def train_xgboost(X_train, y_train, X_val, y_val, hardcode_threshold: float = None,
                  output_dir: str = None, horizon: int = None, search_logs: list = None) -> Tuple:
    """Train XGBoost with random hyperparameter search and early stopping.

    Args:
        search_logs: Optional list to collect iteration logs across horizons
    """
    print("  Training XGBoost (random hyperparameter search)...")

    # Prepare DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Class imbalance weighting
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Seeded RNG for reproducibility
    rng = np.random.default_rng(RANDOM_STATE)

    # Random search for best hyperparameters
    best = {"pr": -1, "params": {}, "best_iter": 0}
    local_logs = []

    for i in range(1, 21):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "seed": RANDOM_STATE,
            "scale_pos_weight": float(scale_pos_weight),
            # Random hyperparameters
            "eta": float(rng.choice([0.01, 0.03, 0.05, 0.1])),
            "max_depth": int(rng.choice([3, 4, 5, 6, 7])),
            "min_child_weight": int(rng.choice([1, 3, 5])),
            "subsample": float(rng.choice([0.6, 0.8, 1.0])),
            "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
            "lambda": float(rng.choice([0.5, 1.0, 1.5])),
            "gamma": float(rng.choice([0.0, 0.5, 1.0])),
        }

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=200,
            verbose_eval=False
        )

        # Get validation predictions (use best iteration)
        proba_val = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        pr = average_precision_score(y_val, proba_val)

        # Log iteration
        log_entry = {
            "horizon": horizon,
            "iter": i,
            "pr_auc": float(pr),
            **params
        }
        local_logs.append(log_entry)
        if search_logs is not None:
            search_logs.append(log_entry)

        if pr > best["pr"]:
            best["pr"] = pr
            best["params"] = params
            best["best_iter"] = booster.best_iteration

        if i % 5 == 0:
            print(f"    Iter {i}/20 best_pr={best['pr']:.4f}")

    # Retrain with best hyperparameters
    model = xgb.train(
        best["params"],
        dtrain,
        num_boost_round=best["best_iter"] + 1,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=200,
        verbose_eval=False
    )

    # Get validation predictions
    proba_val = model.predict(dval, iteration_range=(0, model.best_iteration + 1))

    # Store best params
    best_params_json = {
        "horizon": horizon,
        "best_pr": float(best["pr"]),
        "best_iter": int(best["best_iter"]),
        "params": best["params"]
    }
    print(f"    Best params (PR={best['pr']:.4f}, iter={best['best_iter']})")

    if hardcode_threshold is not None:
        chosen_thr = hardcode_threshold
        print(f"    Using hardcoded threshold: {chosen_thr:.4f}")
    else:
        thr_df = build_threshold_table(y_val, proba_val)
        feasible = thr_df[thr_df["recall"] >= RECALL_TARGET]
        chosen_thr = feasible.sort_values("threshold").iloc[0]["threshold"] if not feasible.empty else 0.5
        print(f"    Chosen threshold: {chosen_thr:.4f}")

    return model, chosen_thr, best_params_json


def train_lightgbm(X_train, y_train, X_val, y_val, hardcode_threshold: float = None,
                   horizon: int = None, search_logs: list = None) -> Tuple:
    """Train LightGBM with early stopping."""
    print("  Training LightGBM...")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Handle imbalance
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(200)]
    )

    proba_val = model.predict(X_val)

    # Log hyperparameters (LightGBM uses fixed config)
    log_entry = {
        "horizon": horizon,
        "model": "lgbm",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "num_boost_round": 500,
        "early_stopping_rounds": 200
    }
    if search_logs is not None:
        search_logs.append(log_entry)

    if hardcode_threshold is not None:
        chosen_thr = hardcode_threshold
        print(f"    Using hardcoded threshold: {chosen_thr:.4f}")
    else:
        thr_df = build_threshold_table(y_val, proba_val)
        feasible = thr_df[thr_df["recall"] >= RECALL_TARGET]
        chosen_thr = feasible.sort_values("threshold").iloc[0]["threshold"] if not feasible.empty else 0.5
        print(f"    Chosen threshold: {chosen_thr:.4f}")

    return model, chosen_thr


# ====================
# EVALUATION & EXPORT
# ====================

def evaluate_and_export(model, X_test, y_test, df_test_meta, horizon: int, thr: float,
                        model_type: str, output_dir: str):
    """Evaluate model and export predictions."""
    proba = get_probabilities(model, X_test, model_type)
    pred = (proba >= thr).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan
    pr = average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan
    acc = accuracy_score(y_test, pred)
    err = calc_type_errors(y_test, proba, thr)

    print(f"\n  === TEST {horizon}Y (thr={thr}) ===")
    print(f"  Accuracy : {acc}")
    print(f"  ROC-AUC  : {auc}")
    print(f"  PR-AUC   : {pr}")
    print(f"  Type I Error : {err['Type_I_error']}")
    print(f"  Type II Error: {err['Type_II_error']}")
    print(f"  Recall       : {err['Recall']}")

    # Build export dataframe
    export_df = df_test_meta[["symbol", "calendar_year", "period", "time"]].copy()
    export_df["tanggal"] = export_df.apply(
        lambda r: quarter_end_date(r["calendar_year"], r["period"]).strftime("%d/%m/%Y"),
        axis=1
    )
    export_df["horizon"] = horizon
    export_df["distress_actual"] = y_test.values
    export_df["prob_distress"] = proba
    export_df["pred_label"] = pred
    export_df["distance_to_threshold"] = proba - thr
    export_df["model_name"] = MODEL_TITLE_MAP[model_type]
    export_df["threshold_used"] = thr

    # Risk buckets
    export_df["risk_bucket"] = pd.cut(
        export_df["prob_distress"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True
    )

    # Confusion type
    def get_confusion(row):
        if row.distress_actual == 1 and row.pred_label == 1:
            return "TP"
        elif row.distress_actual == 1 and row.pred_label == 0:
            return "FN"
        elif row.distress_actual == 0 and row.pred_label == 1:
            return "FP"
        return "TN"

    export_df["confusion_type"] = export_df.apply(get_confusion, axis=1)
    export_df = export_df.sort_values(["calendar_year", "prob_distress"], ascending=[True, False])

    # Save CSV
    output_path = f"{output_dir}/{model_type}_predictions_{horizon}y.csv"
    export_df.to_csv(output_path, index=False)
    print(f"  Exported: {output_path}")

    return export_df
