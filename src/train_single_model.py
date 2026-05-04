"""
Train a single model across all horizons with flexible model selection.
Outputs predictions CSV per horizon.

Usage:
  python train_single_model.py --model ngboost
  python train_single_model.py --model rf --data ./data/processed/data.csv --output ./results
  python train_single_model.py --model xgboost --horizons 1,2,3
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import xgboost as xgb
import lightgbm as lgb
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)


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

def train_ngboost(X_train, y_train, X_val, y_val, hardcode_threshold: float = None) -> Tuple:
    """Train NGBoost. Optionally use hardcoded threshold instead of optimizing."""
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

    # Use hardcoded threshold if provided, otherwise optimize
    if hardcode_threshold is not None:
        chosen_thr = hardcode_threshold
        print(f"    Using hardcoded threshold: {chosen_thr:.4f}")
    else:
        thr_df = build_threshold_table(y_val, proba_val)
        feasible = thr_df[thr_df["recall"] >= RECALL_TARGET]
        chosen_thr = feasible.sort_values("threshold").iloc[0]["threshold"] if not feasible.empty else 0.5
        print(f"    Chosen threshold: {chosen_thr:.4f}")

    return ngb, chosen_thr


def train_random_forest(X_train, y_train, X_val, y_val, hardcode_threshold: float = None) -> Tuple:
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

    if hardcode_threshold is not None:
        chosen_thr = hardcode_threshold
        print(f"    Using hardcoded threshold: {chosen_thr:.4f}")
    else:
        thr_df = build_threshold_table(y_val, proba_val)
        feasible = thr_df[thr_df["recall"] >= RECALL_TARGET]
        chosen_thr = feasible.sort_values("threshold").iloc[0]["threshold"] if not feasible.empty else 0.5
        print(f"    Chosen threshold: {chosen_thr:.4f}")

    return rf, chosen_thr


def train_xgboost(X_train, y_train, X_val, y_val, hardcode_threshold: float = None, random_search_iter: int = 20) -> Tuple:
    """Train XGBoost. Optionally use hardcoded threshold."""
    print("  Training XGBoost (random hyperparameter search)...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    best = {"pr": -1, "params": {}, "best_iter": 0}

    for iter_num in range(random_search_iter):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "seed": RANDOM_STATE,
            "scale_pos_weight": scale_pos_weight,
            "eta": np.random.choice([0.01, 0.03, 0.05]),
            "max_depth": np.random.choice([3, 4, 5]),
            "min_child_weight": np.random.choice([1, 3]),
            "subsample": np.random.choice([0.8, 1.0]),
            "colsample_bytree": np.random.choice([0.6, 0.8]),
            "lambda": np.random.choice([0.5, 1.0]),
            "gamma": np.random.choice([0.0, 0.5]),
        }

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=3000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=200,
            verbose_eval=False
        )

        proba_val = booster.predict(dval)
        pr = average_precision_score(y_val, proba_val)

        if pr > best["pr"]:
            best = {
                "pr": pr,
                "params": params,
                "best_iter": booster.best_iteration
            }

        if (iter_num + 1) % 5 == 0:
            print(f"    Iter {iter_num + 1}/{random_search_iter} best_pr={best['pr']:.4f}")

    # Final train
    booster = xgb.train(
        params=best["params"],
        dtrain=dtrain,
        num_boost_round=best["best_iter"] + 1,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=200,
        verbose_eval=False
    )

    proba_val = booster.predict(dval)

    if hardcode_threshold is not None:
        chosen_thr = hardcode_threshold
        print(f"    Using hardcoded threshold: {chosen_thr:.4f}")
    else:
        thr_df = build_threshold_table(y_val, proba_val)
        feasible = thr_df[thr_df["recall"] >= RECALL_TARGET]
        chosen_thr = feasible.sort_values("threshold").iloc[0]["threshold"] if not feasible.empty else 0.5
        print(f"    Chosen threshold: {chosen_thr:.4f}")

    return booster, chosen_thr


def train_lightgbm(X_train, y_train, X_val, y_val, hardcode_threshold: float = None) -> Tuple:
    """Train LightGBM. Optionally use hardcoded threshold."""
    print("  Training LightGBM...")
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    sample_weight = np.where(y_train == 1, neg / pos, 1.0)

    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": neg / pos,
        "random_state": RANDOM_STATE
    }

    lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weight)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=0)
        ]
    )

    proba_val = booster.predict(X_val)

    if hardcode_threshold is not None:
        chosen_thr = hardcode_threshold
        print(f"    Using hardcoded threshold: {chosen_thr:.4f}")
    else:
        thr_df = build_threshold_table(y_val, proba_val)
        feasible = thr_df[thr_df["recall"] >= RECALL_TARGET]
        chosen_thr = feasible.sort_values("threshold").iloc[0]["threshold"] if not feasible.empty else 0.5
        print(f"    Chosen threshold: {chosen_thr:.4f}")

    return booster, chosen_thr


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

    # Build export dataframe (ordered columns per notebook format)
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


# ====================
# MAIN
# ====================

def main():
    parser = argparse.ArgumentParser(
        description="Train single model across all horizons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_single_model.py --model ngboost
  python train_single_model.py --model rf --horizons 1,2,3
  python train_single_model.py --model xgboost --output ./results
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ngboost", "rf", "xgboost", "lgbm"],
        help="Model to train"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/processed/financial_report_bank_zscore_clean.csv",
        help="Path to cleaned data CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory"
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,2,3,4,5",
        help="Horizons to train (comma-separated, e.g. '1,2,3')"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained model to pickle"
    )

    args = parser.parse_args()

    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    # Validate input
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data}")
        return

    output_dir = Path(args.output) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"TRAINING {args.model.upper()}")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Horizons: {horizons}")
    print()

    # Load data
    print("Loading data...")
    df_h, df_model = load_and_prepare_data(str(data_path))

    # Select training function
    if args.model == "ngboost":
        train_fn = train_ngboost
    elif args.model == "rf":
        train_fn = train_random_forest
    elif (args.model == "xgboost" or args.model == "xgb"):
        train_fn = train_xgboost
    elif args.model == "lgbm":
        train_fn = train_lightgbm
    else:
        print(f"Unknown model: {args.model}")
        return

    # Train for each horizon
    models_dict = {}

    # Normalize model type
    model_key = "xgb" if args.model == "xgboost" else args.model

    for horizon in horizons:
        print(f"\nHORIZON {horizon}Y")
        print("-" * 70)

        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test, df_test_meta) = split_by_horizon(df_h, horizon)

        # Train (hardcode threshold=0.4 for all models)
        model, threshold = train_fn(X_train, y_train, X_val, y_val, hardcode_threshold=0.4)

        # Evaluate & export
        evaluate_and_export(model, X_test, y_test, df_test_meta, horizon, threshold, model_key, str(output_dir))

        models_dict[horizon] = {
            "model": model,
            "threshold": threshold,
            "X_train": X_train,
            "y_train": y_train
        }

    # Optionally save models
    if args.save_model:
        model_path = output_dir / f"{args.model}_models_all_horizons.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(models_dict, f)
        print(f"\nModels saved: {model_path}")

    print("\n" + "=" * 70)
    print("✓ COMPLETE")
    print("=" * 70)
    print(f"Predictions exported to: {output_dir}/")
    print()


if __name__ == "__main__":
    main()
