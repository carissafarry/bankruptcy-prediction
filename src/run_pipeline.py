#!/usr/bin/env python3
"""
Main execution script for the bankruptcy prediction pipeline.
Orchestrates training and optional SHAP analysis.
"""

import sys
import argparse
import pickle
from pathlib import Path

from training_utils import (
    load_and_prepare_data,
    split_by_horizon,
    train_ngboost,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    FEATURE_COLS,
)
from shap_analysis import analyze_all_horizons


def train_all_models(data_path: str, output_dir: str):
    """Train all 4 models sequentially."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_h, df_model = load_and_prepare_data(data_path)

    models_config = [
        ("ngboost", train_ngboost),
        ("rf", train_random_forest),
        ("xgboost", train_xgboost),
        ("lgbm", train_lightgbm),
    ]

    models_store = {}

    for model_name, train_fn in models_config:
        print(f"\nTraining {model_name.upper()}...")
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        export_key = "xgb" if model_name == "xgboost" else model_name

        models_dict = {}
        for horizon in range(1, 6):
            (X_train, y_train), (X_val, y_val), (X_test, y_test, df_test_meta) = split_by_horizon(df_h, horizon)
            result = train_fn(X_train, y_train, X_val, y_val, hardcode_threshold=0.4)
            if model_name == "xgboost":
                model, threshold, best_params = result
            else:
                model, threshold = result

            from training_utils import evaluate_and_export
            evaluate_and_export(model, X_test, y_test, df_test_meta, horizon, threshold, export_key, str(model_output_dir))

            models_dict[horizon] = {
                "model": model,
                "threshold": threshold,
                "X_train": X_train,
                "y_train": y_train
            }

        models_store[model_name] = models_dict

    # Save models
    models_path = output_dir / "models_all_horizons.pkl"
    with open(models_path, "wb") as f:
        pickle.dump(models_store, f)

    return models_store, df_h


def main():
    parser = argparse.ArgumentParser(
        description="Bankruptcy Prediction Pipeline"
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
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP analysis (faster for testing)"
    )

    args = parser.parse_args()

    # Validate input
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BANKRUPTCY PREDICTION PIPELINE")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print()

    # ========== PHASE 1: TRAINING ==========
    print("PHASE 1: Training all models (NGBoost, RF, XGBoost, LightGBM)")
    print("-" * 70)

    models_store, df_h = train_all_models(str(data_path), str(output_dir))

    print("\n✓ Training complete!")
    print(f"  Models saved to: {output_dir}/models_all_horizons.pkl")
    print(f"  Predictions exported to: {output_dir}/<model>/<model>_predictions_<horizon>y.csv")

    # ========== PHASE 2: SHAP ANALYSIS ==========
    if not args.skip_shap:
        print("\n" + "=" * 70)
        print("PHASE 2: SHAP Explainability Analysis")
        print("-" * 70)

        for model_type in ["ngboost", "rf", "xgboost", "lgbm"]:
            X_trains = {}
            models_dict = {}

            for horizon in range(1, 6):
                models_dict[horizon] = models_store[model_type][horizon]["model"]
                X_trains[horizon] = models_store[model_type][horizon]["X_train"]

            analyze_all_horizons(models_dict, X_trains, model_type, FEATURE_COLS, str(output_dir))

        print("\n✓ SHAP analysis complete!")
        print(f"  Plots saved to: {output_dir}/<model>/")
        print(f"  Feature importance CSVs saved to: {output_dir}/<model>/<model>_feature_importance_<horizon>y.csv")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE ✓")
    print("=" * 70)
    print("\nOutput Files:")
    print("  - Trained models: output/models_all_horizons.pkl")
    print("  - Predictions (5 horizons × 4 models): output/<model>/<model>_predictions_<horizon>y.csv")
    if not args.skip_shap:
        print("  - SHAP plots: output/<model>/<model>_shap_summary|importance|dependence_<feature>_<horizon>y.png")
        print("  - Feature importance: output/<model>/<model>_feature_importance_<horizon>y.csv")
    print()


if __name__ == "__main__":
    main()
