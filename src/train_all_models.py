"""
Train all 4 models sequentially (complete each model before starting next).
Each model trains across all horizons (1-5 years).
Outputs predictions CSV per horizon per model.

Usage:
  python train_all_models.py
  python train_all_models.py --data ./data/processed/data.csv --output ./results
"""

import pickle
import argparse
import json
from pathlib import Path
import pandas as pd

from training_utils import (
    load_and_prepare_data,
    split_by_horizon,
    train_ngboost,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    evaluate_and_export,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train all 4 models sequentially across all horizons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_all_models.py
  python train_all_models.py --data ./data/processed/data.csv --output ./results
        """
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

    args = parser.parse_args()

    # Validate input
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BANKRUPTCY PREDICTION PIPELINE - ALL MODELS")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print()

    # Load data once
    print("Loading data...")
    df_h, df_model = load_and_prepare_data(str(data_path))

    # Models to train in sequence
    models_config = [
        ("ngboost", train_ngboost),
        ("rf", train_random_forest),
        ("xgboost", train_xgboost),
        ("lgbm", train_lightgbm),
    ]

    # Train each model sequentially
    models_store = {}

    for model_name, train_fn in models_config:
        print("\n" + "=" * 70)
        print(f"TRAINING {model_name.upper()}")
        print("=" * 70)

        # Create output directory for this model
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize model type for export
        export_key = "xgb" if model_name == "xgboost" else model_name

        # Train for each horizon
        models_dict = {}
        search_logs = []
        best_params_by_horizon = {}

        for horizon in range(1, 6):
            print(f"\nHORIZON {horizon}Y")
            print("-" * 70)

            # Split data
            (X_train, y_train), (X_val, y_val), (X_test, y_test, df_test_meta) = split_by_horizon(df_h, horizon)

            # Train (hardcode threshold=0.4 for all models)
            training_params = {
                "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
                "hardcode_threshold": 0.4, "horizon": horizon, "search_logs": search_logs
            }
            if model_name == "xgboost":
                training_params["output_dir"] = str(model_output_dir)

            result = train_fn(**training_params)
            if model_name == "xgboost":
                model, threshold, best_params = result
                best_params_by_horizon[horizon] = best_params
            else:
                model, threshold = result

            # Evaluate & export
            evaluate_and_export(model, X_test, y_test, df_test_meta, horizon, threshold, export_key, str(model_output_dir))

            models_dict[horizon] = {
                "model": model,
                "threshold": threshold,
                "X_train": X_train,
                "y_train": y_train
            }

        models_store[model_name] = models_dict

        # Save consolidated results for this model
        if model_name == "xgboost":
            json_path = model_output_dir / f"{model_name}_best_params.json"
            with open(json_path, "w") as f:
                json.dump(best_params_by_horizon, f, indent=2)
            print(f"\n  Best params (all horizons) saved: {json_path}")

        # Save search logs CSV (for XGBoost iterations)
        if search_logs:
            csv_path = model_output_dir / f"{model_name}_search_logs.csv"
            df_logs = pd.DataFrame(search_logs)
            df_logs.to_csv(csv_path, index=False)
            print(f"  Search logs saved: {csv_path}")

        print(f"\n✓ {model_name.upper()} complete!")
        print(f"  Predictions exported to: {model_output_dir}/")

    # Save all models
    models_path = output_dir / "models_all_horizons.pkl"
    with open(models_path, "wb") as f:
        pickle.dump(models_store, f)

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"All models saved to: {models_path}")
    print(f"Predictions in: {output_dir}/<model>/<model>_predictions_<horizon>y.csv")
    print()


if __name__ == "__main__":
    main()
