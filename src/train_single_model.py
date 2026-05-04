"""
Train a single model across all horizons with flexible model selection.
Outputs predictions CSV per horizon.

Usage:
  python train_single_model.py --model ngboost
  python train_single_model.py --model rf --data ./data/processed/data.csv --output ./results
  python train_single_model.py --model xgboost --horizons 1,2,3
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

    # Normalize model type for consistent key mapping
    model_key = "xgb" if args.model == "xgboost" else args.model

    # Train for each horizon
    models_dict = {}
    search_logs = []
    best_params_by_horizon = {}

    for horizon in horizons:
        print(f"\nHORIZON {horizon}Y")
        print("-" * 70)

        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test, df_test_meta) = split_by_horizon(df_h, horizon)

        # Train (hardcode threshold=0.4 for all models)
        # Pass horizon for all models, output_dir for XGBoost param logging
        training_params = {
            "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
            "hardcode_threshold": 0.4, "horizon": horizon, "search_logs": search_logs
        }
        if args.model == "xgboost":
            training_params["output_dir"] = str(output_dir)

        result = train_fn(**training_params)
        if args.model == "xgboost":
            model, threshold, best_params = result
            best_params_by_horizon[horizon] = best_params
        else:
            model, threshold = result

        # Evaluate & export
        evaluate_and_export(model, X_test, y_test, df_test_meta, horizon, threshold, model_key, str(output_dir))

        models_dict[horizon] = {
            "model": model,
            "threshold": threshold,
            "X_train": X_train,
            "y_train": y_train
        }

    # Save xgboost best params consolidated results
    if args.model == "xgboost":
        # Consolidated JSON with all horizons
        json_path = output_dir / f"{args.model}_best_params.json"
        with open(json_path, "w") as f:
            json.dump(best_params_by_horizon, f, indent=2)
        print(f"\nBest params (all horizons) saved: {json_path}")

    # Save search logs CSV (for XGBoost with iterations)
    if search_logs:
        csv_path = output_dir / f"{args.model}_search_logs.csv"
        df_logs = pd.DataFrame(search_logs)
        df_logs.to_csv(csv_path, index=False)
        print(f"Search logs saved: {csv_path}")

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
