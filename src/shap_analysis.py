"""
SHAP explainability analysis for bankruptcy prediction models.
Generates summary plots, feature importance, dependence plots, and force plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from typing import Dict, List, Optional, Tuple

from training_utils import MODEL_TITLE_MAP


class SHAPAnalyzer:
    """SHAP analysis for tree-based and gradient boosting models."""

    MODEL_TITLE_MAP = MODEL_TITLE_MAP

    def __init__(self, sample_n: int = 200, background_n: int = 100, random_state: int = 42):
        self.sample_n = sample_n
        self.background_n = background_n
        self.random_state = random_state

    def compute_shap_values(self, model, X_train, X_sample, model_type: str, feature_cols: List[str]) -> np.ndarray:
        """Compute SHAP values for any model type."""
        if model_type == "xgb":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(xgb.DMatrix(X_sample))

        elif model_type in ["rf", "lgbm"]:
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X_sample)

            if isinstance(shap_values_raw, list):
                shap_values = shap_values_raw[1]  # class 1
            else:
                shap_values = shap_values_raw

        elif model_type == "ngboost":
            X_bg = X_train.sample(
                n=min(self.background_n, len(X_train)),
                random_state=self.random_state
            )
            explainer = shap.KernelExplainer(
                lambda X: model.predict_proba(X)[:, 1],
                X_bg,
                link="logit"
            )
            shap_values = explainer.shap_values(X_sample)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return shap_values

    def plot_summary(self, model, X_train, model_type: str, feature_cols: List[str],
                     horizon: int, plot_type: str = "dot", figsize: Tuple[int, int] = (10, 6)):
        """Plot SHAP summary plot (dot or bar)."""
        X_sample = X_train.sample(
            n=min(self.sample_n, len(X_train)),
            random_state=self.random_state
        )

        shap_values = self.compute_shap_values(model, X_train, X_sample, model_type, feature_cols)

        plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type=plot_type, show=False)
        plt.title(f"{self.MODEL_TITLE_MAP[model_type]} SHAP Summary – {horizon}Y Horizon")
        plt.tight_layout()

        return plt.gcf()

    def plot_feature_importance(self, model, X_train, model_type: str, feature_cols: List[str],
                                horizon: int, figsize: Tuple[int, int] = (10, 6)):
        """Plot SHAP feature importance (bar)."""
        X_sample = X_train.sample(
            n=min(self.sample_n, len(X_train)),
            random_state=self.random_state
        )

        shap_values = self.compute_shap_values(model, X_train, X_sample, model_type, feature_cols)

        plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance – {self.MODEL_TITLE_MAP[model_type]} ({horizon}Y Horizon)")
        plt.tight_layout()

        return plt.gcf()

    def plot_dependence(self, model, X_train, model_type: str, feature: str,
                       horizon: int, figsize: Tuple[int, int] = (8, 6)):
        """Plot SHAP dependence plot for a single feature."""
        X_sample = X_train.sample(
            n=min(self.sample_n, len(X_train)),
            random_state=self.random_state
        )

        if feature not in X_sample.columns:
            raise ValueError(f"Feature '{feature}' not found in X_sample")

        shap_values = self.compute_shap_values(model, X_train, X_sample, model_type, X_sample.columns.tolist())

        plt.figure(figsize=figsize)
        shap.dependence_plot(feature, shap_values, X_sample, show=False)
        plt.title(f"{feature} – {self.MODEL_TITLE_MAP[model_type]} SHAP Dependence ({horizon}Y)")
        plt.tight_layout()

        return plt.gcf()

    def plot_force(self, model, X_train, X_row, model_type: str, feature_cols: List[str]):
        """Plot SHAP force plot for a single instance."""
        if model_type == "xgb":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(xgb.DMatrix(X_row, feature_names=feature_cols))
            base_value = explainer.expected_value

        elif model_type in ["rf", "lgbm"]:
            explainer = shap.TreeExplainer(model)
            shap_values_all = explainer.shap_values(X_row)

            if isinstance(shap_values_all, list):
                shap_values = shap_values_all[1]
                base_value = explainer.expected_value[1]
            else:
                shap_values = shap_values_all
                base_value = explainer.expected_value

        elif model_type == "ngboost":
            X_bg = X_train.sample(
                n=min(self.background_n, len(X_train)),
                random_state=self.random_state
            )
            explainer = shap.KernelExplainer(
                lambda X: model.predict_proba(X)[:, 1],
                X_bg,
                link="logit"
            )
            shap_values = explainer.shap_values(X_row)
            base_value = explainer.expected_value

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return shap.force_plot(base_value, shap_values[0], X_row.iloc[0], feature_names=feature_cols)

    def get_feature_importance_df(self, model, X_train, model_type: str, feature_cols: List[str]) -> pd.DataFrame:
        """Get SHAP-based feature importance as DataFrame."""
        X_sample = X_train.sample(
            n=min(self.sample_n, len(X_train)),
            random_state=self.random_state
        )

        shap_values = self.compute_shap_values(model, X_train, X_sample, model_type, feature_cols)

        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)

        df_importance = pd.DataFrame({
            "feature": feature_cols,
            "shap_importance": importance
        }).sort_values("shap_importance", ascending=False)

        return df_importance


def analyze_all_horizons(models_dict: Dict, X_trains: Dict, model_type: str,
                        feature_cols: List[str], output_dir: str = "./output"):
    """Run SHAP analysis for all horizons of a single model type."""

    analyzer = SHAPAnalyzer()

    for horizon in sorted(models_dict.keys()):
        print(f"\nGenerating SHAP analysis for {model_type.upper()} {horizon}Y...")

        model = models_dict[horizon]["model"]
        X_train = X_trains[horizon]

        # Summary plot
        fig = analyzer.plot_summary(model, X_train, model_type, feature_cols, horizon, plot_type="dot")
        fig.savefig(f"{output_dir}/{model_type}/{model_type}_shap_summary_{horizon}y.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Feature importance
        fig = analyzer.plot_feature_importance(model, X_train, model_type, feature_cols, horizon)
        fig.savefig(f"{output_dir}/{model_type}/{model_type}_shap_importance_{horizon}y.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Dependence plots for top features
        importance_df = analyzer.get_feature_importance_df(model, X_train, model_type, feature_cols)
        top_features = importance_df.head(6)["feature"].tolist()

        for feat in top_features:
            try:
                fig = analyzer.plot_dependence(model, X_train, model_type, feat, horizon)
                fig.savefig(f"{output_dir}/{model_type}/{model_type}_dependence_{feat}_{horizon}y.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: Could not generate dependence plot for {feat}: {e}")

        # Save importance table
        importance_df.to_csv(f"{output_dir}/{model_type}/{model_type}_feature_importance_{horizon}y.csv", index=False)

    print(f"✓ SHAP analysis complete for {model_type.upper()}")
