"""Train and persist flight delay classification and regression models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    add_synthetic_operational_features,
    engineer_features,
)
from preprocess import preprocess_flight_data
from utils import RANDOM_STATE, configure_logging, ensure_saved_models_dir, load_flights_dataset


@dataclass
class ModelResult:
    """Container for a fitted model and its validation metrics."""

    name: str
    model: Any
    score: float
    metrics: dict[str, Any]


def _one_hot_encoder() -> OneHotEncoder:
    """Create a version-tolerant OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_encoder() -> ColumnTransformer:
    """Build the preprocessing encoder used by both models."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )


def _candidate_classifiers() -> dict[str, Any]:
    models: dict[str, Any] = {
        "CalibratedRandomForestClassifier": CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            method="sigmoid",
            cv=3,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "CalibratedGradientBoostingClassifier": CalibratedClassifierCV(
            estimator=GradientBoostingClassifier(
                n_estimators=120,
                max_depth=2,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            ),
            method="sigmoid",
            cv=3,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoostClassifier"] = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    except Exception as exc:
        logging.info("XGBoostClassifier unavailable: %s", exc)
    return models


def _candidate_regressors() -> dict[str, Any]:
    models: dict[str, Any] = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoostRegressor"] = XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
        )
    except Exception as exc:
        logging.info("XGBoostRegressor unavailable: %s", exc)
    return models


def _classification_metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else predictions
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities) if y_test.nunique() > 1 else np.nan,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def _regression_metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": float(np.sqrt(mse)),
        "r2_score": r2_score(y_test, predictions),
    }


def select_best_classifier(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> ModelResult:
    """Train candidate classifiers and return the best by validation accuracy."""
    best: ModelResult | None = None
    for name, estimator in _candidate_classifiers().items():
        model = Pipeline(steps=[("encoder", build_encoder()), ("model", estimator)])
        model.fit(x_train, y_train)
        metrics = _classification_metrics(model, x_test, y_test)
        score = float(metrics["accuracy"])
        logging.info("Classifier %s metrics: %s", name, metrics)
        if best is None or score > best.score:
            best = ModelResult(name=name, model=model, score=score, metrics=metrics)
    if best is None:
        raise RuntimeError("No classifier candidates were available.")
    return best


def select_best_regressor(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> ModelResult:
    """Train candidate regressors and return the best by validation RMSE."""
    best: ModelResult | None = None
    for name, estimator in _candidate_regressors().items():
        model = Pipeline(steps=[("encoder", build_encoder()), ("model", estimator)])
        model.fit(x_train, y_train)
        metrics = _regression_metrics(model, x_test, y_test)
        score = float(metrics["rmse"])
        logging.info("Regressor %s metrics: %s", name, metrics)
        if best is None or score < best.score:
            best = ModelResult(name=name, model=model, score=score, metrics=metrics)
    if best is None:
        raise RuntimeError("No regressor candidates were available.")
    return best


def train_and_save_models() -> dict[str, Any]:
    """Run the full training workflow and save reusable artifacts."""
    configure_logging()
    try:
        raw_df = load_flights_dataset()
        clean_df = preprocess_flight_data(raw_df)
        target_df = add_synthetic_operational_features(clean_df)
        engineered_df, route_density_map = engineer_features(target_df, fit_density=True)

        x = engineered_df[FEATURE_COLUMNS]
        y_class = engineered_df["Is_Delayed"].astype(int)
        y_reg = engineered_df["Delay_Duration"].astype(float)

        x_train, x_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            x,
            y_class,
            y_reg,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_class if y_class.nunique() > 1 else None,
        )

        best_classifier = select_best_classifier(x_train, x_test, y_class_train, y_class_test)
        best_regressor = select_best_regressor(x_train, x_test, y_reg_train, y_reg_test)

        output_dir = ensure_saved_models_dir()
        encoder = best_classifier.model.named_steps["encoder"]
        pipeline_bundle = {
            "classifier": best_classifier.model,
            "regressor": best_regressor.model,
            "route_density_map": route_density_map,
            "feature_columns": FEATURE_COLUMNS,
            "classifier_metrics": best_classifier.metrics,
            "regressor_metrics": best_regressor.metrics,
            "best_classifier_name": best_classifier.name,
            "best_regressor_name": best_regressor.name,
        }

        artifacts = {
            "classifier": output_dir / "best_classifier.pkl",
            "regressor": output_dir / "best_regressor.pkl",
            "encoder": output_dir / "encoder.pkl",
            "pipeline": output_dir / "pipeline.pkl",
        }
        joblib.dump(best_classifier.model, artifacts["classifier"])
        joblib.dump(best_regressor.model, artifacts["regressor"])
        joblib.dump(encoder, artifacts["encoder"])
        joblib.dump(pipeline_bundle, artifacts["pipeline"])

        logging.info("Best classifier: %s", best_classifier.name)
        logging.info("Best regressor: %s", best_regressor.name)
        logging.info("Saved artifacts to: %s", output_dir.resolve())

        return {
            "artifacts": {key: str(path) for key, path in artifacts.items()},
            "best_classifier": best_classifier.name,
            "classifier_metrics": best_classifier.metrics,
            "best_regressor": best_regressor.name,
            "regressor_metrics": best_regressor.metrics,
        }
    except Exception:
        logging.exception("Training failed.")
        raise


if __name__ == "__main__":
    results = train_and_save_models()
    print("Training complete.")
    print(results)
