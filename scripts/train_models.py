"""Train legacy MLB models and write validated model manifests."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.features import build_model_features
from src.models.manifest import FeatureSpec, ModelManifest, sha256_file, write_manifest
from src.models.spread_model import MODEL_PATH as SPREAD_MODEL_PATH
from src.models.spread_model import train_spread_model
from src.models.totals_model import train_totals_model
from src.models.underdog_model import MODEL_PATH as MONEYLINE_MODEL_PATH
from src.models.underdog_model import train_moneyline_model

TOTALS_MODEL_PATH = ROOT / "models" / "totals_xgb_v1.joblib"
LOCK_PATH = ROOT / "uv.lock"
FEATURE_SET_VERSION = "mlb_game_v2"
DATA_SCHEMA_VERSION = "2.0.0"
MODEL_VERSION = "1.0.0"


def _write_legacy_manifest(
    *,
    result: dict,
    features: pd.DataFrame,
    artifact_path: Path,
    model_name: str,
    market_id: str,
) -> None:
    model = result["model"]
    feature_columns = list(result["feature_cols"])

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)

    feature_frame = features.loc[:, feature_columns].dropna()
    dates = pd.to_datetime(features.loc[feature_frame.index, "date"], errors="coerce")

    manifest = ModelManifest(
        model_run_id=artifact_path.stem,
        model_name=model_name,
        model_version=MODEL_VERSION,
        market_id=market_id,
        supported_snapshot_types=("morning", "confirmed_lineup", "pregame_30m"),
        feature_set_version=FEATURE_SET_VERSION,
        data_schema_version=DATA_SCHEMA_VERSION,
        python_version=platform.python_version(),
        sklearn_version=sklearn.__version__,
        dependency_lock_sha256=sha256_file(LOCK_PATH),
        dependency_versions={
            "numpy": __import__("numpy").__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        artifact_sha256=sha256_file(artifact_path),
        source_commit=os.getenv("GITHUB_SHA", "local"),
        random_seed=42,
        validation_definition={
            "kind": "legacy_chronological_holdout",
            "test_fraction": 0.20,
            "metrics_split": "most_recent_games",
        },
        training_start=str(dates.min().date()),
        training_end=str(dates.max().date()),
        metrics=result["metrics"],
        features=tuple(
            FeatureSpec(
                name=column,
                dtype=str(feature_frame[column].dtype),
                nullable=bool(feature_frame[column].isna().any()),
            )
            for column in feature_columns
        ),
    )

    manifest_path = artifact_path.with_suffix(".manifest.json")
    write_manifest(manifest_path, manifest)
    print(f"  -> bundle: {artifact_path} + {manifest_path}")


def main(start_year: int = 2020, end_year: int | None = None) -> None:
    if end_year is None:
        end_year = datetime.utcnow().year

    if not LOCK_PATH.is_file():
        raise FileNotFoundError(f"Required dependency lock is missing: {LOCK_PATH}")

    print(f"Building feature matrix for {start_year}-{end_year}…")
    features = build_model_features(start_year, end_year)

    print("Training moneyline model…")
    moneyline = train_moneyline_model(features)
    print(f"  -> ROC-AUC {moneyline['metrics']['roc_auc']:.4f}")
    _write_legacy_manifest(
        result=moneyline,
        features=features,
        artifact_path=MONEYLINE_MODEL_PATH,
        model_name="xgboost_moneyline_home_win",
        market_id="moneyline_full_game",
    )

    print("Training spread model…")
    spread = train_spread_model(features)
    print(f"  -> ROC-AUC {spread['metrics']['roc_auc']:.4f}")
    _write_legacy_manifest(
        result=spread,
        features=features,
        artifact_path=SPREAD_MODEL_PATH,
        model_name="xgboost_run_line_home_cover",
        market_id="run_line_full_game",
    )

    print("Training totals model…")
    totals = train_totals_model(features)
    print(f"  -> ROC-AUC {totals['metrics']['roc_auc']:.4f}")
    _write_legacy_manifest(
        result=totals,
        features=features,
        artifact_path=TOTALS_MODEL_PATH,
        model_name="xgboost_total_over",
        market_id="total_full_game",
    )

    print(f"Training complete. Validated bundles written to {ROOT / 'models'}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train all MLB betting models")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()
    main(start_year=args.start_year, end_year=args.end_year)