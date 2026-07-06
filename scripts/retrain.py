#!/usr/bin/env python3
"""
Nightly retraining pipeline with MLflow tracking and validation gates.

Steps:
  1. Download latest InsideAirbnb NYC listings snapshot (fresh actual prices)
  2. Run through the production cleaning + feature engineering pipeline
  3. Train XGBoost with the same hyperparameters as the production model
  4. Evaluate on held-out test set
  5. Validation gate:
       new R²  >= baseline R²  - 0.01
       new MAE <= baseline MAE + $5.00
  6. If gates pass  → export to ONNX, register in MLflow, promote to Production
  7. If gates fail  → keep current model, register as Staging only, raise alert

WHY fresh data instead of prediction logs?
  Prediction logs contain *predicted* prices, not *actual* prices. Retraining on
  predicted values creates a feedback loop — the model learns its own bias and
  collapses toward its own mean (model collapse). InsideAirbnb provides actual host
  listing prices for every scrape, which is the correct supervision signal.

Usage:
    python scripts/retrain.py                         # download fresh data + retrain
    python scripts/retrain.py --no-download           # use existing engineered_features.csv
    python scripts/retrain.py --force                 # skip validation gate
    python scripts/retrain.py --rollback <version>    # rollback to MLflow version N

MLflow UI:
    mlflow ui --port 5000
    open http://localhost:5000
"""

import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.new_york_workflow.nyc_data_validator import (
    validate_raw_snapshot,
    validate_clean_listings,
    validate_engineered_features,
    DataQualityError,
)

BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / "data" / "airbnb"
MODEL_DIR  = BASE_DIR / "models" / "nyc"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}")
MODEL_NAME = "nyc-airbnb-xgboost"

TARGET      = "log_price"
DROP_COLS   = ["id", "price", "log_price", "neighbourhood_cleansed"]
RANDOM_SEED = 42
TEST_SIZE   = 0.20
PRICE_CAP   = 0.99

INSIDEAIRBNB_BASE = "https://data.insideairbnb.com/united-states/ny/new-york-city"

# Fixed fallback params used when --skip-optuna is set or Optuna is unavailable.
XGB_PARAMS = {
    "n_estimators":     800,
    "max_depth":        8,
    "learning_rate":    0.03,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_alpha":        0.1,
    "reg_lambda":       10.0,
    "gamma":            0,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "verbosity":        0,
}

OPTUNA_N_TRIALS = 20    # ~3-5 min on GH Actions ubuntu-latest; raise for quality, lower for speed
OPTUNA_CV_FOLDS = 3     # cross-validation folds for objective — faster than full train+eval


# ── InsideAirbnb download ──────────────────────────────────────────────────────

def _find_via_index() -> str | None:
    """
    Fetch the InsideAirbnb S3 bucket index and extract the most recent
    NYC listings URL.  Returns None on any network / parse error.
    """
    import xml.etree.ElementTree as ET

    try:
        with urllib.request.urlopen(f"{INSIDEAIRBNB_BASE}/", timeout=20) as r:
            tree = ET.fromstring(r.read())
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        keys = [
            k.text for k in tree.findall(".//s3:Key", ns)
            if k.text and k.text.endswith("data/listings.csv.gz")
        ]
        if keys:
            latest_key = sorted(keys)[-1]   # lexicographic sort — ISO dates sort correctly
            url = f"{INSIDEAIRBNB_BASE}/{latest_key}"
            logger.info("InsideAirbnb index → latest snapshot: %s", url)
            return url
    except Exception as exc:
        logger.debug("Index fetch failed (%s) — falling back to date probing", exc)
    return None


def _find_via_date_probe() -> str | None:
    """
    Probe HEAD requests for the first ~7 days of each of the past 12 months.
    InsideAirbnb NYC snapshots cluster around the 2nd–7th of each month.
    Returns the first URL that responds HTTP 200.
    """
    today = date.today()
    for months_back in range(0, 12):
        raw_month = today.month - months_back
        year  = today.year + (raw_month - 1) // 12
        month = ((raw_month - 1) % 12) + 1
        for day in (4, 3, 5, 2, 6, 7, 1):
            try:
                snap_date = date(year, month, day).isoformat()
            except ValueError:
                continue
            url = f"{INSIDEAIRBNB_BASE}/{snap_date}/data/listings.csv.gz"
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=8) as r:
                    if r.status == 200:
                        logger.info("Found InsideAirbnb snapshot via probe: %s", url)
                        return url
            except Exception:
                pass
    return None


def download_fresh_data() -> Path:
    """
    Download the latest InsideAirbnb NYC listings.csv.gz.

    Strategy:
      1. Parse the S3 bucket index for the newest key.
      2. Fall back to HEAD-probing recent monthly dates.
      3. If both fail, raise RuntimeError (caller can handle with --no-download).

    The file is cached in data/airbnb/raw_snapshots/ — re-running on the same
    day re-uses the cached file without hitting the network again.

    Returns:
        Path to the downloaded (gzipped) CSV file.
    """
    raw_dir = DATA_DIR / "raw_snapshots"
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = _find_via_index() or _find_via_date_probe()
    if url is None:
        raise RuntimeError(
            "Could not locate a valid InsideAirbnb NYC snapshot URL.\n"
            "Check network connectivity, or use --no-download to retrain "
            "on the existing engineered_features.csv."
        )

    # Extract the ISO date from the URL path for a stable filename
    url_parts = [p for p in url.split("/") if len(p) == 10 and p.count("-") == 2]
    date_str  = url_parts[-1] if url_parts else datetime.now().strftime("%Y-%m-%d")
    out_path  = raw_dir / f"listings_raw_{date_str}.csv.gz"

    if out_path.exists():
        logger.info("Re-using cached snapshot: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
        return out_path

    logger.info("Downloading %s …", url)
    urllib.request.urlretrieve(url, out_path)
    logger.info("Saved %.1f MB → %s", out_path.stat().st_size / 1e6, out_path)
    return out_path


# ── production pipeline (cleaning + feature engineering) ──────────────────────

def _load_module(alias: str, rel_path: str):
    """
    Import a source file as a module using its filesystem path.
    Needed because the pipeline files start with digits (1_, 2_) and cannot
    be imported via the normal `import` statement.
    """
    path = BASE_DIR / rel_path
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean_and_engineer(raw_path: Path) -> pd.DataFrame:
    """
    Run a raw InsideAirbnb listings CSV through the production pipeline:
      1_data_cleaning.py   → select columns, parse price, encode categoricals, drop nulls
      2_feature_engineering.py → log transforms, borough dummies, polynomial/interaction/ratio features

    The pipeline functions are loaded dynamically so we always use the exact
    same code as the original training, preventing train/retrain skew.

    Args:
        raw_path: Path to the raw listings CSV (may be .gz — pandas reads it natively).

    Returns:
        Engineered DataFrame with the same column set as engineered_features.csv.
    """
    c = _load_module("cleaning",     "src/new_york_workflow/1_data_cleaning.py")
    e = _load_module("engineering",  "src/new_york_workflow/2_feature_engineering.py")

    # ── Step 1: clean ─────────────────────────────────────────────────────────
    logger.info("=== CLEANING  %s ===", raw_path.name)
    df_raw = pd.read_csv(raw_path)
    logger.info("  Raw input:  %d rows × %d cols", *df_raw.shape)

    # Gate 1: validate raw snapshot before touching any data
    validate_raw_snapshot(df_raw).raise_if_failed()

    available = [col for col in c.KEEP_FEATURES if col in df_raw.columns]
    missing   = set(c.KEEP_FEATURES) - set(available)
    if missing:
        logger.warning("  Columns absent from raw CSV (skipped): %s", sorted(missing))

    df = c.select_features(df_raw, available)
    df = c.parse_price(df)
    df = c.encode_categorical(df)
    df = c.handle_missing_values(df)
    logger.info("  After cleaning: %d rows × %d cols", *df.shape)

    # Gate 2: validate cleaned data before feature engineering
    validate_clean_listings(df).raise_if_failed()

    # ── Step 2: feature engineer ───────────────────────────────────────────────
    logger.info("=== FEATURE ENGINEERING ===")
    df, _ = e.apply_log_transforms(df)
    df, _ = e.encode_neighbourhood(df)
    df, _ = e.create_polynomial_features(df)
    df, _ = e.create_interaction_features(df)
    df, _ = e.create_ratio_features(df)

    redundant = [col for col in e.DROP_REDUNDANT_COLS if col in df.columns]
    df = df.drop(columns=redundant)
    logger.info("  After engineering: %d rows × %d cols", *df.shape)
    return df


# ── training data loader ───────────────────────────────────────────────────────

def load_training_data(no_download: bool = False) -> pd.DataFrame:
    """
    Load the training dataset.

    With no_download=False (default): download the latest InsideAirbnb NYC
    snapshot and run it through the full cleaning + feature engineering pipeline.

    With no_download=True: read the existing engineered_features.csv on disk
    (same fallback behavior as the original retrain.py).

    In both cases, the top 1% of prices is capped to match the original
    training setup.
    """
    if no_download:
        path = DATA_DIR / "engineered_features.csv"
        logger.info("--no-download: using existing %s", path)
        df = pd.read_csv(path)
    else:
        raw_path = download_fresh_data()
        df = clean_and_engineer(raw_path)

    cap = df["price"].quantile(PRICE_CAP)
    df  = df[df["price"] <= cap].copy()
    logger.info("  %d rows after %.0f%% price cap ($%.0f)", len(df), PRICE_CAP * 100, cap)

    # Gate 3: validate engineered features before training
    validate_engineered_features(df).raise_if_failed()

    return df


def prepare_xy(df: pd.DataFrame, feature_list: list[str]):
    X = df[feature_list].fillna(0)
    y = df[TARGET]
    return X, y


# ── train ──────────────────────────────────────────────────────────────────────

def _tune_hyperparams(X_train: np.ndarray, y_train: np.ndarray, n_trials: int = OPTUNA_N_TRIALS) -> dict:
    """
    Run a 20-trial Optuna TPE study over XGBoost hyperparameters.
    Objective: mean CV MAE on log_price (lower = better).
    Returns the best param dict, including fixed params not tuned (n_jobs, random_state, verbosity).
    """
    import optuna
    from sklearn.model_selection import cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int(  "n_estimators",     300,  1500, step=100),
            "max_depth":        trial.suggest_int(  "max_depth",         3,    10),
            "learning_rate":    trial.suggest_float("learning_rate",     0.01, 0.2,  log=True),
            "min_child_weight": trial.suggest_int(  "min_child_weight",  1,    10),
            "subsample":        trial.suggest_float("subsample",         0.5,  1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",  0.5,  1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha",         0.0,  1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda",        1.0,  20.0),
            "gamma":            0,
            "random_state":     RANDOM_SEED,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        model    = XGBRegressor(**params)
        neg_maes = cross_val_score(model, X_train, y_train, cv=OPTUNA_CV_FOLDS,
                                   scoring="neg_mean_absolute_error", n_jobs=1)
        return float(-neg_maes.mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"gamma": 0, "random_state": RANDOM_SEED, "n_jobs": -1, "verbosity": 0})
    logger.info(
        "Optuna best: n_estimators=%d  depth=%d  lr=%.4f  MAE=%.4f (cv, log-scale)",
        best["n_estimators"], best["max_depth"], best["learning_rate"], study.best_value,
    )
    return best


def train(X_train: np.ndarray, y_train: np.ndarray, params: dict | None = None) -> XGBRegressor:
    """Train XGBoost with provided params, or fall back to fixed XGB_PARAMS."""
    model = XGBRegressor(**(params or XGB_PARAMS))
    model.fit(X_train, y_train, verbose=False)
    return model


# ── evaluate ───────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, df_test: pd.DataFrame) -> dict:
    y_pred_log = model.predict(X_test)
    y_pred_usd = np.expm1(y_pred_log)
    y_true_usd = df_test["price"].values

    r2   = float(r2_score(y_test, y_pred_log))
    mae  = float(mean_absolute_error(y_true_usd, y_pred_usd))
    mape = float(np.mean(np.abs((y_true_usd - y_pred_usd) / np.maximum(y_true_usd, 1))) * 100)

    return {"r2": round(r2, 4), "mae_dollar": round(mae, 2), "mape_pct": round(mape, 2)}


# ── validation gate ────────────────────────────────────────────────────────────

def check_gates(new_metrics: dict, baseline: dict, force: bool = False) -> tuple[bool, list[str]]:
    """Returns (passed, list_of_failure_reasons)."""
    if force:
        logger.warning("--force: skipping validation gate")
        return True, []

    reasons = []
    bm = baseline["model_metrics"]
    r2_threshold  = bm["r2"]        - 0.01
    mae_threshold = bm["mae_dollar"] + 5.0

    if new_metrics["r2"] < r2_threshold:
        reasons.append(f"R² {new_metrics['r2']:.4f} < threshold {r2_threshold:.4f}")
    if new_metrics["mae_dollar"] > mae_threshold:
        reasons.append(f"MAE ${new_metrics['mae_dollar']:.2f} > threshold ${mae_threshold:.2f}")

    return len(reasons) == 0, reasons


# ── ONNX export ────────────────────────────────────────────────────────────────

def export_onnx(model: XGBRegressor, feature_list: list[str], out_path: Path) -> None:
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat

    initial_type = [("float_input", OnnxFloat([None, len(feature_list)]))]
    onnx_model   = convert_xgboost(model, initial_types=initial_type)
    out_path.write_bytes(onnx_model.SerializeToString())
    logger.info("ONNX exported: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


# ── MLflow ─────────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def register_with_mlflow(
    model,
    metrics: dict,
    feature_list: list[str],
    scaler,
    stage: str,       # "Production" or "Staging"
    n_samples: int = 0,
    data_date: str = "",
    tuned_params: dict | None = None,
) -> int:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("nyc-airbnb-retraining")

    onnx_path = (
        MODEL_DIR / "nyc_xgb_model.onnx" if stage == "Production"
        else MODEL_DIR / "challenger.onnx"
    )

    with mlflow.start_run(run_name=f"retrain-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        # ── hyperparams (tuned > fixed fallback) ─────────────────────────────
        effective_params = tuned_params if tuned_params else XGB_PARAMS
        mlflow.log_params({k: str(v) for k, v in effective_params.items()})
        mlflow.log_params({
            "n_features":         len(feature_list),
            "n_samples":          n_samples,
            "data_date":          data_date or str(date.today()),
            "gate_stage":         stage,
            "optuna_n_trials":    OPTUNA_N_TRIALS if tuned_params else 0,
            "optuna_skipped":     str(not bool(tuned_params)),
        })

        # ── metrics ───────────────────────────────────────────────────────────
        mlflow.log_metrics(metrics)
        mlflow.log_metric("gate_passed", 1 if stage == "Production" else 0)

        # ── tags for easy filtering in the UI ─────────────────────────────────
        mlflow.set_tags({
            "git_sha":    _git_sha(),
            "pipeline":   "nightly-retrain",
            "stage":      stage,
            "data_date":  data_date or str(date.today()),
        })

        # ── artifacts ─────────────────────────────────────────────────────────
        mlflow.log_artifact(str(MODEL_DIR / "baseline_stats.json"), artifact_path="baseline")
        if onnx_path.exists():
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")

        # ── register XGBoost model in Model Registry ──────────────────────────
        mlflow.xgboost.log_model(
            model, artifact_path="model", registered_model_name=MODEL_NAME
        )

        run_id = run.info.run_id

    client  = MlflowClient(tracking_uri=MLFLOW_URI)
    version = client.get_latest_versions(MODEL_NAME)[0].version

    client.set_model_version_tag(MODEL_NAME, version, "metrics_r2",  str(metrics["r2"]))
    client.set_model_version_tag(MODEL_NAME, version, "metrics_mae", str(metrics["mae_dollar"]))
    client.set_model_version_tag(MODEL_NAME, version, "gate_stage",  stage)
    client.set_model_version_tag(MODEL_NAME, version, "run_id",      run_id)
    client.set_model_version_tag(MODEL_NAME, version, "git_sha",     _git_sha())

    if stage == "Production":
        # Archive current champion as "previous" before promoting new one
        try:
            prev = client.get_model_version_by_alias(MODEL_NAME, "champion")
            client.set_registered_model_alias(MODEL_NAME, "previous", prev.version)
            client.delete_registered_model_alias(MODEL_NAME, "champion")
        except Exception:
            pass
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        logger.info("Promoted v%s to Production (alias: champion)  run_id=%s", version, run_id)
    else:
        try:
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
        except Exception:
            pass
        client.set_registered_model_alias(MODEL_NAME, "challenger", version)
        logger.info("Registered v%s as Staging (alias: challenger)  run_id=%s", version, run_id)

    return int(version)


def rollback(version: int) -> None:
    """Re-promote a previous MLflow version to Production."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    client.set_registered_model_alias(MODEL_NAME, "champion", str(version))
    logger.info("Rolled back to version %d (alias: champion)", version)

    model_uri = f"models:/{MODEL_NAME}/{version}"
    xgb = mlflow.xgboost.load_model(model_uri)
    with open(MODEL_DIR / "nyc_xgb_model.pkl", "wb") as f:
        pickle.dump(xgb, f)
    feature_cols = json.loads((MODEL_DIR / "baseline_stats.json").read_text()).get("feature_cols", [])
    export_onnx(xgb, feature_cols, MODEL_DIR / "nyc_xgb_model.onnx")
    logger.info("Model files restored from MLflow version %d", version)


# ── main ───────────────────────────────────────────────────────────────────────

def main(force: bool = False, rollback_version: int | None = None, no_download: bool = False, skip_optuna: bool = False):
    if rollback_version is not None:
        rollback(rollback_version)
        return

    # Load baseline for gate check
    baseline_path = MODEL_DIR / "baseline_stats.json"
    if not baseline_path.exists():
        logger.error("baseline_stats.json missing — run scripts/register_baseline.py first")
        sys.exit(1)
    baseline = json.loads(baseline_path.read_text())

    # Feature list from original training report
    report       = json.loads((MODEL_DIR / "nyc_training_report.json").read_text())
    feature_list = report["feature_cols"]

    # Load + split
    df = load_training_data(no_download=no_download)
    X, y = prepare_xy(df, feature_list)
    X_train, X_test, y_train, y_test, _, df_test = train_test_split(
        X, y, df, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    if skip_optuna:
        logger.info("Optuna skipped (--skip-optuna). Using fixed XGB_PARAMS.")
        tuned_params = None
    else:
        logger.info("Optuna: running %d-trial hyperparameter search…", OPTUNA_N_TRIALS)
        tuned_params = _tune_hyperparams(X_train_s, y_train)

    logger.info("Training XGBoost (%d features, %d train rows)…", len(feature_list), len(X_train))
    model = train(X_train_s, y_train, params=tuned_params)

    metrics = evaluate(model, X_test_s, y_test, df_test)
    logger.info("New model:  R²=%.4f  MAE=$%.2f  MAPE=%.1f%%",
                metrics["r2"], metrics["mae_dollar"], metrics["mape_pct"])
    logger.info("Baseline:   R²=%.4f  MAE=$%.2f  MAPE=%.1f%%",
                baseline["model_metrics"]["r2"],
                baseline["model_metrics"]["mae_dollar"],
                baseline["model_metrics"]["mape_pct"])

    passed, reasons = check_gates(metrics, baseline, force=force)

    n_samples  = len(df)
    data_date  = str(date.today())

    if passed:
        logger.info("✓ Validation gate PASSED — promoting to Production")
        with open(MODEL_DIR / "nyc_xgb_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(MODEL_DIR / "nyc_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        export_onnx(model, feature_list, MODEL_DIR / "nyc_xgb_model.onnx")
        version = register_with_mlflow(
            model, metrics, feature_list, scaler,
            stage="Production", n_samples=n_samples, data_date=data_date,
            tuned_params=tuned_params,
        )

        # Update baseline metrics so the next gate is relative to this run
        baseline["model_metrics"] = metrics
        baseline["registered_at"] = datetime.now(timezone.utc).isoformat()
        baseline_path.write_text(json.dumps(baseline, indent=2))

        logger.info("✓ Retraining complete — version %d in Production", version)
        logger.info("  Restart the API to load the new model:")
        logger.info("  docker compose restart api")
    else:
        logger.warning("✗ Validation gate FAILED:")
        for reason in reasons:
            logger.warning("  - %s", reason)
        version = register_with_mlflow(
            model, metrics, feature_list, scaler,
            stage="Staging", n_samples=n_samples, data_date=data_date,
            tuned_params=tuned_params,
        )
        logger.warning("Registered as Staging version %d — champion unchanged", version)

        # Save challenger artefacts for shadow deployment.
        # The running API will pick these up and run the challenger in the background
        # on every prediction, logging side-by-side comparisons without exposing the
        # challenger result to users.  Use GET /shadow/stats to see the comparison,
        # then POST /shadow/promote if the challenger looks good.
        try:
            export_onnx(model, feature_list, MODEL_DIR / "challenger.onnx")
            with open(MODEL_DIR / "challenger_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            logger.info(
                "Challenger saved for shadow deployment:\n"
                "  models/nyc/challenger.onnx\n"
                "  models/nyc/challenger_scaler.pkl\n"
                "Restart the API to activate shadow mode:\n"
                "  docker compose restart api\n"
                "Then monitor: GET /shadow/stats\n"
                "Promote when ready: POST /shadow/promote"
            )
        except Exception as exc:
            logger.warning("Could not save challenger artefacts: %s", exc)

        try:
            sys.path.insert(0, str(BASE_DIR))
            from src.new_york_workflow.nyc_alerts import alerts
            alerts.push(
                alert_type="retraining_gate_failed",
                message=f"Retraining gate failed: {'; '.join(reasons)}",
                severity="warning",
                details={
                    "new_metrics": metrics,
                    "baseline": baseline["model_metrics"],
                    "mlflow_version": version,
                    "reasons": reasons,
                    "shadow_deployment": "challenger.onnx saved — restart API to activate",
                },
            )
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nightly NYC Airbnb retraining pipeline")
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the R²/MAE validation gate and promote regardless",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Use existing data/airbnb/engineered_features.csv instead of downloading fresh data",
    )
    parser.add_argument(
        "--rollback", type=int, metavar="VERSION",
        help="Re-promote an existing MLflow version to Production (skips training)",
    )
    parser.add_argument(
        "--skip-optuna", action="store_true",
        help="Skip Optuna hyperparameter search and use fixed XGB_PARAMS (faster runs, CI testing)",
    )
    args = parser.parse_args()
    main(force=args.force, rollback_version=args.rollback, no_download=args.no_download,
         skip_optuna=args.skip_optuna)
