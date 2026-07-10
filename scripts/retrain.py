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
from src.training.data_validator import (
    validate_raw_snapshot,
    validate_clean_listings,
    validate_engineered_features,
)

BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / "data" / "airbnb"
MODEL_DIR  = BASE_DIR / "models" / "nyc"
MLFLOW_URI           = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "")
MODEL_NAME = "nyc-airbnb-xgboost"

# S3 training data bucket — 3 pre-uploaded InsideAirbnb snapshots rotated daily
S3_TRAINING_BUCKET  = "nyc-airbnb-models-698172256228"
S3_TRAINING_PREFIX  = "training-data"
S3_SNAPSHOT_COUNT   = 3   # listings_0.csv.gz, listings_1.csv.gz, listings_2.csv.gz

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

OPTUNA_N_TRIALS = 50    # ~8-12 min on GH Actions ubuntu-latest; 50 gives meaningfully better params than 20
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
    date_str  = url_parts[-1] if url_parts else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path  = raw_dir / f"listings_raw_{date_str}.csv.gz"

    if out_path.exists():
        logger.info("Re-using cached snapshot: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
        return out_path

    logger.info("Downloading %s …", url)
    urllib.request.urlretrieve(url, out_path)
    logger.info("Saved %.1f MB → %s", out_path.stat().st_size / 1e6, out_path)
    return out_path


def download_s3_snapshot(index: int | None = None) -> Path:
    """
    Download one of the three pre-uploaded InsideAirbnb snapshots from S3.

    Index defaults to date.today().toordinal() % S3_SNAPSHOT_COUNT — rotates
    daily so consecutive nightly runs train on different data vintages.

    Snapshots are stored as:
      s3://nyc-airbnb-models-698172256228/training-data/listings_0.csv  (Feb)
      s3://nyc-airbnb-models-698172256228/training-data/listings_1.csv  (March)
      s3://nyc-airbnb-models-698172256228/training-data/listings_2.csv  (April)

    Upload command (one-time):
      aws s3 cp feb.csv   s3://nyc-airbnb-models-698172256228/training-data/listings_0.csv
      aws s3 cp march.csv s3://nyc-airbnb-models-698172256228/training-data/listings_1.csv
      aws s3 cp April.csv s3://nyc-airbnb-models-698172256228/training-data/listings_2.csv

    Returns:
        Path to the downloaded CSV file.

    Raises:
        RuntimeError: If the S3 download fails.
    """
    import boto3
    from botocore.exceptions import ClientError

    if index is None:
        index = date.today().toordinal() % S3_SNAPSHOT_COUNT

    s3_key   = f"{S3_TRAINING_PREFIX}/listings_{index}.csv"
    raw_dir  = DATA_DIR / "raw_snapshots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"listings_s3_{index}.csv"

    if out_path.exists():
        logger.info("Re-using cached S3 snapshot index=%d: %s", index, out_path)
        return out_path

    logger.info("Downloading s3://%s/%s → %s", S3_TRAINING_BUCKET, s3_key, out_path)
    try:
        boto3.client("s3").download_file(S3_TRAINING_BUCKET, s3_key, str(out_path))
    except ClientError as exc:
        raise RuntimeError(
            f"S3 download failed for s3://{S3_TRAINING_BUCKET}/{s3_key}: {exc}\n"
            f"Upload snapshots first:\n"
            f"  aws s3 cp feb.csv   s3://{S3_TRAINING_BUCKET}/{S3_TRAINING_PREFIX}/listings_0.csv\n"
            f"  aws s3 cp march.csv s3://{S3_TRAINING_BUCKET}/{S3_TRAINING_PREFIX}/listings_1.csv\n"
            f"  aws s3 cp April.csv s3://{S3_TRAINING_BUCKET}/{S3_TRAINING_PREFIX}/listings_2.csv"
        ) from exc

    logger.info("Downloaded %.1f MB → %s", out_path.stat().st_size / 1e6, out_path)
    return out_path


def download_all_s3_snapshots() -> list[Path]:
    """Download all S3 snapshots and return their local paths."""
    return [download_s3_snapshot(index=i) for i in range(S3_SNAPSHOT_COUNT)]


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
    c = _load_module("cleaning",     "src/training/cleaning.py")
    e = _load_module("engineering",  "src/training/features.py")

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


# ── data provenance ───────────────────────────────────────────────────────────

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _s3_version_id(s3_key: str) -> str:
    """Return the S3 VersionId of the object, or 'unknown' if versioning is off."""
    try:
        import boto3
        resp = boto3.client("s3").head_object(Bucket=S3_TRAINING_BUCKET, Key=s3_key)
        return resp.get("VersionId", "unversioned")
    except Exception:
        return "unknown"


def _data_provenance(
    raw_path: Path | None,
    df_raw_rows: int,
    df_clean_rows: int,
    df_train_rows: int,
    null_price_pct: float,
    price_median: float,
    price_p10: float,
    price_p90: float,
    s3_key: str | None = None,
) -> dict:
    """
    Build a data-provenance dict that gets logged to MLflow as params + metrics.
    Captures: file identity (hash + S3 version), row counts at each pipeline
    stage, and price distribution of the training set.
    """
    prov: dict = {
        "data_s3_key":       s3_key or "local",
        "data_rows_raw":     df_raw_rows,
        "data_rows_clean":   df_clean_rows,
        "data_rows_train":   df_train_rows,
        "data_null_price_pct": round(null_price_pct * 100, 1),
        "data_price_median": round(price_median, 2),
        "data_price_p10":    round(price_p10, 2),
        "data_price_p90":    round(price_p90, 2),
    }
    if raw_path is not None:
        prov["data_sha256"]    = _sha256(raw_path)
        prov["data_file_mb"]   = round(raw_path.stat().st_size / 1e6, 1)
    if s3_key is not None:
        prov["data_s3_version"] = _s3_version_id(s3_key)
    return prov


# ── null-price fallback ────────────────────────────────────────────────────────

def _probe_null_price_pct(raw_path: Path) -> float:
    """Read only the price column (first 2000 rows) and return the null fraction."""
    try:
        probe = pd.read_csv(raw_path, usecols=["price"], nrows=2000)
        return float(probe["price"].isna().mean())
    except Exception:
        return 1.0  # assume broken if we can't read it


def _with_null_price_fallback(raw_path: Path) -> pd.DataFrame:
    """
    Try to process raw_path.  If the price column is >95% null (InsideAirbnb
    format change — seen in Dec 2025 snapshot where prices were fully removed)
    fall back in order:
      1. S3 pre-uploaded snapshots (listings_0/1/2.csv)
      2. Existing engineered_features.csv on disk
    Normal snapshots have 40-70% null prices (draft/unlisted properties); those
    are handled naturally by clean_and_engineer which drops null-price rows.
    """
    null_pct = _probe_null_price_pct(raw_path)
    if null_pct <= 0.95:
        return clean_and_engineer(raw_path)

    logger.warning(
        "Downloaded snapshot has %.0f%% null prices — InsideAirbnb format change detected. "
        "Prices are no longer included in listings.csv for this snapshot.",
        null_pct * 100,
    )

    # Fallback 1: S3 pre-uploaded snapshots
    try:
        logger.info("Falling back to S3 snapshot rotation...")
        s3_path = download_s3_snapshot()
        logger.info("Using S3 snapshot: %s", s3_path)
        return clean_and_engineer(s3_path)
    except Exception as exc:
        logger.warning("S3 fallback failed: %s", exc)

    # Fallback 2: existing engineered_features.csv
    eng_path = DATA_DIR / "engineered_features.csv"
    if eng_path.exists():
        logger.warning(
            "No usable snapshot available. Retraining on existing %s "
            "(no new data — model stays current but won't reflect market changes).",
            eng_path,
        )
        return pd.read_csv(eng_path)

    raise RuntimeError(
        "Downloaded snapshot has null prices, S3 fallback failed, and no "
        "engineered_features.csv exists on disk. Cannot retrain.\n"
        "Fix: upload snapshots to S3 or run with --no-download."
    )


# ── training data loader ───────────────────────────────────────────────────────

def load_training_data(
    no_download: bool = False,
    s3_snapshot: bool = False,
    s3_index: int | None = None,
    all_months: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the training dataset.  Returns (df, provenance_dict).

    no_download=True  : use existing engineered_features.csv on disk (fastest, no network)
    all_months=True   : combine all 3 S3 snapshots (Feb+March+April) — more data, better model
    s3_snapshot=True  : download one of 3 pre-uploaded InsideAirbnb CSVs from S3
                        (rotates by date unless s3_index is given explicitly)
    default           : auto-download the latest InsideAirbnb NYC snapshot

    In all cases, the top 1% of prices is capped to match the original training setup.
    """
    raw_path: Path | None = None
    s3_key:   str  | None = None
    rows_raw               = 0

    if no_download:
        path = DATA_DIR / "engineered_features.csv"
        logger.info("--no-download: using existing %s", path)
        df = pd.read_csv(path)
    elif all_months:
        logger.info("--all-months: combining all %d S3 snapshots for maximum training data", S3_SNAPSHOT_COUNT)
        paths    = download_all_s3_snapshots()
        rows_raw = sum(pd.read_csv(p, usecols=[0]).shape[0] for p in paths)
        frames   = []
        for p in paths:
            null_pct = _probe_null_price_pct(p)
            if null_pct > 0.95:
                logger.warning("Skipping %s — %.0f%% null prices (format-removed snapshot)", p.name, null_pct * 100)
                continue
            logger.info("Including %s (%.0f%% null prices — normal)", p.name, null_pct * 100)
            frames.append(clean_and_engineer(p))
        if not frames:
            raise RuntimeError("All S3 snapshots have null prices — cannot train. Re-upload valid CSVs.")
        df      = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
        s3_key  = f"{S3_TRAINING_PREFIX}/listings_all"
        logger.info("Combined %d valid snapshots → %d unique listings after dedup", len(frames), len(df))
    elif s3_snapshot:
        if s3_index is None:
            s3_index = date.today().toordinal() % S3_SNAPSHOT_COUNT
        s3_key   = f"{S3_TRAINING_PREFIX}/listings_{s3_index}.csv"
        raw_path = download_s3_snapshot(index=s3_index)
        rows_raw = pd.read_csv(raw_path, usecols=[0]).shape[0]  # accurate row count
        df       = clean_and_engineer(raw_path)
    else:
        raw_path = download_fresh_data()
        rows_raw = pd.read_csv(raw_path, usecols=[0]).shape[0]
        df       = _with_null_price_fallback(raw_path)

    rows_clean = len(df)
    null_pct   = _probe_null_price_pct(raw_path) if raw_path else 0.0

    cap = df["price"].quantile(PRICE_CAP)
    df  = df[df["price"] <= cap].copy()
    logger.info("  %d rows after %.0f%% price cap ($%.0f)", len(df), PRICE_CAP * 100, cap)

    # Gate 3: validate engineered features before training
    validate_engineered_features(df).raise_if_failed()

    prov = _data_provenance(
        raw_path      = raw_path,
        df_raw_rows   = rows_raw,
        df_clean_rows = rows_clean,
        df_train_rows = len(df),
        null_price_pct= null_pct,
        price_median  = float(df["price"].median()),
        price_p10     = float(df["price"].quantile(0.10)),
        price_p90     = float(df["price"].quantile(0.90)),
        s3_key        = s3_key,
    )
    logger.info(
        "Data provenance: %d raw → %d clean → %d train rows | "
        "null_price=%.1f%% | price median=$%.0f (p10=$%.0f p90=$%.0f) | sha256=%s",
        prov["data_rows_raw"], prov["data_rows_clean"], prov["data_rows_train"],
        prov["data_null_price_pct"],
        prov["data_price_median"], prov["data_price_p10"], prov["data_price_p90"],
        prov.get("data_sha256", "n/a")[:12],
    )
    return df, prov


def add_neighbourhood_target_encoding(
    df: pd.DataFrame, train_idx: pd.Index
) -> tuple[pd.DataFrame, dict]:
    """
    Add neighbourhood_price_rank (train-only mean log_price) and
    neighbourhood_median_price (train-only median $ price) columns.

    Must be computed on train_idx only — using the full dataset would leak
    test-set prices into a feature the model sees at train time.

    Returns:
        (df with the two columns added, artefact dict for nyc_neighbourhood_means.pkl)
    """
    neigh_mean    = df.loc[train_idx].groupby("neighbourhood_cleansed")["log_price"].mean()
    neigh_median  = df.loc[train_idx].groupby("neighbourhood_cleansed")["price"].median()
    global_mean   = float(df.loc[train_idx]["log_price"].mean())
    global_median = float(df.loc[train_idx]["price"].median())

    df = df.copy()
    df["neighbourhood_price_rank"] = (
        df["neighbourhood_cleansed"].map(neigh_mean).fillna(global_mean)
    )
    df["neighbourhood_median_price"] = (
        df["neighbourhood_cleansed"].map(neigh_median).fillna(global_median)
    )

    artefact = {
        "means":        neigh_mean.to_dict(),
        "global_mean":  global_mean,
        "medians":      neigh_median.to_dict(),
        "global_median": global_median,
    }
    return df, artefact


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
    data_prov: dict | None = None,
) -> int:
    mlflow.set_tracking_uri(MLFLOW_URI)
    if MLFLOW_ARTIFACT_ROOT:
        # Ensure the experiment is created with the S3 artifact root on first run.
        from mlflow.tracking import MlflowClient as _C
        _client = _C(tracking_uri=MLFLOW_URI)
        existing = _client.get_experiment_by_name("nyc-airbnb-retraining")
        if existing is None:
            _client.create_experiment(
                "nyc-airbnb-retraining",
                artifact_location=MLFLOW_ARTIFACT_ROOT,
            )
    mlflow.set_experiment("nyc-airbnb-retraining")
    data_prov = data_prov or {}

    onnx_path = (
        MODEL_DIR / "nyc_xgb_model.onnx" if stage == "Production"
        else MODEL_DIR / "challenger.onnx"
    )

    with mlflow.start_run(run_name=f"retrain-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}") as run:
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

        # ── data provenance (which file, which S3 version, how clean) ─────────
        str_prov = {k: str(v) for k, v in data_prov.items()}
        mlflow.log_params(str_prov)
        # numeric stats also as metrics so they're queryable / plottable
        for key in ("data_rows_raw", "data_rows_clean", "data_rows_train",
                    "data_null_price_pct", "data_price_median",
                    "data_price_p10", "data_price_p90"):
            if key in data_prov:
                mlflow.log_metric(key, float(data_prov[key]))

        # ── metrics ───────────────────────────────────────────────────────────
        mlflow.log_metrics(metrics)
        mlflow.log_metric("gate_passed", 1 if stage == "Production" else 0)

        # ── tags for easy filtering in the UI ─────────────────────────────────
        mlflow.set_tags({
            "git_sha":          _git_sha(),
            "pipeline":         "nightly-retrain",
            "stage":            stage,
            "data_date":        data_date or str(date.today()),
            "data_s3_key":      data_prov.get("data_s3_key", "local"),
            "data_s3_version":  data_prov.get("data_s3_version", "n/a"),
            "data_sha256":      data_prov.get("data_sha256", "n/a")[:16],
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

        # Save baseline so the NEXT run can compare price distribution drift
        baseline_path = MODEL_DIR / "baseline_stats.json"
        existing = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        existing.update({
            "data_price_median": data_prov.get("data_price_median"),
            "data_price_p10":    data_prov.get("data_price_p10"),
            "data_price_p90":    data_prov.get("data_price_p90"),
            "data_rows_train":   data_prov.get("data_rows_train"),
            "data_null_price_pct": data_prov.get("data_null_price_pct"),
            "data_s3_key":       data_prov.get("data_s3_key"),
            "data_s3_version":   data_prov.get("data_s3_version"),
            "data_sha256":       data_prov.get("data_sha256"),
            "champion_r2":       metrics.get("r2"),
            "champion_mae":      metrics.get("mae_dollar"),
            "champion_version":  int(version),
            "champion_run_id":   run_id,
        })
        baseline_path.write_text(json.dumps(existing, indent=2))
        logger.info("Baseline updated → %s", baseline_path)
        _push_retrain_metrics(metrics, data_prov, int(version), gate_passed=True)
    else:
        try:
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
        except Exception:
            pass
        client.set_registered_model_alias(MODEL_NAME, "challenger", version)
        logger.info("Registered v%s as Staging (alias: challenger)  run_id=%s", version, run_id)

    return int(version)


# ── Prometheus Pushgateway ─────────────────────────────────────────────────────

def _push_retrain_metrics(
    metrics: dict, data_prov: dict, version: int, gate_passed: bool
) -> None:
    """
    Push retrain outcome gauges to the Prometheus Pushgateway so Grafana can
    display a "Model Health" row without needing direct MLflow access.

    Called on BOTH Production (gate_passed=True) and Staging (gate_passed=False)
    so Grafana always shows the outcome of the most recent retrain attempt.

    Pushgateway URL is read from PUSHGATEWAY_URL env var
    (default: http://pushgateway:9091).  Silently skipped if not reachable —
    retrain success does not depend on observability infra being up.
    """
    pushgateway = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    except ImportError:
        logger.warning("prometheus_client not installed — skipping Pushgateway push")
        return

    reg = CollectorRegistry()

    # No extra labels on the metrics — the Pushgateway grouping key (job=nyc-retrain)
    # already adds a job label when Prometheus scrapes. Adding our own job label too
    # would conflict and produce duplicate/overridden labels.
    def g(name, desc, value):
        Gauge(name, desc, registry=reg).set(value)

    g("nyc_model_r2",               "Champion model R²",              metrics.get("r2", 0))
    g("nyc_model_mae_dollar",        "Champion model MAE (USD)",       metrics.get("mae_dollar", 0))
    g("nyc_model_mape_pct",          "Champion model MAPE %",          metrics.get("mape_pct", 0))
    g("nyc_model_gate_passed",       "Validation gate (1=pass 0=fail)",int(gate_passed))
    g("nyc_model_champion_version",  "MLflow champion version number", version)
    g("nyc_model_train_rows",        "Training rows used",             data_prov.get("data_rows_train", 0))
    g("nyc_model_raw_rows",          "Raw snapshot rows before clean", data_prov.get("data_rows_raw", 0))
    g("nyc_model_null_price_pct",    "Null price % in raw snapshot",   data_prov.get("data_null_price_pct", 0))
    g("nyc_model_price_median",      "Training set price median (USD)",data_prov.get("data_price_median", 0))
    g("nyc_model_price_p10",         "Training set price p10 (USD)",   data_prov.get("data_price_p10", 0))
    g("nyc_model_price_p90",         "Training set price p90 (USD)",   data_prov.get("data_price_p90", 0))

    try:
        push_to_gateway(pushgateway, job="nyc-retrain", registry=reg)
        logger.info("Pushgateway metrics pushed → %s (gate_passed=%s)", pushgateway, gate_passed)
    except Exception as exc:
        logger.warning("Pushgateway push failed (non-fatal): %s", exc)


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

def main(
    force: bool = False,
    rollback_version: int | None = None,
    no_download: bool = False,
    skip_optuna: bool = False,
    s3_snapshot: bool = False,
    s3_index: int | None = None,
    all_months: bool = False,
):
    if rollback_version is not None:
        rollback(rollback_version)
        return

    # Load baseline for gate check
    baseline_path = MODEL_DIR / "baseline_stats.json"
    if not baseline_path.exists():
        logger.error("baseline_stats.json missing — run scripts/register_baseline.py first")
        sys.exit(1)
    baseline = json.loads(baseline_path.read_text())

    # Load data, then split BEFORE target-encoding neighbourhood —
    # same random_state/test_size as the split below reproduces identical
    # train row IDs, so the encoding never sees test-set prices.
    df, data_prov = load_training_data(
        no_download=no_download,
        s3_snapshot=s3_snapshot,
        s3_index=s3_index,
        all_months=all_months,
    )
    train_idx, _ = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    df, neighbourhood_means_artefact = add_neighbourhood_target_encoding(df, train_idx)

    # Derived fresh from the engineered dataframe every run (not a frozen JSON
    # snapshot) so any column 2_feature_engineering.py/_engineer() adds is
    # picked up automatically — this is what "activates" dormant features that
    # were added to the engineering step but not yet trained on.
    feature_list = [c for c in df.columns if c not in DROP_COLS]

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
        with open(MODEL_DIR / "nyc_neighbourhood_means.pkl", "wb") as f:
            pickle.dump(neighbourhood_means_artefact, f)
        with open(MODEL_DIR / "nyc_feature_list.pkl", "wb") as f:
            pickle.dump(feature_list, f)
        export_onnx(model, feature_list, MODEL_DIR / "nyc_xgb_model.onnx")
        version = register_with_mlflow(
            model, metrics, feature_list, scaler,
            stage="Production", n_samples=n_samples, data_date=data_date,
            tuned_params=tuned_params, data_prov=data_prov,
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
            tuned_params=tuned_params, data_prov=data_prov,
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
            from src.serving.alerts import alerts
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
        _push_retrain_metrics(metrics, data_prov, int(version), gate_passed=False)
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
    parser.add_argument(
        "--s3-snapshot", action="store_true",
        help=(
            "Download training data from S3 instead of InsideAirbnb. "
            "Rotates through listings_0/1/2.csv daily by default. "
            "Use with --s3-index to pick a specific snapshot."
        ),
    )
    parser.add_argument(
        "--s3-index", type=int, choices=[0, 1, 2], default=None,
        help="Override S3 snapshot index (0=Feb, 1=March, 2=April). Defaults to date-based rotation.",
    )
    parser.add_argument(
        "--all-months", action="store_true",
        help=(
            "Combine all 3 S3 snapshots (Feb+March+April) into one training set. "
            "More data → better generalisation. Deduplicates on listing ID."
        ),
    )
    args = parser.parse_args()
    main(
        force=args.force,
        rollback_version=args.rollback,
        no_download=args.no_download,
        skip_optuna=args.skip_optuna,
        s3_snapshot=args.s3_snapshot,
        s3_index=args.s3_index,
        all_months=args.all_months,
    )
