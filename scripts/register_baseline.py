"""
Register baseline stats from the training dataset.

Run ONCE after training (or whenever the training data changes):
    python scripts/register_baseline.py

Saves:  models/nyc/baseline_stats.json
Used by: nyc_drift.py to detect data/prediction drift in production.

Baseline captures:
  - Feature distributions (mean, std, p25, p50, p75) for key numeric features
  - Categorical distributions (borough, room_type)
  - Prediction (log_price → price_usd) distribution
  - Model metrics from nyc_training_report.json
"""

import json
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parents[1]
DATA_DIR  = BASE_DIR / "data" / "airbnb"
MODEL_DIR = BASE_DIR / "models" / "nyc"

# ── numeric features present in both training data AND SQLite prediction logs ──
MONITOR_FEATURES = [
    "accommodates", "bedrooms", "bathrooms", "minimum_nights",
    "number_of_reviews", "reviews_per_month", "review_scores_rating",
    "amenity_count", "host_listings_count",
]
CATEGORICAL = {
    "borough":   ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
    "room_type": ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
}


def pct(series, q):
    return round(float(np.percentile(series.dropna(), q)), 4)


def feature_stats(df: pd.DataFrame) -> dict:
    out = {}
    for col in MONITOR_FEATURES:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        out[col] = {
            "mean": round(float(s.mean()), 4),
            "std":  round(float(s.std()),  4),
            "p10":  pct(s, 10),
            "p25":  pct(s, 25),
            "p50":  pct(s, 50),
            "p75":  pct(s, 75),
            "p90":  pct(s, 90),
        }
    return out


def categorical_stats(df: pd.DataFrame) -> dict:
    out = {}
    for col, values in CATEGORICAL.items():
        if col not in df.columns:
            continue
        counts = df[col].value_counts(normalize=True)
        out[col] = {v: round(float(counts.get(v, 0.0)), 4) for v in values}
    return out


def price_stats(df: pd.DataFrame) -> dict:
    prices = df["price"].dropna()
    return {
        "mean": round(float(prices.mean()), 2),
        "std":  round(float(prices.std()),  2),
        "p5":   pct(prices, 5),
        "p25":  pct(prices, 25),
        "p50":  pct(prices, 50),
        "p75":  pct(prices, 75),
        "p95":  pct(prices, 95),
    }


def main():
    print("Loading engineered_features.csv …")
    df = pd.read_csv(DATA_DIR / "engineered_features.csv")
    n_total = len(df)
    print(f"  {n_total:,} rows, {len(df.columns)} columns")

    # Apply same price cap as training (top 1%)
    cap = df["price"].quantile(0.99)
    df  = df[df["price"] <= cap].copy()
    print(f"  After 99th-pct cap (${cap:.0f}): {len(df):,} rows")

    # Load borough/room_type from listings.csv (not in engineered_features)
    try:
        rt_cols = ["room_type_Entire home/apt", "room_type_Hotel room",
                   "room_type_Private room", "room_type_Shared room"]
        listings = pd.read_csv(DATA_DIR / "clean_listings.csv",
                               usecols=["id", "neighbourhood_group_cleansed"] + rt_cols)
        listings = listings.rename(columns={"neighbourhood_group_cleansed": "borough"})
        # Reconstruct room_type label from one-hot columns
        listings["room_type"] = listings[rt_cols].idxmax(axis=1).str.replace("room_type_", "")
        df = df.merge(listings[["id", "borough", "room_type"]], on="id", how="left")
        print(f"  Merged borough/room_type from clean_listings.csv")
    except Exception as e:
        print(f"  ⚠  Could not merge borough/room_type: {e}")

    # Load model metrics
    report = json.loads((MODEL_DIR / "nyc_training_report.json").read_text())
    xgb_metrics = report["models"]["XGBoost"]["test"]

    baseline = {
        "registered_at":     datetime.now(timezone.utc).isoformat(),
        "n_training_samples": len(df),
        "price_cap_usd":      round(cap, 2),
        "model_metrics": {
            "r2":        xgb_metrics["r2"],
            "mae_dollar": xgb_metrics["mae_dollar"],
            "mape_pct":  xgb_metrics["mape"],
        },
        "price_dist":        price_stats(df),
        "feature_dist":      feature_stats(df),
        "categorical_dist":  categorical_stats(df),
        # PSI alert thresholds
        "drift_thresholds": {
            "prediction_mean_pct":  0.15,   # 15% shift in mean predicted price → alert
            "feature_psi":         0.20,    # PSI > 0.2 for any feature → alert
            "categorical_psi":     0.20,
            "min_samples_for_check": 50,    # don't check drift on fewer than 50 predictions
        },
    }

    out_path = MODEL_DIR / "baseline_stats.json"
    out_path.write_text(json.dumps(baseline, indent=2))
    print(f"\n✓  Saved: {out_path}")
    print(f"   Price baseline: mean=${baseline['price_dist']['mean']}  "
          f"p50=${baseline['price_dist']['p50']}  p95=${baseline['price_dist']['p95']}")
    print(f"   Model R²={baseline['model_metrics']['r2']}  "
          f"MAE=${baseline['model_metrics']['mae_dollar']}")


if __name__ == "__main__":
    main()
