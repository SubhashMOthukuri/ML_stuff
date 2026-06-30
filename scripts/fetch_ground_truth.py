#!/usr/bin/env python3
"""
Ground truth feedback loop — batch job that closes the prediction loop.

What it does:
  1. Reads all predictions that have a listing_id but no ground truth yet
  2. Downloads the latest InsideAirbnb NYC listings snapshot (actual prices)
  3. Joins on listing_id → gets the real price for each prediction
  4. Writes (predicted_price, actual_price, error) to the ground_truth table
  5. Computes production MAE and compares to training baseline
  6. Pushes an alert if production MAE exceeds training MAE + $10 threshold

Run as a cron job (monthly, after InsideAirbnb publishes a fresh snapshot):
  0 6 1 * * cd /app && python scripts/fetch_ground_truth.py

Manual run:
  python scripts/fetch_ground_truth.py
  python scripts/fetch_ground_truth.py --url https://...listings.csv.gz   # custom snapshot
  python scripts/fetch_ground_truth.py --dry-run                          # print matches, no writes
"""

import argparse
import gzip
import io
import json
import logging
import sys
import urllib.request
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.new_york_workflow.nyc_store import RequestStore
from src.new_york_workflow.nyc_ground_truth import GroundTruthStore
from src.new_york_workflow.nyc_alerts import alerts

# Latest InsideAirbnb NYC listings snapshot (updated monthly)
INSIDEAIRBNB_URL = (
    "https://data.insideairbnb.com/united-states/ny/new-york-city/"
    "2024-09-04/data/listings.csv.gz"
)

# Alert if production MAE exceeds training MAE by more than this
MAE_ALERT_THRESHOLD_DOLLARS = 10.0


def fetch_snapshot(url: str) -> pd.DataFrame:
    logger.info("Downloading InsideAirbnb snapshot: %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read()

    if url.endswith(".gz"):
        raw = gzip.decompress(raw)

    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    logger.info("Snapshot loaded: %d listings", len(df))
    return df


def parse_price(price_str) -> float | None:
    """Convert '$120.00' → 120.0, handling nulls and formatting."""
    if pd.isna(price_str):
        return None
    cleaned = str(price_str).replace("$", "").replace(",", "").strip()
    try:
        v = float(cleaned)
        return v if v > 0 else None
    except ValueError:
        return None


def run(url: str, dry_run: bool = False) -> dict:
    pred_store = RequestStore()
    gt_store   = GroundTruthStore()

    # 1. Get predictions that need ground truth
    pending = pred_store.get_listings_pending_ground_truth()
    if not pending:
        logger.info("No predictions pending ground truth — nothing to do")
        return {"matched": 0, "inserted": 0, "pending": 0}

    logger.info("%d predictions pending ground truth evaluation", len(pending))
    pending_ids = {str(row["listing_id"]): row for row in pending}

    # 2. Download InsideAirbnb snapshot
    df = fetch_snapshot(url)

    # Normalise: id column → string, price → float
    df["id"] = df["id"].astype(str)
    df["actual_price"] = df["price"].apply(parse_price)
    df = df[df["actual_price"].notna() & df["id"].isin(pending_ids)]

    logger.info("Matched %d listing_ids in snapshot", len(df))

    if dry_run:
        logger.info("DRY RUN — no writes")
        for _, row in df.iterrows():
            pred = pending_ids[row["id"]]
            err  = abs(pred["predicted_price"] - row["actual_price"])
            logger.info(
                "  listing %s | predicted=$%.0f actual=$%.0f error=$%.0f",
                row["id"], pred["predicted_price"], row["actual_price"], err,
            )
        return {"matched": len(df), "inserted": 0, "pending": len(pending), "dry_run": True}

    # 3. Write ground truth rows
    inserted = 0
    for _, snapshot_row in df.iterrows():
        pred = pending_ids[snapshot_row["id"]]
        ok = gt_store.insert(
            listing_id      = snapshot_row["id"],
            predicted_price = pred["predicted_price"],
            actual_price    = snapshot_row["actual_price"],
            request_id      = pred.get("request_id"),
            borough         = snapshot_row.get("neighbourhood_group_cleansed"),
            room_type       = snapshot_row.get("room_type"),
        )
        if ok:
            inserted += 1

    logger.info("Inserted %d ground truth rows", inserted)

    # 4. Check production MAE vs training baseline
    stats = gt_store.stats()
    production_mae = stats.get("mae_dollar")

    if production_mae is not None and stats["n_evaluated"] >= 30:
        baseline_path = BASE_DIR / "models" / "nyc" / "nyc_training_report.json"
        if baseline_path.exists():
            baseline     = json.loads(baseline_path.read_text())
            training_mae = baseline.get("models", {}).get("XGBoost", {}).get("test", {}).get("mae_dollar")
            if training_mae:
                drift = production_mae - training_mae
                logger.info(
                    "Production MAE=$%.2f  Training MAE=$%.2f  Drift=$%.2f",
                    production_mae, training_mae, drift,
                )
                if drift > MAE_ALERT_THRESHOLD_DOLLARS:
                    alerts.push_once(
                        alert_type = "production_mae_degraded",
                        message    = (
                            f"Production MAE ${production_mae:.2f} exceeds "
                            f"training MAE ${training_mae:.2f} by ${drift:.2f} "
                            f"(threshold ${MAE_ALERT_THRESHOLD_DOLLARS})"
                        ),
                        severity = "warning",
                        details  = {
                            "production_mae":  production_mae,
                            "training_mae":    training_mae,
                            "drift_dollars":   round(drift, 2),
                            "n_evaluated":     stats["n_evaluated"],
                            "recommendation":  "Consider retraining — run scripts/retrain.py",
                        },
                    )
                    logger.warning("Alert pushed: production MAE degraded")

    return {
        "matched":  len(df),
        "inserted": inserted,
        "pending":  len(pending),
        "production_mae": production_mae,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch ground truth from InsideAirbnb")
    parser.add_argument("--url",     default=INSIDEAIRBNB_URL, help="InsideAirbnb listings CSV URL")
    parser.add_argument("--dry-run", action="store_true",       help="Match and print, no DB writes")
    args = parser.parse_args()

    result = run(url=args.url, dry_run=args.dry_run)
    logger.info("Done: %s", result)


if __name__ == "__main__":
    main()
