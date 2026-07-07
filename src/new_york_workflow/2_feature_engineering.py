"""
PRODUCTION FEATURE ENGINEERING PIPELINE
NYC Airbnb Price Prediction — Phase 3
Input:  clean_listings.csv  (20,692 rows × 25 cols)
Output: engineered_features.csv
        feature_engineering_report.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "airbnb"

# ============================================================================
# CONFIG
# ============================================================================

# Features with high right-skew — apply log1p (handles zeros safely)
# DROPPED: minimum_minimum_nights — corr=1.0 with minimum_nights (identical column)
LOG_TRANSFORM_COLS = [
    'price',                    # target — log for regression
    'host_listings_count',
    'number_of_reviews',
    'number_of_reviews_ltm',
    'reviews_per_month',
    'minimum_nights',
    'minimum_nights_avg_ntm',
    'accommodates',
    'bedrooms',
    'bathrooms',
    'amenity_count',
]

# Polynomial degree-2 on these (non-linear price relationship expected)
# DROPPED: latitude, longitude          — corr=1.0 with their squares (near-constant range)
# DROPPED: review_scores_rating         — corr=0.993 with its square (adds no new info)
# DROPPED: review_scores_cleanliness    — corr=0.992 with its square (0-5 bounded range)
# DROPPED: review_scores_location       — corr=0.994 with its square (0-5 bounded range)
POLY_COLS = [
    'accommodates',
    'bedrooms',
]

# Interaction pairs (domain-driven: what hosts use to justify higher price)
# DROPPED: accommodates × review_scores_cleanliness   — corr=0.997 with accommodates × rating
# DROPPED: bedrooms × review_scores_value             — corr=0.995 with bedrooms × rating
# DROPPED: number_of_reviews × review_scores_rating   — corr=0.999 with number_of_reviews alone
# DROPPED: rating × location                          — corr=0.993 with review_scores_rating alone
INTERACTION_PAIRS = [
    ('accommodates',        'review_scores_rating'),  # bigger + better rated = premium
    ('bedrooms',            'review_scores_rating'),  # more rooms + rating
    ('host_listings_count', 'review_scores_rating'),  # professional host quality
    ('accommodates',        'host_is_superhost'),     # superhost large listing
    ('minimum_nights',      'accommodates'),          # long-stay large apt
    # Overcrowding signal: XGBoost can split on high accommodates with low bedrooms
    ('accommodates',             'bedrooms'),
    # Room-type quality: premium varies by type; shared rooms shouldn't scale with guest count
    ('room_type_Private room',   'review_scores_rating'),
    ('room_type_Entire home/apt','bedrooms'),
    ('room_type_Shared room',    'accommodates'),
]

# Columns to drop from raw data before output (redundant with kept columns)
DROP_REDUNDANT_COLS = [
    'minimum_minimum_nights',          # corr=1.0 with minimum_nights
    'neighbourhood_group_cleansed',    # replaced by borough_* dummies
    # neighbourhood_cleansed is KEPT — target encoding done in training script
    # to avoid data leakage (train-only means applied to test)
]

# Ratio features (price-relevant relationships)
RATIO_DEFINITIONS = {
    'reviews_per_bedroom'    : ('number_of_reviews',  'bedrooms',     1.0),
    'reviews_per_accommodates': ('number_of_reviews', 'accommodates', 1.0),
    'host_density'           : ('host_listings_count','accommodates', 1.0),
    'avg_score'              : None,   # handled separately (mean of all 7 scores)
    'score_consistency'      : None,   # handled separately (std of all 7 scores)
}

REVIEW_SCORE_COLS = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value',
]

# ============================================================================
# STEP 1 — LOAD
# ============================================================================

def load_data():
    logger.info("=" * 80)
    logger.info("PHASE 3: FEATURE ENGINEERING")
    logger.info("=" * 80)

    path = DATA_DIR / "clean_listings.csv"
    df = pd.read_csv(path)
    logger.info(f"\n📂 Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ============================================================================
# STEP 2 — DISTRIBUTION ANALYSIS
# ============================================================================

def analyse_distributions(df):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2 — DISTRIBUTION ANALYSIS")
    logger.info("=" * 80)

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ['id'] and not c.startswith('room_type_') and c != 'host_is_superhost'
    ]

    report = {}
    logger.info(f"\n{'Column':<38} {'Skew':>7}  {'Kurt':>7}  {'Min':>10}  {'Max':>12}  Action")
    logger.info("-" * 90)

    for col in num_cols:
        skew = float(df[col].skew())
        kurt = float(df[col].kurt())
        mn   = float(df[col].min())
        mx   = float(df[col].max())

        if abs(skew) > 1:
            action = "LOG1P"
        elif abs(skew) > 0.5:
            action = "mild skew"
        else:
            action = "normal"

        logger.info(f"  {col:<36} {skew:>7.2f}  {kurt:>7.2f}  {mn:>10.2f}  {mx:>12.2f}  {action}")
        report[col] = {'skew': skew, 'kurtosis': kurt, 'min': mn, 'max': mx, 'action': action}

    return report


# ============================================================================
# STEP 3 — LOG TRANSFORMS
# ============================================================================

def apply_log_transforms(df):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3 — LOG TRANSFORMS  (log1p — safe for zeros)")
    logger.info("=" * 80)

    df = df.copy()
    created = []

    for col in LOG_TRANSFORM_COLS:
        if col not in df.columns:
            logger.warning(f"   ⚠️  {col} not found — skipping")
            continue

        new_col = f'log_{col}'
        before_skew = df[col].skew()
        df[new_col] = np.log1p(df[col])
        after_skew  = df[new_col].skew()
        created.append(new_col)

        logger.info(f"   ✓  log_{col:<32} skew {before_skew:>6.2f} → {after_skew:>6.2f}")

    logger.info(f"\n   Created {len(created)} log features")
    logger.info(f"\n   Top 5 rows (log features only):")
    logger.info("\n" + df[created].head().to_string())
    return df, created


# ============================================================================
# STEP 3.5 — NEIGHBOURHOOD ENCODING
# Must run AFTER log transforms (needs log_price for target encoding)
# ============================================================================

def encode_neighbourhood(df):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3.5 — NEIGHBOURHOOD ENCODING")
    logger.info("=" * 80)

    df = df.copy()
    created = []

    # --- Borough one-hot (5 values → 4 dummies; Bronx = reference category) ---
    if 'neighbourhood_group_cleansed' in df.columns:
        dummies = pd.get_dummies(df['neighbourhood_group_cleansed'], prefix='borough')
        if 'borough_Bronx' in dummies.columns:
            dummies = dummies.drop(columns=['borough_Bronx'])
        df = pd.concat([df, dummies], axis=1)
        created.extend(dummies.columns.tolist())

        logger.info(f"\n   Borough dummies created:")
        for col in dummies.columns:
            cnt = int(dummies[col].sum())
            logger.info(f"      {col:<35}: {cnt:,} listings")

    logger.info(f"\n   neighbourhood_cleansed kept as string column")
    logger.info(f"   Target encoding (mean log_price per neighbourhood) done in training")
    logger.info(f"   script on training data only — prevents data leakage into test set")

    logger.info(f"\n   Created {len(created)} neighbourhood features")
    logger.info(f"\n   Top 5 rows (neighbourhood features only):")
    logger.info("\n" + df[created].head().to_string())
    return df, created


# ============================================================================
# STEP 4 — POLYNOMIAL FEATURES
# ============================================================================

def create_polynomial_features(df):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4 — POLYNOMIAL FEATURES  (degree 2 — captures non-linear price curve)")
    logger.info("=" * 80)

    df = df.copy()
    created = []

    for col in POLY_COLS:
        if col not in df.columns:
            logger.warning(f"   ⚠️  {col} not found — skipping")
            continue

        new_col = f'{col}_sq'
        df[new_col] = df[col] ** 2
        created.append(new_col)
        logger.info(f"   ✓  {new_col}")

    logger.info(f"\n   Created {len(created)} polynomial features")
    logger.info(f"\n   Top 5 rows (polynomial features only):")
    logger.info("\n" + df[created].head().to_string())
    return df, created


# ============================================================================
# STEP 5 — INTERACTION FEATURES
# ============================================================================

def create_interaction_features(df):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5 — INTERACTION FEATURES  (domain-driven cross terms)")
    logger.info("=" * 80)

    df = df.copy()
    created = []

    for col_a, col_b in INTERACTION_PAIRS:
        if col_a not in df.columns or col_b not in df.columns:
            logger.warning(f"   ⚠️  {col_a} × {col_b} — column missing, skipping")
            continue

        new_col = f'{col_a}_x_{col_b}'
        df[new_col] = df[col_a] * df[col_b]
        created.append(new_col)
        logger.info(f"   ✓  {new_col}")

    logger.info(f"\n   Created {len(created)} interaction features")
    logger.info(f"\n   Top 5 rows (interaction features only):")
    logger.info("\n" + df[created].head().to_string())
    return df, created


# ============================================================================
# STEP 6 — RATIO FEATURES
# ============================================================================

def create_ratio_features(df):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6 — RATIO FEATURES")
    logger.info("=" * 80)

    df = df.copy()
    created = []

    # Simple ratios — fill denominator=0 with 1 before dividing
    simple_ratios = [
        ('reviews_per_bedroom',     'number_of_reviews',  'bedrooms'),
        ('reviews_per_accommodates','number_of_reviews',  'accommodates'),
        ('host_density',            'host_listings_count','accommodates'),
    ]
    for name, num, denom in simple_ratios:
        denom_safe = df[denom].replace(0, 1)
        df[name] = df[num] / denom_safe
        created.append(name)
        logger.info(f"   ✓  {name}  =  {num} / {denom}")

    # Score consistency (std — lower = more consistent, 0 = no reviews)
    # NOTE: avg_review_score DROPPED — corr=0.997 with review_scores_rating (redundant)
    reviewed = df[REVIEW_SCORE_COLS].sum(axis=1) > 0
    df['review_score_std'] = np.where(
        reviewed,
        df[REVIEW_SCORE_COLS].std(axis=1),
        0.0
    )
    created.append('review_score_std')
    logger.info(f"   ✓  review_score_std  =  std of 7 review scores")

    # Price-relevant location: distance from NYC center (Midtown Manhattan)
    MIDTOWN_LAT, MIDTOWN_LON = 40.7549, -73.9840
    df['dist_from_midtown'] = np.sqrt(
        (df['latitude']  - MIDTOWN_LAT) ** 2 +
        (df['longitude'] - MIDTOWN_LON) ** 2
    ) * 111   # rough km conversion (1 degree ≈ 111 km)
    created.append('dist_from_midtown')
    logger.info(f"   ✓  dist_from_midtown  =  Euclidean km from Midtown Manhattan")

    # Occupancy density — encodes overcrowding signal the model couldn't learn from
    # raw accommodates/bedrooms separately (6 guests in 1 bed ≠ 6 guests in 3 beds)
    guests_per_bed = df['accommodates'] / df['bedrooms'].replace(0, 1)
    df['guests_per_bedroom']  = guests_per_bed
    df['guests_per_bathroom'] = df['accommodates'] / df['bathrooms'].replace(0, 1)
    df['overcrowding_ratio']  = np.maximum(0.0, guests_per_bed - 2.0)
    created += ['guests_per_bedroom', 'guests_per_bathroom', 'overcrowding_ratio']
    logger.info(f"   ✓  guests_per_bedroom   = accommodates / max(bedrooms, 1)")
    logger.info(f"   ✓  guests_per_bathroom  = accommodates / max(bathrooms, 1)")
    logger.info(f"   ✓  overcrowding_ratio   = max(0, guests_per_bedroom - 2)")

    logger.info(f"\n   Created {len(created)} ratio features")
    logger.info(f"\n   Top 5 rows (ratio features only):")
    logger.info("\n" + df[created].head().to_string())
    return df, created


# ============================================================================
# STEP 7 — VALIDATE & SUMMARISE
# ============================================================================

def validate(df, original_cols, all_new_cols):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7 — VALIDATION")
    logger.info("=" * 80)

    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    infs  = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

    logger.info(f"\n   Rows       : {df.shape[0]:,}")
    logger.info(f"   Cols total : {df.shape[1]}")
    logger.info(f"   Original   : {len(original_cols)}")
    logger.info(f"   New        : {len(all_new_cols)}")
    logger.info(f"   Nulls      : {null_cols.sum()} {'✓' if null_cols.sum()==0 else '✗ — see below'}")
    logger.info(f"   Infs       : {infs} {'✓' if infs==0 else '✗'}")

    if len(null_cols) > 0:
        logger.warning("   Null columns:")
        for col, n in null_cols.items():
            logger.warning(f"      {col}: {n:,}")

    logger.info(f"\n   New features created:")
    for col in all_new_cols:
        logger.info(f"      {col}")

    logger.info(f"\n   ── TOP 5 ROWS — ALL ENGINEERED FEATURES ──")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    logger.info("\n" + df[all_new_cols].head().to_string())


# ============================================================================
# STEP 8 — SAVE
# ============================================================================

def save(df, dist_report, all_new_cols):
    out_path = DATA_DIR / "engineered_features.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\n💾 Saved: {out_path}")

    meta = {
        'timestamp'             : datetime.now().isoformat(),
        'rows'                  : df.shape[0],
        'columns_total'         : df.shape[1],
        'new_features_count'    : len(all_new_cols),
        'new_features'          : all_new_cols,
        'all_columns'           : list(df.columns),
        'distribution_analysis' : dist_report,
    }
    meta_path = DATA_DIR / "feature_engineering_report.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"💾 Report : {meta_path}")
    return meta


# ============================================================================
# MAIN
# ============================================================================

def main():
    df = load_data()
    original_cols = list(df.columns)

    dist_report              = analyse_distributions(df)
    df, log_cols             = apply_log_transforms(df)
    df, neighbourhood_cols   = encode_neighbourhood(df)
    df, poly_cols            = create_polynomial_features(df)
    df, interaction_cols     = create_interaction_features(df)
    df, ratio_cols           = create_ratio_features(df)

    all_new_cols = log_cols + neighbourhood_cols + poly_cols + interaction_cols + ratio_cols

    # Drop redundant raw columns before output
    dropped = [c for c in DROP_REDUNDANT_COLS if c in df.columns]
    df = df.drop(columns=dropped)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7a — DROP REDUNDANT COLUMNS")
    logger.info("=" * 80)
    for c in dropped:
        logger.info(f"   ✗  {c}  (redundant — see CONFIG comments)")

    validate(df, original_cols, all_new_cols)
    meta = save(df, dist_report, all_new_cols)

    logger.info(f"""
================================================================================
✅ FEATURE ENGINEERING COMPLETE
================================================================================
INPUT  : clean_listings.csv      ({len(original_cols)} features)
OUTPUT : engineered_features.csv ({df.shape[1]} features)

  Log transforms    : {len(log_cols)}
  Neighbourhood enc : {len(neighbourhood_cols)}
  Polynomial (deg2) : {len(poly_cols)}
  Interaction terms : {len(interaction_cols)}
  Ratio features    : {len(ratio_cols)}
  Redundant dropped : {len(dropped)}
  ─────────────────────────────
  Total new         : {len(all_new_cols)}

✅ READY FOR: Model training (3_train_model.py)
================================================================================
""")
    return df


if __name__ == "__main__":
    df = main()
