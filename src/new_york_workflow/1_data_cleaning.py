"""
PRODUCTION DATA CLEANING PIPELINE
Phase 2: Clean listings data
Input: listings.csv (35,036 rows × 90 columns)
Output: clean_listings.csv (15,783 rows × 15 features)
Time: 5 minutes
"""

import pandas as pd
import numpy as np
import json
import re
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

KEEP_FEATURES = [
    'id',
    'price',  # target variable
    'accommodates',
    'bedrooms',
    'host_is_superhost',
    'host_listings_count',
    'latitude',
    'longitude',
    'minimum_minimum_nights',
    'minimum_nights',
    'minimum_nights_avg_ntm',
    'number_of_reviews',
    'number_of_reviews_ltm',
    'reviews_per_month',
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value',
    'room_type',
    'neighbourhood_group_cleansed',   # borough: Manhattan/Brooklyn/Queens/Bronx/Staten Island
    'neighbourhood_cleansed',          # specific neighbourhood (221 unique values)
    'bathrooms_text',                  # parsed to bathrooms + is_private_bath
    'amenities',                       # parsed to amenity_count + key amenity flags
]

# Key amenities with notable price correlation
KEY_AMENITIES = ['Gym', 'Elevator', 'Dryer', 'Air conditioning', 'Washer', 'Pool']

# ============================================================================
# PIPELINE STEPS
# ============================================================================

def load_data(filepath):
    logger.info(f"📂 Loading {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def select_features(df, features):
    logger.info(f"\n📋 Selecting {len(features)} features...")
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df_selected = df[features].copy()
    logger.info(f"✓ Selected {len(features)} features")
    return df_selected


def parse_price(df):
    logger.info(f"\n💰 Parsing price column...")
    df = df.copy()
    before = df['price'].isnull().sum()
    df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
    logger.info(f"   ✓ price: string → float ({before:,} nulls, will be dropped)")
    return df


def encode_categorical(df):
    logger.info(f"\n🔄 Encoding categorical features...")
    df = df.copy()

    if 'room_type' in df.columns:
        room_dummies = pd.get_dummies(df['room_type'], prefix='room_type')
        df = pd.concat([df, room_dummies], axis=1).drop('room_type', axis=1)
        logger.info(f"   ✓ room_type: {room_dummies.shape[1]} columns created")

    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
        logger.info(f"   ✓ host_is_superhost: encoded to 0/1")

    # Keep neighbourhood columns as strings — encoded in feature engineering step
    for col in ['neighbourhood_group_cleansed', 'neighbourhood_cleansed']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            logger.info(f"   ✓ {col}: kept as string ({df[col].nunique()} unique values)")

    # Parse bathrooms_text → numeric bathrooms count + is_private_bath flag
    if 'bathrooms_text' in df.columns:
        def _parse_bath(s):
            if pd.isna(s): return np.nan
            nums = re.findall(r'\d+\.?\d*', str(s).lower())
            return float(nums[0]) if nums else np.nan

        df['bathrooms'] = df['bathrooms_text'].apply(_parse_bath)
        df['is_private_bath'] = (~df['bathrooms_text'].str.lower().str.contains(
            'shared', na=True)).astype(int)
        df = df.drop(columns=['bathrooms_text'])
        logger.info(f"   ✓ bathrooms_text → bathrooms (numeric) + is_private_bath (0/1)")

    # Parse amenities → count + key amenity binary flags
    if 'amenities' in df.columns:
        df['amenity_count'] = df['amenities'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 0
        )
        for am in KEY_AMENITIES:
            col = f'has_{am.lower().replace(" ", "_")}'
            df[col] = df['amenities'].str.contains(am, case=False, na=False).astype(int)
            logger.info(f"   ✓ {col}: {int(df[col].sum()):,} listings")
        df = df.drop(columns=['amenities'])
        logger.info(f"   ✓ amenities → amenity_count + {len(KEY_AMENITIES)} binary flags")

    logger.info(f"✓ Shape after encoding: {df.shape}")
    return df


def handle_missing_values(df):
    logger.info(f"\n🧹 Handling missing values...")
    df = df.copy()

    nulls_before = df.isnull().sum()
    cols_with_nulls = nulls_before[nulls_before > 0]
    if len(cols_with_nulls) > 0:
        logger.info("   Nulls BEFORE cleaning:")
        for col, count in cols_with_nulls.items():
            logger.info(f"      {col:40s} | {count:>6,} ({count/len(df)*100:5.1f}%)")

    if 'bedrooms' in df.columns and df['bedrooms'].isnull().sum() > 0:
        median = df['bedrooms'].median()
        df['bedrooms'] = df['bedrooms'].fillna(median)
        logger.info(f"   ✓ bedrooms: filled nulls with median ({median:.1f})")

    if 'bathrooms' in df.columns and df['bathrooms'].isnull().sum() > 0:
        median = df['bathrooms'].median()
        df['bathrooms'] = df['bathrooms'].fillna(median)
        logger.info(f"   ✓ bathrooms: filled nulls with median ({median:.1f})")

    if 'reviews_per_month' in df.columns and df['reviews_per_month'].isnull().sum() > 0:
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
        logger.info(f"   ✓ reviews_per_month: filled nulls with 0")

    if 'host_listings_count' in df.columns:
        df['host_listings_count'] = df['host_listings_count'].fillna(1).clip(lower=1)
        logger.info(f"   ✓ host_listings_count: filled nulls with 1, clipped zeros to 1")

    review_score_cols = [
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value'
    ]
    for col in review_score_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(0)
    logger.info(f"   ✓ review_scores_*: filled nulls with 0 (listings with no reviews)")

    rows_before = len(df)
    df = df.dropna()
    dropped = rows_before - len(df)
    logger.info(f"   Dropped {dropped:,} rows with remaining nulls → {len(df):,} rows remaining")

    # Drop junk-price listings (< $3/night are placeholder/test listings)
    if "price" in df.columns:
        price_floor_mask = df["price"] >= 3
        n_dropped = (~price_floor_mask).sum()
        if n_dropped > 0:
            df = df[price_floor_mask].copy()
            logger.info(f"   Dropped {n_dropped} listings with price < $3/night")

    return df


def validate_output(df):
    logger.info(f"\n✅ VALIDATING OUTPUT...")
    logger.info(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    nulls = df.isnull().sum().sum()
    if nulls == 0:
        logger.info(f"   ✓ NO NULLS!")
    else:
        logger.warning(f"   ⚠️  {nulls} nulls remain")
    logger.info(f"\n   Data types:\n{df.dtypes.to_string()}")
    logger.info(f"\n   Statistics:\n{df.describe().to_string()}")


def save_data(df, output_path):
    logger.info(f"\n💾 Saving cleaned data...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved: {output_path}")

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'rows': df.shape[0],
        'columns': df.shape[1],
        'column_names': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    metadata_path = str(output_path).replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved: {metadata_path}")
    return metadata

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("PHASE 2 STEP 1: LISTINGS DATA CLEANING")
    logger.info("=" * 80)

    df = load_data(DATA_DIR / "listings.csv")
    df = select_features(df, KEEP_FEATURES)
    df = parse_price(df)
    df = encode_categorical(df)
    df = handle_missing_values(df)
    validate_output(df)
    metadata = save_data(df, DATA_DIR / "clean_listings.csv")

    logger.info("\n" + "=" * 80)
    logger.info("✅ CLEANING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"""
INPUT:  listings.csv       (35,036 rows × 90 columns)
OUTPUT: clean_listings.csv ({metadata['rows']:,} rows × {metadata['columns']} columns)

✅ READY FOR: Model training (review scores embedded from listings.csv)
""")
    return df


if __name__ == "__main__":
    df_clean = main()
