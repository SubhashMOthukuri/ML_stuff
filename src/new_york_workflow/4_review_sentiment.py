"""
PRODUCTION REVIEW SENTIMENT PIPELINE
NYC Airbnb Price Prediction — Phase 5
Input:  reviews.csv  (468,009 rows × 6 cols)
Output: data/airbnb/review_sentiment_features.csv
        (one row per listing_id, 11 aggregated sentiment features)

Method: VADER (Valence Aware Dictionary and sEntiment Reasoner)
        Rule-based NLP — no GPU needed, ~5 min for 468K reviews.
        Best for short social/review text (same domain VADER was designed for).

NOTE ON IDs:
  reviews.csv uses new-format Airbnb IDs (19-digit).
  listings.csv uses old-format IDs.
  Direct join is not possible with the April 2026 InsideAirbnb snapshot.
  Output is saved as a standalone file for future use when data sources align,
  or for neighbourhood-level aggregation (group by listing borough/neighbourhood).
"""

import pandas as pd
import numpy as np
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("/Users/subhashmothukurigmail.com/Projects/ML_stuff")
DATA_DIR = BASE_DIR / "data" / "airbnb"

# VADER compound thresholds (standard from original VADER paper)
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

LOG_BATCH = 50_000   # log progress every N reviews


# ============================================================================
# STEP 1 — SETUP
# ============================================================================

def setup_vader():
    logger.info("=" * 80)
    logger.info("PHASE 5: REVIEW SENTIMENT ANALYSIS  (VADER)")
    logger.info("=" * 80)

    logger.info("\n   Downloading VADER lexicon if needed...")
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

    # Smoke test
    test = sia.polarity_scores("Absolutely amazing — clean, central, superhost!")
    logger.info(f"   ✓ VADER ready  (smoke test compound={test['compound']:.3f})")
    return sia


# ============================================================================
# STEP 2 — LOAD
# ============================================================================

def load_reviews():
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2 — LOAD REVIEWS")
    logger.info("=" * 80)

    path = DATA_DIR / "reviews.csv"
    df = pd.read_csv(path, parse_dates=['date'])
    logger.info(f"\n   📂 Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"   Columns: {list(df.columns)}")

    nulls = df['comments'].isnull().sum()
    logger.info(f"   Null comments  : {nulls:,} ({nulls/len(df)*100:.1f}%) — will score as neutral")
    logger.info(f"   Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
    logger.info(f"   Unique listings: {df['listing_id'].nunique():,}")

    return df


# ============================================================================
# STEP 3 — VADER SCORING
# ============================================================================

def score_reviews(df, sia):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3 — VADER SENTIMENT SCORING")
    logger.info("=" * 80)

    logger.info(f"\n   Scoring {len(df):,} reviews...")
    logger.info(f"   Progress logged every {LOG_BATCH:,} reviews\n")

    compound_scores = []
    pos_scores      = []
    neg_scores      = []
    neu_scores      = []

    t0 = datetime.now()

    for i, comment in enumerate(df['comments']):
        if pd.isna(comment) or str(comment).strip() == '':
            compound_scores.append(0.0)
            pos_scores.append(0.0)
            neg_scores.append(0.0)
            neu_scores.append(1.0)
        else:
            scores = sia.polarity_scores(str(comment)[:1024])
            compound_scores.append(scores['compound'])
            pos_scores.append(scores['pos'])
            neg_scores.append(scores['neg'])
            neu_scores.append(scores['neu'])

        if (i + 1) % LOG_BATCH == 0:
            elapsed = (datetime.now() - t0).seconds
            pct = (i + 1) / len(df) * 100
            logger.info(f"   ✓ {i+1:>7,} / {len(df):,} reviews scored  ({pct:.1f}%)  [{elapsed}s elapsed]")

    elapsed_total = (datetime.now() - t0).seconds
    logger.info(f"\n   ✅ Scoring complete — {len(df):,} reviews in {elapsed_total}s")

    df = df.copy()
    df['compound']  = compound_scores
    df['pos']       = pos_scores
    df['neg']       = neg_scores
    df['neu']       = neu_scores
    df['sentiment'] = df['compound'].apply(
        lambda c: 'positive' if c >= POSITIVE_THRESHOLD
                  else ('negative' if c <= NEGATIVE_THRESHOLD else 'neutral')
    )

    # Distribution summary
    dist = df['sentiment'].value_counts()
    logger.info(f"\n   Sentiment distribution:")
    for label, cnt in dist.items():
        bar = '█' * int(cnt / len(df) * 60)
        logger.info(f"   {label.upper():<10}: {cnt:>7,}  ({cnt/len(df)*100:5.1f}%)  {bar}")

    logger.info(f"\n   Compound score stats:")
    logger.info(f"   Mean   : {df['compound'].mean():.4f}")
    logger.info(f"   Std    : {df['compound'].std():.4f}")
    logger.info(f"   Min    : {df['compound'].min():.4f}")
    logger.info(f"   Max    : {df['compound'].max():.4f}")

    return df


# ============================================================================
# STEP 4 — AGGREGATE PER LISTING
# ============================================================================

def aggregate_per_listing(df):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4 — AGGREGATE SENTIMENT PER LISTING")
    logger.info("=" * 80)

    logger.info(f"\n   Grouping {len(df):,} reviews by listing_id...")

    # Core sentiment aggregates
    agg = df.groupby('listing_id').agg(
        review_count         = ('compound', 'count'),
        sentiment_mean       = ('compound', 'mean'),
        sentiment_std        = ('compound', 'std'),
        sentiment_min        = ('compound', 'min'),
        sentiment_max        = ('compound', 'max'),
        avg_pos              = ('pos',      'mean'),
        avg_neg              = ('neg',      'mean'),
        avg_neu              = ('neu',      'mean'),
    ).reset_index()

    # Positive / negative review percentage
    pct = df.groupby('listing_id')['sentiment'].agg(
        positive_review_pct = lambda x: (x == 'positive').sum() / len(x) * 100,
        negative_review_pct = lambda x: (x == 'negative').sum() / len(x) * 100,
    ).reset_index()
    agg = agg.merge(pct, on='listing_id', how='left')

    # Sentiment consistency: inverse of std (0 std → perfect consistency)
    agg['sentiment_consistency'] = 1 / (1 + agg['sentiment_std'].fillna(0))

    # Recency: days since last review (scraped April 14, 2026)
    SCRAPE_DATE = pd.Timestamp('2026-04-14')
    recency = df.groupby('listing_id')['date'].max().reset_index(name='last_review_date')
    recency['days_since_last_review'] = (SCRAPE_DATE - recency['last_review_date']).dt.days.clip(lower=0)
    agg = agg.merge(recency[['listing_id', 'days_since_last_review']], on='listing_id', how='left')

    # Activity span: number of distinct months with reviews
    df['year_month'] = df['date'].dt.to_period('M')
    span = df.groupby('listing_id')['year_month'].nunique().reset_index(name='months_active')
    agg = agg.merge(span, on='listing_id', how='left')

    # Fill std=NaN for single-review listings
    agg['sentiment_std'] = agg['sentiment_std'].fillna(0.0)

    logger.info(f"\n   ✓ {len(agg):,} listings with sentiment features")
    logger.info(f"\n   Features created ({agg.shape[1] - 1} total):")
    for col in agg.columns[1:]:
        logger.info(f"      {col}")

    # Top 5 preview
    logger.info(f"\n   Top 5 most positive listings (by sentiment_mean):")
    logger.info("\n" + agg.nlargest(5, 'sentiment_mean')[
        ['listing_id', 'sentiment_mean', 'positive_review_pct', 'review_count']
    ].to_string(index=False))

    logger.info(f"\n   Top 5 most negative listings (by sentiment_mean):")
    logger.info("\n" + agg.nsmallest(5, 'sentiment_mean')[
        ['listing_id', 'sentiment_mean', 'negative_review_pct', 'review_count']
    ].to_string(index=False))

    return agg


# ============================================================================
# STEP 5 — VALIDATE
# ============================================================================

def validate(df):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5 — VALIDATION")
    logger.info("=" * 80)

    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    logger.info(f"\n   Rows   : {df.shape[0]:,}")
    logger.info(f"   Cols   : {df.shape[1]}")
    logger.info(f"   Nulls  : {null_cols.sum()} {'✓' if null_cols.sum() == 0 else '✗'}")

    if len(null_cols):
        for col, n in null_cols.items():
            logger.warning(f"   ⚠️  {col}: {n} nulls")

    infs = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    logger.info(f"   Infs   : {infs} {'✓' if infs == 0 else '✗'}")


# ============================================================================
# STEP 6 — SAVE
# ============================================================================

def save(df):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6 — SAVE")
    logger.info("=" * 80)

    out_path = DATA_DIR / "review_sentiment_features.csv"
    df.to_csv(out_path, index=False)
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"\n   💾 Saved: {out_path}  ({size_kb:.1f} KB)")

    meta = {
        'timestamp'      : datetime.now().isoformat(),
        'method'         : 'VADER (nltk)',
        'input_reviews'  : None,
        'output_listings': int(df.shape[0]),
        'features'       : list(df.columns),
        'thresholds'     : {
            'positive': POSITIVE_THRESHOLD,
            'negative': NEGATIVE_THRESHOLD,
        },
        'id_note': (
            'listing_ids in reviews.csv use new-format Airbnb IDs (19-digit). '
            'listings.csv uses old-format IDs. Direct join not possible with '
            'April 2026 InsideAirbnb snapshot. Use for neighbourhood aggregation '
            'or future data refresh.'
        ),
    }
    meta_path = DATA_DIR / "review_sentiment_report.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"   💾 Report: {meta_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    sia      = setup_vader()
    reviews  = load_reviews()
    scored   = score_reviews(reviews, sia)
    features = aggregate_per_listing(scored)
    validate(features)
    save(features)

    pos_pct = (scored['sentiment'] == 'positive').mean() * 100
    neg_pct = (scored['sentiment'] == 'negative').mean() * 100
    neu_pct = (scored['sentiment'] == 'neutral').mean() * 100

    logger.info(f"""
================================================================================
✅ REVIEW SENTIMENT PIPELINE COMPLETE
================================================================================
INPUT  : reviews.csv           ({len(scored):,} reviews)
OUTPUT : review_sentiment_features.csv  ({len(features):,} listings)

  Positive reviews : {pos_pct:.1f}%
  Neutral reviews  : {neu_pct:.1f}%
  Negative reviews : {neg_pct:.1f}%

  Features per listing:
    sentiment_mean / std / min / max
    avg_pos / avg_neg / avg_neu
    positive_review_pct / negative_review_pct
    sentiment_consistency
    days_since_last_review
    months_active
    review_count

⚠️  NOTE: listing_id in output uses new-format Airbnb IDs — not directly
    joinable with listings.csv in the April 2026 snapshot.
    Use neighbourhood-level aggregation or await data refresh.

✅ READY FOR: Joining with engineered_features.csv when IDs align
================================================================================
""")
    return features


if __name__ == "__main__":
    main()
