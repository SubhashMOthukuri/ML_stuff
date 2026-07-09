# Model Card — NYC Airbnb Price Prediction

**Model name:** `nyc-airbnb-xgboost-v1`
**Model type:** XGBoost gradient-boosted trees (ONNX Runtime inference)
**Task:** Regression — predict nightly listing price (USD) from listing features
**Version:** See `models/nyc/nyc_training_report.json` for the deployed version timestamp

---

## Model Description

Predicts the nightly price of a New York City Airbnb listing given its
structural characteristics (size, location, room type, amenities, reviews).
The model outputs a dollar price for a single night's stay.

Predictions are for **pricing guidance**, not authoritative valuations.
Hosts, guests, and analysts should treat model output as one input among many.

---

## Training Data

**Source:** [InsideAirbnb](http://insideairbnb.com/get-the-data/) — New York City snapshot
**Snapshot date:** April 14, 2026
**License:** InsideAirbnb data is licensed under Creative Commons Attribution 4.0

| Split | Rows |
|-------|------|
| Raw listings | 35,036 |
| After cleaning | 20,692 |
| Core market (≤99th pct price) | 20,485 |
| Training set (80%) | 16,388 |
| Test set (20%) | 4,097 |

**Price range (core market):** $4.47 – $1,562 / night

**Cleaning steps:**
- Removed listings with missing price, zero price, or price above the 99th percentile ($1,562/night)
- Removed listings with missing or invalid room type, borough, or location
- Imputed missing review scores with per-borough medians
- Removed duplicate columns (e.g. `minimum_minimum_nights` was identical to `minimum_nights`)

**Target:** `log_price` = log1p of the raw nightly USD price.
Log-transformation is applied because raw price is heavily right-skewed (skew=14.21, max=$15,075/night).
The model outputs log_price and the API applies `np.expm1()` to recover the dollar price.

**Target encoding note:** `neighbourhood_price_rank` encodes the mean log_price of each neighbourhood.
This encoding is fitted **on training data only** to prevent data leakage.
At inference, new or unseen neighbourhoods fall back to the global training mean.

---

## Features (58)

### Core listing attributes

| Feature | Description |
|---------|-------------|
| `accommodates` | Maximum number of guests |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `minimum_nights` | Minimum stay requirement |
| `minimum_nights_avg_ntm` | Average minimum nights across the listing's calendar (training mean: 25.0 — NYC 30-day minimum law skews the distribution) |
| `room_type_*` | One-hot: Entire home/apt, Private room, Hotel room, Shared room |
| `instant_bookable` | Whether the listing can be booked without host approval |

### Host signals

| Feature | Description |
|---------|-------------|
| `host_is_superhost` | Superhost status (binary) |
| `host_listings_count` | Number of listings managed by this host |

### Review scores

| Feature | Description |
|---------|-------------|
| `review_scores_rating` | Overall rating (0–5) |
| `review_scores_accuracy` | Accuracy sub-score |
| `review_scores_cleanliness` | Cleanliness sub-score |
| `review_scores_checkin` | Check-in sub-score |
| `review_scores_communication` | Communication sub-score |
| `review_scores_location` | Location sub-score |
| `review_scores_value` | Value sub-score |
| `number_of_reviews` | Total review count |
| `number_of_reviews_ltm` | Reviews in the last 12 months |
| `reviews_per_month` | Review rate (proxy for occupancy) |

### Amenities

| Feature | Description |
|---------|-------------|
| `amenity_count` | Total number of listed amenities |
| `has_gym` | Building has gym (corr=0.31 with log_price) |
| `has_elevator` | Building has elevator |
| `has_dryer` | In-unit dryer |
| `has_air_conditioning` | Air conditioning |
| `has_washer` | In-unit washer |
| `has_pool` | Pool access |
| `is_private_bath` | Private (not shared) bathroom — single strongest pricing signal |

### Location

| Feature | Description |
|---------|-------------|
| `latitude`, `longitude` | GPS coordinates |
| `dist_from_midtown` | Distance from Midtown Manhattan (km) |
| `borough_*` | One-hot: Manhattan, Brooklyn, Queens, Staten Island (Bronx = baseline) |
| `neighbourhood_price_rank` | Mean log_price of the listing's neighbourhood (221 NYC neighbourhoods) |

### Engineered features (log-transforms, interactions, polynomials)

| Feature | Description |
|---------|-------------|
| `log_accommodates`, `log_bedrooms`, `log_bathrooms` | Log1p-transformed capacity features |
| `log_host_listings_count`, `log_number_of_reviews`, `log_number_of_reviews_ltm` | Log1p of count features |
| `log_reviews_per_month`, `log_minimum_nights`, `log_minimum_nights_avg_ntm` | Log1p transforms |
| `log_amenity_count` | Log1p of amenity count |
| `accommodates_sq`, `bedrooms_sq` | Quadratic terms — non-linear price curve |
| `accommodates_x_review_scores_rating` | Interaction: capacity × quality |
| `bedrooms_x_review_scores_rating` | Interaction: bedrooms × quality |
| `host_listings_count_x_review_scores_rating` | Interaction: host scale × quality |
| `accommodates_x_host_is_superhost` | Interaction: capacity × superhost |
| `minimum_nights_x_accommodates` | Interaction: stay length × size |
| `reviews_per_bedroom` | Review density per bedroom |
| `reviews_per_accommodates` | Review density per guest |
| `host_density` | Host activity signal |
| `review_score_std` | Std deviation across sub-scores (consistency signal) |

---

## Performance

Evaluated on a held-out test set of 4,097 listings not seen during training.

| Model | R² | MAE | RMSE | MAPE | Median AE |
|-------|-----|-----|------|------|-----------|
| Dummy baseline (always predict mean) | -0.00 | $131 | $218 | 72.4% | $84 |
| Ridge Regression | 0.729 | $75 | $139 | 31.1% | $36 |
| Random Forest | 0.816 | $59 | $114 | 24.2% | $28 |
| **XGBoost (champion)** | **0.824** | **$58** | **$111** | **23.6%** | **$28** |

**Overfitting analysis:** XGBoost train R²=0.94, test R²=0.82 (gap=0.11). This is
structural for boosted trees (they memorise training leaves). Cross-validation R²≈0.81
closely matches the unseen test score (CV→test gap=0.013), confirming the model
generalises correctly.

**Top feature importances (XGBoost, gain-based):**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `is_private_bath` | 0.2587 |
| 2 | `accommodates_sq` | 0.1074 |
| 3 | `log_minimum_nights` | 0.0621 |
| 4 | `log_minimum_nights_avg_ntm` | 0.0542 |
| 5 | `minimum_nights_avg_ntm` | 0.0499 |

---

## Model Selection Rationale

Before selecting models, formal assumption tests were run on the training data:

- **Linearity:** mean |correlation| with log_price = 0.195 (weak). Linear models underperform.
- **Multicollinearity:** VIF > 40 on `accommodates`, `log_accommodates`, `log_minimum_nights`. OLS produces unstable coefficients.
- **Heteroscedasticity:** price variance differs across price bands (low-price std=0.30, mid-price std=0.19). Hurts OLS.

OLS is ruled out. Tree ensembles (RF, XGBoost) handle all three naturally.
XGBoost wins over RF due to built-in L1/L2 regularisation, `gamma` (minimum split gain),
and `min_child_weight` — smaller train→test gap (0.11 vs 0.14).

---

## Inference Pipeline

1. Receive listing features from the API request
2. Fill `minimum_nights_avg_ntm` with training mean (25.0) if not provided — prevents the NYC 30-day-law distribution shift from inflating predictions
3. Apply `neighbourhood_price_rank` from saved neighbourhood means (fallback: global mean)
4. Apply StandardScaler (fitted on training data only)
5. Run ONNX Runtime inference
6. Apply `np.expm1()` to recover USD price from log_price

**Cache:** Predictions are MD5-keyed and cached in Redis for 5 minutes (~40–50% hit rate in production).

---

## SHAP Explanations

`POST /v1/predict?explain=true` returns per-feature SHAP values (TreeExplainer, log-price space).

Features excluded from the explanation UI because they are not user-controllable:
`is_private_bath`, `minimum_nights_avg_ntm`, `log_minimum_nights_avg_ntm`,
`minimum_minimum_nights`, `days_to_checkin`, `is_peak_season`, `is_weekend`,
`month`, `day_of_week`.

---

## Known Limitations

**Price range:** The model is trained on the 0–99th percentile price range ($4–$1,562/night).
Performance degrades for ultra-luxury listings. Do not use for properties expected to
price above $1,562/night.

**NYC only:** Trained exclusively on New York City listings. Neighbourhood-level features
(`neighbourhood_price_rank`, `dist_from_midtown`, `borough_*`) are meaningless outside NYC.
Do not use for other cities without retraining.

**No demand data:** The following signals are unavailable from InsideAirbnb and would
materially improve accuracy if added:

| Missing signal | Estimated R² gain |
|----------------|------------------|
| Historical booking rate / occupancy | +3–4% |
| Seasonal pricing (summer vs winter) | +2–3% |
| Detailed amenity list (# bathrooms, balcony, view) | +2–3% |
| Competitive density (similar listings within 500m) | +1–2% |
| Photo quality score | +1–2% |

**Minimum nights distribution:** `minimum_nights_avg_ntm` has a training mean of ~25 nights
due to NYC's 30-day minimum night law skewing the distribution. Listings outside NYC or
with unusual policies may see prediction shifts from this feature. The API uses the
training mean as a fallback, which is safer than using the raw minimum_nights input.

**Static neighbourhood encoding:** `neighbourhood_price_rank` is computed from the
April 2026 InsideAirbnb snapshot. Neighbourhood price levels shift over time.
The retraining pipeline updates this encoding when new data is ingested.

---

## Ethical Considerations

**Feedback loops:** If hosts price based on this model and Airbnb search ranking
rewards higher-priced listings, model predictions could reinforce existing pricing
disparities between neighbourhoods.

**Neighbourhood bias:** `neighbourhood_price_rank` encodes historical average prices.
Neighbourhoods that are historically underpriced (often lower-income areas) will receive
lower predicted prices, which could perpetuate underpricing rather than correcting it.

**Exclusions:** Luxury listings (>$1,562/night) are excluded from training. Predictions
for this segment are unreliable and should not be used.

This model is intended as a pricing **reference tool**, not an automated pricing system.
Hosts retain full control over their prices.

---

## Retraining Schedule

Retrained automatically when nightly drift check detects critical PSI drift
(Population Stability Index) against the April 2026 baseline.

Manual retrain: `python scripts/retrain.py`

Model versions are tracked in MLflow. Promotion follows shadow → A/B → canary
before replacing the champion. See [README.md](README.md) for the full rollout workflow.

---

## Further Reading

| Document | What it covers |
|----------|---------------|
| [README.md](README.md) | Quick start, API reference, deployment guide, environment variables |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system walkthrough — component map, design decisions, critical invariants |
| [docs/benchmark.md](docs/benchmark.md) | Dataset stats, model comparison (Dummy / Ridge / RF / XGBoost), feature engineering rationale |
| [docs/fix.md](docs/fix.md) | Historical record of all 22 production problems found and how they were resolved |
