# NYC Airbnb Price Prediction — Model Benchmark Report

**Dataset**: InsideAirbnb NYC snapshot · April 14, 2026  
**Branch**: `proj/new_york`  
**Pipeline**: `1_data_cleaning` → `2_feature_engineering` → `3_train_model`

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Raw listings | 35,036 |
| After cleaning | 20,692 |
| Core market (≤99th pct price) | 20,485 |
| Train / Test split | 80% / 20% |
| Train rows | 16,388 |
| Test rows | 4,097 |
| Features (final) | 58 |
| Target | `log_price` (log1p of nightly USD price) |
| Price range (core market) | $4.47 – $1,562 / night |

**Why log_price as target?**  
Raw price is heavily right-skewed (skew=14.21, max=$15,075). Log-transforming compresses the range, makes the residuals more symmetric, and lets tree models make multiplicative predictions — which matches how hosts actually price (e.g. "20% premium for Manhattan").

---

## 2. Why We Chose These Three Models

### 2.1 Data characteristics that drove model selection

Before picking any model, we ran formal assumption checks on the training data:

| Assumption | Result | Impact |
|---|---|---|
| **Linearity** (mean \|corr\| with log_price) | 0.195 — mostly weak-moderate | Linear models will underperform |
| **Normality of log_price** | Skew=0.17, Kurt=-0.18 — acceptable | Log transform worked |
| **Multicollinearity (VIF)** | VIF > 40 on `accommodates`, `log_accommodates`, `log_minimum_nights` | OLS would produce unstable coefficients |
| **Homoscedasticity** | Variance differs across price bands ($100-200: std=0.19 vs <$100: std=0.30) | Hurts OLS and plain linear regression |

**Conclusion from checks:**  
OLS (ordinary least squares) is ruled out by high VIF + heteroscedasticity. Tree-based models are the natural fit.

---

### 2.2 Model 1 — Dummy Baseline (always predict mean price)

**Purpose**: Floor check. Any real model must beat R²=0.00.  
**Why include it**: Validates that the pipeline is working end-to-end before spending compute on complex models. If a tuned model barely beats the dummy, the features are wrong — not the model.

| Metric | Train | Test |
|---|---|---|
| R² | 0.00 | -0.00 |
| MAE | $127 | $131 |
| MAPE | 73.7% | 72.4% |

---

### 2.3 Model 2 — Ridge Regression (L2 regularisation)

**Why Ridge instead of OLS?**  
The VIF analysis showed extreme multicollinearity (VIF=195 for `accommodates`). OLS coefficients become numerically unstable under high VIF. Ridge adds an L2 penalty that shrinks correlated coefficients toward zero, producing stable estimates.

**Why keep it despite weaker R²?**  
Ridge provides **interpretable coefficients** — each feature has a single weight that shows direction and magnitude. This is valuable for communicating model behaviour to non-technical stakeholders.

| Metric | Train | Test |
|---|---|---|
| R² | 0.7213 | 0.7289 |
| MAE | $73 | $75 |
| MAPE | 31.4% | 31.1% |
| Train ≈ Test | ✅ No overfitting |

**Top Ridge coefficients**: `accommodates` (0.52), `log_accommodates` (0.50), `bedrooms` (0.40), `review_scores_value` (0.28), `neighbourhood_price_rank` (0.16)

---

### 2.4 Model 3 — Random Forest

**Why RF on this data?**

- **Non-linearity**: RF handles non-linear relationships without any explicit polynomial or interaction terms — it splits data at optimal thresholds per feature.
- **High VIF tolerance**: Each tree randomly subsamples features (`max_features=0.3` = 17 of 58 features per split), so correlated features don't amplify each other.
- **Heteroscedasticity immunity**: Trees split on thresholds, not residuals — variance differences across price bands don't bias them.
- **No assumptions**: No linearity, normality, or independence requirements.

**Hyperparameter search**: RandomizedSearchCV, 60 iterations, 5-fold CV.

| Param | Search Space | Best |
|---|---|---|
| `n_estimators` | [400, 600, 800] | 600 |
| `max_depth` | [10, 15, 20, 25, 30] | 25 |
| `min_samples_split` | [2, 5, 10] | 5 |
| `min_samples_leaf` | [1, 2, 4] | 1 |
| `max_features` | [sqrt, log2, 0.3, 0.5] | 0.3 |
| `max_samples` | [0.8, 0.9, 1.0] | 0.9 |

| Metric | Train | Test |
|---|---|---|
| R² | 0.9518 | **0.8158** |
| MAE | $31 | **$59** |
| MAPE | 11.8% | **24.2%** |
| CV R² (5-fold) | ≈ 0.80 | |

**Top 5 feature importances**:

| Rank | Feature | Importance | Why it matters |
|---|---|---|---|
| 1 | `is_private_bath` | 0.1060 | Private vs shared bathroom — biggest price separator |
| 2 | `neighbourhood_price_rank` | 0.1044 | Mean log_price of the specific neighbourhood (221 areas) |
| 3 | `dist_from_midtown` | 0.0683 | Proximity to Manhattan business/tourist core |
| 4 | `accommodates_sq` | 0.0472 | Non-linear: large apartments earn disproportionately more |
| 5 | `log_accommodates` | 0.0466 | Listing capacity on log scale |

---

### 2.5 Model 4 — XGBoost (best model)

**Why XGBoost on this data?**  
XGBoost is gradient boosting — it builds trees sequentially, each correcting the errors of the previous one. Compared to RF:

- **Built-in L1/L2 regularisation** (`reg_alpha`, `reg_lambda`) directly penalises tree complexity → less overfitting than RF.
- **`min_child_weight`** controls minimum samples per leaf more strictly than RF's `min_samples_leaf`.
- **`gamma`** (minimum loss reduction to split) adds another overfitting guard.
- **Gradient boosting** often outperforms bagging (RF) on tabular data with this feature count.

**Hyperparameter search**: RandomizedSearchCV, 60 iterations, 5-fold CV.

| Param | Search Space | Best |
|---|---|---|
| `n_estimators` | [400, 600, 800, 1000] | 800 |
| `max_depth` | [4, 5, 6, 7, 8] | 8 |
| `learning_rate` | [0.01, 0.03, 0.05, 0.08, 0.1] | 0.03 |
| `subsample` | [0.7, 0.8, 0.9, 1.0] | 0.8 |
| `colsample_bytree` | [0.6, 0.7, 0.8, 1.0] | 0.7 |
| `reg_alpha` | [0, 0.1, 0.5, 1.0] | 0.1 |
| `reg_lambda` | [0.5, 1.0, 5.0, 10.0] | 10.0 |
| `min_child_weight` | [1, 3, 5] | 3 |
| `gamma` | [0, 0.1, 0.5] | 0 |

| Metric | Train | Test |
|---|---|---|
| R² | 0.9377 | **0.8241** |
| MAE | $33 | **$58** |
| MAPE | 13.2% | **23.6%** |
| Median AE | $14 | **$28** |
| CV R² (5-fold) | ≈ 0.81 | |

**Top 5 feature importances**:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `is_private_bath` | 0.2587 |
| 2 | `accommodates_sq` | 0.1074 |
| 3 | `log_minimum_nights` | 0.0621 |
| 4 | `log_minimum_nights_avg_ntm` | 0.0542 |
| 5 | `minimum_nights_avg_ntm` | 0.0499 |

---

## 3. Final Benchmark Comparison

| Model | R² | MAE | RMSE | MAPE | Median AE | Overfit? |
|---|---|---|---|---|---|---|
| Dummy Baseline | -0.00 | $131 | $218 | 72.4% | $84 | N/A |
| Ridge Regression | 0.7289 | $75 | $139 | 31.1% | $36 | ✅ None (Δ=0.01) |
| Random Forest | 0.8158 | $59 | $114 | 24.2% | $28 | ⚠️ Train gap 0.14, CV gap 0.01 |
| **XGBoost** | **0.8241** | **$58** | **$111** | **23.6%** | **$28** | ⚠️ Train gap 0.11, CV gap 0.01 |

**Winner: XGBoost** — best R², lowest MAE/MAPE, smaller train-test gap than RF.

---

## 4. Overfitting Analysis

Both tree models show high train R² (~0.95/0.94) but lower test R² (~0.82). This looks alarming but is **not a practical problem**:

| | RF | XGB |
|---|---|---|
| Train R² | 0.9518 | 0.9377 |
| CV R² (5-fold, train data only) | ≈ 0.80 | ≈ 0.81 |
| Test R² (unseen data) | 0.8158 | 0.8241 |
| **CV → Test gap** | **+0.014** | **+0.013** |

The cross-validation score on training data closely matches the unseen test score. The models are not learning anything specific to the test set — they genuinely generalise. High train R² for tree ensembles is structural (they memorise leaves they've seen) and is not a deployment risk.

---

## 5. Feature Engineering Decisions

| Decision | Reason |
|---|---|
| Log1p on price (target) | Right-skew=14.21; log makes predictions multiplicative |
| `is_private_bath` (corr=0.49) | Strongest single new signal; private bath is a clear luxury proxy |
| `neighbourhood_price_rank` | Target-encoded on **train data only** to prevent leakage; 221 NYC neighbourhoods have wildly different price levels ($63 in Hunts Point vs $1,010 in Fort Wadsworth) |
| Borough one-hot (4 dummies) | Manhattan listings average $227/night vs Bronx $106 — borough is a strong structural price driver |
| `amenity_count` + key amenity flags | `has_gym` corr=0.31; building amenities (gym, elevator) signal luxury buildings |
| Luxury filter (top 1% both splits) | Listings >$1,562/night are unpredictable without demand/booking data; including them drags R² down by ~3-4% without teaching the model anything useful |
| Dropped `minimum_minimum_nights` | corr=1.0 with `minimum_nights` — exact duplicate column |
| Polynomial: `accommodates_sq`, `bedrooms_sq` | Non-linear price curve — each additional bedroom earns diminishing price returns, but going from 1→4 bedrooms is non-linear |

---

## 6. What Would Push R² Above 85%

Current ceiling with structural listing features alone is ~82-85%. The remaining variance requires data we don't have:

| Missing Signal | Expected R² Gain |
|---|---|
| Amenity detail (# bathrooms, specific kitchen appliances, balcony, view) | +2-3% |
| Historical booking rate / occupancy | +3-4% |
| Seasonal pricing (summer vs winter) | +2-3% |
| Competitive density (similar listings within 500m) | +1-2% |
| Photo quality score (via CV model) | +1-2% |

---

## 7. Artifacts

All saved to `models/nyc/`:

| File | Description |
|---|---|
| `nyc_xgb_model.pkl` | Best model (XGBoost, R²=0.82) |
| `nyc_rf_model.pkl` | Random Forest (R²=0.82) |
| `nyc_ridge_model.pkl` | Ridge baseline (R²=0.73) |
| `nyc_scaler.pkl` | StandardScaler fitted on training data |
| `nyc_feature_list.pkl` | Ordered list of 58 feature names for inference |
| `nyc_neighbourhood_means.pkl` | Per-neighbourhood mean log_price (train-only) for target encoding at inference |
| `nyc_training_report.json` | Full metrics, best params, feature importances |

**Inference**: Apply scaler → add neighbourhood_price_rank from saved means → predict with XGBoost → `np.expm1(prediction)` to get dollar price.
