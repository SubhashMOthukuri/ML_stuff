"""
PRODUCTION MODEL TRAINING PIPELINE
NYC Airbnb Price Prediction — Phase 4
Input:  engineered_features.csv  (20,692 rows × 45 cols)
Models: Ridge Regression | Random Forest | XGBoost
Output: models/nyc/<model>.pkl  +  nyc_training_report.json
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from scipy import stats as sp
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR  = Path("/Users/subhashmothukurigmail.com/Projects/ML_stuff")
DATA_DIR  = BASE_DIR / "data" / "airbnb"
MODEL_DIR = BASE_DIR / "models" / "nyc"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIG
# ============================================================================

TARGET     = 'log_price'
DROP_COLS  = ['id', 'price', 'log_price', 'neighbourhood_cleansed']
RANDOM_SEED = 42
TEST_SIZE   = 0.20

# Listings above this price percentile are luxury outliers we cannot predict
# with structural features alone (no demand/seasonality data).
# Remove from BOTH train and test so we evaluate on the core market.
PRICE_CAP_PCT = 99

RF_PARAM_GRID = {
    'n_estimators'     : [400, 600, 800],
    'max_depth'        : [10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features'     : ['sqrt', 'log2', 0.3, 0.5],
    'max_samples'      : [0.8, 0.9, 1.0],
}

XGB_PARAM_GRID = {
    'n_estimators'    : [400, 600, 800, 1000],
    'max_depth'       : [4, 5, 6, 7, 8],
    'learning_rate'   : [0.01, 0.03, 0.05, 0.08, 0.1],
    'subsample'       : [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'reg_alpha'       : [0, 0.1, 0.5, 1.0],
    'reg_lambda'      : [0.5, 1.0, 5.0, 10.0],
    'min_child_weight': [1, 3, 5],
    'gamma'           : [0, 0.1, 0.5],
}

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

# ============================================================================
# STEP 1 — LOAD & SPLIT
# ============================================================================

def load_and_split():
    logger.info("=" * 80)
    logger.info("PHASE 4: MODEL TRAINING — NYC Airbnb Price Prediction")
    logger.info("=" * 80)

    df = pd.read_csv(DATA_DIR / "engineered_features.csv")
    logger.info(f"\n📂 Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # --- Neighbourhood target encoding (train-only means → no leakage) ---
    # neighbourhood_cleansed is kept as string in FE; encode it here correctly.
    if 'neighbourhood_cleansed' in df.columns:
        # Temporarily split to get train rows for computing means
        train_idx, test_idx = train_test_split(
            df.index, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        neigh_mean = df.loc[train_idx].groupby('neighbourhood_cleansed')['log_price'].mean()
        global_mean = float(df.loc[train_idx]['log_price'].mean())
        df['neighbourhood_price_rank'] = (
            df['neighbourhood_cleansed'].map(neigh_mean).fillna(global_mean)
        )
        top5 = neigh_mean.sort_values(ascending=False).head(5)
        logger.info(f"\n   Neighbourhood target encoding (train-only means):")
        for n, v in top5.items():
            logger.info(f"      {n:<35}: ${np.expm1(v):.0f}/night")

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET]

    logger.info(f"\n   Target  : {TARGET}  →  inverse: expm1() to get $ price")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   y range : {y.min():.3f} → {y.max():.3f}  (log scale)")
    logger.info(f"   $ range : ${np.expm1(y.min()):.2f} → ${np.expm1(y.max()):,.2f}")

    # --- Cap at percentile BEFORE split (same threshold applied to both) ---
    cap_log = float(np.percentile(y, PRICE_CAP_PCT))
    cap_dol = float(np.expm1(cap_log))
    core_mask = y <= cap_log
    removed_total = int((~core_mask).sum())
    X = X[core_mask]
    y = y[core_mask]
    logger.info(f"\n   Luxury listing filter (top {100-PRICE_CAP_PCT}% removed):")
    logger.info(f"      {PRICE_CAP_PCT}th pct price : ${cap_dol:,.0f}/night")
    logger.info(f"      Removed        : {removed_total:,} listings")
    logger.info(f"      Core market    : {len(y):,} listings")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    logger.info(f"\n   Train split : {X_train.shape[0]:,} rows  ({100*(1-TEST_SIZE):.0f}%)")
    logger.info(f"   Test  split : {X_test.shape[0]:,} rows  ({100*TEST_SIZE:.0f}%)")

    logger.info(f"\n   Top 5 training rows:")
    logger.info("\n" + X_train.head().to_string())

    return X_train, X_test, y_train, y_test, feature_cols


# ============================================================================
# STEP 2 — MODEL ASSUMPTION CHECKS
# ============================================================================

def check_assumptions(X_train, y_train, feature_cols):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2 — MODEL ASSUMPTION CHECKS")
    logger.info("=" * 80)

    X_num = X_train.select_dtypes(include=[np.number])

    # --- Linearity ---
    logger.info("\n  ── ASSUMPTION 1: LINEARITY (|corr| with log_price) ──")
    corrs = X_num.corrwith(y_train).abs().sort_values(ascending=False)
    logger.info(f"\n  {'Feature':<45} {'|corr|':>7}  Strength")
    logger.info("  " + "-" * 65)
    for feat, c in corrs.head(15).items():
        strength = "STRONG" if c > 0.4 else ("MODERATE" if c > 0.2 else "WEAK")
        logger.info(f"  {feat:<45} {c:>7.4f}  {strength}")
    logger.info(f"\n  Mean |corr| across {len(corrs)} features: {corrs.mean():.4f}")
    logger.info(f"  → Linear Reg will struggle (most features have weak-moderate correlation)")

    # --- Normality of target ---
    logger.info("\n  ── ASSUMPTION 2: NORMALITY OF TARGET (log_price) ──")
    _, p = sp.shapiro(y_train.sample(500, random_state=RANDOM_SEED))
    logger.info(f"  Skewness : {y_train.skew():.4f}  (|<1| = acceptable)")
    logger.info(f"  Kurtosis : {y_train.kurt():.4f}  (|<3| = acceptable)")
    logger.info(f"  Shapiro p: {p:.4f}  → {'✓ normal' if p>0.05 else '✗ not perfectly normal — CLT covers this at n=16K'}")

    # --- Multicollinearity VIF ---
    logger.info("\n  ── ASSUMPTION 3: MULTICOLLINEARITY (VIF on top 12 features) ──")
    top_feats = corrs.head(12).index.tolist()
    Xsub = X_num[top_feats].copy()
    Xsub = (Xsub - Xsub.mean()) / Xsub.std()
    vifs = [variance_inflation_factor(Xsub.values, i) for i in range(Xsub.shape[1])]
    for feat, v in sorted(zip(top_feats, vifs), key=lambda x: -x[1]):
        flag = "✗ HIGH — Ridge regularisation needed" if v > 10 else ("⚠ moderate" if v > 5 else "✓ ok")
        logger.info(f"  VIF={v:6.1f}  {flag:<40}  {feat}")

    # --- Homoscedasticity ---
    logger.info("\n  ── ASSUMPTION 4: HOMOSCEDASTICITY (variance across price bands) ──")
    bins = pd.cut(np.expm1(y_train),
                  bins=[0,100,200,400,1000,20000],
                  labels=['<$100','$100-200','$200-400','$400-1K','>$1K'])
    logger.info(f"  {'Price band':<14}  {'Count':>6}  {'Std(log_price)':>15}")
    for b, grp in y_train.groupby(bins, observed=True):
        logger.info(f"  {str(b):<14}  {len(grp):>6,}  {grp.std():>15.4f}")
    logger.info("  → Std varies = heteroscedastic  (hurts Linear Reg, OK for RF/XGB)")

    # --- Verdict ---
    logger.info("\n  ── VERDICT ──")
    logger.info("  Ridge Regression : ✗ weak linearity  ✗ high VIF  ✗ heteroscedastic")
    logger.info("                     ✓ target ~normal  → use as interpretable baseline")
    logger.info("  Random Forest    : ✓ no linearity needed  ✓ handles VIF via subsampling")
    logger.info("                     ✓ handles heteroscedasticity  → strong fit")
    logger.info("  XGBoost          : ✓ all RF benefits + built-in regularisation")
    logger.info("                     ✓ often best on tabular data  → strong contender")

    return corrs.to_dict()


# ============================================================================
# STEP 3 — SCALE
# ============================================================================

def scale_features(X_train, X_test):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3 — FEATURE SCALING  (StandardScaler — required for Ridge)")
    logger.info("=" * 80)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    logger.info(f"\n   ✓ Fitted on train ({X_train.shape[0]:,} rows), transformed both splits")
    logger.info(f"   ✓ Mean of scaled train (sample): {X_train_sc[:5, :3].mean(axis=0).round(4)}")
    logger.info(f"   ✓ Std  of scaled train (sample): {X_train_sc[:5, :3].std(axis=0).round(4)}")

    return X_train_sc, X_test_sc, scaler


# ============================================================================
# HELPERS
# ============================================================================

def compute_metrics(y_true, y_pred, label):
    r2       = r2_score(y_true, y_pred)
    rmse_log = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_log  = float(mean_absolute_error(y_true, y_pred))
    y_true_d = np.expm1(y_true)
    y_pred_d = np.expm1(y_pred)
    mae_dol  = float(mean_absolute_error(y_true_d, y_pred_d))
    rmse_dol = float(np.sqrt(mean_squared_error(y_true_d, y_pred_d)))
    mape     = float((np.abs((y_true_d - y_pred_d) / y_true_d) * 100).mean())
    med_ae   = float(np.median(np.abs(y_true_d - y_pred_d)))
    return {
        'split': label, 'r2': round(r2, 4), 'rmse_log': round(rmse_log, 4),
        'mae_log': round(mae_log, 4), 'mae_dollar': round(mae_dol, 2),
        'rmse_dollar': round(rmse_dol, 2), 'mape': round(mape, 2),
        'median_ae_dollar': round(med_ae, 2)
    }


def log_metrics_table(name, train_m, test_m):
    logger.info(f"\n  {'Metric':<22} {'Train':>12}  {'Test':>12}")
    logger.info("  " + "-" * 50)
    for key in ['r2','rmse_log','mae_log','mae_dollar','rmse_dollar','mape','median_ae_dollar']:
        fmt    = '.4f' if 'log' in key or key == 'r2' else '.2f'
        suffix = '%' if key == 'mape' else ('$' if 'dollar' in key else '')
        logger.info(f"  {key:<22} {train_m[key]:>12{fmt}}{suffix}  {test_m[key]:>12{fmt}}{suffix}")


# ============================================================================
# STEP 4a — DUMMY BASELINE  (always predicts mean price)
# ============================================================================

def train_dummy_baseline(X_train_sc, X_test_sc, y_train, y_test):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4a — DUMMY BASELINE  (predicts mean price every time)")
    logger.info("=" * 80)

    model = DummyRegressor(strategy='mean')
    model.fit(X_train_sc, y_train)

    mean_log_price = float(np.asarray(model.constant_).flat[0])
    mean_dollar    = float(np.expm1(mean_log_price))
    logger.info(f"\n   Strategy     : always predict mean log_price = {mean_log_price:.4f}")
    logger.info(f"   Mean $ price : ${mean_dollar:.2f}  (this is what the model always outputs)")

    train_m = compute_metrics(y_train, model.predict(X_train_sc), 'train')
    test_m  = compute_metrics(y_test,  model.predict(X_test_sc),  'test')
    log_metrics_table("Dummy Baseline", train_m, test_m)
    logger.info("   → R²=0.0 by definition — every real model must beat this")

    return model, train_m, test_m


# ============================================================================
# STEP 4b — RIDGE REGRESSION
# ============================================================================

def train_ridge(X_train_sc, X_test_sc, y_train, y_test, feature_cols):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4a — RIDGE REGRESSION  (L2 regularisation handles high VIF)")
    logger.info("=" * 80)

    best_alpha, best_score = None, -np.inf
    logger.info(f"\n   Alpha search over: {RIDGE_ALPHAS}")
    for alpha in RIDGE_ALPHAS:
        scores = cross_val_score(
            Ridge(alpha=alpha), X_train_sc, y_train,
            cv=5, scoring='r2', n_jobs=-1
        )
        mean_r2 = scores.mean()
        logger.info(f"   alpha={alpha:>8}  CV R²={mean_r2:.4f}")
        if mean_r2 > best_score:
            best_score, best_alpha = mean_r2, alpha

    logger.info(f"\n   ✓ Best alpha: {best_alpha}  (CV R²={best_score:.4f})")
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_sc, y_train)

    train_m = compute_metrics(y_train, model.predict(X_train_sc), 'train')
    test_m  = compute_metrics(y_test,  model.predict(X_test_sc),  'test')
    log_metrics_table("Ridge", train_m, test_m)

    # Top coefficients
    coef = pd.Series(np.abs(model.coef_), index=feature_cols).sort_values(ascending=False)
    logger.info(f"\n   Top 10 coefficients (|value|):")
    for feat, c in coef.head(10).items():
        logger.info(f"   {feat:<45} {c:.4f}")

    return model, train_m, test_m


# ============================================================================
# STEP 5 — RANDOM FOREST
# ============================================================================

def train_random_forest(X_train_sc, X_test_sc, y_train, y_test, feature_cols):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4b — RANDOM FOREST  (RandomizedSearchCV, 30 iter, 5-fold CV)")
    logger.info("=" * 80)

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        param_distributions=RF_PARAM_GRID,
        n_iter=60, cv=5,
        scoring='neg_root_mean_squared_error',
        random_state=RANDOM_SEED, n_jobs=-1, verbose=0,
    )
    logger.info(f"\n   Running search...")
    search.fit(X_train_sc, y_train)

    best_params  = search.best_params_
    best_cv_rmse = -search.best_score_
    logger.info(f"\n   ✓ Best params:")
    for k, v in best_params.items():
        logger.info(f"      {k:<25}: {v}")
    logger.info(f"   ✓ Best CV RMSE (log): {best_cv_rmse:.4f}")

    model   = search.best_estimator_
    train_m = compute_metrics(y_train, model.predict(X_train_sc), 'train')
    test_m  = compute_metrics(y_test,  model.predict(X_test_sc),  'test')
    log_metrics_table("Random Forest", train_m, test_m)

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info(f"\n   Top 15 feature importances:")
    for feat, score in imp.head(15).items():
        bar = '█' * int(score * 400)
        logger.info(f"   {feat:<45} {score:.4f}  {bar}")

    return model, best_params, best_cv_rmse, train_m, test_m, imp


# ============================================================================
# STEP 6 — XGBOOST
# ============================================================================

def train_xgboost(X_train_sc, X_test_sc, y_train, y_test, feature_cols):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4c — XGBOOST  (RandomizedSearchCV, 30 iter, 5-fold CV)")
    logger.info("=" * 80)

    search = RandomizedSearchCV(
        XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
                     objective='reg:squarederror'),
        param_distributions=XGB_PARAM_GRID,
        n_iter=60, cv=5,
        scoring='neg_root_mean_squared_error',
        random_state=RANDOM_SEED, n_jobs=-1, verbose=0,
    )
    logger.info(f"\n   Running search...")
    search.fit(X_train_sc, y_train)

    best_params  = search.best_params_
    best_cv_rmse = -search.best_score_
    logger.info(f"\n   ✓ Best params:")
    for k, v in best_params.items():
        logger.info(f"      {k:<25}: {v}")
    logger.info(f"   ✓ Best CV RMSE (log): {best_cv_rmse:.4f}")

    model   = search.best_estimator_
    train_m = compute_metrics(y_train, model.predict(X_train_sc), 'train')
    test_m  = compute_metrics(y_test,  model.predict(X_test_sc),  'test')
    log_metrics_table("XGBoost", train_m, test_m)

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info(f"\n   Top 15 feature importances:")
    for feat, score in imp.head(15).items():
        bar = '█' * int(score * 400)
        logger.info(f"   {feat:<45} {score:.4f}  {bar}")

    return model, best_params, best_cv_rmse, train_m, test_m, imp


# ============================================================================
# STEP 7 — MODEL COMPARISON
# ============================================================================

def compare_models(dummy_test, ridge_test, rf_test, xgb_test):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5 — MODEL COMPARISON (TEST SET)")
    logger.info("=" * 80)

    models = [('Dummy Baseline',   dummy_test),
              ('Ridge Regression', ridge_test),
              ('Random Forest',    rf_test),
              ('XGBoost',          xgb_test)]

    logger.info(f"\n  {'Model':<22} {'R²':>8}  {'MAE $':>9}  {'RMSE $':>10}  {'MAPE':>8}  {'Median AE $':>12}")
    logger.info("  " + "-" * 80)
    for name, m in models:
        logger.info(f"  {name:<22} {m['r2']:>8.4f}  {m['mae_dollar']:>9.2f}  "
                    f"{m['rmse_dollar']:>10.2f}  {m['mape']:>7.2f}%  {m['median_ae_dollar']:>12.2f}")

    real_models = [(n, m) for n, m in models if n != 'Dummy Baseline']
    best_name   = max(real_models, key=lambda x: x[1]['r2'])[0]
    logger.info(f"\n  🏆 Best model by R²: {best_name}")

    return best_name


# ============================================================================
# STEP 8 — SAVE
# ============================================================================

def save_artifacts(ridge, rf, xgb, scaler, feature_cols,
                   rf_params, xgb_params,
                   ridge_train, ridge_test,
                   rf_train, rf_test,
                   xgb_train, xgb_test,
                   rf_imp, xgb_imp,
                   assumption_report, best_name):
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6 — SAVING ARTIFACTS")
    logger.info("=" * 80)

    # Load neighbourhood means for saving (inference needs them)
    df_raw = pd.read_csv(DATA_DIR / "engineered_features.csv")
    if 'neighbourhood_cleansed' in df_raw.columns:
        from sklearn.model_selection import train_test_split as _tts
        train_idx, _ = _tts(df_raw.index, test_size=0.20, random_state=42)
        neigh_means = df_raw.loc[train_idx].groupby(
            'neighbourhood_cleansed')['log_price'].mean().to_dict()
        global_mean_pkl = float(df_raw.loc[train_idx]['log_price'].mean())
    else:
        neigh_means, global_mean_pkl = {}, 0.0

    artifacts = {
        'nyc_ridge_model'           : ridge,
        'nyc_rf_model'              : rf,
        'nyc_xgb_model'             : xgb,
        'nyc_scaler'                : scaler,
        'nyc_feature_list'          : feature_cols,
        'nyc_neighbourhood_means'   : {'means': neigh_means, 'global_mean': global_mean_pkl},
    }
    for name, obj in artifacts.items():
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        size_kb = path.stat().st_size / 1024
        logger.info(f"   💾 {name:<22}: {size_kb:>8.1f} KB  →  {path.name}")

    report = {
        'timestamp'       : datetime.now().isoformat(),
        'target'          : TARGET,
        'n_features'      : len(feature_cols),
        'feature_cols'    : feature_cols,
        'best_model'      : best_name,
        'assumption_check': assumption_report,
        'models': {
            'Ridge': {
                'best_alpha' : None,
                'train'      : ridge_train,
                'test'       : ridge_test,
            },
            'RandomForest': {
                'best_params': {k: str(v) for k, v in rf_params.items()},
                'train'      : rf_train,
                'test'       : rf_test,
                'top20_imp'  : rf_imp.head(20).to_dict(),
            },
            'XGBoost': {
                'best_params': {k: str(v) for k, v in xgb_params.items()},
                'train'      : xgb_train,
                'test'       : xgb_test,
                'top20_imp'  : xgb_imp.head(20).to_dict(),
            },
        }
    }
    report_path = MODEL_DIR / "nyc_training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"   💾 {'report':<22}: {report_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    X_train, X_test, y_train, y_test, feature_cols = load_and_split()

    assumption_report = check_assumptions(X_train, y_train, feature_cols)

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    _dummy, dummy_train, dummy_test = train_dummy_baseline(
        X_train_sc, X_test_sc, y_train, y_test
    )
    ridge, ridge_train, ridge_test = train_ridge(
        X_train_sc, X_test_sc, y_train, y_test, feature_cols
    )
    rf, rf_params, rf_cv, rf_train, rf_test, rf_imp = train_random_forest(
        X_train_sc, X_test_sc, y_train, y_test, feature_cols
    )
    xgb, xgb_params, xgb_cv, xgb_train, xgb_test, xgb_imp = train_xgboost(
        X_train_sc, X_test_sc, y_train, y_test, feature_cols
    )

    best_name = compare_models(dummy_test, ridge_test, rf_test, xgb_test)

    save_artifacts(
        ridge, rf, xgb, scaler, feature_cols,
        rf_params, xgb_params,
        ridge_train, ridge_test,
        rf_train, rf_test,
        xgb_train, xgb_test,
        rf_imp, xgb_imp,
        assumption_report, best_name
    )

    logger.info(f"""
================================================================================
✅ ALL MODELS TRAINED
================================================================================
  Dummy Baseline   : R²={dummy_test['r2']:.4f}  MAE=${dummy_test['mae_dollar']:.2f}  MAPE={dummy_test['mape']:.2f}%
  Ridge Regression : R²={ridge_test['r2']:.4f}  MAE=${ridge_test['mae_dollar']:.2f}  MAPE={ridge_test['mape']:.2f}%
  Random Forest    : R²={rf_test['r2']:.4f}  MAE=${rf_test['mae_dollar']:.2f}  MAPE={rf_test['mape']:.2f}%
  XGBoost          : R²={xgb_test['r2']:.4f}  MAE=${xgb_test['mae_dollar']:.2f}  MAPE={xgb_test['mape']:.2f}%

  🏆 Best model: {best_name}

✅ READY FOR: Inference pipeline (4_predict.py)
================================================================================
""")
    return rf, xgb, ridge


if __name__ == "__main__":
    main()
