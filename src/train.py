"""
Training pipeline
Data preparation → Feature engineering → Scale → Train → Evaluate → Save
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from data_prep import prepare_data
from feature_engineering import engineer_features
from config_loader import get_config

logger = logging.getLogger(__name__)


def prepare_features(df):
    """Prepare X and y (drop unnecessary columns)."""
    cfg = get_config()
    target = cfg['data']['target_column']
    X = df.drop(['Unnamed: 0', target, 'price'], axis=1, errors='ignore')
    y = df[target]
    logger.info("Features prepared — target: '%s', X: %s, y: %s", target, X.shape, y.shape)
    return X, y


def split_data(X, y, test_size, random_state):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Split complete — train: %s, test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Fit scaler on train only, apply to both."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Scaling complete — X_train_scaled: %s, X_test_scaled: %s",
                X_train_scaled.shape, X_test_scaled.shape)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train_scaled, y_train, model_params):
    """Train Random Forest model using params from config."""
    logger.info("Training RandomForestRegressor with params: %s", model_params)
    model = RandomForestRegressor(**model_params)
    model.fit(X_train_scaled, y_train)
    logger.info("Model training complete")
    return model


def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    error_pct = (rmse / y_test.mean()) * 100

    metrics = {
        'r2': float(r2),
        'mse': float(mse),
        'rmse': float(rmse),
        'error_pct': float(error_pct),
        'avg_price': float(y_test.mean())
    }

    logger.info("Evaluation — R²: %.4f, RMSE: $%.0f, Error%%: %.1f%%",
                r2, rmse * 1000, error_pct)
    return metrics


def save_model(model, scaler, features, metrics, output_dir):
    """Save model, scaler, features list, and metrics config."""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model,    os.path.join(output_dir, 'model.pkl'))
    joblib.dump(scaler,   os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(features, os.path.join(output_dir, 'features.pkl'))

    cfg = get_config()
    config = {
        'model_type': cfg['model']['type'],
        'features_count': len(features),
        'features': features,
        'metrics': metrics
    }
    joblib.dump(config, os.path.join(output_dir, 'config.pkl'))

    logger.info("Model artifacts saved to '%s'", output_dir)
    return config


def train_pipeline():
    """Complete training pipeline — all values driven by config.yaml."""
    cfg = get_config()

    data_path  = cfg['data']['raw_path']
    output_dir = cfg['output']['model_dir']
    test_size  = cfg['train']['test_size']
    random_state = cfg['train']['random_state']
    model_params = cfg['model']['params']

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE START")
    logger.info("=" * 60)

    logger.info("Step 1: Data preparation")
    df_clean = prepare_data(filepath=data_path)

    logger.info("Step 2: Feature engineering")
    df_clean = engineer_features(df_clean)

    logger.info("Step 3: Preparing features")
    X, y = prepare_features(df_clean)

    logger.info("Step 4: Splitting data")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    logger.info("Step 5: Scaling features")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    logger.info("Step 6: Training model")
    model = train_model(X_train_scaled, y_train, model_params)

    logger.info("Step 7: Evaluating model")
    metrics = evaluate_model(model, X_test_scaled, y_test)

    logger.info("Step 8: Saving model")
    features = X_train.columns.tolist()
    config = save_model(model, scaler, features, metrics, output_dir)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return model, scaler, config


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    model, scaler, config = train_pipeline()
    logger.info("Model ready for production — R²: %.4f", config['metrics']['r2'])
