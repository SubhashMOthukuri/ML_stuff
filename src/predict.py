"""
Production prediction module
Load model and make predictions
"""

import joblib
import pandas as pd
import numpy as np
import logging

from config_loader import get_config
from feature_engineering import (
    create_polynomial_features,
    create_interaction_features,
    create_ratio_features,
)

logger = logging.getLogger(__name__)

# The 13 base features a caller provides — engineered features are derived internally
RAW_FEATURE_NAMES = [
    'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
    'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'
]


class HousePricePredictor:
    """Production-grade house price predictor."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            cfg = get_config()
            model_dir = cfg['output']['model_dir']

        self.model   = joblib.load(f'{model_dir}/model.pkl')
        self.scaler  = joblib.load(f'{model_dir}/scaler.pkl')
        self.features = joblib.load(f'{model_dir}/features.pkl')
        self.config  = joblib.load(f'{model_dir}/config.pkl')

        logger.info(
            "Model loaded from '%s' — features: %d, R²: %.4f",
            model_dir, len(self.features), self.config['metrics']['r2']
        )

    def _validate_raw(self, raw_dict):
        """Raise ValueError only for values that cause computation errors."""
        checks = {
            'crim':    (lambda v: v >= 0,    "must be >= 0"),
            'rm':      (lambda v: v > 0,     "must be > 0 (used as divisor in tax_per_room)"),
            'dis':     (lambda v: v > 0,     "must be > 0"),
            'tax':     (lambda v: v > 0,     "must be > 0"),
            'ptratio': (lambda v: v > 0,     "must be > 0"),
            'nox':     (lambda v: 0 <= v <= 1, "must be between 0 and 1"),
            'age':     (lambda v: v >= 0,    "must be >= 0"),
        }
        for field, (check, msg) in checks.items():
            value = raw_dict.get(field)
            if value is not None and not check(value):
                raise ValueError(f"Invalid value for '{field}': {value} — {msg}")

    def predict_raw(self, raw_features_dict):
        """
        Accept the 13 base features, apply the same feature engineering pipeline
        used during training, then scale and predict.

        This is the preferred endpoint — callers never need to pre-compute
        age_squared, rm_squared, etc.
        """
        self._validate_raw(raw_features_dict)

        df = pd.DataFrame([raw_features_dict])[RAW_FEATURE_NAMES]
        df = create_polynomial_features(df)
        df = create_interaction_features(df)
        df = create_ratio_features(df)

        df = df[self.features]  # reorder to exactly match training column order
        df_scaled = self.scaler.transform(df)
        prediction = self.model.predict(df_scaled)[0]
        logger.debug("Raw prediction: %.4f", prediction)
        return float(prediction)

    def predict(self, features_dict):
        """
        Make a single prediction from a pre-computed 18-feature dictionary.
        Prefer predict_raw() — this exists for callers that already have all features.
        """
        df = pd.DataFrame([features_dict])
        df = df[self.features]
        df_scaled = self.scaler.transform(df)
        prediction = self.model.predict(df_scaled)[0]
        logger.debug("Prediction: %.4f", prediction)
        return float(prediction)

    def predict_batch_raw(self, raw_features_list):
        """
        Batch version of predict_raw(). Applies feature engineering to each row,
        logs and skips invalid rows rather than failing the whole batch.
        """
        results = []
        for i, raw in enumerate(raw_features_list):
            try:
                pred = self.predict_raw(raw)
                results.append({"index": i, "prediction": pred})
            except Exception as e:
                logger.error("Batch raw prediction failed for index %d: %s", i, e)
        logger.info("Batch raw complete — %d/%d succeeded", len(results), len(raw_features_list))
        return results

    def predict_batch(self, features_list):
        """
        Batch version of predict(). Logs and skips failed rows.
        """
        results = []
        for i, features in enumerate(features_list):
            try:
                pred = self.predict(features)
                results.append({"index": i, "prediction": pred})
            except Exception as e:
                logger.error("Batch prediction failed for index %d: %s", i, e)
        logger.info("Batch complete — %d/%d succeeded", len(results), len(features_list))
        return results

    def get_model_info(self):
        """Return model performance metadata."""
        return {
            'R²': f"{self.config['metrics']['r2']:.4f}",
            'RMSE': f"${self.config['metrics']['rmse'] * 1000:,.0f}",
            'Error%': f"{self.config['metrics']['error_pct']:.1f}%",
            'Features': len(self.features),
            'Model_Type': self.config['model_type']
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    predictor = HousePricePredictor()

    logger.info("Model info: %s", predictor.get_model_info())

    example = {
        'crim': 0.5, 'zn': 18.0, 'indus': 2.3, 'chas': 0,
        'nox': 0.54, 'rm': 6.5, 'age': 65, 'dis': 4.1,
        'rad': 1, 'tax': 296, 'ptratio': 15.3, 'b': 400,
        'lstat': 5.0,
        'age_squared': 4225, 'rm_squared': 42.25,
        'rm_lstat_interaction': 32.5, 'tax_per_room': 45.5,
        'dis_squared': 16.81
    }

    price = predictor.predict(example)
    logger.info("Predicted price: $%,.0f", price * 1000)
