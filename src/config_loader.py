"""
Load config.yaml once, then apply environment variable overrides.

Environment variables win over config.yaml — use them to change behaviour
between dev / staging / prod without editing any file.

Supported overrides:
  DATA_RAW_PATH        → data.raw_path
  DATA_CLEANED_PATH    → data.cleaned_path
  OUTLIER_THRESHOLD    → data.outlier_threshold  (int)
  MODEL_DIR            → output.model_dir
  API_HOST             → api.host
  API_PORT             → api.port                (int)
  API_TITLE            → api.title
  TEST_SIZE            → train.test_size         (float)
  RANDOM_STATE         → train.random_state      (int)
"""

import os
import yaml
import logging

logger = logging.getLogger(__name__)

_config = None


def get_config(config_path=None):
    global _config
    if _config is not None:
        return _config

    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

    with open(config_path) as f:
        _config = yaml.safe_load(f)

    _apply_env_overrides(_config)
    return _config


def _apply_env_overrides(cfg):
    """Override config values from environment variables."""
    overrides = {
        ('data',   'raw_path'):          ('DATA_RAW_PATH',     str),
        ('data',   'cleaned_path'):      ('DATA_CLEANED_PATH', str),
        ('data',   'outlier_threshold'): ('OUTLIER_THRESHOLD', int),
        ('output', 'model_dir'):         ('MODEL_DIR',         str),
        ('api',    'host'):              ('API_HOST',          str),
        ('api',    'port'):              ('API_PORT',          int),
        ('api',    'title'):             ('API_TITLE',         str),
        ('train',  'test_size'):         ('TEST_SIZE',         float),
        ('train',  'random_state'):      ('RANDOM_STATE',      int),
    }

    for (section, key), (env_var, cast) in overrides.items():
        value = os.getenv(env_var)
        if value is not None:
            cfg[section][key] = cast(value)
            logger.info("Config override from env: %s.%s = %s", section, key, value)
