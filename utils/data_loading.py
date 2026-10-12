"""Data I/O helpers: .gpkg loading, model persistence, and per-run config loading."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from utils.train_config import CATEGORICAL_COLS


def sanitize_numeric(df, verbose=True):
    """
    Replace values XGBoost cannot ingest with NaN, which it handles natively as missing.
    """
    float32_max = np.finfo(np.float32).max
    bad_counts = {}

    for col in df.select_dtypes(include=["floating"]).columns:
        values = df[col].to_numpy()
        bad = np.isinf(values) | (np.abs(values) > float32_max)
        n_bad = int(bad.sum())
        if n_bad:
            bad_counts[col] = n_bad
            df.loc[bad, col] = np.nan

    if bad_counts and verbose:
        total = sum(bad_counts.values())
        print(f"Replaced {total} non-finite / out-of-float32-range values with NaN:")
        for col, n in sorted(bad_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {col}: {n}")

    return df


def load_gpkg(path, layer=None, geometry_col="geom", categories=None):
    """
    Read a .gpkg into a plain pandas DataFrame with categorical dtypes applied.
    Drops the geometry column since the model consumes tabular features only.
    """
    gdf = gpd.read_file(path, layer=layer)
    df = gdf.drop(columns=[geometry_col], errors="ignore")

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if categories and col in categories:
            was_na = df[col].isna()
            df[col] = df[col].astype(pd.CategoricalDtype(categories=categories[col]))
            unseen = df[col].isna() & ~was_na
            if unseen.any():
                print(f"WARNING: {unseen.sum()} rows have '{col}' values not seen during "
                      f"training; set to NaN (XGBoost treats them as missing)")
        else:
            df[col] = df[col].astype("category")

    return sanitize_numeric(df)


def save_model(model, path):
    """
    Save a fitted XGBRegressor in XGBoost's native format.
    """
    model.save_model(str(path))


def load_model(path, device="cuda"):
    """
    Load an XGBRegressor saved with save_model().
    """
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    model.set_params(device=device)
    return model


def save_feature_list(features, categorical_cols, path, best_iteration=None, categories=None):
    """
    Write the exact feature list a model was trained on, as a JSON artifact.
    """
    payload = {
        "features": list(features),
        "categorical_cols": list(categorical_cols),
        "best_iteration": None if best_iteration is None else int(best_iteration),
        "categories": None if categories is None else {k: list(v) for k, v in categories.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_feature_list(path):
    """
    Load a feature-list JSON artifact written by save_feature_list().
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run_config(path):
    """
    Load a per-run YAML config.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)