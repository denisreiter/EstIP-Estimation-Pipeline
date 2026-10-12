"""Correlation and grouped error-statistics helpers shared across scripts/notebooks."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(x, y):
    """
    Cramer's V for a pair of categorical variables.
    """
    valid_idx = (~x.isna()) & (~y.isna())
    x = x[valid_idx]
    y = y[valid_idx]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    confusion = pd.crosstab(x, y)

    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return np.nan

    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return np.sqrt(phi2 / min(r - 1, k - 1))


def correlation_ratio(categories, values):
    """
    Correlation ratio (eta) for a categorical/numeric variable pair.
    """
    valid_idx = categories.notna() & values.notna()
    categories = categories[valid_idx]
    values = values[valid_idx]

    if len(values) == 0 or categories.nunique() < 2:
        return np.nan

    group_stats = values.groupby(categories, observed=True).agg(["count", "mean"])
    group_stats = group_stats[group_stats["count"] > 0]
    if len(group_stats) < 2:
        return np.nan

    mean_total = values.mean()
    ss_between = (group_stats["count"] * (group_stats["mean"] - mean_total) ** 2).sum()
    ss_total = ((values - mean_total) ** 2).sum()

    return np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0


def mixed_corr(
        df, 
        cat_cols=None, 
        sample_size=None, 
        random_state=147
    ):
    """
    Mixed correlation matrix: Pearson (numeric-numeric), Cramer's V
    (categorical-categorical), correlation ratio (mixed).
    """
    df = df.copy()

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].cat.remove_unused_categories()

    num_cols = [c for c in df.columns if c not in cat_cols]
    all_cols = num_cols + cat_cols
    corr_matrix = pd.DataFrame(np.zeros((len(all_cols), len(all_cols))),
                               index=all_cols, columns=all_cols)

    for i, col1 in enumerate(all_cols):
        for j, col2 in enumerate(all_cols):
            if i == j:
                corr_matrix.loc[col1, col2] = 1.0
            elif i < j:
                try:
                    if col1 in num_cols and col2 in num_cols:
                        if df[col1].nunique(dropna=True) < 2 or df[col2].nunique(dropna=True) < 2:
                            v = np.nan
                        else:
                            v = df[col1].corr(df[col2], method="pearson")
                    elif col1 in cat_cols and col2 in cat_cols:
                        v = cramers_v(df[col1], df[col2])
                    else:
                        cat_var, num_var = (col1, col2) if col1 in cat_cols else (col2, col1)
                        v = correlation_ratio(df[cat_var], df[num_var])
                except Exception:
                    v = np.nan

                corr_matrix.loc[col1, col2] = v
                corr_matrix.loc[col2, col1] = v

    return corr_matrix


# Shared aggregation spec for the grouped error-stats tables
BASE_ERROR_AGG = {
    "MEAN_INHABITANTS": ("Einwohner", "mean"),
    "MEAN_POPULATION_EST": ("POPULATION_EST", "mean"),
    "MEAN_ERROR": ("ERROR", "mean"),
    "MEAN_ABS_ERROR": ("ABS_ERROR", "mean"),
    "MEAN_REL_ERROR": ("REL_ERROR", "mean"),
    "MEAN_ABS_PERC_ERROR": ("ABS_PERC_ERROR", "mean"),
    "MEAN_IHSE": ("IHSE", "mean"),
    "MEAN_ABS_IHSE": ("ABS_IHSE", "mean"),
    "MEAN_FOLD_ERROR": ("FOLD_ERROR", "mean"),
}


def grouped_error_stats(
        df, 
        by, 
        extra_agg=None, 
        sort_by="MEAN_IHSE", 
        ascending=True
    ):
    """
    Mean actual/predicted/error stats grouped by `by`.
    extra_agg: optional dict of additional named aggregations to merge in
    (e.g. {"n_cells": ("Einwohner", "size")}).
    """
    agg = dict(BASE_ERROR_AGG)
    if extra_agg:
        agg.update(extra_agg)

    grouped = df.groupby(by, observed=True).agg(**agg).reset_index()

    if sort_by:
        grouped = grouped.sort_values(by=sort_by, ascending=ascending)

    return grouped


# German federal state (Bundesland) codes -> names, shared by test.ipynb and analysis.ipynb
STATE_CODE_MAP = {
    "01": "Schleswig-Holstein",
    "02": "Hamburg",
    "03": "Niedersachsen",
    "04": "Bremen",
    "05": "Nordrhein-Westfalen",
    "06": "Hessen",
    "07": "Rheinland-Pfalz",
    "08": "Baden-Württemberg",
    "09": "Bayern",
    "10": "Saarland",
    "11": "Berlin",
    "12": "Brandenburg",
    "13": "Mecklenburg-Vorpommern",
    "14": "Sachsen",
    "15": "Sachsen-Anhalt",
    "16": "Thüringen",
}

# RegioStar17 region-type codes -> names, shared by test.ipynb and analysis.ipynb
REGION_CODE_MAP = {
    "111": "Metropole",
    "112": "Großstadt einer Metropolitanen Stadtregion",
    "113": "Mittelstadt einer Metropolitanen Stadtregion",
    "114": "Städtischer Raum einer Metropolitanen Stadtregion",
    "115": "Kleinstädtischer, dörflicher Raum einer Metropolitanen Stadtregion",
    "121": "Regiopole",
    "123": "Mittelstadt einer Regiopolitanen Stadtregion",
    "124": "Städtischer Raum einer Regiopolitanen Stadtregion",
    "125": "Kleinstädtischer, dörflicher Raum einer Regiopolitanen Stadtregion",
    "211": "Zentrale Stadt einer Stadtregionsnahen ländlichen Region",
    "213": "Mittelstadt einer Stadtregionsnahen ländlichen Region",
    "214": "Städtischer Raum einer Stadtregionsnahen ländlichen Region",
    "215": "Kleinstädtischer, dörflicher Raum einer Stadtregionsnahen ländlichen Region",
    "221": "Zentrale Stadt einer Peripheren ländlichen Region",
    "223": "Mittelstadt einer Peripheren ländlichen Region",
    "224": "Städtischer Raum einer Peripheren ländlichen Region",
    "225": "Kleinstädtischer, dörflicher Raum einer Peripheren ländlichen Region",
}