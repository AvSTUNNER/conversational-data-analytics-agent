"""Statistical utilities for outlier detection, correlations, and descriptive distributions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def calculate_outliers_iqr(
    series: pd.Series, multiplier: float = 1.5
) -> Dict[str, Any]:
    """Detects outliers in a numeric Series using the Interquartile Range (IQR) rule.

    Args:
        series: Numeric Pandas Series.
        multiplier: IQR multiplier (default 1.5 for standard outliers, 3.0 for extreme).

    Returns:
        Dict[str, Any]: Detailed outlier statistics and bounding limits.
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if clean_series.empty:
        return {
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "outlier_count": 0,
            "total_count": 0,
            "outlier_percentage": 0.0,
            "outlier_values": [],
            "outlier_mask": pd.Series(False, index=series.index),
        }

    q1 = float(np.percentile(clean_series, 25))
    q3 = float(np.percentile(clean_series, 75))
    iqr = float(q3 - q1)

    lower_bound = float(q1 - (multiplier * iqr))
    upper_bound = float(q3 + (multiplier * iqr))

    outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
    outlier_count = int(outlier_mask.sum())
    total_count = len(clean_series)
    outlier_pct = float((outlier_count / max(1, total_count)) * 100.0)

    outlier_vals = clean_series[outlier_mask].head(10).tolist()

    # Reindex mask to original series index
    full_mask = pd.Series(False, index=series.index)
    full_mask.loc[clean_series.index] = outlier_mask

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "total_count": total_count,
        "outlier_percentage": round(outlier_pct, 2),
        "outlier_values": outlier_vals,
        "outlier_mask": full_mask,
    }


def calculate_outliers_zscore(
    series: pd.Series, threshold: float = 3.0
) -> Dict[str, Any]:
    """Detects outliers in a numeric Series using standard Z-score deviations."""
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean_series) < 2:
        return {
            "mean": 0.0,
            "std": 0.0,
            "threshold": threshold,
            "outlier_count": 0,
            "total_count": len(clean_series),
            "outlier_percentage": 0.0,
            "outlier_values": [],
            "outlier_mask": pd.Series(False, index=series.index),
        }

    mean_val = float(clean_series.mean())
    std_val = float(clean_series.std())

    if std_val == 0.0 or np.isnan(std_val):
        return {
            "mean": mean_val,
            "std": 0.0,
            "threshold": threshold,
            "outlier_count": 0,
            "total_count": len(clean_series),
            "outlier_percentage": 0.0,
            "outlier_values": [],
            "outlier_mask": pd.Series(False, index=series.index),
        }

    z_scores = np.abs((clean_series - mean_val) / std_val)
    outlier_mask = z_scores > threshold
    outlier_count = int(outlier_mask.sum())
    total_count = len(clean_series)
    outlier_pct = float((outlier_count / max(1, total_count)) * 100.0)
    outlier_vals = clean_series[outlier_mask].head(10).tolist()

    full_mask = pd.Series(False, index=series.index)
    full_mask.loc[clean_series.index] = outlier_mask

    return {
        "mean": mean_val,
        "std": std_val,
        "threshold": threshold,
        "outlier_count": outlier_count,
        "total_count": total_count,
        "outlier_percentage": round(outlier_pct, 2),
        "outlier_values": outlier_vals,
        "outlier_mask": full_mask,
    }


def calculate_correlation(
    series_a: pd.Series, series_b: pd.Series, method: str = "pearson"
) -> Dict[str, Any]:
    """Computes pairwise correlation between two numerical series."""
    df_pair = pd.DataFrame({"a": pd.to_numeric(series_a, errors="coerce"), "b": pd.to_numeric(series_b, errors="coerce")}).dropna()
    if len(df_pair) < 2:
        return {
            "correlation": 0.0,
            "sample_size": len(df_pair),
            "strength": "Insufficient data",
            "method": method,
        }

    corr = float(df_pair["a"].corr(df_pair["b"], method=method))
    if np.isnan(corr):
        corr = 0.0

    # Interpret strength
    abs_c = abs(corr)
    if abs_c >= 0.8:
        strength = "Very strong positive" if corr > 0 else "Very strong negative"
    elif abs_c >= 0.6:
        strength = "Strong positive" if corr > 0 else "Strong negative"
    elif abs_c >= 0.4:
        strength = "Moderate positive" if corr > 0 else "Moderate negative"
    elif abs_c >= 0.2:
        strength = "Weak positive" if corr > 0 else "Weak negative"
    else:
        strength = "Negligible / No linear correlation"

    return {
        "correlation": round(corr, 4),
        "sample_size": len(df_pair),
        "strength": strength,
        "method": method,
    }


def calculate_descriptive_stats(series: pd.Series) -> Dict[str, Any]:
    """Computes full descriptive distribution summary for a numeric Series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}

    return {
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "25%": float(np.percentile(clean, 25)),
        "50%": float(np.percentile(clean, 50)),
        "75%": float(np.percentile(clean, 75)),
        "max": float(clean.max()),
        "skew": float(clean.skew()) if len(clean) > 2 else 0.0,
    }
