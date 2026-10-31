"""Non-destructive data cleaning and type normalization utilities."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class CleaningSummary:
    """Records transformations applied to a DataFrame during cleaning."""

    def __init__(self) -> None:
        self.renamed_columns: Dict[str, str] = {}
        self.whitespace_cleaned_columns: List[str] = []
        self.coerced_numeric_columns: List[str] = []
        self.coerced_datetime_columns: List[str] = []
        self.coerced_boolean_columns: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "renamed_columns": self.renamed_columns,
            "whitespace_cleaned_columns": self.whitespace_cleaned_columns,
            "coerced_numeric_columns": self.coerced_numeric_columns,
            "coerced_datetime_columns": self.coerced_datetime_columns,
            "coerced_boolean_columns": self.coerced_boolean_columns,
        }


class DataCleaner:
    """Performs safe, non-destructive data cleaning and type normalization."""

    @classmethod
    def clean(
        cls,
        df: pd.DataFrame,
        strip_column_names: bool = True,
        clean_whitespace_strings: bool = True,
        infer_numeric: bool = True,
        infer_datetime: bool = True,
    ) -> Tuple[pd.DataFrame, CleaningSummary]:
        """Cleans and standardizes DataFrame columns without corrupting underlying records.

        Args:
            df: Input raw DataFrame.
            strip_column_names: Whether to strip leading/trailing whitespace from column names.
            clean_whitespace_strings: Replace blank strings and whitespace with NaN.
            infer_numeric: Attempt numeric conversion for object columns that are numbers with formatting.
            infer_datetime: Attempt datetime conversion for date-like string columns.

        Returns:
            Tuple[pd.DataFrame, CleaningSummary]: Cleaned DataFrame and transformation summary.
        """
        cleaned_df = df.copy()
        summary = CleaningSummary()

        # 1. Normalize Column Names
        if strip_column_names:
            new_columns = []
            seen: Dict[str, int] = {}
            for col in cleaned_df.columns:
                col_str = str(col).strip()
                if not col_str:
                    col_str = "Unnamed"
                # Handle duplicate column names
                if col_str in seen:
                    seen[col_str] += 1
                    renamed = f"{col_str}_{seen[col_str]}"
                else:
                    seen[col_str] = 0
                    renamed = col_str

                if str(col) != renamed:
                    summary.renamed_columns[str(col)] = renamed
                new_columns.append(renamed)
            cleaned_df.columns = new_columns

        # 2. Process Each Column
        for col in cleaned_df.columns:
            series = cleaned_df[col]

            # If already numeric or datetime, skip heavy string checks
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                continue

            # Object / String handling
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                # Clean whitespace strings to NaN
                if clean_whitespace_strings:
                    # Replace pure whitespace with NaN
                    cleaned_series = series.apply(
                        lambda x: np.nan if (isinstance(x, str) and x.strip() == "") else x
                    )
                    if not cleaned_series.equals(series):
                        summary.whitespace_cleaned_columns.append(col)
                        series = cleaned_series
                        cleaned_df[col] = series

                # Infer Numeric
                if infer_numeric:
                    coerced, was_coerced = cls._try_infer_numeric(series)
                    if was_coerced:
                        cleaned_df[col] = coerced
                        summary.coerced_numeric_columns.append(col)
                        continue

                # Infer Datetime
                if infer_datetime:
                    coerced_dt, was_coerced_dt = cls._try_infer_datetime(series, col)
                    if was_coerced_dt:
                        cleaned_df[col] = coerced_dt
                        summary.coerced_datetime_columns.append(col)
                        continue

        return cleaned_df, summary

    @staticmethod
    def _try_infer_numeric(series: pd.Series) -> Tuple[pd.Series, bool]:
        """Attempts to safely convert an object series with numeric-like values to float/int."""
        non_null = series.dropna()
        if non_null.empty:
            return series, False

        # If it's already mixed non-strings and non-numbers, check sample
        sample = non_null.iloc[: min(100, len(non_null))]

        # Quick check: does sample look numeric or currency/percentage?
        looks_numeric = False
        for val in sample:
            if isinstance(val, (int, float)):
                looks_numeric = True
                break
            if isinstance(val, str):
                s = val.strip().replace("$", "").replace("€", "").replace("£", "").replace(",", "").replace("%", "")
                if s.startswith("(") and s.endswith(")"):
                    s = "-" + s[1:-1]
                if re.match(r"^-?\d+(\.\d+)?$", s):
                    looks_numeric = True
                    break

        if not looks_numeric:
            return series, False

        try:
            # Strip common currency symbols, accounting parentheses, and commas
            def clean_num_str(val: Any) -> Any:
                if isinstance(val, str):
                    s = val.strip().replace("$", "").replace("€", "").replace("£", "").replace(",", "")
                    if s.startswith("(") and s.endswith(")"):
                        s = "-" + s[1:-1].strip()
                    if s.endswith("%"):
                        try:
                            return float(s[:-1]) / 100.0
                        except ValueError:
                            return s
                    return s
                return val

            cleaned_vals = series.apply(clean_num_str)
            numeric_coerced = pd.to_numeric(cleaned_vals, errors="coerce")
            
            # Check conversion ratio on non-null items
            valid_ratio = numeric_coerced.count() / max(1, non_null.count())
            if valid_ratio >= 0.90:  # If at least 90% of non-nulls convert cleanly
                # Downcast to int if all valid are integers
                valid_num = numeric_coerced.dropna()
                if (valid_num % 1 == 0).all() and not numeric_coerced.isna().any():
                    # Can convert to int64 if no NaNs
                    try:
                        return numeric_coerced.astype("int64"), True
                    except Exception:
                        pass
                return numeric_coerced, True
        except Exception:
            pass

        return series, False

    @staticmethod
    def _try_infer_datetime(series: pd.Series, col_name: str) -> Tuple[pd.Series, bool]:
        """Attempts to safely convert a series to datetime if column name or values strongly indicate dates."""
        non_null = series.dropna()
        if non_null.empty:
            return series, False

        col_lower = str(col_name).lower()
        date_keywords = ["date", "time", "timestamp", "year", "month", "day", "created", "updated", "invoicedate"]
        name_suggests_date = any(k in col_lower for k in date_keywords)

        # Sample check
        sample = non_null.iloc[: min(50, len(non_null))]
        # Check if sample strings match standard date patterns
        date_pattern = re.compile(
            r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"  # 2023-01-01
            r"|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"  # 01/01/2023
            r"|^\w{3,9}\s+\d{1,2},\s*\d{4}"  # Jan 1, 2023
        )
        sample_matches = sum(1 for v in sample if isinstance(v, str) and date_pattern.search(v.strip()))

        if not name_suggests_date and sample_matches < (len(sample) * 0.7):
            return series, False

        try:
            # Attempt parsing
            converted = pd.to_datetime(series, errors="coerce")
            valid_ratio = converted.count() / max(1, non_null.count())
            if valid_ratio >= 0.85:
                return converted, True
        except Exception:
            pass

        return series, False


def clean_dataset(
    df: pd.DataFrame,
    strip_column_names: bool = True,
    clean_whitespace_strings: bool = True,
    infer_numeric: bool = True,
    infer_datetime: bool = True,
) -> Tuple[pd.DataFrame, CleaningSummary]:
    """Convenience helper function for cleaning datasets."""
    return DataCleaner.clean(
        df=df,
        strip_column_names=strip_column_names,
        clean_whitespace_strings=clean_whitespace_strings,
        infer_numeric=infer_numeric,
        infer_datetime=infer_datetime,
    )
