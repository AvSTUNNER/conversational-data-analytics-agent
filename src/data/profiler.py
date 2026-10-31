"""Dynamic dataset profiler for schema extraction, type inference, and LLM context generation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class NumericStats(BaseModel):
    """Basic statistical summary for numeric columns."""
    min: float
    max: float
    mean: float
    median: float
    std: Optional[float] = None


class DateRange(BaseModel):
    """Range boundaries for datetime columns."""
    min_date: str
    max_date: str


class ColumnProfile(BaseModel):
    """Profile metadata for a single column."""
    name: str
    data_type: str  # "numeric", "categorical", "datetime", "boolean", "id_or_text"
    raw_dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any] = Field(default_factory=list)
    numeric_stats: Optional[NumericStats] = None
    date_range: Optional[DateRange] = None


class DatasetProfile(BaseModel):
    """Comprehensive dataset metadata structured for deterministic validation and LLM planning."""
    row_count: int
    column_count: int
    duplicate_rows: int
    total_null_values: int
    columns: Dict[str, ColumnProfile]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    boolean_columns: List[str]
    id_or_text_columns: List[str]

    def get_column(self, col_name: str) -> Optional[ColumnProfile]:
        """Case-insensitive column lookup."""
        col_lower = col_name.strip().lower()
        for name, profile in self.columns.items():
            if name.lower() == col_lower:
                return profile
        return None

    def to_llm_schema_prompt(self, max_samples: int = 4) -> str:
        """Constructs a compact, high-signal schema summary tailored for LLM prompt context."""
        lines = [
            f"Dataset Summary: {self.row_count} rows, {self.column_count} columns",
            "Columns & Schema:",
        ]

        for col_name, prof in self.columns.items():
            type_info = f"{prof.data_type.upper()} ({prof.raw_dtype})"
            null_info = f"nulls: {prof.null_count} ({prof.null_percentage:.1f}%)"
            uniq_info = f"uniques: {prof.unique_count}"

            detail_parts = []
            if prof.numeric_stats:
                st = prof.numeric_stats
                detail_parts.append(f"range: [{st.min:g} to {st.max:g}], mean: {st.mean:.2f}, median: {st.median:g}")
            elif prof.date_range:
                detail_parts.append(f"span: {prof.date_range.min_date} to {prof.date_range.max_date}")

            if prof.sample_values:
                # Format samples cleanly
                samples_str = ", ".join(repr(v) for v in prof.sample_values[:max_samples])
                detail_parts.append(f"samples: [{samples_str}]")

            details = f" | {', '.join(detail_parts)}" if detail_parts else ""
            lines.append(f" - {col_name}: {type_info} | {uniq_info} | {null_info}{details}")

        return "\n".join(lines)


class DatasetProfiler:
    """Extracts dynamic statistical profiles and type classifications from DataFrames."""

    @classmethod
    def profile(cls, df: pd.DataFrame, max_sample_values: int = 5) -> DatasetProfile:
        """Profiles a DataFrame into a structured DatasetProfile object.

        Args:
            df: Cleaned or raw DataFrame.
            max_sample_values: Maximum distinct non-null sample values to record.

        Returns:
            DatasetProfile: Structured profile metadata.
        """
        row_count = len(df)
        column_count = len(df.columns)
        duplicate_rows = int(df.duplicated().sum())
        total_null_values = int(df.isnull().sum().sum())

        columns_dict: Dict[str, ColumnProfile] = {}
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []
        boolean_cols: List[str] = []
        id_text_cols: List[str] = []

        for col in df.columns:
            series = df[col]
            null_cnt = int(series.isnull().sum())
            null_pct = float((null_cnt / max(1, row_count)) * 100.0)
            unique_cnt = int(series.nunique(dropna=True))

            # Sample values (non-null, converted to standard Python primitives)
            non_null_series = series.dropna()
            samples: List[Any] = []
            if not non_null_series.empty:
                unique_vals = non_null_series.unique()[:max_sample_values]
                for v in unique_vals:
                    if isinstance(v, (np.integer, int)):
                        samples.append(int(v))
                    elif isinstance(v, (np.floating, float)):
                        samples.append(float(v))
                    elif isinstance(v, (pd.Timestamp, np.datetime64)):
                        samples.append(str(v))
                    else:
                        samples.append(str(v))

            # Detect data type category
            data_type, num_stats, date_range = cls._classify_column(series, row_count, unique_cnt)

            if data_type == "numeric":
                numeric_cols.append(col)
            elif data_type == "datetime":
                datetime_cols.append(col)
            elif data_type == "boolean":
                boolean_cols.append(col)
            elif data_type == "categorical":
                categorical_cols.append(col)
            else:
                id_text_cols.append(col)

            columns_dict[col] = ColumnProfile(
                name=col,
                data_type=data_type,
                raw_dtype=str(series.dtype),
                null_count=null_cnt,
                null_percentage=null_pct,
                unique_count=unique_cnt,
                sample_values=samples,
                numeric_stats=num_stats,
                date_range=date_range,
            )

        return DatasetProfile(
            row_count=row_count,
            column_count=column_count,
            duplicate_rows=duplicate_rows,
            total_null_values=total_null_values,
            columns=columns_dict,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            boolean_columns=boolean_cols,
            id_or_text_columns=id_text_cols,
        )

    @classmethod
    def _classify_column(
        cls, series: pd.Series, row_count: int, unique_cnt: int
    ) -> tuple[str, Optional[NumericStats], Optional[DateRange]]:
        """Classifies column into semantic category and computes domain stats."""
        non_null = series.dropna()
        if non_null.empty:
            return "id_or_text", None, None

        # 1. Boolean check
        if pd.api.types.is_bool_dtype(series) or (unique_cnt <= 2 and set(non_null.unique()).issubset({True, False, 0, 1, "0", "1", "True", "False", "true", "false", "yes", "no"})):
            return "boolean", None, None

        # 2. Datetime check
        if pd.api.types.is_datetime64_any_dtype(series):
            min_d = str(non_null.min())
            max_d = str(non_null.max())
            return "datetime", None, DateRange(min_date=min_d, max_date=max_d)

        # 3. Numeric check
        if pd.api.types.is_numeric_dtype(series):
            # Check if this numeric column is actually a high-cardinality row ID
            col_name_lower = str(series.name).lower()
            is_id_name = any(k in col_name_lower for k in ["row id", "row_id", "index", "id"])
            if is_id_name and unique_cnt == row_count and row_count > 100:
                # Treat as ID
                return "id_or_text", None, None

            # Calculate numeric stats
            val_min = float(non_null.min())
            val_max = float(non_null.max())
            val_mean = float(non_null.mean())
            val_median = float(non_null.median())
            val_std = float(non_null.std()) if len(non_null) > 1 else 0.0

            stats = NumericStats(
                min=val_min,
                max=val_max,
                mean=val_mean,
                median=val_median,
                std=val_std if not np.isnan(val_std) else 0.0,
            )
            return "numeric", stats, None

        # 4. Categorical vs ID / Free Text check
        # High cardinality ratio check
        cardinality_ratio = unique_cnt / max(1, row_count)
        if unique_cnt <= 100 or cardinality_ratio < 0.20:
            return "categorical", None, None

        return "id_or_text", None, None


def profile_dataset(df: pd.DataFrame, max_sample_values: int = 5) -> DatasetProfile:
    """Convenience helper function for profiling datasets."""
    return DatasetProfiler.profile(df, max_sample_values=max_sample_values)
