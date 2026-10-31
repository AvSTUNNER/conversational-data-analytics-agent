"""Rigid validation layer for analysis plans against live DataFrame schema."""
from __future__ import annotations

import difflib
from typing import List, Optional, Set
import pandas as pd

from src.analysis.schemas import (
    AnalysisPlan,
    OperationType,
    AggregationType,
    FilterCondition,
    FilterOperator,
)


class ValidationError(Exception):
    """Raised when an AnalysisPlan fails structural or schema validation."""
    pass


class PlanValidator:
    """Validates AnalysisPlans against DataFrame schemas, data types, and logical constraints."""

    @classmethod
    def validate(cls, plan: AnalysisPlan, df: pd.DataFrame) -> None:
        """Validates the plan against the provided DataFrame.

        Args:
            plan: The structured AnalysisPlan to inspect.
            df: The live Pandas DataFrame.

        Raises:
            ValidationError: If any column, type, aggregation, or constraint is invalid.
        """
        if plan.operation == OperationType.UNSUPPORTED:
            return

        if df is None or df.empty:
            raise ValidationError("Cannot execute plan on an empty DataFrame.")

        df_columns = list(df.columns)
        df_columns_lower = {c.lower(): c for c in df_columns}

        # 1. Validate Filters
        if plan.filters:
            for idx, flt in enumerate(plan.filters):
                cls._validate_filter(flt, df, df_columns, df_columns_lower, idx)

        # 2. Validate Limits
        if plan.limit is not None:
            if plan.limit <= 0:
                raise ValidationError(f"Plan limit must be a positive integer > 0, got {plan.limit}.")

        # 3. Operation-specific validation
        if plan.operation == OperationType.COUNT:
            if plan.target_column:
                cls._resolve_column(plan.target_column, df_columns, df_columns_lower)

        elif plan.operation == OperationType.AGGREGATION:
            if not plan.target_column:
                raise ValidationError("Operation 'aggregation' requires a 'target_column'.")
            actual_target = cls._resolve_column(plan.target_column, df_columns, df_columns_lower)
            if not plan.aggregation:
                raise ValidationError("Operation 'aggregation' requires an 'aggregation' method (e.g. sum, mean).")
            if plan.aggregation not in {AggregationType.COUNT, AggregationType.NUNIQUE}:
                if not pd.api.types.is_numeric_dtype(df[actual_target]):
                    raise ValidationError(
                        f"Target column '{actual_target}' for aggregation '{plan.aggregation.value}' must be numeric, but is '{df[actual_target].dtype}'."
                    )

        elif plan.operation == OperationType.GROUPBY_AGGREGATE:
            if not plan.group_by_columns:
                raise ValidationError("Operation 'groupby_aggregate' requires 'group_by_columns'.")
            for grp_col in plan.group_by_columns:
                cls._resolve_column(grp_col, df_columns, df_columns_lower)

            if not plan.target_column:
                raise ValidationError("Operation 'groupby_aggregate' requires a 'target_column'.")
            actual_target = cls._resolve_column(plan.target_column, df_columns, df_columns_lower)

            if not plan.aggregation:
                plan.aggregation = AggregationType.SUM  # default if omitted

            resolved_grp_cols = [cls._resolve_column(g, df_columns, df_columns_lower) for g in plan.group_by_columns]
            if actual_target in resolved_grp_cols and plan.aggregation not in {AggregationType.COUNT, AggregationType.NUNIQUE}:
                raise ValidationError(f"Target column '{actual_target}' cannot be identical to a group_by column.")

            if plan.aggregation not in {AggregationType.COUNT, AggregationType.NUNIQUE}:
                if not pd.api.types.is_numeric_dtype(df[actual_target]):
                    raise ValidationError(
                        f"Target column '{actual_target}' for groupby aggregation '{plan.aggregation.value}' must be numeric, but is '{df[actual_target].dtype}'."
                    )

        elif plan.operation == OperationType.TREND:
            if not plan.date_column:
                raise ValidationError("Operation 'trend' requires a 'date_column'.")
            actual_date = cls._resolve_column(plan.date_column, df_columns, df_columns_lower)
            if not pd.api.types.is_datetime64_any_dtype(df[actual_date]):
                raise ValidationError(
                    f"Date column '{actual_date}' must be of datetime type, but is '{df[actual_date].dtype}'."
                )

            if not plan.target_column:
                raise ValidationError("Operation 'trend' requires a 'target_column'.")
            actual_target = cls._resolve_column(plan.target_column, df_columns, df_columns_lower)

            if plan.aggregation not in {AggregationType.COUNT, AggregationType.NUNIQUE}:
                if not pd.api.types.is_numeric_dtype(df[actual_target]):
                    raise ValidationError(
                        f"Target column '{actual_target}' for trend must be numeric, but is '{df[actual_target].dtype}'."
                    )

        elif plan.operation == OperationType.CORRELATION:
            cols_to_check: List[str] = []
            if plan.target_columns and len(plan.target_columns) >= 2:
                cols_to_check = plan.target_columns[:2]
            elif plan.target_column and plan.group_by_columns:
                cols_to_check = [plan.target_column, plan.group_by_columns[0]]
            else:
                raise ValidationError(
                    "Operation 'correlation' requires two numeric columns specified in 'target_columns' or ('target_column' and 'group_by_columns')."
                )

            for col in cols_to_check:
                actual_col = cls._resolve_column(col, df_columns, df_columns_lower)
                if not pd.api.types.is_numeric_dtype(df[actual_col]):
                    raise ValidationError(
                        f"Column '{actual_col}' in correlation must be numeric, but is '{df[actual_col].dtype}'."
                    )

        elif plan.operation == OperationType.OUTLIERS:
            if not plan.target_column:
                raise ValidationError("Operation 'outliers' requires a 'target_column'.")
            actual_target = cls._resolve_column(plan.target_column, df_columns, df_columns_lower)
            if not pd.api.types.is_numeric_dtype(df[actual_target]):
                raise ValidationError(
                    f"Target column '{actual_target}' for outlier analysis must be numeric, but is '{df[actual_target].dtype}'."
                )

        elif plan.operation == OperationType.FILTER_PREVIEW:
            if plan.target_column:
                cls._resolve_column(plan.target_column, df_columns, df_columns_lower)
            if plan.group_by_columns:
                for grp_col in plan.group_by_columns:
                    cls._resolve_column(grp_col, df_columns, df_columns_lower)

    @classmethod
    def _validate_filter(
        cls,
        flt: FilterCondition,
        df: pd.DataFrame,
        df_columns: List[str],
        df_columns_lower: dict,
        idx: int,
    ) -> None:
        """Validates a single FilterCondition."""
        if not flt.column:
            raise ValidationError(f"Filter at index {idx} is missing a column name.")
        actual_col = cls._resolve_column(flt.column, df_columns, df_columns_lower)

        # Value check
        if flt.value is None and flt.operator not in {FilterOperator.EQUALS, FilterOperator.NOT_EQUALS}:
            raise ValidationError(f"Filter on '{actual_col}' with operator '{flt.operator}' cannot have a None value.")

        # Numeric comparison on non-numeric column check
        if flt.operator in {
            FilterOperator.GREATER_THAN,
            FilterOperator.GREATER_EQUAL,
            FilterOperator.LESS_THAN,
            FilterOperator.LESS_EQUAL,
        }:
            if not pd.api.types.is_numeric_dtype(df[actual_col]) and not pd.api.types.is_datetime64_any_dtype(df[actual_col]):
                raise ValidationError(
                    f"Relational operator '{flt.operator.value}' cannot be applied to non-numeric/non-datetime column '{actual_col}' ({df[actual_col].dtype})."
                )

    @staticmethod
    def _resolve_column(col_name: str, df_columns: List[str], df_columns_lower: dict) -> str:
        """Verifies column exists, allowing case-insensitive matching, or raises with helpful suggestions."""
        if col_name in df_columns:
            return col_name

        col_lower = col_name.strip().lower()
        if col_lower in df_columns_lower:
            return df_columns_lower[col_lower]

        # Suggest closest match
        matches = difflib.get_close_matches(col_name, df_columns, n=3, cutoff=0.6)
        suggestion = f" Did you mean: {', '.join(matches)}?" if matches else ""
        raise ValidationError(f"Column '{col_name}' not found in dataset.{suggestion}")
