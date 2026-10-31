"""Deterministic Pandas analysis executor for structured AnalysisPlans."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.analysis.schemas import (
    AnalysisPlan,
    OperationType,
    AggregationType,
    FilterCondition,
    FilterOperator,
    ChartRecommendation,
    TimeGranularity,
    ExecutionResult,
)
from src.analysis.validator import PlanValidator, ValidationError
from src.analysis.statistics import (
    calculate_outliers_iqr,
    calculate_correlation,
    calculate_descriptive_stats,
)


class AnalysisExecutor:
    """Executes validated AnalysisPlans deterministically using Pandas."""

    @classmethod
    def execute(cls, plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
        """Validates and executes an AnalysisPlan on a DataFrame.

        Args:
            plan: The AnalysisPlan containing operation, target, group by, and filters.
            df: The cleaned Pandas DataFrame.

        Returns:
            ExecutionResult: Deterministically computed result, evidence table, and summary.
        """
        start_time = time.perf_counter()

        # Handle unsupported queries
        if plan.operation == OperationType.UNSUPPORTED:
            return ExecutionResult(
                operation=OperationType.UNSUPPORTED,
                is_unsupported=True,
                unsupported_reason=plan.unsupported_reason or "The question cannot be answered from the provided dataset schema.",
                evidence_summary="Unsupported query.",
                chart_type=ChartRecommendation.NONE,
            )

        # Pre-execution validation
        PlanValidator.validate(plan, df)

        # 1. Apply Filters
        filtered_df = cls._apply_filters(df, plan.filters)

        # 2. Execute Operation
        if plan.operation == OperationType.COUNT:
            result = cls._execute_count(plan, filtered_df)
        elif plan.operation == OperationType.AGGREGATION:
            result = cls._execute_aggregation(plan, filtered_df)
        elif plan.operation == OperationType.GROUPBY_AGGREGATE:
            result = cls._execute_groupby(plan, filtered_df)
        elif plan.operation == OperationType.TREND:
            result = cls._execute_trend(plan, filtered_df)
        elif plan.operation == OperationType.CORRELATION:
            result = cls._execute_correlation(plan, filtered_df)
        elif plan.operation == OperationType.OUTLIERS:
            result = cls._execute_outliers(plan, filtered_df)
        elif plan.operation == OperationType.FILTER_PREVIEW:
            result = cls._execute_filter_preview(plan, filtered_df)
        else:
            raise ValidationError(f"Unhandled operation type: {plan.operation}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        result.metadata["execution_time_ms"] = round(elapsed_ms, 2)
        result.metadata["total_dataset_rows"] = len(df)
        result.metadata["filtered_rows"] = len(filtered_df)

        return result

    @classmethod
    def _apply_filters(cls, df: pd.DataFrame, filters: Optional[List[FilterCondition]]) -> pd.DataFrame:
        """Applies a list of FilterConditions deterministically."""
        if not filters:
            return df

        mask = pd.Series(True, index=df.index)
        for flt in filters:
            col_name = cls._match_col(flt.column, df)
            series = df[col_name]
            val = flt.value

            # Type coercion for comparison values if needed
            if pd.api.types.is_numeric_dtype(series) and isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    pass
            elif pd.api.types.is_datetime64_any_dtype(series) and isinstance(val, str):
                try:
                    val = pd.to_datetime(val)
                except Exception:
                    pass

            if flt.operator == FilterOperator.EQUALS:
                # Type-tolerant equality
                if isinstance(val, str) and (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
                    curr_mask = series.astype(str).str.strip().str.lower() == val.strip().lower()
                else:
                    curr_mask = series == val
            elif flt.operator == FilterOperator.NOT_EQUALS:
                if isinstance(val, str) and (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
                    curr_mask = series.astype(str).str.strip().str.lower() != val.strip().lower()
                else:
                    curr_mask = series != val
            elif flt.operator == FilterOperator.GREATER_THAN:
                curr_mask = series > val
            elif flt.operator == FilterOperator.GREATER_EQUAL:
                curr_mask = series >= val
            elif flt.operator == FilterOperator.LESS_THAN:
                curr_mask = series < val
            elif flt.operator == FilterOperator.LESS_EQUAL:
                curr_mask = series <= val
            elif flt.operator == FilterOperator.IN:
                val_list = val if isinstance(val, list) else [val]
                if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                    val_lower = [str(x).strip().lower() for x in val_list]
                    curr_mask = series.astype(str).str.strip().str.lower().isin(val_lower)
                else:
                    curr_mask = series.isin(val_list)
            elif flt.operator == FilterOperator.NOT_IN:
                val_list = val if isinstance(val, list) else [val]
                if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                    val_lower = [str(x).strip().lower() for x in val_list]
                    curr_mask = ~series.astype(str).str.strip().str.lower().isin(val_lower)
                else:
                    curr_mask = ~series.isin(val_list)
            elif flt.operator == FilterOperator.CONTAINS:
                curr_mask = series.astype(str).str.contains(str(val), case=False, na=False)
            else:
                curr_mask = pd.Series(True, index=df.index)

            mask = mask & curr_mask

        return df[mask]

    @classmethod
    def _execute_count(cls, plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
        """Executes a count or distinct count query."""
        if plan.target_column and plan.aggregation == AggregationType.NUNIQUE:
            col_name = cls._match_col(plan.target_column, df)
            cnt = int(df[col_name].nunique(dropna=True))
            summary = f"Distinct count of '{col_name}': {cnt:,}"
            evidence_df = pd.DataFrame([{"Metric": f"Distinct {col_name}", "Count": cnt}])
        elif plan.target_column and plan.aggregation == AggregationType.COUNT:
            col_name = cls._match_col(plan.target_column, df)
            cnt = int(df[col_name].count())
            summary = f"Non-null count of '{col_name}': {cnt:,}"
            evidence_df = pd.DataFrame([{"Metric": f"Non-null {col_name}", "Count": cnt}])
        else:
            cnt = int(len(df))
            summary = f"Total matching records: {cnt:,}"
            evidence_df = pd.DataFrame([{"Metric": "Total Records", "Count": cnt}])

        return ExecutionResult(
            operation=OperationType.COUNT,
            main_value=cnt,
            evidence_df=evidence_df,
            evidence_summary=summary,
            chart_type=ChartRecommendation.NONE,
        )

    @classmethod
    def _execute_aggregation(cls, plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
        """Executes a single-metric aggregation (sum, mean, median, min, max, std)."""
        target_col = cls._match_col(plan.target_column, df)  # type: ignore
        series = pd.to_numeric(df[target_col], errors="coerce").dropna()
        agg = plan.aggregation or AggregationType.SUM

        if series.empty:
            return ExecutionResult(
                operation=OperationType.AGGREGATION,
                main_value=0.0,
                evidence_df=pd.DataFrame([{"Metric": f"{agg.value.title()} of {target_col}", "Value": 0.0, "Count": 0}]),
                evidence_summary=f"No matching non-null records found for '{target_col}'.",
                chart_type=ChartRecommendation.NONE,
            )

        if agg == AggregationType.SUM:
            val = float(series.sum())
        elif agg == AggregationType.MEAN:
            val = float(series.mean())
        elif agg == AggregationType.MEDIAN:
            val = float(series.median())
        elif agg == AggregationType.MIN:
            val = float(series.min())
        elif agg == AggregationType.MAX:
            val = float(series.max())
        elif agg == AggregationType.STD:
            val = float(series.std()) if len(series) > 1 else 0.0
        elif agg == AggregationType.COUNT:
            val = int(len(series))
        else:
            val = float(series.sum())

        val_rounded = round(val, 2) if isinstance(val, float) else val
        formatted_val = f"{val_rounded:,.2f}" if isinstance(val_rounded, float) else f"{val_rounded:,}"
        metric_label = f"{agg.value.title()} of {target_col}"
        summary = f"{metric_label}: {formatted_val} (based on {len(series):,} records)"

        evidence_df = pd.DataFrame([{
            "Metric": metric_label,
            "Value": val_rounded,
            "Record Count": len(series),
        }])

        return ExecutionResult(
            operation=OperationType.AGGREGATION,
            main_value=val_rounded,
            evidence_df=evidence_df,
            evidence_summary=summary,
            chart_type=ChartRecommendation.NONE,
        )

    @classmethod
    def _execute_groupby(cls, plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
        """Executes group-by aggregation, sorting, and limit/top-N selection."""
        group_cols = [cls._match_col(c, df) for c in plan.group_by_columns]  # type: ignore
        target_col = cls._match_col(plan.target_column, df)  # type: ignore
        agg = plan.aggregation or AggregationType.SUM
        sort_asc = plan.sort_ascending if plan.sort_ascending is not None else False

        # Build aggregation mapping
        agg_func = agg.value
        if agg == AggregationType.COUNT:
            agg_func = "count"
        elif agg == AggregationType.NUNIQUE:
            agg_func = "nunique"

        # Group and aggregate
        grouped = df.groupby(group_cols, dropna=False)[target_col].agg(agg_func).reset_index()
        
        # Name aggregated column clearly
        agg_col_name = f"{agg.value.title()} of {target_col}"
        grouped.rename(columns={target_col: agg_col_name}, inplace=True)

        # Sort
        grouped.sort_values(by=agg_col_name, ascending=sort_asc, inplace=True)

        # Apply Limit if requested
        if plan.limit and plan.limit > 0:
            evidence_df = grouped.head(plan.limit).copy()
        else:
            evidence_df = grouped.copy()

        # Format evidence summary
        summary_items = []
        for _, row in evidence_df.head(5).iterrows():
            grp_val = " / ".join(str(row[c]) for c in group_cols)
            metric_val = row[agg_col_name]
            formatted_num = f"{metric_val:,.2f}" if isinstance(metric_val, float) else f"{metric_val:,}"
            summary_items.append(f"{grp_val}: {formatted_num}")

        evidence_summary = " · ".join(summary_items)
        if len(evidence_df) > 5:
            evidence_summary += f" · ... ({len(evidence_df)} total groups)"

        # Determine main value (top ranked item)
        main_value = None
        if not evidence_df.empty:
            top_row = evidence_df.iloc[0]
            top_grp = " / ".join(str(top_row[c]) for c in group_cols)
            top_metric = top_row[agg_col_name]
            main_value = {
                "top_group": top_grp,
                "top_value": round(top_metric, 2) if isinstance(top_metric, float) else top_metric,
                "metric_name": agg_col_name,
            }

        # Chart selection
        chart_type = plan.chart if plan.chart != ChartRecommendation.NONE else ChartRecommendation.BAR

        return ExecutionResult(
            operation=OperationType.GROUPBY_AGGREGATE,
            main_value=main_value,
            evidence_df=evidence_df,
            evidence_summary=evidence_summary,
            chart_type=chart_type,
        )

    @classmethod
    def _execute_trend(cls, plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
        """Executes time-series aggregation grouped by date granularity."""
        date_col = cls._match_col(plan.date_column, df)  # type: ignore
        target_col = cls._match_col(plan.target_column, df)  # type: ignore
        agg = plan.aggregation or AggregationType.SUM
        granularity = plan.time_granularity or TimeGranularity.MONTH

        # Ensure datetime
        clean_df = df.dropna(subset=[date_col]).copy()
        clean_df[date_col] = pd.to_datetime(clean_df[date_col], errors="coerce")
        clean_df = clean_df.dropna(subset=[date_col])

        # Resample / Period grouping
        clean_df.set_index(date_col, inplace=True)
        resampled = clean_df[target_col].resample(granularity.value).agg(agg.value).reset_index()

        # Format date column as string for clean display
        resampled["Period"] = resampled[date_col].dt.strftime("%Y-%m-%d")
        metric_col_name = f"{agg.value.title()} of {target_col}"
        resampled.rename(columns={target_col: metric_col_name}, inplace=True)

        evidence_df = resampled[["Period", metric_col_name]].copy()

        # Summary
        summary_items = []
        for _, row in evidence_df.head(4).iterrows():
            m_val = row[metric_col_name]
            f_val = f"{m_val:,.2f}" if isinstance(m_val, float) else f"{m_val:,}"
            summary_items.append(f"{row['Period']}: {f_val}")
        if len(evidence_df) > 4:
            summary_items.append(f"... ({len(evidence_df)} periods)")

        evidence_summary = " · ".join(summary_items)

        return ExecutionResult(
            operation=OperationType.TREND,
            main_value={"period_count": len(evidence_df), "metric_name": metric_col_name},
            evidence_df=evidence_df,
            evidence_summary=evidence_summary,
            chart_type=ChartRecommendation.LINE,
        )

    @classmethod
    def _execute_correlation(cls, plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
        """Executes pairwise correlation between two numerical columns."""
        cols = []
        if plan.target_columns and len(plan.target_columns) >= 2:
            cols = [cls._match_col(c, df) for c in plan.target_columns[:2]]
        elif plan.target_column and plan.group_by_columns:
            cols = [cls._match_col(plan.target_column, df), cls._match_col(plan.group_by_columns[0], df)]

        col_a, col_b = cols[0], cols[1]
        corr_data = calculate_correlation(df[col_a], df[col_b])

        evidence_df = pd.DataFrame([{
            "Column A": col_a,
            "Column B": col_b,
            "Pearson Correlation": corr_data["correlation"],
            "Strength": corr_data["strength"],
            "Sample Size": corr_data["sample_size"],
        }])

        summary = f"Correlation between '{col_a}' and '{col_b}': {corr_data['correlation']} ({corr_data['strength']})"

        return ExecutionResult(
            operation=OperationType.CORRELATION,
            main_value=corr_data["correlation"],
            evidence_df=evidence_df,
            evidence_summary=summary,
            chart_type=ChartRecommendation.SCATTER,
        )

    @classmethod
    def _execute_outliers(cls, plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
        """Executes outlier detection using the IQR method."""
        target_col = cls._match_col(plan.target_column, df)  # type: ignore
        outlier_res = calculate_outliers_iqr(df[target_col])

        evidence_df = pd.DataFrame([{
            "Column": target_col,
            "Outlier Count": outlier_res["outlier_count"],
            "Total Records": outlier_res["total_count"],
            "Outlier %": f"{outlier_res['outlier_percentage']}%",
            "Lower Bound": round(outlier_res["lower_bound"], 2),
            "Upper Bound": round(outlier_res["upper_bound"], 2),
            "Q1 (25%)": round(outlier_res["q1"], 2),
            "Q3 (75%)": round(outlier_res["q3"], 2),
        }])

        summary = (
            f"Detected {outlier_res['outlier_count']:,} outliers ({outlier_res['outlier_percentage']}%) "
            f"in '{target_col}' (bounds: [{outlier_res['lower_bound']:,.2f}, {outlier_res['upper_bound']:,.2f}])"
        )

        return ExecutionResult(
            operation=OperationType.OUTLIERS,
            main_value={
                "outlier_count": outlier_res["outlier_count"],
                "outlier_percentage": outlier_res["outlier_percentage"],
            },
            evidence_df=evidence_df,
            evidence_summary=summary,
            chart_type=ChartRecommendation.BOX,
        )

    @classmethod
    def _execute_filter_preview(cls, plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
        """Returns filtered rows preview."""
        limit = plan.limit or 10
        evidence_df = df.head(limit).copy()
        summary = f"Showing top {len(evidence_df)} filtered records."

        return ExecutionResult(
            operation=OperationType.FILTER_PREVIEW,
            main_value=len(evidence_df),
            evidence_df=evidence_df,
            evidence_summary=summary,
            chart_type=ChartRecommendation.NONE,
        )

    @staticmethod
    def _match_col(name: str, df: pd.DataFrame) -> str:
        """Resolves column name against DataFrame with case-insensitive matching."""
        if name in df.columns:
            return name
        name_lower = name.strip().lower()
        for col in df.columns:
            if col.lower() == name_lower:
                return col
        return name


def execute_plan(plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
    """Convenience helper function for executing plans."""
    return AnalysisExecutor.execute(plan, df)
