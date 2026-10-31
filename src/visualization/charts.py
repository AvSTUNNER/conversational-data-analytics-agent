"""Dynamic Plotly chart generator for deterministic analysis results."""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analysis.schemas import AnalysisPlan, ChartRecommendation, ExecutionResult, OperationType


class ChartGenerator:
    """Creates beautiful, truthful Plotly charts matching deterministic ExecutionResults."""

    CHART_THEME = "plotly_white"
    PRIMARY_COLOR = "#2563eb"  # Royal blue
    COLOR_SEQUENCE = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]

    @classmethod
    def create_chart(
        cls,
        plan: AnalysisPlan,
        result: ExecutionResult,
        raw_df: Optional[pd.DataFrame] = None,
    ) -> Optional[go.Figure]:
        """Generates a Plotly Figure matching the analysis plan and execution evidence.

        Args:
            plan: The executed AnalysisPlan.
            result: The deterministic ExecutionResult containing evidence_df.
            raw_df: Optional raw or filtered DataFrame (useful for scatter and box plots).

        Returns:
            Optional[go.Figure]: Configured Plotly figure, or None if no chart is needed.
        """
        if result.is_unsupported or result.chart_type == ChartRecommendation.NONE:
            return None

        evidence_df = result.evidence_df
        if evidence_df is None or evidence_df.empty:
            return None

        chart_type = result.chart_type

        try:
            if chart_type == ChartRecommendation.BAR:
                return cls._create_bar_chart(plan, result, evidence_df)
            elif chart_type == ChartRecommendation.LINE:
                return cls._create_line_chart(plan, result, evidence_df)
            elif chart_type == ChartRecommendation.SCATTER:
                return cls._create_scatter_chart(plan, result, evidence_df, raw_df)
            elif chart_type == ChartRecommendation.BOX:
                return cls._create_box_chart(plan, result, evidence_df, raw_df)
            elif chart_type == ChartRecommendation.PIE:
                return cls._create_pie_chart(plan, result, evidence_df)
            elif chart_type == ChartRecommendation.HISTOGRAM:
                return cls._create_histogram(plan, result, evidence_df, raw_df)
            else:
                return None
        except Exception:
            # Fallback gracefully if any charting dimension is irregular
            return None

    @classmethod
    def _create_bar_chart(
        cls, plan: AnalysisPlan, result: ExecutionResult, evidence_df: pd.DataFrame
    ) -> Optional[go.Figure]:
        """Creates vertical or horizontal bar chart for grouped aggregations."""
        cols = list(evidence_df.columns)
        if len(cols) < 2:
            return None

        x_col = cols[0]
        y_col = cols[1]

        # Ensure numeric Y axis
        if not pd.api.types.is_numeric_dtype(evidence_df[y_col]):
            # Try swapping if x is numeric
            if pd.api.types.is_numeric_dtype(evidence_df[x_col]):
                x_col, y_col = y_col, x_col
            else:
                return None

        title = f"{y_col} by {x_col}"
        fig = px.bar(
            evidence_df,
            x=x_col,
            y=y_col,
            title=title,
            text_auto=".2s",
            color_discrete_sequence=[cls.PRIMARY_COLOR],
            template=cls.CHART_THEME,
        )

        fig.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis_title=x_col,
            yaxis_title=y_col,
            hoverlabel=dict(bgcolor="white", font_size=12),
        )
        return fig

    @classmethod
    def _create_line_chart(
        cls, plan: AnalysisPlan, result: ExecutionResult, evidence_df: pd.DataFrame
    ) -> Optional[go.Figure]:
        """Creates line chart for time-series trend analysis."""
        cols = list(evidence_df.columns)
        if len(cols) < 2:
            return None

        x_col = "Period" if "Period" in cols else cols[0]
        y_col = [c for c in cols if c != x_col][0]

        title = f"{y_col} Trend Over Time"
        fig = px.line(
            evidence_df,
            x=x_col,
            y=y_col,
            title=title,
            markers=True,
            color_discrete_sequence=[cls.PRIMARY_COLOR],
            template=cls.CHART_THEME,
        )

        fig.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis_title="Time Period",
            yaxis_title=y_col,
            hovermode="x unified",
        )
        return fig

    @classmethod
    def _create_scatter_chart(
        cls,
        plan: AnalysisPlan,
        result: ExecutionResult,
        evidence_df: pd.DataFrame,
        raw_df: Optional[pd.DataFrame],
    ) -> Optional[go.Figure]:
        """Creates scatter plot for correlation analysis."""
        col_a, col_b = None, None

        if plan.target_columns and len(plan.target_columns) >= 2:
            col_a, col_b = plan.target_columns[0], plan.target_columns[1]
        elif "Column A" in evidence_df.columns and "Column B" in evidence_df.columns:
            col_a = str(evidence_df["Column A"].iloc[0])
            col_b = str(evidence_df["Column B"].iloc[0])

        if raw_df is not None and col_a in raw_df.columns and col_b in raw_df.columns:
            # Sample if dataset is very large (> 2000 points) for fast rendering
            sample_data = raw_df[[col_a, col_b]].dropna()
            if len(sample_data) > 2000:
                sample_data = sample_data.sample(n=2000, random_state=42)

            corr_val = result.main_value if isinstance(result.main_value, (float, int)) else 0.0
            title = f"{col_a} vs {col_b} (Pearson Correlation: {corr_val})"

            fig = px.scatter(
                sample_data,
                x=col_a,
                y=col_b,
                title=title,
                opacity=0.6,
                trendline="ols",
                color_discrete_sequence=[cls.PRIMARY_COLOR],
                template=cls.CHART_THEME,
            )
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
            return fig

        return None

    @classmethod
    def _create_box_chart(
        cls,
        plan: AnalysisPlan,
        result: ExecutionResult,
        evidence_df: pd.DataFrame,
        raw_df: Optional[pd.DataFrame],
    ) -> Optional[go.Figure]:
        """Creates box plot for outlier distribution inspection."""
        target_col = plan.target_column or (evidence_df["Column"].iloc[0] if "Column" in evidence_df.columns else None)
        if not target_col:
            return None

        if raw_df is not None and target_col in raw_df.columns:
            clean_series = pd.to_numeric(raw_df[target_col], errors="coerce").dropna()
            title = f"Distribution and Outliers in {target_col}"

            fig = px.box(
                clean_series,
                y=target_col,
                title=title,
                points="outliers",
                color_discrete_sequence=[cls.PRIMARY_COLOR],
                template=cls.CHART_THEME,
            )
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
            return fig

        return None

    @classmethod
    def _create_pie_chart(
        cls, plan: AnalysisPlan, result: ExecutionResult, evidence_df: pd.DataFrame
    ) -> Optional[go.Figure]:
        """Creates pie chart only for small categorical splits (<= 7 categories) with positive values."""
        cols = list(evidence_df.columns)
        if len(cols) < 2 or len(evidence_df) > 7:
            return None

        names_col = cols[0]
        values_col = cols[1]

        # Values must be numeric and non-negative
        if not pd.api.types.is_numeric_dtype(evidence_df[values_col]):
            return None
        if (evidence_df[values_col] < 0).any():
            return None

        title = f"{values_col} Share by {names_col}"
        fig = px.pie(
            evidence_df,
            names=names_col,
            values=values_col,
            title=title,
            color_discrete_sequence=cls.COLOR_SEQUENCE,
            template=cls.CHART_THEME,
            hole=0.3,
        )
        fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def _create_histogram(
        cls,
        plan: AnalysisPlan,
        result: ExecutionResult,
        evidence_df: pd.DataFrame,
        raw_df: Optional[pd.DataFrame],
    ) -> Optional[go.Figure]:
        """Creates histogram for value distributions."""
        target_col = plan.target_column
        if raw_df is not None and target_col and target_col in raw_df.columns:
            clean_series = pd.to_numeric(raw_df[target_col], errors="coerce").dropna()
            fig = px.histogram(
                clean_series,
                x=target_col,
                title=f"Histogram of {target_col}",
                nbins=30,
                color_discrete_sequence=[cls.PRIMARY_COLOR],
                template=cls.CHART_THEME,
            )
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
            return fig
        return None


def create_chart(
    plan: AnalysisPlan,
    result: ExecutionResult,
    raw_df: Optional[pd.DataFrame] = None,
) -> Optional[go.Figure]:
    """Convenience helper function for chart creation."""
    return ChartGenerator.create_chart(plan, result, raw_df=raw_df)
