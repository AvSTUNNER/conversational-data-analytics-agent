"""Unit tests for the Plotly ChartGenerator."""
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.analysis.schemas import (
    AnalysisPlan,
    ExecutionResult,
    OperationType,
    AggregationType,
    ChartRecommendation,
)
from src.visualization.charts import ChartGenerator, create_chart


class TestChartGenerator:
    """Test suite for truthful Plotly chart rendering."""

    def test_bar_chart_generation(self):
        plan = AnalysisPlan(
            operation=OperationType.GROUPBY_AGGREGATE,
            group_by_columns=["Category"],
            target_column="Profit",
            chart=ChartRecommendation.BAR,
        )
        evidence_df = pd.DataFrame({
            "Category": ["Technology", "Furniture", "Office Supplies"],
            "Sum of Profit": [145000.0, 18000.0, 122000.0],
        })
        result = ExecutionResult(
            operation=OperationType.GROUPBY_AGGREGATE,
            evidence_df=evidence_df,
            chart_type=ChartRecommendation.BAR,
        )
        fig = create_chart(plan, result)
        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert "Sum of Profit by Category" in fig.layout.title.text

    def test_line_chart_generation(self):
        plan = AnalysisPlan(
            operation=OperationType.TREND,
            date_column="Order Date",
            target_column="Sales",
            chart=ChartRecommendation.LINE,
        )
        evidence_df = pd.DataFrame({
            "Period": ["2023-01-01", "2023-02-01", "2023-03-01"],
            "Sum of Sales": [10000.0, 15000.0, 12000.0],
        })
        result = ExecutionResult(
            operation=OperationType.TREND,
            evidence_df=evidence_df,
            chart_type=ChartRecommendation.LINE,
        )
        fig = create_chart(plan, result)
        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert "Trend Over Time" in fig.layout.title.text

    def test_scatter_chart_generation(self):
        plan = AnalysisPlan(
            operation=OperationType.CORRELATION,
            target_columns=["Sales", "Profit"],
            chart=ChartRecommendation.SCATTER,
        )
        evidence_df = pd.DataFrame([{
            "Column A": "Sales",
            "Column B": "Profit",
            "Pearson Correlation": 0.48,
        }])
        raw_df = pd.DataFrame({
            "Sales": [100, 200, 300, 400],
            "Profit": [10, 25, 40, 60],
        })
        result = ExecutionResult(
            operation=OperationType.CORRELATION,
            main_value=0.48,
            evidence_df=evidence_df,
            chart_type=ChartRecommendation.SCATTER,
        )
        fig = create_chart(plan, result, raw_df=raw_df)
        assert fig is not None
        assert "Sales vs Profit" in fig.layout.title.text

    def test_box_chart_generation(self):
        plan = AnalysisPlan(
            operation=OperationType.OUTLIERS,
            target_column="Sales",
            chart=ChartRecommendation.BOX,
        )
        evidence_df = pd.DataFrame([{"Column": "Sales", "Outlier Count": 5}])
        raw_df = pd.DataFrame({"Sales": [10, 20, 30, 40, 500]})
        result = ExecutionResult(
            operation=OperationType.OUTLIERS,
            evidence_df=evidence_df,
            chart_type=ChartRecommendation.BOX,
        )
        fig = create_chart(plan, result, raw_df=raw_df)
        assert fig is not None
        assert "Distribution and Outliers in Sales" in fig.layout.title.text

    def test_pie_chart_generation(self):
        plan = AnalysisPlan(
            operation=OperationType.GROUPBY_AGGREGATE,
            group_by_columns=["Region"],
            target_column="Sales",
            chart=ChartRecommendation.PIE,
        )
        evidence_df = pd.DataFrame({
            "Region": ["North", "South", "East", "West"],
            "Sales": [100.0, 200.0, 300.0, 400.0],
        })
        result = ExecutionResult(
            operation=OperationType.GROUPBY_AGGREGATE,
            evidence_df=evidence_df,
            chart_type=ChartRecommendation.PIE,
        )
        fig = create_chart(plan, result)
        assert fig is not None
        assert "Sales Share by Region" in fig.layout.title.text

    def test_no_chart_for_count_and_unsupported(self):
        plan = AnalysisPlan(operation=OperationType.COUNT, chart=ChartRecommendation.NONE)
        result = ExecutionResult(
            operation=OperationType.COUNT,
            main_value=100,
            evidence_df=pd.DataFrame([{"Count": 100}]),
            chart_type=ChartRecommendation.NONE,
        )
        fig = create_chart(plan, result)
        assert fig is None

        # Unsupported query
        res_unsupp = ExecutionResult(
            operation=OperationType.UNSUPPORTED,
            is_unsupported=True,
            chart_type=ChartRecommendation.NONE,
        )
        assert create_chart(plan, res_unsupp) is None
