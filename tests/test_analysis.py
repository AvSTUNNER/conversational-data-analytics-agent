"""Unit tests for deterministic AnalysisExecutor with hand-calculated ground truth comparisons."""
import pandas as pd
import pytest

from src.data.loader import load_dataset
from src.data.cleaner import clean_dataset
from src.analysis.schemas import (
    AnalysisPlan,
    OperationType,
    AggregationType,
    FilterCondition,
    FilterOperator,
    TimeGranularity,
    ChartRecommendation,
)
from src.analysis.executor import AnalysisExecutor, execute_plan
from src.analysis.statistics import calculate_outliers_iqr, calculate_correlation


class TestAnalysisExecutor:
    """Test suite for deterministic analysis calculations and ground truth verification."""

    @pytest.fixture(scope="class")
    def superstore_df(self):
        """Loads and cleans the real Superstore dataset once for the test class."""
        df = load_dataset("datasets/demo/sample_-_superstore.xls", sheet_name="Orders")
        cleaned_df, _ = clean_dataset(df)
        return cleaned_df

    def test_count_operation_total_and_distinct(self, superstore_df):
        """Verify: 'How many orders are there?' total rows and distinct order IDs."""
        # Total rows
        plan_total = AnalysisPlan(operation=OperationType.COUNT)
        res_total = execute_plan(plan_total, superstore_df)
        assert res_total.main_value == 10194
        assert "Total matching records: 10,194" in res_total.evidence_summary

        # Distinct Order IDs
        plan_distinct = AnalysisPlan(
            operation=OperationType.COUNT,
            target_column="Order ID",
            aggregation=AggregationType.NUNIQUE,
        )
        res_distinct = execute_plan(plan_distinct, superstore_df)
        assert res_distinct.main_value == 5111
        assert "Distinct count of 'Order ID': 5,111" in res_distinct.evidence_summary

    def test_aggregation_mean_and_median_sales(self, superstore_df):
        """Verify: 'What is the average sales?' and 'What is the median discount?'"""
        # Mean Sales
        plan_mean = AnalysisPlan(
            operation=OperationType.AGGREGATION,
            target_column="Sales",
            aggregation=AggregationType.MEAN,
        )
        res_mean = execute_plan(plan_mean, superstore_df)
        expected_mean = round(float(superstore_df["Sales"].mean()), 2)
        assert res_mean.main_value == expected_mean
        assert res_mean.evidence_df is not None
        assert res_mean.evidence_df["Value"].iloc[0] == expected_mean

        # Median Discount
        plan_median = AnalysisPlan(
            operation=OperationType.AGGREGATION,
            target_column="Discount",
            aggregation=AggregationType.MEDIAN,
        )
        res_median = execute_plan(plan_median, superstore_df)
        expected_median = float(superstore_df["Discount"].median())
        assert res_median.main_value == expected_median

    def test_groupby_top_category_by_profit(self, superstore_df):
        """Verify: 'Which category has the highest profit?' (Group-by + Top 1)."""
        plan = AnalysisPlan(
            operation=OperationType.GROUPBY_AGGREGATE,
            group_by_columns=["Category"],
            target_column="Profit",
            aggregation=AggregationType.SUM,
            sort_ascending=False,
            limit=1,
            chart=ChartRecommendation.BAR,
        )
        res = execute_plan(plan, superstore_df)

        # Hand-calculated check
        profit_by_cat = superstore_df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
        top_cat_name = profit_by_cat.index[0]
        top_cat_val = round(float(profit_by_cat.iloc[0]), 2)

        assert res.main_value["top_group"] == top_cat_name
        assert res.main_value["top_value"] == top_cat_val
        assert top_cat_name == "Technology"
        assert res.chart_type == ChartRecommendation.BAR
        assert res.evidence_df is not None
        assert len(res.evidence_df) == 1

    def test_groupby_with_filter(self, superstore_df):
        """Verify: 'Which region generated highest profit for Technology?'"""
        plan = AnalysisPlan(
            operation=OperationType.GROUPBY_AGGREGATE,
            group_by_columns=["Region"],
            target_column="Profit",
            aggregation=AggregationType.SUM,
            filters=[
                FilterCondition(column="Category", operator=FilterOperator.EQUALS, value="Technology")
            ],
            sort_ascending=False,
            limit=1,
        )
        res = execute_plan(plan, superstore_df)
        tech_df = superstore_df[superstore_df["Category"].str.lower() == "technology"]
        expected_top_region = tech_df.groupby("Region")["Profit"].sum().idxmax()

        assert res.main_value["top_group"] == expected_top_region
        assert res.metadata["filtered_rows"] == len(tech_df)

    def test_trend_monthly_sales(self, superstore_df):
        """Verify: 'Show monthly sales.'"""
        plan = AnalysisPlan(
            operation=OperationType.TREND,
            date_column="Order Date",
            target_column="Sales",
            aggregation=AggregationType.SUM,
            time_granularity=TimeGranularity.MONTH,
        )
        res = execute_plan(plan, superstore_df)
        assert res.operation == OperationType.TREND
        assert res.chart_type == ChartRecommendation.LINE
        assert res.evidence_df is not None
        assert "Period" in res.evidence_df.columns
        assert len(res.evidence_df) > 12

    def test_correlation_sales_and_profit(self, superstore_df):
        """Verify: 'Show the relationship between sales and profit.'"""
        plan = AnalysisPlan(
            operation=OperationType.CORRELATION,
            target_columns=["Sales", "Profit"],
        )
        res = execute_plan(plan, superstore_df)
        expected_corr = round(float(superstore_df["Sales"].corr(superstore_df["Profit"])), 4)
        assert res.main_value == expected_corr
        assert res.chart_type == ChartRecommendation.SCATTER
        assert "Correlation between 'Sales' and 'Profit'" in res.evidence_summary

    def test_outlier_detection_sales(self, superstore_df):
        """Verify: 'Are there obvious outliers in sales?'"""
        plan = AnalysisPlan(
            operation=OperationType.OUTLIERS,
            target_column="Sales",
        )
        res = execute_plan(plan, superstore_df)
        raw_outliers = calculate_outliers_iqr(superstore_df["Sales"])

        assert res.main_value["outlier_count"] == raw_outliers["outlier_count"]
        assert res.chart_type == ChartRecommendation.BOX
        assert "Detected" in res.evidence_summary

    def test_unsupported_query_handling(self, superstore_df):
        """Verify explicit unsupported query plan returns clean explanation."""
        plan = AnalysisPlan(
            operation=OperationType.UNSUPPORTED,
            unsupported_reason="The dataset does not contain weather or employee salary data.",
        )
        res = execute_plan(plan, superstore_df)
        assert res.is_unsupported is True
        assert "weather or employee salary" in res.unsupported_reason

    def test_filter_preview_and_various_operators(self, superstore_df):
        """Verify all filter operators: in, not_in, !=, >, <=, contains."""
        # 'in' operator
        plan_in = AnalysisPlan(
            operation=OperationType.COUNT,
            filters=[
                FilterCondition(column="Region", operator=FilterOperator.IN, value=["West", "East"])
            ],
        )
        res_in = execute_plan(plan_in, superstore_df)
        expected_in_count = len(superstore_df[superstore_df["Region"].isin(["West", "East"])])
        assert res_in.main_value == expected_in_count

        # 'contains' operator
        plan_contains = AnalysisPlan(
            operation=OperationType.COUNT,
            filters=[
                FilterCondition(column="Product Name", operator=FilterOperator.CONTAINS, value="Phone")
            ],
        )
        res_contains = execute_plan(plan_contains, superstore_df)
        expected_contains_count = len(superstore_df[superstore_df["Product Name"].str.contains("Phone", case=False, na=False)])
        assert res_contains.main_value == expected_contains_count

        # Filter preview
        plan_preview = AnalysisPlan(
            operation=OperationType.FILTER_PREVIEW,
            limit=5,
        )
        res_preview = execute_plan(plan_preview, superstore_df)
        assert len(res_preview.evidence_df) == 5

    def test_statistics_zscore_and_descriptive(self, superstore_df):
        """Verify z-score outlier calculation and descriptive stats functions."""
        from src.analysis.statistics import calculate_outliers_zscore, calculate_descriptive_stats

        z_res = calculate_outliers_zscore(superstore_df["Sales"], threshold=3.0)
        assert "outlier_count" in z_res
        assert z_res["total_count"] == len(superstore_df)

        stats = calculate_descriptive_stats(superstore_df["Sales"])
        assert stats["count"] == len(superstore_df)
        assert "mean" in stats
        assert "50%" in stats

    def test_schema_aliases_and_string_comparison_filters(self, superstore_df):
        """Verify alias normalization (group_by, target, sort) and string-formatted numeric comparisons."""
        plan_dict = {
            "operation": "groupby_aggregate",
            "group_by": "Region",
            "target": "Sales",
            "aggregation": "avg",
            "sort": "descending",
            "limit": 2,
            "chart": "bar_chart",
            "filters": [
                {"column": "Sales", "operator": ">", "value": "100.0"}  # string representation of number
            ]
        }
        plan = AnalysisPlan.model_validate(plan_dict)
        assert plan.group_by_columns == ["Region"]
        assert plan.target_column == "Sales"
        assert plan.aggregation == AggregationType.MEAN
        assert plan.sort_ascending is False
        assert plan.chart == ChartRecommendation.BAR

        res = execute_plan(plan, superstore_df)
        assert res.chart_type == ChartRecommendation.BAR
        assert len(res.evidence_df) == 2
