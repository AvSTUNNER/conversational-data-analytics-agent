"""Unit tests for the rigid PlanValidator."""
import pandas as pd
import pytest

from src.analysis.schemas import (
    AnalysisPlan,
    OperationType,
    AggregationType,
    FilterCondition,
    FilterOperator,
    TimeGranularity,
)
from src.analysis.validator import PlanValidator, ValidationError


class TestPlanValidator:
    """Test suite for plan validation and error detection."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "Region": ["North", "South", "East", "West"],
            "Sales": [100.0, 200.0, 300.0, 400.0],
            "OrderDate": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
            "CustomerName": ["Alice", "Bob", "Charlie", "David"],
        })

    def test_valid_count_plan(self, sample_df):
        plan = AnalysisPlan(operation=OperationType.COUNT)
        PlanValidator.validate(plan, sample_df)

    def test_valid_aggregation_plan(self, sample_df):
        plan = AnalysisPlan(
            operation=OperationType.AGGREGATION,
            target_column="Sales",
            aggregation=AggregationType.MEAN,
        )
        PlanValidator.validate(plan, sample_df)

    def test_missing_target_column_raises_error(self, sample_df):
        plan = AnalysisPlan(
            operation=OperationType.AGGREGATION,
            target_column="NonExistentCol",
            aggregation=AggregationType.SUM,
        )
        with pytest.raises(ValidationError, match="not found in dataset"):
            PlanValidator.validate(plan, sample_df)

    def test_close_match_suggestion(self, sample_df):
        plan = AnalysisPlan(
            operation=OperationType.AGGREGATION,
            target_column="Sale",  # typo for Sales
            aggregation=AggregationType.SUM,
        )
        with pytest.raises(ValidationError, match="Did you mean: Sales"):
            PlanValidator.validate(plan, sample_df)

    def test_numeric_aggregation_on_string_column_raises_error(self, sample_df):
        plan = AnalysisPlan(
            operation=OperationType.AGGREGATION,
            target_column="CustomerName",
            aggregation=AggregationType.SUM,
        )
        with pytest.raises(ValidationError, match="must be numeric"):
            PlanValidator.validate(plan, sample_df)

    def test_groupby_missing_group_columns_raises_error(self, sample_df):
        plan = AnalysisPlan(
            operation=OperationType.GROUPBY_AGGREGATE,
            target_column="Sales",
            aggregation=AggregationType.SUM,
        )
        with pytest.raises(ValidationError, match="requires 'group_by_columns'"):
            PlanValidator.validate(plan, sample_df)

    def test_trend_on_non_date_column_raises_error(self, sample_df):
        plan = AnalysisPlan(
            operation=OperationType.TREND,
            date_column="Region",
            target_column="Sales",
            time_granularity=TimeGranularity.MONTH,
        )
        with pytest.raises(ValidationError, match="must be of datetime type"):
            PlanValidator.validate(plan, sample_df)

    def test_invalid_negative_limit_raises_error(self, sample_df):
        plan = AnalysisPlan(
            operation=OperationType.COUNT,
            limit=-5,
        )
        with pytest.raises(ValidationError, match="must be a positive integer"):
            PlanValidator.validate(plan, sample_df)

    def test_relational_filter_on_string_column_raises_error(self, sample_df):
        plan = AnalysisPlan(
            operation=OperationType.COUNT,
            filters=[
                FilterCondition(column="Region", operator=FilterOperator.GREATER_THAN, value=50)
            ],
        )
        with pytest.raises(ValidationError, match="Relational operator.*cannot be applied to non-numeric"):
            PlanValidator.validate(plan, sample_df)

    def test_empty_dataframe_raises_error(self):
        plan = AnalysisPlan(operation=OperationType.COUNT)
        with pytest.raises(ValidationError, match="empty DataFrame"):
            PlanValidator.validate(plan, pd.DataFrame())
