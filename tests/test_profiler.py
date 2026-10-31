"""Unit tests for the dynamic DatasetProfiler."""
import pandas as pd
import pytest

from src.data.loader import load_dataset
from src.data.cleaner import clean_dataset
from src.data.profiler import DatasetProfiler, profile_dataset


class TestDatasetProfiler:
    """Test suite for dataset profiling and schema generation."""

    def test_profile_synthetic_business_dataset(self):
        """Verify profiling on the clean synthetic business dataset."""
        df = load_dataset("datasets/synthetic/business.csv")
        cleaned_df, _ = clean_dataset(df)
        profile = profile_dataset(cleaned_df)

        assert profile.row_count == 200
        assert profile.column_count == 9
        assert "Sales" in profile.numeric_columns
        assert "Profit" in profile.numeric_columns
        assert "Quantity" in profile.numeric_columns
        assert "Discount" in profile.numeric_columns
        assert "Region" in profile.categorical_columns
        assert "Category" in profile.categorical_columns
        assert "OrderDate" in profile.datetime_columns

        # Verify numeric stats
        sales_prof = profile.get_column("sales")
        assert sales_prof is not None
        assert sales_prof.data_type == "numeric"
        assert sales_prof.numeric_stats is not None
        assert sales_prof.numeric_stats.min >= 10.0
        assert sales_prof.numeric_stats.max <= 1500.0

    def test_profile_real_superstore_dataset(self):
        """Verify profiling on the real Superstore dataset."""
        df = load_dataset("datasets/demo/sample_-_superstore.xls", sheet_name="Orders")
        cleaned_df, _ = clean_dataset(df)
        profile = profile_dataset(cleaned_df)

        assert profile.row_count == 10194
        assert profile.column_count == 21

        # Check case-insensitive lookup
        profit_col = profile.get_column("profit")
        assert profit_col is not None
        assert profit_col.name == "Profit"
        assert profit_col.data_type == "numeric"

        cat_col = profile.get_column("category")
        assert cat_col is not None
        assert cat_col.data_type == "categorical"
        assert cat_col.unique_count == 3

        date_col = profile.get_column("order date")
        assert date_col is not None
        assert date_col.data_type == "datetime"
        assert date_col.date_range is not None

    def test_llm_schema_prompt_generation(self):
        """Verify that the generated prompt is compact, informative, and token-efficient."""
        df = pd.DataFrame({
            "Region": ["North", "South", "East", "West"],
            "Sales": [100.5, 200.0, 300.2, 400.0],
            "OrderDate": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
        })
        profile = profile_dataset(df)
        prompt_text = profile.to_llm_schema_prompt()

        assert "Dataset Summary: 4 rows, 3 columns" in prompt_text
        assert "Region: CATEGORICAL" in prompt_text
        assert "Sales: NUMERIC" in prompt_text
        assert "OrderDate: DATETIME" in prompt_text
        assert "samples:" in prompt_text
        assert "range: [100.5 to 400]" in prompt_text

    def test_profile_boolean_and_edge_types(self):
        """Verify boolean column detection."""
        df = pd.DataFrame({
            "IsActive": [True, False, True, True],
            "Flag": [0, 1, 0, 0],
            "ConstantCol": ["A", "A", "A", "A"],
        })
        profile = profile_dataset(df)
        assert "IsActive" in profile.boolean_columns
        assert "ConstantCol" in profile.categorical_columns
