"""Unit tests for the non-destructive DataCleaner."""
import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import DataCleaner, clean_dataset
from src.data.loader import load_dataset


class TestDataCleaner:
    """Test suite for data cleaning and type normalization."""

    def test_strip_column_whitespace_and_handle_duplicates(self):
        """Verify column names are trimmed and duplicates disambiguated."""
        df_raw = pd.DataFrame({
            "  Region  ": ["North", "South"],
            "Region": ["East", "West"],
            " Sales ": [100, 200],
        })
        cleaned_df, summary = clean_dataset(df_raw)
        assert list(cleaned_df.columns) == ["Region", "Region_1", "Sales"]
        assert "  Region  " in summary.renamed_columns
        assert summary.renamed_columns["  Region  "] == "Region"

    def test_whitespace_strings_to_nan(self):
        """Verify whitespace strings are converted to NaN without corrupting non-empty strings."""
        df_raw = pd.DataFrame({
            "TextCol": ["Valid", "  ", "", "Another Valid", None],
        })
        cleaned_df, summary = clean_dataset(df_raw)
        assert pd.isna(cleaned_df["TextCol"].iloc[1])
        assert pd.isna(cleaned_df["TextCol"].iloc[2])
        assert cleaned_df["TextCol"].iloc[0] == "Valid"
        assert cleaned_df["TextCol"].iloc[3] == "Another Valid"
        assert "TextCol" in summary.whitespace_cleaned_columns

    def test_currency_and_formatted_number_coercion(self):
        """Verify currency strings ($1,200.50) are coerced to numeric floats."""
        df_raw = pd.DataFrame({
            "Revenue": ["$1,200.50", "$350.00", "$4,500.25", "$50.00"],
            "DiscountPct": ["10%", "20%", "0%", "15%"],
        })
        cleaned_df, summary = clean_dataset(df_raw)
        assert pd.api.types.is_numeric_dtype(cleaned_df["Revenue"])
        assert cleaned_df["Revenue"].iloc[0] == 1200.50
        assert cleaned_df["Revenue"].iloc[2] == 4500.25
        assert cleaned_df["DiscountPct"].iloc[0] == 0.10
        assert "Revenue" in summary.coerced_numeric_columns
        assert "DiscountPct" in summary.coerced_numeric_columns

    def test_date_string_coercion(self):
        """Verify date-like strings are converted to Datetime."""
        df_raw = pd.DataFrame({
            "OrderDate": ["2023-01-01", "2023-01-02", "2023-01-15"],
            "Ship_Date": ["01/05/2023", "01/06/2023", "01/20/2023"],
        })
        cleaned_df, summary = clean_dataset(df_raw)
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df["OrderDate"])
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df["Ship_Date"])
        assert "OrderDate" in summary.coerced_datetime_columns

    def test_cleaning_real_telco_churn_dataset(self):
        """Verify TotalCharges in telco_churn (which has whitespace for 0 tenure) is cleaned into float."""
        df_raw = load_dataset("datasets/validation/telco_churn.csv")
        assert not pd.api.types.is_numeric_dtype(df_raw["TotalCharges"])

        cleaned_df, summary = clean_dataset(df_raw)
        assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])
        # Exactly 11 rows in telco_churn have 0 tenure and blank TotalCharges
        assert cleaned_df["TotalCharges"].isnull().sum() == 11
        assert "TotalCharges" in summary.coerced_numeric_columns
        summary_dict = summary.to_dict()
        assert "coerced_numeric_columns" in summary_dict

    def test_cleaning_unnamed_and_empty_column_names(self):
        """Verify empty string column names are renamed to Unnamed."""
        df_raw = pd.DataFrame([["val1", "val2"]], columns=["", "  "])
        cleaned_df, summary = clean_dataset(df_raw)
        assert list(cleaned_df.columns) == ["Unnamed", "Unnamed_1"]

    def test_accounting_parentheses_coercion(self):
        """Verify accounting formatted negatives ($150.00) convert to -150.0."""
        df_raw = pd.DataFrame({"Profit": ["($150.50)", "$200.50", "(50.00)", "0.00"]})
        cleaned_df, summary = clean_dataset(df_raw)
        assert pd.api.types.is_numeric_dtype(cleaned_df["Profit"])
        assert cleaned_df["Profit"].iloc[0] == -150.50
        assert cleaned_df["Profit"].iloc[2] == -50.00
        assert cleaned_df["Profit"].sum() == 0.0
