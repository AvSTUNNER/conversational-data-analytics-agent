"""Unit tests for the multi-format DataLoader."""
import io
from pathlib import Path
import pytest
import pandas as pd

from src.data.loader import (
    DataLoader,
    load_dataset,
    DataLoadError,
    UnsupportedFormatError,
    EmptyFileError,
)


class TestDataLoader:
    """Test suite for DataLoader with real and edge case datasets."""

    def test_load_real_superstore_xls(self):
        """Verify loading the real BIFF8 .xls Superstore dataset."""
        path = "datasets/demo/sample_-_superstore.xls"
        sheet_names = DataLoader.get_sheet_names(path)
        assert "Orders" in sheet_names
        assert "People" in sheet_names
        assert "Returns" in sheet_names

        df = load_dataset(path, sheet_name="Orders")
        assert not df.empty
        assert len(df) == 10194
        assert len(df.columns) == 21
        assert "Sales" in df.columns
        assert "Profit" in df.columns

    def test_load_real_online_shoppers_csv(self):
        """Verify loading the real CSV dataset."""
        path = "datasets/validation/online_shoppers.csv"
        df = load_dataset(path)
        assert len(df) == 12330
        assert len(df.columns) == 18
        assert "BounceRates" in df.columns

    def test_load_real_online_retail_xlsx_sheets(self):
        """Verify sheet detection and loading on large XLSX workbook."""
        path = "datasets/validation/online_retail.xlsx"
        sheet_names = DataLoader.get_sheet_names(path)
        assert "Online Retail" in sheet_names

    def test_load_real_telco_churn_csv(self):
        """Verify loading the real Telco churn dataset."""
        path = "datasets/validation/telco_churn.csv"
        df = load_dataset(path)
        assert len(df) == 7043
        assert len(df.columns) == 21
        assert "MonthlyCharges" in df.columns

    def test_load_from_bytes_io_buffer(self):
        """Verify loading from in-memory BytesIO (Streamlit file uploader simulation)."""
        csv_bytes = b"Category,Sales\nElectronics,500.0\nFurniture,300.0\n"
        buffer = io.BytesIO(csv_bytes)
        df = load_dataset(buffer, filename_hint="upload.csv")
        assert len(df) == 2
        assert list(df.columns) == ["Category", "Sales"]
        assert df["Sales"].sum() == 800.0

    def test_empty_file_raises_empty_file_error(self):
        """Verify 0-byte file raises EmptyFileError."""
        with pytest.raises(EmptyFileError):
            load_dataset("datasets/edge_cases/empty.csv")

    def test_headers_only_file_raises_empty_file_error(self):
        """Verify file with headers but 0 rows raises EmptyFileError."""
        with pytest.raises(EmptyFileError):
            load_dataset("datasets/edge_cases/headers_only.csv")

    def test_get_sheet_names_on_csv_and_invalid(self):
        """Verify get_sheet_names returns empty list on CSV or non-excel files."""
        assert DataLoader.get_sheet_names("datasets/validation/online_shoppers.csv") == []
        assert DataLoader.get_sheet_names("non_existent_file.xyz") == []

    def test_get_sheet_names_on_bytes_io_buffer(self):
        """Verify get_sheet_names works with BytesIO buffer."""
        with open("datasets/demo/sample_-_superstore.xls", "rb") as f:
            buf = io.BytesIO(f.read())
        sheet_names = DataLoader.get_sheet_names(buf)
        assert "Orders" in sheet_names
        assert "People" in sheet_names

    def test_sniff_and_load_csv_without_extension(self):
        """Verify auto-sniffing when no filename extension is present."""
        csv_bytes = b"ColA,ColB\n10,20\n30,40\n"
        buf = io.BytesIO(csv_bytes)
        df = DataLoader.load(buf)
        assert len(df) == 2
        assert list(df.columns) == ["ColA", "ColB"]

    def test_unsupported_format_raises_error(self):
        """Verify unsupported file extensions are rejected."""
        dummy_path = Path("datasets/edge_cases/dummy.txt")
        dummy_path.write_text("hello world")
        try:
            with pytest.raises(UnsupportedFormatError):
                load_dataset(str(dummy_path))
        finally:
            if dummy_path.exists():
                dummy_path.unlink()

    def test_missing_file_raises_load_error(self):
        """Verify non-existent file path raises DataLoadError."""
        with pytest.raises(DataLoadError):
            load_dataset("datasets/non_existent_file_xyz.csv")
