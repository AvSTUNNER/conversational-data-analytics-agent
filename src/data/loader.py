"""Robust multi-format dataset loader supporting CSV, XLS, and XLSX."""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, List, Optional, Union
import pandas as pd


class DataLoadError(Exception):
    """Base exception for all data loading failures."""
    pass


class UnsupportedFormatError(DataLoadError):
    """Raised when the uploaded or specified file format is not supported."""
    pass


class EmptyFileError(DataLoadError):
    """Raised when the loaded file contains zero data rows or is empty."""
    pass


class DataLoader:
    """Handles secure, deterministic ingestion of CSV, XLS, and XLSX files."""

    SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

    @staticmethod
    def get_sheet_names(source: Union[str, Path, BinaryIO]) -> List[str]:
        """Returns the list of sheet names if the source is an Excel file, else empty list."""
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                ext = path.suffix.lower()
                if ext in {".xls", ".xlsx"}:
                    excel_file = pd.ExcelFile(path)
                    return excel_file.sheet_names
            elif hasattr(source, "read") and hasattr(source, "seek"):
                # File-like object
                curr_pos = source.tell()
                try:
                    excel_file = pd.ExcelFile(source)
                    sheet_names = excel_file.sheet_names
                    source.seek(curr_pos)
                    return sheet_names
                except Exception:
                    source.seek(curr_pos)
                    return []
        except Exception:
            return []
        return []

    @classmethod
    def load(
        cls,
        source: Union[str, Path, BinaryIO, io.StringIO, io.BytesIO],
        sheet_name: Optional[Union[str, int]] = 0,
        filename_hint: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads a dataset into a Pandas DataFrame from a path or file-like object.

        Args:
            source: Path to file or binary/text file-like buffer.
            sheet_name: Sheet index or name if loading an Excel workbook (default 0).
            filename_hint: Optional filename to assist format detection for stream buffers.

        Returns:
            pd.DataFrame: Cleanly ingested DataFrame.

        Raises:
            EmptyFileError: If the file has no data rows.
            UnsupportedFormatError: If format is not supported.
            DataLoadError: If loading fails due to parsing or I/O issues.
        """
        # Determine format
        extension = ""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise DataLoadError(f"File not found at path: {path}")
            if path.stat().st_size == 0:
                raise EmptyFileError(f"File is completely empty (0 bytes): {path}")
            extension = path.suffix.lower()
        elif filename_hint:
            extension = Path(filename_hint).suffix.lower()

        try:
            df: pd.DataFrame
            if extension == ".csv":
                df = cls._load_csv(source)
            elif extension == ".xls":
                df = cls._load_excel_xls(source, sheet_name=sheet_name)
            elif extension == ".xlsx":
                df = cls._load_excel_xlsx(source, sheet_name=sheet_name)
            elif not extension:
                # Try sniffing magic bytes / trying CSV then Excel
                df = cls._sniff_and_load(source, sheet_name=sheet_name)
            else:
                raise UnsupportedFormatError(
                    f"Unsupported file extension '{extension}'. Supported formats: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
                )

            if df is None or df.empty:
                raise EmptyFileError("The loaded dataset is empty (contains 0 rows).")

            return df

        except (EmptyFileError, UnsupportedFormatError):
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to load dataset: {str(e)}") from e

    @staticmethod
    def _load_csv(source: Union[str, Path, BinaryIO, io.StringIO, io.BytesIO]) -> pd.DataFrame:
        """Loads CSV with fallback encoding strategies."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
        last_err = None

        for enc in encodings:
            try:
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_csv(source, encoding=enc)
                return df
            except UnicodeDecodeError as ue:
                last_err = ue
                continue
            except pd.errors.EmptyDataError:
                raise EmptyFileError("CSV file contains no data or headers.")
            except Exception as e:
                # If error is not encoding related, raise immediately
                if "codec" in str(e).lower() or "decode" in str(e).lower():
                    last_err = e
                    continue
                raise

        raise DataLoadError(f"Failed to decode CSV across supported encodings: {last_err}")

    @staticmethod
    def _load_excel_xls(
        source: Union[str, Path, BinaryIO, io.BytesIO],
        sheet_name: Optional[Union[str, int]] = 0,
    ) -> pd.DataFrame:
        """Loads legacy Excel (.xls) using xlrd engine."""
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            return pd.read_excel(source, sheet_name=sheet_name, engine="xlrd")
        except Exception as e:
            # Fallback to default engine if xlrd fails
            if hasattr(source, "seek"):
                source.seek(0)
            try:
                return pd.read_excel(source, sheet_name=sheet_name)
            except Exception:
                raise DataLoadError(f"Error reading legacy Excel .xls file: {str(e)}") from e

    @staticmethod
    def _load_excel_xlsx(
        source: Union[str, Path, BinaryIO, io.BytesIO],
        sheet_name: Optional[Union[str, int]] = 0,
    ) -> pd.DataFrame:
        """Loads modern Excel (.xlsx) using openpyxl engine."""
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            return pd.read_excel(source, sheet_name=sheet_name, engine="openpyxl")
        except Exception as e:
            raise DataLoadError(f"Error reading Excel .xlsx file: {str(e)}") from e

    @classmethod
    def _sniff_and_load(
        cls,
        source: Union[str, Path, BinaryIO, io.BytesIO],
        sheet_name: Optional[Union[str, int]] = 0,
    ) -> pd.DataFrame:
        """Attempts loading by sniffing content when extension is absent."""
        # Try CSV first
        try:
            return cls._load_csv(source)
        except Exception:
            pass

        # Try XLSX
        try:
            return cls._load_excel_xlsx(source, sheet_name=sheet_name)
        except Exception:
            pass

        # Try XLS
        try:
            return cls._load_excel_xls(source, sheet_name=sheet_name)
        except Exception:
            pass

        raise UnsupportedFormatError("Unable to detect valid CSV or Excel format from source content.")


def load_dataset(
    source: Union[str, Path, BinaryIO, io.StringIO, io.BytesIO],
    sheet_name: Optional[Union[str, int]] = 0,
    filename_hint: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience helper function for loading datasets."""
    return DataLoader.load(source, sheet_name=sheet_name, filename_hint=filename_hint)
