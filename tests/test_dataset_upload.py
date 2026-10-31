"""Tests for Phase 2: Dataset Management & Local Upload (CSV, XLS, XLSX)."""
import io
import os
import pandas as pd
import pytest
from src.web.app import create_app
from src.web.state import app_state, PRESET_DATASETS


@pytest.fixture
def client():
    """Configures test client for Flask application."""
    os.environ["LLM_PROVIDER"] = "mock"
    app_state.active_provider = "mock"
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    with app.test_client() as client:
        yield client


def test_upload_valid_csv(client):
    """Verifies uploading a valid CSV file."""
    csv_content = b"Product,Category,Revenue\nWidget A,Hardware,150.0\nWidget B,Software,320.5\nWidget C,Hardware,85.0\n"
    data = {
        "file": (io.BytesIO(csv_content), "custom_data.csv"),
    }
    res = client.post("/dataset/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert res.status_code == 200
    assert app_state.is_uploaded is True
    assert app_state.dataset_name == "custom_data.csv"
    assert app_state.profile.row_count == 3
    assert app_state.profile.column_count == 3
    assert "Revenue" in app_state.profile.numeric_columns
    assert b"custom_data.csv" in res.data


def test_upload_and_query_custom_csv(client):
    """Verifies running an analysis query against an uploaded custom dataset."""
    csv_content = b"Department,Budget\nEngineering,500000\nMarketing,200000\nSales,350000\n"
    data = {
        "file": (io.BytesIO(csv_content), "departments.csv"),
    }
    client.post("/dataset/upload", data=data, content_type="multipart/form-data", follow_redirects=True)

    # Ask query on the uploaded dataset
    res = client.post("/ask/run", data={"question": "What is the average budget?"}, follow_redirects=True)
    assert res.status_code == 200
    assert app_state.last_response is not None
    assert app_state.last_response.error_message is None
    assert "350,000" in app_state.last_response.direct_answer or "350000" in app_state.last_response.direct_answer


def test_upload_multi_sheet_excel_and_switch(client):
    """Verifies uploading an Excel workbook with multiple sheets and switching between them."""
    # Create in-memory multi-sheet Excel file
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df_sheet1 = pd.DataFrame({"Customer": ["Alice", "Bob"], "Spend": [100, 200]})
        df_sheet2 = pd.DataFrame({"Supplier": ["Vendor X", "Vendor Y", "Vendor Z"], "Cost": [50, 75, 90]})
        df_sheet1.to_excel(writer, sheet_name="Customers", index=False)
        df_sheet2.to_excel(writer, sheet_name="Suppliers", index=False)

    excel_buffer.seek(0)
    data = {
        "file": (excel_buffer, "multi_sheet_test.xlsx"),
    }
    res = client.post("/dataset/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert res.status_code == 200
    assert app_state.is_uploaded is True
    assert app_state.available_sheets == ["Customers", "Suppliers"]
    assert app_state.profile.row_count == 2
    assert "Customer" in app_state.profile.columns

    # Switch sheet to "Suppliers"
    res_switch = client.post("/dataset/sheet", data={"sheet_name": "Suppliers"}, follow_redirects=True)
    assert res_switch.status_code == 200
    assert app_state.selected_sheet == "Suppliers"
    assert app_state.profile.row_count == 3
    assert "Supplier" in app_state.profile.columns


def test_upload_empty_file_fails_safely(client):
    """Verifies uploading an empty file produces an error notification and does not crash."""
    data = {
        "file": (io.BytesIO(b""), "empty.csv"),
    }
    res = client.post("/dataset/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert res.status_code == 200
    assert b"Upload Error" in res.data or b"empty" in res.data


def test_upload_unsupported_format_fails_safely(client):
    """Verifies uploading an unsupported file format produces a user-friendly error message."""
    data = {
        "file": (io.BytesIO(b"fake pdf binary data"), "document.pdf"),
    }
    res = client.post("/dataset/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert res.status_code == 200
    assert b"Format Error" in res.data or b"Unsupported" in res.data


def test_all_presets_still_load_cleanly(client):
    """Verifies that all 6 preset datasets still load and profile without regressions."""
    for preset_name in PRESET_DATASETS.keys():
        res = client.post("/dataset/switch", data={"preset_name": preset_name}, follow_redirects=True)
        assert res.status_code == 200
        assert app_state.dataset_name == preset_name
        assert app_state.profile.row_count > 0
        assert app_state.profile.column_count > 0
