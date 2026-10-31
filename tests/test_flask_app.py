"""Unit and integration tests for Flask web UI (src/web/app.py)."""
import json
import os
import pytest
from src.web.app import create_app
from src.web.state import app_state


@pytest.fixture
def client():
    """Configures test client for Flask application."""
    os.environ["LLM_PROVIDER"] = "mock"
    app_state.active_provider = "mock"
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    with app.test_client() as client:
        yield client


def test_index_redirects_to_ask(client):
    """Verifies that the root URL redirects to /ask."""
    res = client.get("/")
    assert res.status_code == 302
    assert "/ask" in res.headers["Location"]


def test_navigation_pages(client):
    """Verifies that all navigation pages load with status 200."""
    routes = [
        "/ask",
        "/dashboard",
        "/dataset",
        "/visualizations",
        "/knowledge",
        "/pipeline",
        "/settings",
    ]
    for route in routes:
        res = client.get(route)
        assert res.status_code == 200, f"Failed on route {route}"
        # Ensure brand and layout are present
        assert b"DataAnalyst" in res.data
        assert b"HCL P_123" in res.data
        assert b"sidebar" in res.data


def test_dataset_switch(client):
    """Verifies switching preset datasets."""
    res = client.post("/dataset/switch", data={"preset_name": "Telco Customer Churn (.csv)"}, follow_redirects=True)
    assert res.status_code == 200
    assert app_state.dataset_name == "Telco Customer Churn (.csv)"
    assert app_state.profile.row_count == 7043


def test_settings_update(client):
    """Verifies updating LLM settings."""
    res = client.post("/settings", data={"provider": "mock"}, follow_redirects=True)
    assert res.status_code == 200
    assert app_state.active_provider == "mock"


def test_flask_analysis_regression_top_5_products(client):
    """Runs regression test on 'Top 5 products by total sales' and checks against Phase 0 baseline."""
    # Ensure Superstore is loaded
    app_state.load_preset("Superstore Retail (.xls)")
    app_state.active_provider = "mock"

    res = client.post("/ask/run", data={"question": "Top 5 products by total sales"}, follow_redirects=True)
    assert res.status_code == 200
    assert app_state.last_response is not None

    resp = app_state.last_response
    assert resp.error_message is None
    assert not resp.is_unsupported
    assert "Canon imageCLASS 2200 Advanced Copier" in resp.direct_answer
    assert resp.plan.chart.value == "bar"
    assert resp.plan.operation.value == "groupby_aggregate"
    assert resp.plan.limit == 5

    # Verify baseline match
    with open("evaluation/regression_baseline.json", "r", encoding="utf-8") as f:
        baseline = json.load(f)

    baseline_top5 = next(item for item in baseline if item["query"] == "Top 5 products by total sales")
    assert resp.direct_answer == baseline_top5["direct_answer"]
    assert resp.plan.model_dump(exclude_none=True) == baseline_top5["validated_plan"]


def test_flask_all_baseline_regression_queries(client):
    """Runs all 6 representative baseline queries through the Flask interface and asserts 100% equivalence."""
    app_state.load_preset("Superstore Retail (.xls)")
    app_state.active_provider = "mock"

    with open("evaluation/regression_baseline.json", "r", encoding="utf-8") as f:
        baseline = json.load(f)

    for item in baseline:
        q = item["query"]
        res = client.post("/ask/run", data={"question": q}, follow_redirects=True)
        assert res.status_code == 200
        resp = app_state.last_response
        assert resp is not None
        assert resp.error_message is None
        assert resp.direct_answer == item["direct_answer"]
        assert resp.plan.chart.value == item["visualization_type"]

