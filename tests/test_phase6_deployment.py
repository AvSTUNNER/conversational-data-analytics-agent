"""Tests for Phase 6: Dashboard, Pipeline Page, Settings, Security Hardening, and Deployment Verification."""
from __future__ import annotations

import io
import os
import pytest
import pandas as pd
from src.web.app import create_app
from src.web.state import app_state, PRESET_DATASETS
from src.llm.client import get_llm_client, MockLLMClient
from src.rag.router import QueryIntent
from src.assistant import ConversationalAssistant


@pytest.fixture
def client():
    """Configures test client with mock LLM and test configuration."""
    os.environ["LLM_PROVIDER"] = "mock"
    app_state.active_provider = "mock"
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret-key"})
    with app.test_client() as test_client:
        yield test_client


def test_dashboard_page_rendering(client):
    """Verifies that the dashboard page renders dataset metrics, RAG index status, and capabilities."""
    res = client.get("/dashboard")
    assert res.status_code == 200
    assert b"Active Dataset" in res.data
    assert b"Knowledge Layer" in res.data
    assert b"RAG Vector Store" in res.data
    assert b"Analysis &amp; Reasoning Capabilities" in res.data or b"Analysis & Reasoning Capabilities" in res.data
    assert b"Available Benchmark Datasets" in res.data


def test_pipeline_page_rendering(client):
    """Verifies that the pipeline page displays the 3 pathways, trust boundary, and decision matrix."""
    res = client.get("/pipeline")
    assert res.status_code == 200
    # Check 3 distinct execution paths
    assert b"KNOWLEDGE" in res.data
    assert b"ANALYTICAL" in res.data
    assert b"HYBRID" in res.data
    # Check the critical trust boundary
    assert b"Retrieved knowledge provides semantic context only" in res.data
    assert b"Numerical truth comes from the active DataFrame through deterministic Pandas execution" in res.data
    # Check decision matrix
    assert b"Execution &amp; Responsibility Decision Matrix" in res.data or b"Execution & Responsibility Decision Matrix" in res.data


def test_settings_page_security_no_secrets_exposed(client):
    """Verifies settings page lists providers and runtime info without leaking secrets."""
    res = client.get("/settings")
    assert res.status_code == 200
    assert b"Active LLM Provider Selection" in res.data
    assert b"Provider Configuration &amp; Readiness Matrix" in res.data or b"Provider Configuration & Readiness Matrix" in res.data
    assert b"Runtime Environment &amp; Security Policies" in res.data or b"Runtime Environment & Security Policies" in res.data
    # Verify no raw secrets or passwords in HTML
    html_text = res.data.decode("utf-8")
    assert "gsk_" not in html_text
    assert "AIzaSy" not in html_text
    assert "password" not in html_text.lower() or "reset" in html_text.lower()


def test_settings_update_provider(client):
    """Verifies updating active provider and rejecting invalid ones."""
    res = client.post("/settings", data={"provider": "mock"}, follow_redirects=True)
    assert res.status_code == 200
    assert app_state.active_provider == "mock"

    res = client.post("/settings", data={"provider": "invalid_provider"}, follow_redirects=True)
    assert res.status_code == 200
    assert b"Invalid provider" in res.data


def test_security_headers_present(client):
    """Verifies defensive HTTP security headers are set on responses."""
    res = client.get("/dashboard")
    assert res.status_code == 200
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert res.headers.get("X-XSS-Protection") == "1; mode=block"


def test_security_filename_sanitization_on_dataset_upload(client):
    """Verifies that uploaded dataset filenames with path traversal are sanitized."""
    csv_bytes = b"ColA,ColB\n1,2\n3,4\n"
    # Filename with path traversal
    data = {
        "file": (io.BytesIO(csv_bytes), "../../suspicious_name.csv"),
    }
    res = client.post("/dataset/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert res.status_code == 200
    # secure_filename sanitizes '../../suspicious_name.csv' into 'suspicious_name.csv'
    assert app_state.dataset_name == "suspicious_name.csv"
    assert app_state.profile.row_count == 2


def test_security_unsupported_dataset_extension_rejected(client):
    """Verifies that executable or unsupported formats are rejected."""
    bad_bytes = b"print('malicious')"
    data = {
        "file": (io.BytesIO(bad_bytes), "payload.py"),
    }
    res = client.post("/dataset/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert res.status_code == 200
    assert b"Unsupported dataset format" in res.data


def test_security_unsupported_knowledge_extension_rejected(client):
    """Verifies that invalid extensions for knowledge files are rejected."""
    bad_bytes = b"some executable data"
    data = {
        "file": (io.BytesIO(bad_bytes), "payload.exe"),
    }
    res = client.post("/knowledge/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert res.status_code == 200
    assert b"Unsupported knowledge format" in res.data


def test_custom_error_pages(client):
    """Verifies custom 404 page is rendered cleanly without leaking internal paths."""
    res = client.get("/non_existent_page_path")
    assert res.status_code == 404
    assert b"404" in res.data
    assert b"Page Not Found" in res.data
    assert b"Traceback" not in res.data


def test_provider_missing_key_clear_error(monkeypatch):
    """Verifies get_llm_client fails clearly when required credentials are missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        get_llm_client("gemini")

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        get_llm_client("groq")


def test_all_six_benchmark_datasets_lifecycle():
    """Verifies all six benchmark datasets load, profile, generate knowledge, and sync RAG indices."""
    manager = app_state

    for name in PRESET_DATASETS.keys():
        success = manager.load_preset(name)
        assert success is True, f"Failed to load preset: {name}"
        assert manager.cleaned_df is not None, f"Cleaned df is None for {name}"
        assert len(manager.cleaned_df) > 0, f"Cleaned df empty for {name}"
        assert manager.profile is not None, f"Profile is None for {name}"
        assert manager.profile.row_count > 0, f"Profile row count 0 for {name}"

        # Check knowledge artifact
        artifact = manager.current_knowledge
        assert artifact is not None, f"Knowledge artifact missing for {name}"
        assert len(artifact.observed.columns) > 0, f"Knowledge artifact has no observed columns for {name}"
        assert len(artifact.get_full_text()) > 0, f"Knowledge artifact text is empty for {name}"

        # Check RAG vector index sync
        index, created = manager.sync_active_index()
        assert index is not None, f"RAG index failed to sync for {name}"
        assert index.chunk_count > 0, f"RAG index has 0 chunks for {name}"
        assert index.dataset_name == name, f"RAG index dataset name mismatch for {name}"


def test_representative_queries_across_intents():
    """Verifies representative KNOWLEDGE, ANALYTICAL, and HYBRID queries execute successfully."""
    # Ensure superstore is loaded
    app_state.load_preset("Superstore Retail (.xls)")
    app_state.sync_active_index()

    assistant = ConversationalAssistant(
        llm_client=MockLLMClient(),
        retriever=app_state.retriever,
        router=app_state.query_router,
    )

    # 1. KNOWLEDGE Query
    resp_k = assistant.analyze(
        question="What does Customer Segment mean in this dataset?",
        df=app_state.cleaned_df,
        dataset_name=app_state.dataset_name,
    )
    assert resp_k.intent == QueryIntent.KNOWLEDGE
    assert resp_k.retrieval_used is True
    assert resp_k.plan is None
    assert len(resp_k.direct_answer) > 0

    # 2. ANALYTICAL Query
    resp_a = assistant.analyze(
        question="What is the average Sales?",
        df=app_state.cleaned_df,
        dataset_name=app_state.dataset_name,
    )
    assert resp_a.intent == QueryIntent.ANALYTICAL
    assert resp_a.retrieval_used is False
    assert resp_a.plan is not None
    assert "228.23" in resp_a.direct_answer or "228" in resp_a.direct_answer

    # 3. HYBRID Query
    resp_h = assistant.analyze(
        question="Explain what customer segment means and calculate total sales by segment.",
        df=app_state.cleaned_df,
        dataset_name=app_state.dataset_name,
    )
    assert resp_h.intent == QueryIntent.HYBRID
    assert resp_h.retrieval_used is True
    assert resp_h.plan is not None
    assert resp_h.evidence_df is not None
    assert len(resp_h.direct_answer) > 0
