"""Tests for Phase 3: Dynamic Knowledge System (Generation, Upload, Isolation, Lifecycle)."""
import io
import os
import pytest
from src.web.app import create_app
from src.web.state import app_state, PRESET_DATASETS
from src.knowledge.artifact import KnowledgeSourceType
from src.knowledge.manager import KnowledgeManager


@pytest.fixture
def client():
    """Configures test client for Flask application with mock LLM provider."""
    os.environ["LLM_PROVIDER"] = "mock"
    app_state.active_provider = "mock"
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    with app.test_client() as client:
        yield client


def test_generated_knowledge_from_preset_dataset(client):
    """Verifies that knowledge is automatically generated for the default preset dataset."""
    res = client.get("/knowledge")
    assert res.status_code == 200

    artifact = app_state.current_knowledge
    assert artifact is not None
    assert artifact.source_type == KnowledgeSourceType.GENERATED
    assert artifact.dataset_name == "Superstore Retail (.xls)"
    assert artifact.observed.row_count > 9000
    assert "Sales" in artifact.observed.columns
    assert "Profit" in artifact.observed.columns
    # Check that observed facts match profiler
    assert artifact.observed.row_count == app_state.profile.row_count
    assert artifact.observed.column_count == app_state.profile.column_count
    # Inferred metadata must have markers
    assert artifact.inferred.suggested_domain is not None
    assert b"Auto-Generated (Facts-Only)" in res.data
    assert b"[INFERRED]" in res.data


def test_supplied_knowledge_upload(client):
    """Verifies uploading a custom knowledge document for the active dataset."""
    knowledge_doc = (
        "# Superstore Business Glossary\n\n"
        "### Metric Definitions\n"
        "- **Sales**: Gross transaction amount in USD before returns.\n"
        "- **Profit**: Net margin after subtracting COGS and regional overhead.\n"
        "- **Discount**: Rate applied to unit list price (0.0 to 0.8).\n"
    )
    data = {
        "file": (io.BytesIO(knowledge_doc.encode("utf-8")), "superstore_glossary.md"),
    }
    res = client.post("/knowledge/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert res.status_code == 200

    artifact = app_state.current_knowledge
    assert artifact is not None
    assert artifact.source_type == KnowledgeSourceType.SUPPLIED
    assert artifact.supplied_filename == "superstore_glossary.md"
    assert "Gross transaction amount" in artifact.supplied_content
    # Underlying observed facts must still be preserved
    assert artifact.observed.row_count == app_state.profile.row_count
    assert "Sales" in artifact.observed.columns

    # UI displays user-supplied badge and content
    assert b"User-Supplied Knowledge" in res.data
    assert b"Superstore Business Glossary" in res.data


def test_supplied_knowledge_remains_distinguishable(client):
    """Ensures supplied knowledge, observed facts, and inferred metadata remain distinct."""
    # Ensure superstore is loaded with supplied knowledge
    knowledge_doc = "Customer Tier 1 means annual spend exceeding $5,000."
    data = {"file": (io.BytesIO(knowledge_doc.encode("utf-8")), "tiers.txt")}
    client.post("/knowledge/upload", data=data, content_type="multipart/form-data", follow_redirects=True)

    artifact = app_state.current_knowledge
    assert artifact.source_type == KnowledgeSourceType.SUPPLIED

    # Rendered markdown text must separate them cleanly
    full_text = artifact.get_full_text()
    assert "## User-Supplied Knowledge Document" in full_text
    assert "Customer Tier 1 means annual spend" in full_text
    assert "## Underlying Observed Dataset Facts (Authoritative Ground Truth)" in full_text
    assert "## Inferred Semantic Notes (Non-Authoritative)" in full_text
    assert "[INFERRED]" in full_text


def test_dataset_isolation_on_preset_switch(client):
    """Verifies that switching datasets does NOT leak previous knowledge to the new dataset."""
    # 1. Attach supplied knowledge to Superstore
    knowledge_doc = "Confidential Superstore Margin Target: 25%."
    data = {"file": (io.BytesIO(knowledge_doc.encode("utf-8")), "superstore_secrets.md")}
    client.post("/knowledge/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert "superstore_secrets.md" in app_state.current_knowledge.supplied_filename

    # 2. Switch to Telco Churn dataset
    res_switch = client.post("/dataset/switch", data={"preset_name": "Telco Customer Churn (.csv)"}, follow_redirects=True)
    assert res_switch.status_code == 200
    assert app_state.dataset_name == "Telco Customer Churn (.csv)"

    # 3. Check active knowledge for Telco
    telco_artifact = app_state.current_knowledge
    assert telco_artifact is not None
    assert telco_artifact.dataset_name == "Telco Customer Churn (.csv)"
    # Must NOT have Superstore's supplied document
    assert telco_artifact.source_type == KnowledgeSourceType.GENERATED
    assert telco_artifact.supplied_content is None
    assert telco_artifact.supplied_filename is None
    # Telco's observed facts must be accurate
    assert "Churn" in telco_artifact.observed.columns or "churn" in [c.lower() for c in telco_artifact.observed.columns]

    # 4. Switch back to Superstore — its associated knowledge is remembered and re-activated
    client.post("/dataset/switch", data={"preset_name": "Superstore Retail (.xls)"}, follow_redirects=True)
    superstore_artifact = app_state.current_knowledge
    assert superstore_artifact.dataset_name == "Superstore Retail (.xls)"
    assert superstore_artifact.source_type == KnowledgeSourceType.SUPPLIED
    assert "Confidential Superstore Margin" in superstore_artifact.supplied_content


def test_uploaded_dataset_knowledge_lifecycle(client):
    """Verifies knowledge generation and lifecycle when uploading a custom CSV dataset."""
    custom_csv = b"EmployeeID,Department,Salary,HireDate\n101,Engineering,120000,2022-01-15\n102,HR,65000,2021-06-01\n"
    data = {"file": (io.BytesIO(custom_csv), "staff.csv")}
    client.post("/dataset/upload", data=data, content_type="multipart/form-data", follow_redirects=True)

    artifact = app_state.current_knowledge
    assert artifact is not None
    assert artifact.dataset_name == "staff.csv"
    assert artifact.source_type == KnowledgeSourceType.GENERATED
    assert artifact.observed.row_count == 2
    assert artifact.observed.column_count == 4
    assert "Salary" in artifact.observed.columns
    assert "EmployeeID" in artifact.observed.columns

    # Now upload knowledge for this custom dataset
    staff_doc = b"Engineering salaries follow Band E4 compensation guidelines."
    client.post("/knowledge/upload", data={"file": (io.BytesIO(staff_doc), "bands.txt")}, content_type="multipart/form-data", follow_redirects=True)

    artifact_updated = app_state.current_knowledge
    assert artifact_updated.source_type == KnowledgeSourceType.SUPPLIED
    assert artifact_updated.supplied_filename == "bands.txt"
    assert "Band E4 compensation guidelines" in artifact_updated.supplied_content


def test_knowledge_download_route(client):
    """Verifies that knowledge artifact can be downloaded as a Markdown file."""
    res = client.get("/knowledge/download")
    assert res.status_code == 200
    assert "text/markdown" in res.content_type
    assert "attachment; filename=" in res.headers.get("Content-Disposition", "")
    assert b"# Knowledge Artifact:" in res.data
    assert b"Authoritative Ground Truth" in res.data


def test_knowledge_reset_to_generated(client):
    """Verifies resetting user-supplied knowledge back to purely auto-generated facts."""
    # 1. Supply knowledge
    client.post("/knowledge/upload", data={"file": (io.BytesIO(b"Temporary notes"), "notes.txt")}, content_type="multipart/form-data", follow_redirects=True)
    assert app_state.current_knowledge.source_type == KnowledgeSourceType.SUPPLIED

    # 2. Reset
    res_reset = client.post("/knowledge/reset", follow_redirects=True)
    assert res_reset.status_code == 200
    assert app_state.current_knowledge.source_type == KnowledgeSourceType.GENERATED
    assert app_state.current_knowledge.supplied_content is None


def test_empty_or_unsupported_knowledge_upload_fails_safely(client):
    """Verifies that empty files or invalid extensions fail safely without crashing."""
    # Empty file
    res_empty = client.post("/knowledge/upload", data={"file": (io.BytesIO(b""), "empty.md")}, content_type="multipart/form-data", follow_redirects=True)
    assert res_empty.status_code == 200
    assert b"empty" in res_empty.data

    # Unsupported extension (e.g. .exe)
    res_unsupported = client.post("/knowledge/upload", data={"file": (io.BytesIO(b"malicious"), "danger.exe")}, content_type="multipart/form-data", follow_redirects=True)
    assert res_unsupported.status_code == 200
    assert b"Unsupported knowledge format" in res_unsupported.data
