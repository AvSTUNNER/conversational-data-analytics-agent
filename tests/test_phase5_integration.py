"""Focused end-to-end integration tests for Phase 5: RAG -> Existing Planner -> Pandas Integration."""
import io
import os
import pytest
import pandas as pd

from src.assistant import ConversationalAssistant, AssistantResponse
from src.analysis.schemas import AnalysisPlan, OperationType, ChartRecommendation
from src.analysis.validator import PlanValidator, ValidationError
from src.llm.client import get_llm_client
from src.rag.router import QueryIntent, QueryRouter
from src.rag.retriever import KnowledgeRetriever
from src.web.app import create_app
from src.web.state import app_state, PRESET_DATASETS


@pytest.fixture
def client():
    """Configures test client for Flask application with mock LLM provider."""
    os.environ["LLM_PROVIDER"] = "mock"
    app_state.active_provider = "mock"
    app_state.reset_state()
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    with app.test_client() as client:
        yield client


# ==============================================================================
# 1. KNOWLEDGE-ORIENTED FLOW TESTS
# ==============================================================================

def test_knowledge_query_retrieval_only_and_no_planner_call(client):
    """Verifies KNOWLEDGE query retrieves context directly and intentionally bypasses the analytical planner."""
    # Ensure Superstore is loaded
    client.post("/dataset/switch", data={"preset_name": "Superstore Retail (.xls)"}, follow_redirects=True)

    # Attach a domain definition
    doc = (
        "# Superstore Definitions\n\n"
        "Customer Segment defines the purchasing tier of the client: Consumer, Corporate, or Home Office."
    )
    client.post("/knowledge/upload", data={"file": (io.BytesIO(doc.encode("utf-8")), "defs.md")}, content_type="multipart/form-data", follow_redirects=True)

    # Run query via assistant
    assistant = ConversationalAssistant(
        llm_client=get_llm_client("mock"),
        retriever=app_state.retriever,
        router=app_state.query_router,
    )
    res = assistant.analyze("What does Customer Segment mean?", df=app_state.cleaned_df, dataset_name=app_state.dataset_name)

    assert res.intent == "knowledge"
    assert res.retrieval_used is True
    assert res.plan is None  # Planner must NOT be called for pure knowledge
    assert "Customer Segment defines the purchasing tier" in res.direct_answer or "Customer Segment defines" in res.explanation
    assert len(res.retrieved_chunks) > 0


def test_knowledge_query_no_retrieval_does_not_hallucinate(client):
    """Verifies that an ungrounded knowledge query with no relevant matches returns a clear insufficient-knowledge response without hallucination."""
    client.post("/dataset/switch", data={"preset_name": "Superstore Retail (.xls)"}, follow_redirects=True)

    assistant = ConversationalAssistant(
        llm_client=get_llm_client("mock"),
        retriever=app_state.retriever,
        router=app_state.query_router,
    )
    # Ask about something completely absent
    res = assistant.analyze("Who is the chief operating officer and author of this dataset?", df=app_state.cleaned_df, dataset_name=app_state.dataset_name)

    assert res.intent == "knowledge"
    assert "does not contain sufficient information" in res.direct_answer
    assert res.plan is None


# ==============================================================================
# 2. ANALYTICAL FLOW TESTS
# ==============================================================================

def test_analytical_query_uses_planner_and_pandas_without_retrieval(client):
    """Verifies that pure analytical questions run through existing planner and Pandas without calling RAG retrieval."""
    client.post("/dataset/switch", data={"preset_name": "Superstore Retail (.xls)"}, follow_redirects=True)

    assistant = ConversationalAssistant(
        llm_client=get_llm_client("mock"),
        retriever=app_state.retriever,
        router=app_state.query_router,
    )
    res = assistant.analyze("What is the average Sales?", df=app_state.cleaned_df, dataset_name=app_state.dataset_name)

    assert res.intent == "analytical"
    assert res.retrieval_used is False
    assert len(res.retrieved_chunks) == 0
    assert res.plan is not None
    assert res.plan.operation == OperationType.AGGREGATION
    assert "229.86" in res.direct_answer or "228.23" in res.direct_answer  # Exact verified mean


def test_analytical_query_works_when_retriever_unavailable():
    """Verifies that pure analytical queries continue to work even when RAG retriever is None or unavailable."""
    df = pd.DataFrame({"Category": ["Tech", "Office"], "Sales": [100.0, 200.0]})
    assistant = ConversationalAssistant(
        llm_client=get_llm_client("mock"),
        retriever=None,  # No retriever provided
        router=QueryRouter(),
    )
    res = assistant.analyze("What is the average Sales?", df=df, dataset_name="Test")

    assert res.intent == "analytical"
    assert res.retrieval_used is False
    assert res.plan is not None
    assert "150.00" in res.direct_answer


# ==============================================================================
# 3. HYBRID FLOW TESTS & NUMERICAL SAFETY
# ==============================================================================

def test_hybrid_query_retrieval_and_planner_integration(client):
    """Verifies that HYBRID questions invoke RAG retrieval, supply context to planner, and execute via Pandas."""
    client.post("/dataset/switch", data={"preset_name": "Superstore Retail (.xls)"}, follow_redirects=True)

    # Attach glossary
    glossary = (
        "# Financial Glossary\n\n"
        "Profit represents net margin after regional distribution and taxes."
    )
    client.post("/knowledge/upload", data={"file": (io.BytesIO(glossary.encode("utf-8")), "financial.md")}, content_type="multipart/form-data", follow_redirects=True)

    assistant = ConversationalAssistant(
        llm_client=get_llm_client("mock"),
        retriever=app_state.retriever,
        router=app_state.query_router,
    )
    res = assistant.analyze(
        "Explain what profit means and calculate the total profit by region.",
        df=app_state.cleaned_df,
        dataset_name=app_state.dataset_name,
    )

    assert res.intent == "hybrid"
    assert res.retrieval_used is True
    assert len(res.retrieved_chunks) > 0
    assert res.plan is not None
    assert res.plan.operation == OperationType.GROUPBY_AGGREGATE
    assert "Region" in res.plan.group_by_columns or "region" in [c.lower() for c in res.plan.group_by_columns]
    assert "Context Grounding" in res.explanation


def test_hybrid_numerical_safety_misleading_knowledge_never_overrides_dataframe(client):
    """CRITICAL ACCEPTANCE TEST:
    A user-supplied knowledge document makes an intentionally misleading numerical claim
    ('Average Sales is $999,999.00'). The assistant's final computed answer MUST reflect
    the true Pandas DataFrame calculation (~$229.86), NOT the false claim in the text.
    """
    client.post("/dataset/switch", data={"preset_name": "Superstore Retail (.xls)"}, follow_redirects=True)

    misleading_doc = (
        "# Metric Guidance\n\n"
        "Average Sales is $999,999.00 according to legacy annual report.\n"
        "Profit ratio is fixed at 85%.\n"
    )
    client.post("/knowledge/upload", data={"file": (io.BytesIO(misleading_doc.encode("utf-8")), "misleading.txt")}, content_type="multipart/form-data", follow_redirects=True)

    assistant = ConversationalAssistant(
        llm_client=get_llm_client("mock"),
        retriever=app_state.retriever,
        router=app_state.query_router,
    )
    res = assistant.analyze(
        "Based on the definition, what is the average sales?",
        df=app_state.cleaned_df,
        dataset_name=app_state.dataset_name,
    )

    # 1. Must NOT report $999,999
    assert "999,999" not in res.direct_answer
    assert "999999" not in res.direct_answer

    # 2. Must report the actual DataFrame calculation from Pandas
    actual_mean = app_state.cleaned_df["Sales"].mean()
    assert f"{actual_mean:.2f}" in res.direct_answer or f"{round(actual_mean, 2):,}" in res.direct_answer


# ==============================================================================
# 4. CROSS-DATASET ISOLATION & SECOND DATASET VALIDATION
# ==============================================================================

def test_hybrid_and_knowledge_on_second_dataset_telco(client):
    """Verifies Phase 5 works identically on a second dataset (Telco Churn)."""
    # Switch to Telco
    client.post("/dataset/switch", data={"preset_name": "Telco Customer Churn (.csv)"}, follow_redirects=True)
    assert app_state.dataset_name == "Telco Customer Churn (.csv)"

    # Attach Telco policy
    telco_policy = (
        "# Telco Churn Definition\n\n"
        "Churn indicates whether a subscriber discontinued service within the last billing cycle."
    )
    client.post("/knowledge/upload", data={"file": (io.BytesIO(telco_policy.encode("utf-8")), "telco_def.md")}, content_type="multipart/form-data", follow_redirects=True)

    assistant = ConversationalAssistant(
        llm_client=get_llm_client("mock"),
        retriever=app_state.retriever,
        router=app_state.query_router,
    )

    # 1. Knowledge query
    res_k = assistant.analyze("What is the definition of churn?", df=app_state.cleaned_df, dataset_name=app_state.dataset_name)
    assert res_k.intent == "knowledge"
    assert "discontinued service" in res_k.direct_answer or "discontinued service" in res_k.explanation

    # 2. Analytical query on Telco
    res_a = assistant.analyze("What is the average MonthlyCharges?", df=app_state.cleaned_df, dataset_name=app_state.dataset_name)
    assert res_a.intent == "analytical"
    assert res_a.plan.operation == OperationType.AGGREGATION
    assert "64.76" in res_a.direct_answer  # Ground truth Telco mean MonthlyCharges


# ==============================================================================
# 5. UI ENDPOINT INTEGRATION TEST
# ==============================================================================

def test_flask_ask_run_end_to_end_knowledge_and_hybrid(client):
    """Verifies that the /ask/run web endpoint successfully renders transparency audit with intent and retrieval tags."""
    client.post("/dataset/switch", data={"preset_name": "Superstore Retail (.xls)"}, follow_redirects=True)

    # Ask knowledge query via web POST
    res_post = client.post("/ask/run", data={"question": "Describe the dataset schema and columns"}, follow_redirects=True)
    assert res_post.status_code == 200
    assert b"Detected Intent" in res_post.data
    assert b"KNOWLEDGE" in res_post.data or b"knowledge" in res_post.data
    assert b"RAG Retrieval" in res_post.data

    # Ask analytical query via web POST
    res_post2 = client.post("/ask/run", data={"question": "What is the average sales?"}, follow_redirects=True)
    assert res_post2.status_code == 200
    assert b"ANALYTICAL" in res_post2.data or b"analytical" in res_post2.data
    assert b"NOT USED" in res_post2.data
