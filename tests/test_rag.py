"""Comprehensive test suite for Phase 4: RAG Indexing, Retrieval & Query Routing."""
import io
import os
import pytest

from src.knowledge.artifact import KnowledgeArtifact, KnowledgeSourceType
from src.knowledge.manager import KnowledgeManager
from src.rag.chunker import DocumentChunker, DocumentChunk
from src.rag.embeddings import LocalTfidfEmbeddingModel
from src.rag.vector_store import LocalVectorStore, DatasetVectorIndex, RetrievalResult
from src.rag.retriever import KnowledgeRetriever
from src.rag.router import QueryRouter, QueryIntent, RoutingDecision
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
# 1. Chunking & Provenance Tests
# ==============================================================================

def test_chunk_artifact_generated_facts(client):
    """Verifies that DocumentChunker splits generated knowledge into overview, column, and inferred chunks."""
    artifact = app_state.current_knowledge
    assert artifact is not None

    chunks = DocumentChunker.chunk_artifact(artifact)
    assert len(chunks) > 10  # Overview + 21 columns + inferred notes

    # Check overview chunk
    overview = next(c for c in chunks if c.metadata.get("type") == "overview")
    assert "Dataset: Superstore Retail (.xls)" in overview.content
    assert overview.dataset_name == "Superstore Retail (.xls)"
    assert overview.source_type == KnowledgeSourceType.GENERATED

    # Check column dictionary chunk
    sales_col = next(c for c in chunks if c.metadata.get("column_name") == "Sales")
    assert "Column: Sales" in sales_col.content
    assert "NUMERIC" in sales_col.content
    assert sales_col.dataset_name == "Superstore Retail (.xls)"


def test_chunk_artifact_supplied_document(client):
    """Verifies chunking user-supplied markdown documents with header and paragraph preservation."""
    supplied_text = (
        "# Glossary\n\n"
        "Customer Tier 1 means annual purchases above $10,000.\n\n"
        "## Order Status Codes\n\n"
        "Status Code A indicates active processing.\n"
        "Status Code C indicates cancelled order.\n"
    )
    data = {"file": (io.BytesIO(supplied_text.encode("utf-8")), "policy.md")}
    client.post("/knowledge/upload", data=data, content_type="multipart/form-data", follow_redirects=True)

    artifact = app_state.current_knowledge
    assert artifact.source_type == KnowledgeSourceType.SUPPLIED

    chunks = DocumentChunker.chunk_artifact(artifact)
    supplied_chunks = [c for c in chunks if c.metadata.get("type") == "user_supplied"]
    assert len(supplied_chunks) >= 2
    assert any("Customer Tier 1" in c.content for c in supplied_chunks)
    assert any("Status Code A" in c.content for c in supplied_chunks)
    for c in supplied_chunks:
        assert c.metadata.get("filename") == "policy.md"
        assert c.source_type == KnowledgeSourceType.SUPPLIED


# ==============================================================================
# 2. Vector Store, Embedding & Indexing Lifecycle Tests
# ==============================================================================

def test_embedding_model_fit_transform_and_transform():
    """Verifies local TF-IDF embedding fitting and transformation."""
    model = LocalTfidfEmbeddingModel(ngram_range=(1, 2))
    docs = [
        "Sales represents gross revenue before discounts.",
        "Profit is net income after operating expenses and overhead.",
    ]
    matrix = model.fit_transform(docs)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0

    # Query transform
    q_vec = model.transform(["What is net income profit?"])
    assert q_vec.shape == (1, matrix.shape[1])


def test_vector_store_build_and_similarity_search():
    """Verifies vector store indexing and cosine similarity retrieval."""
    store = LocalVectorStore()
    km = KnowledgeManager()

    # Create dummy artifact
    client_app = create_app({"TESTING": True})
    with client_app.app_context():
        # Using Superstore from state
        artifact = app_state.current_knowledge
        index, was_rebuilt = store.sync_artifact(artifact)
        assert was_rebuilt is True
        assert index.chunk_count > 0

        # Query for sales
        results = store.similarity_search(
            dataset_name=artifact.dataset_name,
            query="Tell me about Sales column data type and range",
            top_k=2,
        )
        assert len(results) > 0
        top_result = results[0]
        assert top_result.score > 0.0
        assert "Sales" in top_result.section_title or "Sales" in top_result.chunk.content


def test_index_caching_and_invalidation(client):
    """Verifies that unchanged knowledge reuses index while modified knowledge rebuilds it."""
    store = LocalVectorStore()
    artifact = app_state.current_knowledge

    # 1. Initial build
    index1, was_rebuilt1 = store.sync_artifact(artifact)
    assert was_rebuilt1 is True

    # 2. Re-sync identical artifact -> MUST REUSE (no rebuild)
    index2, was_rebuilt2 = store.sync_artifact(artifact)
    assert was_rebuilt2 is False
    assert index1.fingerprint == index2.fingerprint

    # 3. Force rebuild
    index3, was_rebuilt3 = store.sync_artifact(artifact, force=True)
    assert was_rebuilt3 is True


def test_knowledge_reset_invalidation(client):
    """Verifies that uploading knowledge and then resetting triggers index rebuilds appropriately."""
    store = LocalVectorStore()

    # Auto-generated index
    index_gen, _ = store.sync_artifact(app_state.current_knowledge)

    # Upload user knowledge
    km = KnowledgeManager()
    km.set_active_dataset(app_state.dataset_name, app_state.profile, app_state.cleaned_df)
    supplied_art = km.upload_supplied_knowledge(
        dataset_name=app_state.dataset_name,
        content="Custom VIP definition",
        filename="vip.txt",
        profile=app_state.profile,
        df=app_state.cleaned_df,
    )
    index_sup, rebuilt_sup = store.sync_artifact(supplied_art)
    assert rebuilt_sup is True
    assert index_sup.fingerprint != index_gen.fingerprint

    # Reset to generated
    reset_art = km.reset_to_generated(app_state.dataset_name, app_state.profile, app_state.cleaned_df)
    index_reset, rebuilt_reset = store.sync_artifact(reset_art)
    assert rebuilt_reset is True
    assert index_reset.fingerprint == index_gen.fingerprint


# ==============================================================================
# 3. Dataset Isolation & Cross-Dataset Leakage Prevention Tests
# ==============================================================================

def test_dataset_isolation_and_cross_dataset_prevention(client):
    """Verifies that querying dataset A never returns chunks from dataset B."""
    # 1. Load Superstore with custom secret policy
    client.post("/dataset/switch", data={"preset_name": "Superstore Retail (.xls)"}, follow_redirects=True)
    superstore_doc = "Confidential Superstore KPI: Target profit ratio is 30%."
    client.post("/knowledge/upload", data={"file": (io.BytesIO(superstore_doc.encode("utf-8")), "superstore_kpi.txt")}, content_type="multipart/form-data", follow_redirects=True)

    # Verify retrieval on Superstore
    results_superstore = app_state.retriever.retrieve("Superstore Retail (.xls)", "Target profit ratio")
    assert len(results_superstore) > 0
    assert any("Confidential Superstore KPI" in r.chunk.content for r in results_superstore)

    # 2. Switch to Telco Churn
    client.post("/dataset/switch", data={"preset_name": "Telco Customer Churn (.csv)"}, follow_redirects=True)
    assert app_state.dataset_name == "Telco Customer Churn (.csv)"

    # Query Telco with Superstore query -> MUST NOT return Superstore chunks
    results_telco = app_state.retriever.retrieve("Telco Customer Churn (.csv)", "Target profit ratio")
    for r in results_telco:
        assert r.dataset_name == "Telco Customer Churn (.csv)"
        assert "Confidential Superstore KPI" not in r.chunk.content

    # Querying a non-existent dataset returns empty list
    assert app_state.retriever.retrieve("NonExistentDataset", "any query") == []


# ==============================================================================
# 4. Query Router Classification Tests
# ==============================================================================

def test_query_router_knowledge_queries():
    """Verifies that definitional and metadata questions are classified as KNOWLEDGE intent."""
    knowledge_queries = [
        "What does Customer Segment mean?",
        "What is the definition of churn?",
        "Explain the business meaning of this column",
        "What does the Discount column represent?",
        "Describe the dataset schema and columns",
        "Who is the author of this dataset?",
        "Show me the glossary definitions",
    ]
    for q in knowledge_queries:
        decision = QueryRouter.route(q)
        assert decision.intent == QueryIntent.KNOWLEDGE, f"Failed on: {q}"
        assert decision.needs_retrieval is True, f"Failed on: {q}"


def test_query_router_analytical_queries():
    """Verifies that mathematical and data-filtering questions are classified as ANALYTICAL intent."""
    analytical_queries = [
        "What is the average sales?",
        "Which region has the highest profit?",
        "Show monthly sales trends over time",
        "Top 5 products by sales",
        "How many rows are in the dataset?",
        "What is the correlation between sales and profit?",
        "Find outliers in the budget column",
        "Total revenue by department",
    ]
    for q in analytical_queries:
        decision = QueryRouter.route(q)
        assert decision.intent == QueryIntent.ANALYTICAL, f"Failed on: {q}"
        assert decision.needs_retrieval is False, f"Failed on: {q}"


def test_query_router_hybrid_queries():
    """Verifies that questions needing both semantic grounding and calculation are classified as HYBRID intent."""
    hybrid_queries = [
        "What is the total sales for VIP customers according to the tier definition?",
        "Calculate revenue for churned users based on the business definition of churn",
        "Explain what profit margin is and calculate the average profit",
        "What is the average tenure for active accounts according to the status definition?",
    ]
    for q in hybrid_queries:
        decision = QueryRouter.route(q)
        assert decision.intent == QueryIntent.HYBRID, f"Failed on: {q}"
        assert decision.needs_retrieval is True, f"Failed on: {q}"


# ==============================================================================
# 5. UI & Live Retrieval Debug Route Tests
# ==============================================================================

def test_knowledge_page_displays_rag_index_status(client):
    """Verifies that the Knowledge page displays active RAG index status and chunk counts."""
    res = client.get("/knowledge")
    assert res.status_code == 200
    assert b"RAG Vector Index" in res.data
    assert b"Local TF-IDF N-Gram Space" in res.data
    assert b"Active (" in res.data


def test_knowledge_search_endpoint_returns_results_and_routing(client):
    """Verifies the /knowledge/search debug endpoint produces retrieved chunks and routing decisions."""
    res = client.post("/knowledge/search", data={"query": "What is the definition of Sales?"}, follow_redirects=True)
    assert res.status_code == 200
    assert b"Routing Decision:" in res.data
    assert b"KNOWLEDGE" in res.data or b"knowledge" in res.data
    assert b"Retrieved Chunks for Active Dataset" in res.data
    assert b"Cosine Score:" in res.data
    assert b"Sales" in res.data


def test_empty_knowledge_search_fails_safely(client):
    """Verifies that searching with an empty query produces safe output without error."""
    res = client.post("/knowledge/search", data={"query": ""}, follow_redirects=True)
    assert res.status_code == 200
