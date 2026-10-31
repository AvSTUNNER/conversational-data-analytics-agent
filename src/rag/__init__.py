"""RAG indexing, retrieval, and query routing package."""
from src.rag.chunker import DocumentChunk, DocumentChunker
from src.rag.embeddings import BaseEmbeddingModel, LocalTfidfEmbeddingModel
from src.rag.vector_store import DatasetVectorIndex, LocalVectorStore, RetrievalResult
from src.rag.retriever import KnowledgeRetriever
from src.rag.router import QueryIntent, QueryRouter, RoutingDecision

__all__ = [
    "DocumentChunk",
    "DocumentChunker",
    "BaseEmbeddingModel",
    "LocalTfidfEmbeddingModel",
    "DatasetVectorIndex",
    "LocalVectorStore",
    "RetrievalResult",
    "KnowledgeRetriever",
    "QueryIntent",
    "QueryRouter",
    "RoutingDecision",
]
