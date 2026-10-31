"""Retriever interface for querying dataset-specific knowledge indices."""
from __future__ import annotations

from typing import List, Optional, Tuple
from src.knowledge.artifact import KnowledgeArtifact
from src.rag.vector_store import DatasetVectorIndex, LocalVectorStore, RetrievalResult


class KnowledgeRetriever:
    """High-level retriever that manages vector store synchronization and queries with dataset provenance."""

    def __init__(self, vector_store: Optional[LocalVectorStore] = None) -> None:
        self.vector_store = vector_store or LocalVectorStore()

    def sync_dataset_knowledge(
        self,
        artifact: KnowledgeArtifact,
        force: bool = False,
    ) -> Tuple[DatasetVectorIndex, bool]:
        """Synchronizes active knowledge artifact into the isolated vector index."""
        return self.vector_store.sync_artifact(artifact, force=force)

    def retrieve(
        self,
        dataset_name: str,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Performs similarity retrieval on the dataset's vector index."""
        return self.vector_store.similarity_search(
            dataset_name=dataset_name,
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

    def get_index_status(self, dataset_name: str) -> Optional[dict]:
        """Returns metadata regarding active index for diagnostics and UI display."""
        index = self.vector_store.get_index(dataset_name)
        if index is None:
            return None
        return {
            "dataset_name": index.dataset_name,
            "chunk_count": index.chunk_count,
            "created_at": index.created_at,
            "fingerprint": index.fingerprint[:12],
        }
