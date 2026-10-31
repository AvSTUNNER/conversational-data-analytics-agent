"""Lightweight local vector store with dataset-specific indexing, cosine similarity, and source-aware caching."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from src.knowledge.artifact import KnowledgeArtifact
from src.rag.chunker import DocumentChunk, DocumentChunker
from src.rag.embeddings import BaseEmbeddingModel, LocalTfidfEmbeddingModel


class RetrievalResult(BaseModel):
    """Result of a similarity search containing the matched chunk, score, and provenance."""
    chunk: DocumentChunk
    score: float
    dataset_name: str
    source_type: str
    section_title: str

    @property
    def provenance_label(self) -> str:
        """Human-readable provenance tag."""
        source_label = "Supplied Document" if self.source_type == "supplied" else "Observed Facts"
        return f"[{self.dataset_name}] {source_label} > {self.section_title} (score: {self.score:.3f})"


class DatasetVectorIndex:
    """Isolated vector index for a specific dataset."""

    def __init__(
        self,
        dataset_name: str,
        fingerprint: str,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
        model: BaseEmbeddingModel,
    ) -> None:
        self.dataset_name = dataset_name
        self.fingerprint = fingerprint
        self.chunks = chunks
        self.embeddings = embeddings
        self.model = model
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.chunk_count = len(chunks)


class LocalVectorStore:
    """Local, in-memory vector store that strictly isolates indexes by dataset."""

    def __init__(self, embedding_model_factory=LocalTfidfEmbeddingModel) -> None:
        self._indices: Dict[str, DatasetVectorIndex] = {}
        self.embedding_model_factory = embedding_model_factory

    @staticmethod
    def compute_artifact_fingerprint(artifact: KnowledgeArtifact) -> str:
        """Computes a SHA-256 fingerprint of the knowledge artifact to detect mutations."""
        hasher = hashlib.sha256()
        hasher.update(artifact.dataset_name.encode("utf-8"))
        hasher.update(artifact.source_type.value.encode("utf-8"))
        if artifact.supplied_content:
            hasher.update(artifact.supplied_content.encode("utf-8"))
        if artifact.supplied_filename:
            hasher.update(artifact.supplied_filename.encode("utf-8"))
        hasher.update(str(artifact.observed.row_count).encode("utf-8"))
        hasher.update(str(artifact.observed.column_count).encode("utf-8"))
        hasher.update(str(artifact.observed.total_null_values).encode("utf-8"))
        # Add column names and types
        for col_name, col_meta in sorted(artifact.observed.columns.items()):
            hasher.update(col_name.encode("utf-8"))
            hasher.update(col_meta.data_type.encode("utf-8"))
            hasher.update(str(col_meta.null_count).encode("utf-8"))
        return hasher.hexdigest()

    def get_index(self, dataset_name: str) -> Optional[DatasetVectorIndex]:
        """Retrieves active index for dataset without modifying it."""
        return self._indices.get(dataset_name)

    def sync_artifact(
        self,
        artifact: KnowledgeArtifact,
        force: bool = False,
    ) -> Tuple[DatasetVectorIndex, bool]:
        """Synchronizes index with the artifact, reusing the index if unchanged, or rebuilding if modified.

        Returns:
            Tuple of (DatasetVectorIndex, was_rebuilt: bool)
        """
        fingerprint = self.compute_artifact_fingerprint(artifact)
        existing_index = self._indices.get(artifact.dataset_name)

        if not force and existing_index is not None and existing_index.fingerprint == fingerprint:
            # Reusing existing index (no rebuild)
            return existing_index, False

        # Rebuilding index
        chunks = DocumentChunker.chunk_artifact(artifact)
        if not chunks:
            # Fallback for empty knowledge
            chunks = [
                DocumentChunk(
                    dataset_name=artifact.dataset_name,
                    source_type=artifact.source_type,
                    section_title="Empty Knowledge",
                    content=f"Dataset: {artifact.dataset_name} (No metadata available)",
                )
            ]

        doc_texts = [chunk.full_representation for chunk in chunks]
        model = self.embedding_model_factory()
        embeddings = model.fit_transform(doc_texts)

        new_index = DatasetVectorIndex(
            dataset_name=artifact.dataset_name,
            fingerprint=fingerprint,
            chunks=chunks,
            embeddings=embeddings,
            model=model,
        )
        self._indices[artifact.dataset_name] = new_index
        return new_index, True

    def similarity_search(
        self,
        dataset_name: str,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Performs cosine similarity search against the active dataset's isolated index."""
        index = self.get_index(dataset_name)
        if index is None or not index.chunks or not query.strip():
            return []

        # Transform query into normalized vector
        query_vec = index.model.transform([query.strip()])  # shape: (1, dim)
        if query_vec.shape[0] == 0 or query_vec.shape[1] == 0:
            return []

        # Cosine similarity between query_vec and all doc embeddings (both are L2-normalized)
        # score = dot(A, B.T)
        scores = np.dot(index.embeddings, query_vec.T).flatten()  # shape: (num_docs,)

        # Rank indices descending
        ranked_indices = np.argsort(scores)[::-1]

        results: List[RetrievalResult] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            chunk = index.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    dataset_name=chunk.dataset_name,
                    source_type=chunk.source_type.value,
                    section_title=chunk.section_title,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def clear(self) -> None:
        """Clears all stored indices."""
        self._indices.clear()
