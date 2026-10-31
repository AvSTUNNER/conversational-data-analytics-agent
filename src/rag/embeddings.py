"""Local, deterministic, zero-cost embedding abstraction for dataset knowledge retrieval."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseEmbeddingModel(ABC):
    """Abstract interface for local and remote embedding providers."""

    @abstractmethod
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fits vocabulary on knowledge document chunks and returns normalized vectors."""
        pass

    @abstractmethod
    def transform(self, queries: List[str]) -> np.ndarray:
        """Transforms query strings into normalized vectors using fitted vocabulary."""
        pass


class LocalTfidfEmbeddingModel(BaseEmbeddingModel):
    """Deterministic, local TF-IDF vector space model with sublinear term-frequency scaling and n-grams.

    - 100% local (0 API calls, 0 cost, 0 network dependencies)
    - Sub-millisecond execution
    - Deterministic across test suites and platforms
    """

    def __init__(self, ngram_range: tuple = (1, 2)) -> None:
        self.ngram_range = ngram_range
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.fitted: bool = False

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        if not documents:
            return np.zeros((0, 1))

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            norm="l2",
            lowercase=True,
            token_pattern=r"(?u)\b[\w-]+\b",  # Capture hyphenated and alphanumeric terms
        )
        matrix = self.vectorizer.fit_transform(documents)
        self.fitted = True
        return matrix.toarray()

    def transform(self, queries: List[str]) -> np.ndarray:
        if not self.fitted or self.vectorizer is None:
            raise RuntimeError("Embedding model must be fitted on documents before transforming queries.")
        if not queries:
            return np.zeros((0, len(self.vectorizer.get_feature_names_out())))

        matrix = self.vectorizer.transform(queries)
        return matrix.toarray()
