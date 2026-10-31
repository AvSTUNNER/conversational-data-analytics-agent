"""Local, deterministic, zero-cost embedding abstraction for dataset knowledge retrieval."""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np


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

    - 100% pure Python + numpy (Zero scipy/scikit-learn requirement, keeps Lambda bundle < 150MB)
    - Sub-millisecond execution
    - Deterministic across test suites, OS platforms, and serverless runtimes
    """

    def __init__(self, ngram_range: tuple = (1, 2)) -> None:
        self.ngram_range = ngram_range
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.fitted: bool = False
        self.doc_count: int = 0

    def _extract_ngrams(self, text: str) -> List[str]:
        """Extracts 1-gram and 2-gram word tokens matching r'(?u)\b[\w-]+\b'."""
        tokens = re.findall(r"[\w-]+", text.lower())
        results: List[str] = []
        n_min, n_max = self.ngram_range
        for n in range(n_min, n_max + 1):
            for i in range(len(tokens) - n + 1):
                results.append(" ".join(tokens[i : i + n]))
        return results

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        if not documents:
            return np.zeros((0, 1))

        self.doc_count = len(documents)
        doc_ngrams = [self._extract_ngrams(doc) for doc in documents]

        # Calculate document frequencies
        df: Counter[str] = Counter()
        for ngrams in doc_ngrams:
            df.update(set(ngrams))

        sorted_features = sorted(df.keys())
        self.vocab = {term: idx for idx, term in enumerate(sorted_features)}

        # Smooth IDF: ln((1 + N) / (1 + df)) + 1
        self.idf = {
            term: math.log((1.0 + self.doc_count) / (1.0 + df[term])) + 1.0
            for term in sorted_features
        }

        # Build vectors with sublinear TF: 1 + ln(tf)
        n_features = len(sorted_features)
        vectors = np.zeros((self.doc_count, n_features), dtype=np.float64)

        for doc_idx, ngrams in enumerate(doc_ngrams):
            tf_counts = Counter(ngrams)
            for term, count in tf_counts.items():
                if term in self.vocab:
                    col_idx = self.vocab[term]
                    sublinear_tf = 1.0 + math.log(count) if count > 0 else 0.0
                    vectors[doc_idx, col_idx] = sublinear_tf * self.idf[term]

            # L2 Normalize
            norm = np.linalg.norm(vectors[doc_idx])
            if norm > 0:
                vectors[doc_idx] /= norm

        self.fitted = True
        return vectors

    def transform(self, queries: List[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Embedding model must be fitted on documents before transforming queries.")
        if not queries:
            return np.zeros((0, len(self.vocab)), dtype=np.float64)

        n_features = len(self.vocab)
        vectors = np.zeros((len(queries), n_features), dtype=np.float64)

        for q_idx, q_text in enumerate(queries):
            ngrams = self._extract_ngrams(q_text)
            tf_counts = Counter(ngrams)
            for term, count in tf_counts.items():
                if term in self.vocab:
                    col_idx = self.vocab[term]
                    sublinear_tf = 1.0 + math.log(count) if count > 0 else 0.0
                    vectors[q_idx, col_idx] = sublinear_tf * self.idf[term]

            # L2 Normalize
            norm = np.linalg.norm(vectors[q_idx])
            if norm > 0:
                vectors[q_idx] /= norm

        return vectors
