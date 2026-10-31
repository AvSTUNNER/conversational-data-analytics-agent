"""Lightweight, deterministic query router for classifying knowledge vs analytical vs hybrid intent."""
from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class QueryIntent(str, Enum):
    """High-level query classification determining pipeline execution path."""
    KNOWLEDGE = "knowledge"    # Context, definitions, business glossary, metadata (Retrieval needed)
    ANALYTICAL = "analytical"  # Pure mathematical / aggregation / grouping / filtering / trend (Pandas needed)
    HYBRID = "hybrid"          # Domain context + numerical computation (Retrieval + Pandas needed)


class RoutingDecision(BaseModel):
    """Structured decision produced by the Query Router."""
    intent: QueryIntent
    needs_retrieval: bool
    reason: str
    confidence: float = 1.0


class QueryRouter:
    """Classifies natural language questions into knowledge, analytical, or hybrid categories."""

    # Explicit phrases asking for semantic meaning, definition, glossary or metadata
    KNOWLEDGE_PATTERNS = [
        r"\bwhat (is|does|are)\b.*\b(mean|meaning|definition|represent|signify|stand for)\b",
        r"\b(define|definition of|meaning of|glossary|data dictionary|business rule|business rules)\b",
        r"\bexplain\b.*\b(meaning|definition|rule|rules|glossary|dictionary|metadata|context)\b",
        r"\bwhat is meant by\b",
        r"\bexplain what [a-z\s_]+ is\b",
        r"\bexplain (the|what) [a-z\s_]+ (means|signifies|represents)\b",
        r"\bwhat does the [a-z\s_]+ column\b",
        r"\bdescribe the (dataset|table|schema|columns?)\b",
        r"\bwhat kind of data\b",
        r"\bwhere did this data come from\b",
        r"\bwho (is|was)\b.*\b(author|source|creator|collector|officer|owner)\b",
    ]

    # Explicit calculation or analytical verbs
    CALC_PATTERNS = [
        r"\b(calculate|compute|find the total|aggregate|plot|chart|graph|visualize|breakdown|correlate)\b",
        r"\b(top|bottom|highest|lowest|maximum|max|minimum|min|average|avg|median|sum|total|how many|count|percentage|std|variance|trend|monthly|yearly|daily|correlation|outlier|outliers|anomaly)\b",
    ]

    # Hybrid trigger phrases that explicitly connect business rules to calculations
    HYBRID_CONNECTORS = [
        r"\baccording to (the|our)\b",
        r"\bbased on (the|our) (definition|rule|glossary|tier|policy)\b",
        r"\bexplain .* and (calculate|find|compute|show)\b",
        r"\bdefinition of .* and (calculate|find|compute|show|what is the total|what is the average)\b",
    ]

    @classmethod
    def route(cls, query: str, column_names: Optional[List[str]] = None) -> RoutingDecision:
        """Determines the appropriate routing path for a query."""
        q = query.strip()
        q_lower = q.lower()

        if not q:
            return RoutingDecision(
                intent=QueryIntent.ANALYTICAL,
                needs_retrieval=False,
                reason="Empty query defaulted to analytical pipeline.",
                confidence=1.0,
            )

        # 1. Check for Hybrid Intent
        is_hybrid_connector = any(re.search(pat, q_lower) for pat in cls.HYBRID_CONNECTORS)
        has_knowledge_pattern = any(re.search(pat, q_lower) for pat in cls.KNOWLEDGE_PATTERNS)
        has_calc_pattern = any(re.search(pat, q_lower) for pat in cls.CALC_PATTERNS)

        if is_hybrid_connector:
            return RoutingDecision(
                intent=QueryIntent.HYBRID,
                needs_retrieval=True,
                reason="Query requests both semantic business definition grounding and numerical data calculation.",
                confidence=0.95,
            )

        if has_knowledge_pattern and has_calc_pattern:
            # Check if it's purely asking what a metric name means, e.g. "What does Customer Segment mean?"
            # "mean" here is the verb to mean, not mathematical mean.
            if re.search(r"\bwhat (is|does|are)\b.*\b(mean|meaning|definition|represent|signify)\b", q_lower) and not any(
                re.search(r"\b(calculate|compute|sum of|average of|total of|highest|lowest|top \d+|bottom \d+|trend|plot|chart)\b", q_lower)
                for _ in [1]
            ):
                return RoutingDecision(
                    intent=QueryIntent.KNOWLEDGE,
                    needs_retrieval=True,
                    reason="Query is asking for semantic definitions, column meanings, or metadata context.",
                    confidence=0.95,
                )
            return RoutingDecision(
                intent=QueryIntent.HYBRID,
                needs_retrieval=True,
                reason="Query requests both semantic business definition grounding and numerical data calculation.",
                confidence=0.95,
            )

        # 2. Check for Pure Knowledge Intent
        if has_knowledge_pattern or "glossary" in q_lower or "data dictionary" in q_lower:
            return RoutingDecision(
                intent=QueryIntent.KNOWLEDGE,
                needs_retrieval=True,
                reason="Query is asking for semantic definitions, column meanings, or metadata context.",
                confidence=0.95,
            )

        # 3. Default: Pure Analytical Intent
        return RoutingDecision(
            intent=QueryIntent.ANALYTICAL,
            needs_retrieval=False,
            reason="Query requires numerical computation, filtering, or aggregations handled by the deterministic Pandas engine.",
            confidence=0.98,
        )
