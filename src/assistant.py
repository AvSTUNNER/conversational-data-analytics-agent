"""End-to-End Conversational Data Analysis Assistant pipeline."""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict, Field

from src.data.loader import DataLoader, load_dataset
from src.data.cleaner import DataCleaner, clean_dataset
from src.data.profiler import DatasetProfiler, DatasetProfile, profile_dataset
from src.analysis.schemas import (
    AnalysisPlan,
    ExecutionResult,
    OperationType,
    AggregationType,
    ChartRecommendation,
)
from src.analysis.validator import PlanValidator, ValidationError
from src.analysis.executor import AnalysisExecutor, execute_plan
from src.visualization.charts import ChartGenerator, create_chart
from src.llm.client import LLMClient, get_llm_client
from src.llm.planner import AnalysisPlanner, PlanningError
from src.llm.prompts import build_explanation_prompt
from src.rag.router import QueryRouter, QueryIntent, RoutingDecision
from src.rag.retriever import KnowledgeRetriever
from src.rag.vector_store import RetrievalResult


class AssistantResponse(BaseModel):
    """Structured 4-part response returned to the UI and user."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    direct_answer: str
    evidence_summary: str
    evidence_df: Optional[pd.DataFrame] = None
    chart_figure: Optional[go.Figure] = None
    explanation: str
    plan: Optional[AnalysisPlan] = None
    is_unsupported: bool = False
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    intent: Optional[str] = None
    retrieval_used: bool = False
    retrieved_chunks: List[Any] = Field(default_factory=list)


class ConversationalAssistant:
    """Orchestrates ingestion, dynamic profiling, LLM planning, deterministic execution, and visualization."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[KnowledgeRetriever] = None,
        router: Optional[QueryRouter] = None,
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.planner = AnalysisPlanner(client=self.llm_client)
        self.retriever = retriever
        self.router = router or QueryRouter()

    def analyze(
        self,
        question: str,
        df: Optional[pd.DataFrame] = None,
        source: Optional[Union[str, Path, io.BytesIO, io.StringIO]] = None,
        sheet_name: Optional[Union[str, int]] = 0,
        filename_hint: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> AssistantResponse:
        """Executes the complete end-to-end question answering pipeline.

        Args:
            question: Plain English query from the user.
            df: Optional already-loaded DataFrame.
            source: Path or buffer to ingest if df is not provided.
            sheet_name: Excel sheet index or name if loading a workbook.
            filename_hint: Filename hint for stream buffers.
            dataset_name: Optional active dataset name for RAG index association.

        Returns:
            AssistantResponse: Structured 4-part answer with evidence and chart.
        """
        start_time = time.perf_counter()

        # 1. Ingestion Phase
        active_df: pd.DataFrame
        if df is not None:
            active_df = df
        elif source is not None:
            try:
                active_df = load_dataset(source, sheet_name=sheet_name, filename_hint=filename_hint)
            except Exception as e:
                return AssistantResponse(
                    question=question,
                    direct_answer="Failed to load dataset.",
                    evidence_summary="",
                    explanation="",
                    error_message=f"Data Ingestion Error: {str(e)}",
                    execution_time_ms=self._elapsed_ms(start_time),
                )
        else:
            return AssistantResponse(
                question=question,
                direct_answer="No dataset provided.",
                evidence_summary="",
                explanation="Please upload a CSV or Excel dataset to begin analysis.",
                error_message="Dataset source is missing.",
                execution_time_ms=self._elapsed_ms(start_time),
            )

        # 2. Cleaning & Dynamic Profiling Phase
        cleaned_df, _ = clean_dataset(active_df)
        profile = profile_dataset(cleaned_df)

        # 3. Query Routing Phase
        decision = self.router.route(question)
        retrieved_results: List[RetrievalResult] = []
        reference_context: Optional[str] = None

        # A. KNOWLEDGE-ORIENTED FLOW
        if decision.intent == QueryIntent.KNOWLEDGE:
            if self.retriever and dataset_name:
                try:
                    retrieved_results = self.retriever.retrieve(
                        dataset_name=dataset_name,
                        query=question,
                        top_k=3,
                        min_score=0.15,
                    )
                except Exception:
                    retrieved_results = []

            # If retrieval produces no sufficiently relevant result
            if not retrieved_results:
                return AssistantResponse(
                    question=question,
                    direct_answer="The available knowledge does not contain sufficient information to answer this question.",
                    evidence_summary="No relevant domain definitions, data dictionary entries, or glossary records found in active dataset knowledge.",
                    explanation="Please upload a domain specification or glossary for this dataset on the Knowledge page to ground semantic questions.",
                    plan=None,
                    intent=QueryIntent.KNOWLEDGE.value,
                    retrieval_used=True,
                    retrieved_chunks=[],
                    execution_time_ms=self._elapsed_ms(start_time),
                )

            # Formulate grounded knowledge response directly from retrieved chunks
            top_chunk = retrieved_results[0].chunk
            provenance_tag = retrieved_results[0].provenance_label
            direct_answer = self._format_knowledge_direct_answer(question, retrieved_results)
            explanation = (
                f"Source: {provenance_tag}\n\n"
                f"{top_chunk.content}"
            )

            return AssistantResponse(
                question=question,
                direct_answer=direct_answer,
                evidence_summary=f"Grounded in {len(retrieved_results)} retrieved knowledge chunk(s).",
                explanation=explanation,
                plan=None,
                intent=QueryIntent.KNOWLEDGE.value,
                retrieval_used=True,
                retrieved_chunks=[r.model_dump() for r in retrieved_results],
                execution_time_ms=self._elapsed_ms(start_time),
            )

        # B & C. RETRIEVAL FOR HYBRID (NOT FOR PURE ANALYTICAL)
        if decision.intent == QueryIntent.HYBRID and self.retriever and dataset_name:
            try:
                retrieved_results = self.retriever.retrieve(dataset_name=dataset_name, query=question, top_k=3)
                if retrieved_results:
                    context_lines = [f"[{r.section_title}]: {r.chunk.content}" for r in retrieved_results]
                    reference_context = "\n\n".join(context_lines)
            except Exception:
                retrieved_results = []
                reference_context = None

        # 4. LLM Planning Phase
        plan: AnalysisPlan
        try:
            plan = self.planner.plan(
                question=question,
                profile=profile,
                reference_context=reference_context,
            )
        except Exception as pe:
            return AssistantResponse(
                question=question,
                direct_answer="Unable to generate a valid analysis plan for this question.",
                evidence_summary="",
                explanation="The assistant encountered an error while interpreting the question schema.",
                error_message=f"Planner Error: {str(pe)}",
                intent=decision.intent.value,
                retrieval_used=bool(reference_context),
                retrieved_chunks=[r.model_dump() for r in retrieved_results],
                execution_time_ms=self._elapsed_ms(start_time),
            )

        # 5. Deterministic Execution Phase (PANDAS REMAINS THE NUMERICAL AUTHORITY)
        result: ExecutionResult
        try:
            result = execute_plan(plan, cleaned_df)
        except Exception as ee:
            return AssistantResponse(
                question=question,
                direct_answer="Execution failed.",
                evidence_summary="",
                explanation=f"A mathematical or validation error occurred during execution: {str(ee)}",
                plan=plan,
                error_message=f"Execution Error: {str(ee)}",
                intent=decision.intent.value,
                retrieval_used=bool(reference_context),
                retrieved_chunks=[r.model_dump() for r in retrieved_results],
                execution_time_ms=self._elapsed_ms(start_time),
            )

        # 6. Handle Unsupported Queries
        if result.is_unsupported:
            reason = result.unsupported_reason or "This question cannot be answered from the provided dataset."
            return AssistantResponse(
                question=question,
                direct_answer="Query cannot be answered from dataset.",
                evidence_summary="No matching columns or data available.",
                explanation=reason,
                plan=plan,
                is_unsupported=True,
                intent=decision.intent.value,
                retrieval_used=bool(reference_context),
                retrieved_chunks=[r.model_dump() for r in retrieved_results],
                execution_time_ms=self._elapsed_ms(start_time),
            )

        # 7. Visualization Phase
        fig = ChartGenerator.create_chart(plan, result, raw_df=cleaned_df)

        # 8. Formulate Direct Answer & Grounded Explanation
        direct_answer = self._generate_direct_answer(plan, result)
        explanation = self._generate_grounded_explanation(
            question=question,
            plan=plan,
            result=result,
            retrieved_results=retrieved_results if decision.intent == QueryIntent.HYBRID else None,
        )

        return AssistantResponse(
            question=question,
            direct_answer=direct_answer,
            evidence_summary=result.evidence_summary,
            evidence_df=result.evidence_df,
            chart_figure=fig,
            explanation=explanation,
            plan=plan,
            intent=decision.intent.value,
            retrieval_used=bool(reference_context),
            retrieved_chunks=[r.model_dump() for r in retrieved_results],
            execution_time_ms=self._elapsed_ms(start_time),
        )

    @classmethod
    def _generate_direct_answer(cls, plan: AnalysisPlan, result: ExecutionResult) -> str:
        """Constructs a concise, mathematically grounded direct answer."""
        op = plan.operation
        val = result.main_value

        if op == OperationType.COUNT:
            if plan.aggregation == AggregationType.NUNIQUE and plan.target_column:
                return f"There are {val:,} distinct {plan.target_column}s."
            return f"There are {val:,} total records matching your query."

        elif op == OperationType.AGGREGATION:
            agg_name = (plan.aggregation or AggregationType.SUM).value.title()
            tgt = plan.target_column
            f_val = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"
            return f"The {agg_name} {tgt} is {f_val}."

        elif op == OperationType.GROUPBY_AGGREGATE:
            if isinstance(val, dict) and "top_group" in val:
                grp = val["top_group"]
                top_v = val["top_value"]
                metric = val.get("metric_name", "Value")
                f_val = f"{top_v:,.2f}" if isinstance(top_v, float) else f"{top_v:,}"
                if plan.limit == 1 or plan.sort_ascending is False:
                    return f"{grp} had the highest {metric} ({f_val})."
                return f"Top result: {grp} with {metric} of {f_val}."
            return "Groupby aggregation completed successfully."

        elif op == OperationType.TREND:
            if isinstance(val, dict):
                p_cnt = val.get("period_count", len(result.evidence_df) if result.evidence_df is not None else 0)
                metric = val.get("metric_name", "Metric")
                return f"Time-series trend of {metric} computed across {p_cnt} periods."
            return "Time-series trend computed successfully."

        elif op == OperationType.CORRELATION:
            return f"The Pearson correlation is {val} ({result.evidence_df['Strength'].iloc[0] if result.evidence_df is not None else ''})."

        elif op == OperationType.OUTLIERS:
            if isinstance(val, dict):
                return f"Detected {val.get('outlier_count', 0):,} outliers ({val.get('outlier_percentage', 0)}%) in {plan.target_column}."
            return f"Outlier analysis completed for {plan.target_column}."

        elif op == OperationType.FILTER_PREVIEW:
            return f"Retrieved {val} matching records."

        return "Analysis completed."

    @classmethod
    def _format_knowledge_direct_answer(
        cls,
        question: str,
        retrieved_results: List[RetrievalResult],
    ) -> str:
        """Formulates a concise direct answer strictly from retrieved knowledge chunks."""
        if not retrieved_results:
            return "No matching definition found in dataset knowledge."

        top_chunk = retrieved_results[0].chunk
        # Extract first key sentence or definition line
        lines = [line.strip() for line in top_chunk.content.splitlines() if line.strip()]
        for line in lines:
            if not line.startswith("#") and len(line) > 10:
                # Clean markdown list markers if present
                clean_line = line.lstrip("-* ").strip()
                return clean_line

        return f"Refer to {top_chunk.section_title} for domain definition."

    def _generate_grounded_explanation(
        self,
        question: str,
        plan: AnalysisPlan,
        result: ExecutionResult,
        retrieved_results: Optional[List[RetrievalResult]] = None,
    ) -> str:
        """Generates grounded explanation using computed evidence and optional hybrid RAG context."""
        base_exp = ""
        if result.evidence_summary:
            base_exp = f"Based on the analysis, {result.evidence_summary}."
            if plan.filters:
                filter_descs = [f"{f.column} {f.operator.value} '{f.value}'" for f in plan.filters]
                base_exp += f" Filtered by {', '.join(filter_descs)}."
        else:
            base_exp = "The computed metrics reflect the exact values calculated from the dataset records."

        # If hybrid context was retrieved, append semantic grounding notice with provenance
        if retrieved_results:
            top_res = retrieved_results[0]
            base_exp += (
                f"\n\nContext Grounding ({top_res.provenance_label}): "
                f"Calculation was aligned with '{top_res.section_title}' business context."
            )

        return base_exp

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000.0, 2)
