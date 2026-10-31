"""Analysis Planner translating user questions into validated AnalysisPlans via LLM."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from src.analysis.schemas import AnalysisPlan, OperationType, ChartRecommendation
from src.data.profiler import DatasetProfile
from src.llm.client import LLMClient, get_llm_client
from src.llm.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt


class PlanningError(Exception):
    """Raised when plan generation or JSON parsing fails."""
    pass


class AnalysisPlanner:
    """Coordinates LLM prompt construction, invocation, and structured plan parsing."""

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.client = client or get_llm_client()

    def plan(
        self,
        question: str,
        profile: DatasetProfile,
        reference_context: Optional[str] = None,
    ) -> AnalysisPlan:
        """Generates a structured AnalysisPlan from a user question, dataset profile, and optional reference context.

        Args:
            question: Plain English question from the user.
            profile: Profile metadata of the dataset being queried.
            reference_context: Optional retrieved RAG context for hybrid questions.

        Returns:
            AnalysisPlan: Validated intermediate representation.
        """
        schema_text = profile.to_llm_schema_prompt()
        prompt = build_planner_prompt(
            question=question,
            schema_prompt=schema_text,
            reference_context=reference_context,
        )

        raw_response = self.client.generate(
            prompt=prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.0,
            response_format="json",
        )

        plan_dict = self._extract_json(raw_response)
        try:
            return AnalysisPlan.model_validate(plan_dict)
        except Exception as e:
            raise PlanningError(f"Failed to validate generated JSON into AnalysisPlan: {str(e)}\nRaw Response: {raw_response}") from e

    @classmethod
    def _extract_json(cls, raw_text: str) -> Dict[str, Any]:
        """Extracts JSON dictionary from raw model text, handling markdown fences and pre/post text."""
        cleaned = raw_text.strip()

        # Handle ```json ... ```
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()

        # Direct JSON load
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Regex search for first outer brace pair
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise PlanningError(f"Could not extract valid JSON from LLM response:\n{raw_text}")
