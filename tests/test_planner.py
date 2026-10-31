"""Unit tests for the LLM Analysis Planner and Prompt Engine."""
import json
import pytest

from src.data.loader import load_dataset
from src.data.cleaner import clean_dataset
from src.data.profiler import profile_dataset
from src.analysis.schemas import OperationType, AggregationType, ChartRecommendation
from src.llm.client import MockLLMClient, get_llm_client
from src.llm.planner import AnalysisPlanner, PlanningError
from src.llm.prompts import build_planner_prompt, build_explanation_prompt, PLANNER_SYSTEM_PROMPT


class TestAnalysisPlanner:
    """Test suite for LLM prompt engineering, planning, and JSON parsing."""

    @pytest.fixture(scope="class")
    def superstore_profile(self):
        df = load_dataset("datasets/demo/sample_-_superstore.xls", sheet_name="Orders")
        cleaned_df, _ = clean_dataset(df)
        return profile_dataset(cleaned_df)

    def test_planner_prompt_construction(self, superstore_profile):
        """Verify prompt formatting combines schema and user question."""
        prompt = build_planner_prompt(
            question="What is the total sales in the West region?",
            schema_prompt=superstore_profile.to_llm_schema_prompt(),
        )
        assert "DATASET SCHEMA & SUMMARY:" in prompt
        assert "USER QUESTION:" in prompt
        assert "What is the total sales in the West region?" in prompt
        assert "Sales: NUMERIC" in prompt

    def test_plan_generation_mock_client(self, superstore_profile):
        """Verify plan generation with the Mock client across core operations."""
        planner = AnalysisPlanner(client=MockLLMClient())

        # Count
        plan_count = planner.plan("How many total orders are there?", superstore_profile)
        assert plan_count.operation == OperationType.COUNT

        # Average Sales
        plan_avg = planner.plan("What is the average sales?", superstore_profile)
        assert plan_avg.operation == OperationType.AGGREGATION
        assert plan_avg.target_column == "Sales"
        assert plan_avg.aggregation == AggregationType.MEAN

        # Top Category by Profit
        plan_cat = planner.plan("Which category has the highest profit?", superstore_profile)
        assert plan_cat.operation == OperationType.GROUPBY_AGGREGATE
        assert plan_cat.target_column == "Profit"
        assert plan_cat.group_by_columns == ["Category"]

        # Unsupported
        plan_unsupp = planner.plan("What will the weather be tomorrow?", superstore_profile)
        assert plan_unsupp.operation == OperationType.UNSUPPORTED
        assert plan_unsupp.unsupported_reason is not None

    def test_extract_json_from_markdown_code_fences(self):
        """Verify JSON extraction handles markdown fences ```json ... ```."""
        raw_text = """Here is your requested plan:
```json
{
  "operation": "groupby_aggregate",
  "group_by_columns": ["Region"],
  "target_column": "Sales",
  "aggregation": "sum",
  "chart": "bar"
}
```
Hope this helps!"""
        extracted = AnalysisPlanner._extract_json(raw_text)
        assert extracted["operation"] == "groupby_aggregate"
        assert extracted["target_column"] == "Sales"

    def test_extract_json_invalid_raises_planning_error(self):
        """Verify unparseable text raises PlanningError."""
        with pytest.raises(PlanningError, match="Could not extract valid JSON"):
            AnalysisPlanner._extract_json("I cannot help you with that plain text response.")

    def test_explanation_prompt_builder(self):
        """Verify grounded explanation prompt generation."""
        prompt = build_explanation_prompt(
            question="Which category has highest profit?",
            evidence_summary="Technology: $145,454.89 · Office Supplies: $122,490.80",
            table_preview="Category | Profit\nTechnology | 145454.89",
            plan_dict={"operation": "groupby_aggregate", "target_column": "Profit", "group_by_columns": ["Category"]},
        )
        assert "COMPUTED EVIDENCE SUMMARY:" in prompt
        assert "STRICT GROUNDING RULES:" in prompt
        assert "Technology: $145,454.89" in prompt

    def test_ollama_client_success_and_connection_error(self, monkeypatch):
        """Verify OllamaClient post request and connection error handling."""
        from unittest.mock import MagicMock
        from src.llm.client import OllamaClient

        # Mock successful post
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '{"operation": "count"}'}
        mock_resp.raise_for_status.return_value = None
        monkeypatch.setattr("requests.post", lambda *args, **kwargs: mock_resp)

        client = OllamaClient(base_url="http://mock-ollama:11434")
        resp = client.generate("Test prompt", system_prompt="Test sys", response_format="json")
        assert resp == '{"operation": "count"}'

        # Mock connection error
        import requests
        def mock_error(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Connection refused")
        monkeypatch.setattr("requests.post", mock_error)

        with pytest.raises(ConnectionError, match="Failed to connect to local Ollama"):
            client.generate("Test prompt")

    def test_gemini_client_success_and_missing_key(self, monkeypatch):
        """Verify GeminiClient generates response with api key and errors without it."""
        from unittest.mock import MagicMock
        from src.llm.client import GeminiClient

        with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
            GeminiClient(api_key="").generate("Test prompt")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"operation": "count"}'}]}}]
        }
        mock_resp.raise_for_status.return_value = None
        monkeypatch.setattr("requests.post", lambda *args, **kwargs: mock_resp)

        client = GeminiClient(api_key="mock_key")
        resp = client.generate("Test prompt", system_prompt="Sys")
        assert resp == '{"operation": "count"}'

    def test_get_llm_client_factory(self, monkeypatch):
        """Verify get_llm_client routes appropriately."""
        from src.llm.client import get_llm_client, MockLLMClient, GeminiClient

        client_mock = get_llm_client(provider="mock")
        assert isinstance(client_mock, MockLLMClient)

        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        client_gemini = get_llm_client(provider="gemini")
        assert isinstance(client_gemini, GeminiClient)
