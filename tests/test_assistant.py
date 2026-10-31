"""Unit tests for the end-to-end ConversationalAssistant pipeline."""
import io
import pandas as pd
import pytest

from src.assistant import ConversationalAssistant, AssistantResponse
from src.llm.client import MockLLMClient


class TestConversationalAssistant:
    """Test suite for complete end-to-end question answering and grounding."""

    @pytest.fixture(scope="class")
    def assistant(self):
        return ConversationalAssistant(llm_client=MockLLMClient())

    def test_e2e_count_query_on_superstore(self, assistant):
        """Verify: 'How many total orders are in the dataset?'"""
        response = assistant.analyze(
            question="How many total orders are in the dataset?",
            source="datasets/demo/sample_-_superstore.xls",
            sheet_name="Orders",
        )
        assert isinstance(response, AssistantResponse)
        assert "10,194" in response.direct_answer
        assert "10,194" in response.evidence_summary
        assert response.error_message is None
        assert response.is_unsupported is False
        assert response.plan is not None

    def test_e2e_average_sales_query(self, assistant):
        """Verify: 'What is the average sales amount?'"""
        response = assistant.analyze(
            question="What is the average sales amount?",
            source="datasets/demo/sample_-_superstore.xls",
            sheet_name="Orders",
        )
        assert "228.23" in response.direct_answer
        assert response.evidence_df is not None
        assert "228.23" in response.evidence_summary

    def test_e2e_top_category_by_profit_with_chart(self, assistant):
        """Verify: 'Which category generated the highest profit?' produces answer, evidence, and chart."""
        response = assistant.analyze(
            question="Which category generated the highest profit?",
            source="datasets/demo/sample_-_superstore.xls",
            sheet_name="Orders",
        )
        assert "Technology" in response.direct_answer
        assert "146,543.38" in response.direct_answer
        assert response.chart_figure is not None  # Bar chart generated
        assert response.evidence_df is not None

    def test_e2e_trend_analysis_with_line_chart(self, assistant):
        """Verify: 'Show monthly sales trends over time' produces line chart."""
        response = assistant.analyze(
            question="Show monthly sales trends over time",
            source="datasets/demo/sample_-_superstore.xls",
            sheet_name="Orders",
        )
        assert "Time-series trend" in response.direct_answer
        assert response.chart_figure is not None

    def test_e2e_correlation_analysis_with_scatter_chart(self, assistant):
        """Verify: 'What is the correlation between sales and profit?' produces scatter chart."""
        response = assistant.analyze(
            question="What is the correlation between sales and profit?",
            source="datasets/demo/sample_-_superstore.xls",
            sheet_name="Orders",
        )
        assert "0.4815" in response.direct_answer
        assert response.chart_figure is not None

    def test_e2e_outlier_detection_with_box_chart(self, assistant):
        """Verify: 'Are there outliers in the sales numbers?' produces box chart."""
        response = assistant.analyze(
            question="Are there outliers in the sales numbers?",
            source="datasets/demo/sample_-_superstore.xls",
            sheet_name="Orders",
        )
        assert "1,183" in response.direct_answer
        assert response.chart_figure is not None

    def test_e2e_unsupported_question_rejection(self, assistant):
        """Verify out-of-scope query is safely refused without inventing numbers."""
        response = assistant.analyze(
            question="What is the weather in Tokyo tomorrow?",
            source="datasets/demo/sample_-_superstore.xls",
            sheet_name="Orders",
        )
        assert response.is_unsupported is True
        assert "cannot be answered" in response.direct_answer.lower()
        assert response.chart_figure is None

    def test_e2e_in_memory_dataframe(self, assistant):
        """Verify passing DataFrame directly works cleanly."""
        df = pd.DataFrame({
            "Product": ["Laptop", "Mouse", "Keyboard"],
            "Revenue": [1200.0, 50.0, 100.0],
        })
        response = assistant.analyze(
            question="What is the average revenue?",
            df=df,
        )
        assert "450.00" in response.direct_answer

    def test_missing_dataset_source(self, assistant):
        """Verify clean error message when no dataset is provided."""
        response = assistant.analyze(question="What is the total sales?")
        assert response.error_message is not None
        assert "missing" in response.error_message.lower()
