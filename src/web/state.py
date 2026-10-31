"""Session and application state management for the Flask web UI."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.data.loader import DataLoader, load_dataset, EmptyFileError, UnsupportedFormatError, DataLoadError
from src.data.cleaner import clean_dataset
from src.data.profiler import DatasetProfile, profile_dataset
from src.assistant import ConversationalAssistant, AssistantResponse
from src.llm.client import get_llm_client
from src.knowledge.manager import KnowledgeManager
from src.knowledge.artifact import KnowledgeArtifact
from src.rag.retriever import KnowledgeRetriever
from src.rag.router import QueryRouter, QueryIntent, RoutingDecision

PRESET_DATASETS: Dict[str, Dict[str, Any]] = {
    "Superstore Retail (.xls)": {
        "id": "superstore",
        "path": "datasets/demo/sample_-_superstore.xls",
        "sheet": "Orders",
        "domain": "Sales & Logistics",
        "description": "Global retail sales, profits, customer segments, and order items.",
        "examples": [
            "Top 5 products by total sales",
            "What is the average sales amount?",
            "Which category generated the highest profit?",
            "Which region generated the highest total profit in the Technology category?",
            "Show monthly sales trends over time",
            "What is the correlation between sales and profit?",
            "Are there outliers in the sales numbers?",
            "How many distinct orders were placed?",
        ],
    },
    "Telco Customer Churn (.csv)": {
        "id": "telco",
        "path": "datasets/validation/telco_churn.csv",
        "sheet": 0,
        "domain": "Telecommunications",
        "description": "Customer demographics, account contracts, monthly charges, and churn status.",
        "examples": [
            "How many total customers are in the telco dataset?",
            "What is the average monthly charge by contract type?",
            "What is the relationship between tenure and monthly charges?",
            "Are there outliers in monthly charges?",
            "Average tenure by internet service provider",
            "Average tenure by payment method for churned customers",
        ],
    },
    "Online Shoppers (.csv)": {
        "id": "shoppers",
        "path": "datasets/validation/online_shoppers.csv",
        "sheet": 0,
        "domain": "E-Commerce",
        "description": "Online shopper browsing sessions, page values, bounce and exit rates.",
        "examples": [
            "How many visitor sessions are recorded?",
            "What is the average bounce rate?",
            "What is the average page value by visitor type?",
            "What is the relationship between bounce rates and exit rates?",
            "Average product related duration by month",
        ],
    },
    "Online Retail (.xlsx)": {
        "id": "retail",
        "path": "datasets/validation/online_retail.xlsx",
        "sheet": 0,
        "domain": "Retail E-Commerce",
        "description": "Transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based online retail.",
        "examples": [
            "How many total transactions are in the retail dataset?",
            "What is the average unit price?",
            "Total quantity of items sold by country",
        ],
    },
    "Synthetic Business (.csv)": {
        "id": "synthetic",
        "path": "datasets/synthetic/business.csv",
        "sheet": 0,
        "domain": "Enterprise Finance",
        "description": "Clean synthetic sales transactions across departments and regions.",
        "examples": [
            "How many records are in the business dataset?",
            "Total sales by category",
            "Correlation between sales and profit in business data",
        ],
    },
    "Messy Business (.csv)": {
        "id": "messy",
        "path": "datasets/edge_cases/messy_business.csv",
        "sheet": 0,
        "domain": "Edge Case Data",
        "description": "Transactions with formatted currency strings, whitespace, and negative parentheses.",
        "examples": [
            "How many rows are in this table?",
            "What is the average sales amount?",
            "Total sales by department",
        ],
    },
}


class AppStateManager:
    """Manages active dataset, cleaned DataFrame, dataset profile, and query history."""

    def __init__(self) -> None:
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.profile: Optional[DatasetProfile] = None
        self.dataset_name: Optional[str] = None
        self.dataset_path: Optional[str] = None
        self.selected_sheet: Optional[Union[str, int]] = 0
        self.available_sheets: List[str] = []
        self.is_uploaded: bool = False
        self.uploaded_bytes: Optional[bytes] = None
        self.uploaded_filename: Optional[str] = None
        self.last_response: Optional[AssistantResponse] = None
        self.query_history: List[Dict[str, Any]] = []
        self.active_provider: str = os.getenv("LLM_PROVIDER", "ollama")
        self.knowledge_manager = KnowledgeManager()
        self.retriever = KnowledgeRetriever()
        self.query_router = QueryRouter()

        # Automatically load default dataset (Superstore)
        self.load_preset("Superstore Retail (.xls)")

    @property
    def current_knowledge(self) -> Optional[KnowledgeArtifact]:
        """Returns the active knowledge artifact for the current dataset."""
        return self.knowledge_manager.current_artifact

    def sync_active_index(self, force: bool = False):
        """Synchronizes the RAG vector index for the active dataset."""
        if self.current_knowledge:
            return self.retriever.sync_dataset_knowledge(self.current_knowledge, force=force)
        return None, False

    def load_preset(self, preset_name: str, sheet_name: Optional[Union[str, int]] = None) -> bool:
        """Loads a preset dataset by name."""
        if preset_name not in PRESET_DATASETS:
            return False

        preset_info = PRESET_DATASETS[preset_name]
        path = preset_info["path"]
        sheet = sheet_name if sheet_name is not None else preset_info["sheet"]

        try:
            self.available_sheets = DataLoader.get_sheet_names(path)
            raw = DataLoader.load(path, sheet_name=sheet)
            cleaned, _ = clean_dataset(raw)
            prof = profile_dataset(cleaned)

            self.raw_df = raw
            self.cleaned_df = cleaned
            self.profile = prof
            self.dataset_name = preset_name
            self.dataset_path = path
            self.selected_sheet = sheet
            self.is_uploaded = False
            self.uploaded_bytes = None
            self.uploaded_filename = None
            self.last_response = None

            # Synchronize isolated knowledge and RAG vector index for this dataset
            self.knowledge_manager.set_active_dataset(preset_name, prof, cleaned)
            self.sync_active_index()
            return True
        except Exception:
            return False

    def load_uploaded_data(
        self,
        file_bytes: bytes,
        filename: str,
        sheet_name: Optional[Union[str, int]] = 0,
    ) -> bool:
        """Loads an uploaded dataset from raw bytes, extracting sheets and profiling cleanly."""
        if not file_bytes:
            raise EmptyFileError("Uploaded file is empty (0 bytes).")

        ext = Path(filename).suffix.lower()
        if ext not in DataLoader.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file format '{ext}'. Allowed formats: {', '.join(DataLoader.SUPPORTED_EXTENSIONS)}"
            )

        # Cache uploaded bytes for workbook sheet switching
        self.uploaded_bytes = file_bytes
        self.uploaded_filename = filename

        # Create BytesIO stream
        stream = io.BytesIO(file_bytes)
        available_sheets = DataLoader.get_sheet_names(stream)
        stream.seek(0)

        raw = DataLoader.load(stream, sheet_name=sheet_name, filename_hint=filename)
        cleaned, _ = clean_dataset(raw)
        prof = profile_dataset(cleaned)

        self.raw_df = raw
        self.cleaned_df = cleaned
        self.profile = prof
        self.dataset_name = filename
        self.dataset_path = f"Uploaded: {filename}"
        self.selected_sheet = sheet_name
        self.available_sheets = available_sheets
        self.is_uploaded = True
        self.last_response = None

        # Synchronize isolated knowledge and RAG vector index for this uploaded dataset
        self.knowledge_manager.set_active_dataset(filename, prof, cleaned)
        self.sync_active_index()
        return True

    def switch_uploaded_sheet(self, sheet_name: Union[str, int]) -> bool:
        """Switches the active sheet for the currently loaded uploaded workbook."""
        if not self.uploaded_bytes or not self.uploaded_filename:
            return False
        return self.load_uploaded_data(self.uploaded_bytes, self.uploaded_filename, sheet_name=sheet_name)

    def reset_state(self) -> None:
        """Resets all state to initial defaults."""
        self.raw_df = None
        self.cleaned_df = None
        self.profile = None
        self.dataset_name = None
        self.dataset_path = None
        self.selected_sheet = 0
        self.available_sheets = []
        self.is_uploaded = False
        self.uploaded_bytes = None
        self.uploaded_filename = None
        self.last_response = None
        self.query_history = []
        self.knowledge_manager = KnowledgeManager()
        self.load_preset("Superstore Retail (.xls)")


# Global state instance for Flask application
app_state = AppStateManager()
