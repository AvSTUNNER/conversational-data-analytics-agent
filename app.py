"""Conversational Data Analysis Assistant - Streamlit Application Entrypoint (HCL P_123)."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.data.loader import DataLoader, load_dataset, DataLoadError, EmptyFileError, UnsupportedFormatError
from src.data.cleaner import DataCleaner, clean_dataset
from src.data.profiler import DatasetProfiler, DatasetProfile, profile_dataset
from src.assistant import ConversationalAssistant, AssistantResponse
from src.llm.client import get_llm_client, OllamaClient, GeminiClient, MockLLMClient

# Streamlit Page Configuration
st.set_page_config(
    page_title="Conversational Data Analysis Assistant (HCL P_123)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .direct-answer-card {
        background-color: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
    }
    .direct-answer-title {
        font-weight: 700;
        color: #1e40af;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }
    .direct-answer-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
    }
    .stat-badge {
        display: inline-block;
        background-color: #f3f4f6;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #374151;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PRESET_DATASETS = {
    "Demo: Superstore Retail (.xls)": {
        "path": "datasets/demo/sample_-_superstore.xls",
        "sheet": "Orders",
        "examples": [
            "Which category generated the highest profit?",
            "What is the average sales amount?",
            "How many total orders are in the dataset?",
            "Top 5 products by total sales",
            "Show monthly sales trends over time",
            "What is the correlation between sales and profit?",
            "Are there outliers in the sales numbers?",
            "Which region generated the highest profit in Technology?",
        ],
    },
    "Validation: Telco Customer Churn (.csv)": {
        "path": "datasets/validation/telco_churn.csv",
        "sheet": 0,
        "examples": [
            "How many total customers are in the telco dataset?",
            "What is the average monthly charge?",
            "What is the average monthly charge by contract type?",
            "What is the relationship between tenure and monthly charges?",
            "Are there outliers in monthly charges?",
            "Average tenure by internet service provider",
        ],
    },
    "Validation: Online Shoppers Sessions (.csv)": {
        "path": "datasets/validation/online_shoppers.csv",
        "sheet": 0,
        "examples": [
            "How many visitor sessions are recorded?",
            "What is the average bounce rate?",
            "What is the average page value by visitor type?",
            "What is the relationship between bounce rates and exit rates?",
            "Average product related duration by month",
        ],
    },
    "Synthetic: Business Transactions (.csv)": {
        "path": "datasets/synthetic/business.csv",
        "sheet": 0,
        "examples": [
            "How many records are in the business dataset?",
            "Total sales by category",
            "Correlation between sales and profit in business data",
        ],
    },
}


def initialize_session_state() -> None:
    """Initializes Streamlit session state variables."""
    if "current_df" not in st.session_state:
        st.session_state.current_df = None
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "dataset_profile" not in st.session_state:
        st.session_state.dataset_profile = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "selected_example" not in st.session_state:
        st.session_state.selected_example = ""


def load_selected_dataset(
    source: Any,
    dataset_label: str,
    sheet_name: Any = 0,
    filename_hint: Optional[str] = None,
) -> bool:
    """Loads, cleans, and profiles a dataset into session state."""
    try:
        raw_df = load_dataset(source, sheet_name=sheet_name, filename_hint=filename_hint)
        cleaned_df, _ = clean_dataset(raw_df)
        profile = profile_dataset(cleaned_df)

        st.session_state.current_df = raw_df
        st.session_state.cleaned_df = cleaned_df
        st.session_state.dataset_profile = profile
        st.session_state.dataset_name = dataset_label
        st.session_state.last_response = None
        return True
    except EmptyFileError:
        st.error("Uploaded file is empty (contains 0 rows). Please upload a valid dataset.")
    except UnsupportedFormatError:
        st.error("Unsupported file format. Please upload a CSV (.csv) or Excel (.xls, .xlsx) file.")
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
    return False


def main() -> None:
    initialize_session_state()

    # App Header
    st.markdown('<div class="main-header">📊 Conversational Data Analysis Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header"><strong>HCL Project P_123</strong> — 100% Mathematically Grounded Analysis (LLM Analysis Planner + Deterministic Pandas Engine)</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Data Ingestion & Configuration
    with st.sidebar:
        st.header("📁 Dataset Source")
        source_mode = st.radio("Choose Input Mode:", ["Preset Local Datasets", "Upload CSV / Excel File"], index=0)

        if source_mode == "Preset Local Datasets":
            chosen_preset = st.selectbox("Select Benchmark Dataset:", list(PRESET_DATASETS.keys()))
            preset_info = PRESET_DATASETS[chosen_preset]
            preset_path = preset_info["path"]

            # Sheet selection for Excel
            sheet_choice = preset_info["sheet"]
            available_sheets = DataLoader.get_sheet_names(preset_path)
            if len(available_sheets) > 1:
                sheet_choice = st.selectbox("Select Sheet:", available_sheets, index=0)

            if st.button("Load Selected Dataset", type="primary", use_container_width=True) or (
                st.session_state.dataset_name != chosen_preset and st.session_state.current_df is None
            ):
                with st.spinner("Loading and profiling dataset..."):
                    load_selected_dataset(preset_path, chosen_preset, sheet_name=sheet_choice)

        else:
            uploaded_file = st.file_uploader("Upload CSV, XLS, or XLSX", type=["csv", "xls", "xlsx"])
            if uploaded_file is not None:
                file_sheets = DataLoader.get_sheet_names(uploaded_file)
                chosen_sheet = 0
                if len(file_sheets) > 1:
                    chosen_sheet = st.selectbox("Select Workbook Sheet:", file_sheets)

                if st.button("Ingest Uploaded File", type="primary", use_container_width=True):
                    with st.spinner("Processing uploaded dataset..."):
                        load_selected_dataset(
                            uploaded_file,
                            uploaded_file.name,
                            sheet_name=chosen_sheet,
                            filename_hint=uploaded_file.name,
                        )

        st.divider()
        st.header("⚙️ Engine & LLM Settings")
        provider_choice = st.selectbox(
            "Active LLM Provider:",
            ["Automatic (Ollama -> Gemini -> Mock)", "Local Ollama", "Google Gemini Free API", "Mock Parser (Offline)"],
            index=0,
        )

        provider_map = {
            "Automatic (Ollama -> Gemini -> Mock)": os.getenv("LLM_PROVIDER", "ollama"),
            "Local Ollama": "ollama",
            "Google Gemini Free API": "gemini",
            "Groq Free API": "groq",
            "Mock Parser (Offline)": "mock",
        }
        selected_provider = provider_map[provider_choice]

        if st.button("Reset State & Clear Cache", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # Check if dataset is loaded
    if st.session_state.cleaned_df is None or st.session_state.dataset_profile is None:
        st.info("👈 Please select or upload a dataset in the sidebar to begin analysis.")
        return

    df: pd.DataFrame = st.session_state.cleaned_df
    profile: DatasetProfile = st.session_state.dataset_profile

    # Dataset Profile & Schema Inspector
    with st.expander(f"📋 Dataset Overview: {st.session_state.dataset_name} ({profile.row_count:,} rows × {profile.column_count} columns)", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{profile.row_count:,}")
        col2.metric("Total Columns", profile.column_count)
        col3.metric("Duplicate Rows", f"{profile.duplicate_rows:,}")
        col4.metric("Total Missing Values", f"{profile.total_null_values:,}")

        tab_preview, tab_schema = st.tabs(["Raw Data Preview", "Inferred Schema & Types"])

        with tab_preview:
            st.dataframe(df.head(10), use_container_width=True)

        with tab_schema:
            schema_data = []
            for col_name, col_prof in profile.columns.items():
                schema_data.append({
                    "Column Name": col_name,
                    "Inferred Type": col_prof.data_type.upper(),
                    "Raw Dtype": col_prof.raw_dtype,
                    "Nulls": f"{col_prof.null_count} ({col_prof.null_percentage:.1f}%)",
                    "Unique Values": col_prof.unique_count,
                    "Sample Values": ", ".join(str(s) for s in col_prof.sample_values[:4]),
                })
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True)

    # Interactive Query Interface
    st.markdown("### 💬 Ask a Question")

    # Example question buttons
    current_preset = st.session_state.dataset_name
    if current_preset in PRESET_DATASETS:
        examples = PRESET_DATASETS[current_preset]["examples"]
        st.caption("💡 Quick Suggestions:")
        cols = st.columns(min(4, len(examples)))
        for idx, ex in enumerate(examples[:4]):
            col_target = cols[idx % len(cols)]
            if col_target.button(ex, key=f"ex_{idx}", use_container_width=True):
                st.session_state.selected_example = ex
                st.rerun()

    # Input Box
    default_text = st.session_state.selected_example or ""
    user_query = st.text_input(
        "Enter your query in plain English:",
        value=default_text,
        placeholder="e.g., Which category has the highest total profit? / What is the average sales?",
        key="user_query_input",
    )

    col_btn, _ = st.columns([1, 5])
    run_query = col_btn.button("Run Analysis 🚀", type="primary", use_container_width=True)

    if (run_query or (default_text and default_text == user_query)) and user_query:
        # Clear selected example flag
        st.session_state.selected_example = ""

        with st.spinner("Analyzing dataset deterministically..."):
            client = get_llm_client(provider=selected_provider)
            assistant = ConversationalAssistant(llm_client=client)
            response: AssistantResponse = assistant.analyze(
                question=user_query,
                df=df,
            )
            st.session_state.last_response = response

    # Render 4-Part Response
    if st.session_state.last_response is not None:
        resp: AssistantResponse = st.session_state.last_response
        st.divider()

        if resp.error_message:
            st.error(f"⚠️ **Error Encountered**: {resp.error_message}")
            if resp.explanation:
                st.info(resp.explanation)
            return

        if resp.is_unsupported:
            st.warning(f"🚫 **Unsupported Query**: {resp.direct_answer}")
            st.info(f"**Reason**: {resp.explanation}")
            return

        # --- PART 1: DIRECT ANSWER ---
        st.markdown(
            f"""
            <div class="direct-answer-card">
                <div class="direct-answer-title">📌 Direct Answer</div>
                <div class="direct-answer-text">{resp.direct_answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- PART 2 & PART 3: SUPPORTING EVIDENCE & VISUALIZATION ---
        evidence_col, chart_col = st.columns([1, 1] if resp.chart_figure is not None else [1, 0.01])

        with evidence_col:
            st.markdown("#### 🔢 Supporting Evidence")
            st.caption(f"**Summary**: {resp.evidence_summary}")
            if resp.evidence_df is not None and not resp.evidence_df.empty:
                st.dataframe(resp.evidence_df, use_container_width=True)

        if resp.chart_figure is not None:
            with chart_col:
                st.markdown("#### 📈 Interactive Visualization")
                st.plotly_chart(resp.chart_figure, use_container_width=True)

        # --- PART 4: GROUNDED EXPLANATION & AUDIT INSPECTOR ---
        st.markdown("#### 💡 Context & Explanation")
        st.write(resp.explanation)

        # Audit & Transparency Accordion
        with st.expander("🔍 Plan Transparency & Execution Audit (Click to inspect)", expanded=False):
            if resp.plan:
                st.json(resp.plan.model_dump(exclude_none=True))
            st.caption(f"⏱️ Engine Execution Time: **{resp.execution_time_ms:.2f} ms**")


if __name__ == "__main__":
    main()
