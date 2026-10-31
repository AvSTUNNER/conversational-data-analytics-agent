# Project Specification: Conversational Data Analysis Assistant (HCL P_123)

## 1. Executive Summary & Objective
The **Conversational Data Analysis Assistant** is a deterministic, LLM-guided data exploration platform designed to allow non-technical business users and data analysts to query structured tabular datasets (CSV, XLS, XLSX) using plain natural language. 

The system guarantees **100% numerical correctness** by strictly separating **intent planning** (performed by an LLM) from **numerical computation** (performed deterministically by a verified Pandas/NumPy execution engine). 

> **Core Architectural Rule**:
> *The LLM decides what analysis to run; Pandas determines the numbers. Never let the LLM calculate an answer directly, and never execute arbitrary LLM-generated code.*

---

## 2. Target Users & Use Cases
- **Business Analysts & Managers**: Quickly query sales, profit, customer churn, and web traffic metrics without writing SQL or Pandas scripts.
- **Students & Evaluators**: Review a clear, transparent, audit-ready data analysis workflow that produces explicit evidence, deterministic figures, and matching Plotly charts.
- **Data Teams**: Triage ad-hoc data questions with structured plans and instant visualization.

---

## 3. Tech Stack & Cost Constraints
- **Core Runtime**: Python 3.10+ (tested on Python 3.12).
- **Data & Computation**: `pandas`, `numpy`, `openpyxl`, `xlrd`.
- **Validation & Schemas**: `pydantic` (v2) for rigid plan validation and type safety.
- **Visualization**: `plotly` (interactive HTML/SVG charts).
- **User Interface**: `streamlit` for responsive, interactive web application.
- **LLM Layer**: Provider-agnostic client abstraction:
  - **Development Mode**: Local Ollama (e.g., `qwen2.5:7b`, `llama3.2:3b`, `llama3.1:8b`).
  - **Deployment / Cloud Mode**: Free-tier hosted APIs (Google Gemini API / Groq / OpenRouter) configured purely via environment variables (`.env`).
  - **Zero Paid Dependencies**: Absolute ban on mandatory paid API keys or proprietary subscriptions.

---

## 4. Dataset Catalog & In-Situ Inspection (Phase 0 Findings)
The system is built to ingest arbitrary structured datasets dynamically without hardcoded column schemas. The canonical dev and validation datasets located in `datasets/` have been inspected on disk without modification:

| Directory | Exact Filename | Format / Magic | Size (Bytes) | Rows / Cols | Missing Values | Duplicate Rows | Key Characteristics |
|---|---|---|---|---|---|---|---|
| `datasets/demo/` | `sample_-_superstore.xls` | BIFF8 Excel (OLE2) | 3,422,208 | 10,194 × 21 | 0 | 0 | 3 sheets (`Orders`, `People`, `Returns`). Contains sales, profit, discount, regions, dates. |
| `datasets/validation/` | `online_retail.xlsx` | OOXML Excel (Zip) | 23,715,344 | 541,909 × 8 | 136,534 | 5,268 | Large transaction dataset. InvoiceDate timestamp, CustomerID with nulls, UnitPrice, Quantity. |
| `datasets/validation/` | `online_shoppers.csv` | ASCII CSV | 1,072,063 | 12,330 × 18 | 0 | 125 | Web session metrics: bounce rates, exit rates, page values, browser, revenue (bool). |
| `datasets/validation/` | `telco_churn.csv` | ASCII CSV | 970,457 | 7,043 × 21 | 0 | 0 | Telecom customer metrics. Note: `TotalCharges` is stored as string with blank whitespace for 0-tenure rows. |
| `datasets/synthetic/` | *(To be generated)* | CSV | - | - | - | - | Planned clean multi-table business dataset for benchmark test suites. |
| `datasets/edge_cases/` | *(To be generated)* | CSV / XLS | - | - | - | - | Planned edge cases: missing columns, malformed dates, extreme outliers, unicode characters. |

---

## 5. Supported Operations & Numerical Engine Scope
The deterministic analysis engine supports 8 standard operations:

1. **`count`**:
   - Total records, filtered record count, distinct count (`nunique`), null count.
2. **`aggregation`**:
   - `sum`, `mean`, `median`, `min`, `max`, `std`, `var` over a single target numerical column with optional conditional filters.
3. **`groupby_aggregate`**:
   - Grouping by one or more categorical/discrete columns, aggregating target column with (`sum`, `mean`, `median`, `min`, `max`, `count`), sorting (`descending` or `ascending`), and limiting top-N / bottom-N.
4. **`trend`**:
   - Time-series metric aggregation over datetime columns grouped by time granularity (`day`, `week`, `month`, `quarter`, `year`).
5. **`correlation`**:
   - Pairwise correlation (Pearson / Spearman) between two numerical columns or correlation matrix across numeric features.
6. **`outliers` / `distribution`**:
   - Interquartile Range (IQR) and Z-score outlier detection, descriptive summary quartiles (25%, 50%, 75%, min, max).
7. **`filter_preview` / `top_n`**:
   - Multi-column filtered record retrieval and ranking.
8. **`unsupported`**:
   - Graceful refusal when the question cannot be answered from the schema, is completely out-of-scope, or is ambiguous.

---

## 6. Strict Four-Part Response Format
Every response generated for user queries must contain the following components:
1. **Direct Answer**: Concise, unambiguous answer answering the prompt (e.g. *"The Technology category generated the highest total profit ($145,454.89)."*).
2. **Supporting Evidence**: Tabular or key-value summary of computed numbers directly from Pandas (e.g. *Technology: $145,454.89 | Office Supplies: $122,490.80 | Furniture: $18,451.27*).
3. **Interactive Visualization**: Appropriate Plotly chart (`bar`, `line`, `scatter`, `pie`, `box`/`histogram`), rendered only when visually informative.
4. **Grounded Explanation**: Short contextual summary explaining what the numbers indicate without inventing ungrounded metrics.

---

## 7. Quality Gates & Phased Milestones

| Phase | Title | Gate Criteria |
|---|---|---|
| **Phase 0** | Spec & Architecture | Full project specification frozen, architecture documented, datasets audited. |
| **Phase 1** | Data Layer & Profiling | Robust loader for CSV/XLS/XLSX, type inference, handling of malformed inputs, dynamic dataset profiler. |
| **Phase 2** | Deterministic Analysis Engine | 100% test coverage on all 8 analytical operations, zero execution errors, verified against hand-calculated ground truth. |
| **Phase 3** | LLM Planner | Benchmark of 50-100 questions across easy/medium/hard tiers reaching ≥85% plan accuracy. |
| **Phase 4** | End-to-End Assistant Pipeline | Integrated Planner → Validator → Executor → Chart Generator → Grounded Explainer. |
| **Phase 5** | UI Polish & Stability | Streamlit application with drag-and-drop file upload, dynamic schema inspector, error boundaries, and state reset. |
| **Phase 6** | Deployment & Documentation | Complete README, zero-cost deployment configuration (Streamlit Community Cloud / Render / local Ollama), clean virtualenv installation. |
