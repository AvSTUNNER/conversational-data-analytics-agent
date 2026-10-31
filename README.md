# Conversational Data Analysis Assistant (HCL Project P_123)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-70%20Passed%20(83%25%20Coverage)-brightgreen.svg)]()
[![Benchmark](https://img.shields.io/badge/Benchmark-50%2F50%20(100%25)-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade conversational data analysis system designed to translate natural language questions into structured, mathematically grounded analytical insights with supporting data tables and interactive charts.

---

## 🌟 Core Architecture & Guarantees

> **The Fundamental Rule**: The LLM decides *what* analysis to run; Pandas deterministically calculates *all numbers*.

```
+------------------+      +-------------------+      +---------------------+
| User Query & CSV | ---> | Dataset Profiler  | ---> | LLM Analysis Planner|
+------------------+      +-------------------+      +---------------------+
                                                                |
                                                                v Structured AnalysisPlan (JSON)
                                                     +---------------------+
                                                     | Strict Plan Validator|
                                                     +---------------------+
                                                                |
                                                                v Validated Plan
+------------------+      +-------------------+      +---------------------+
| 4-Part Response  | <--- | Chart Generator   | <--- | Deterministic Pandas|
| (Answer+Table+UI)|      | (Plotly Engine)   |      | Execution Engine    |
+------------------+      +-------------------+      +---------------------+
```

1. **Zero LLM Code Execution**: No `eval()`, `exec()`, or runtime script execution.
2. **100% Deterministic Mathematical Grounding**: All calculations (sums, averages, correlations, outliers, groupings) are executed by verified Pandas/NumPy routines.
3. **Structured Intermediate Representation (IR)**: The LLM outputs a strict Pydantic JSON schema (`AnalysisPlan`) validated against dataset columns and data types before execution.
4. **Resilient Data Ingestion**: Robust handling for `.csv`, `.xls`, `.xlsx`, accounting format parentheses `($150.00)`, currency symbols, percentages, whitespace, and dirty strings.
5. **Standardized 4-Part Output**:
   - 📌 **Direct Answer**: Factual, high-contrast primary finding.
   - 🔢 **Supporting Evidence**: Full tabular data breakdown and summary statistics.
   - 📈 **Interactive Visualization**: Full-featured Plotly charts (Bar, Line, Scatter + Trendline, Box, Pie).
   - 💡 **Grounded Explanation**: Contextual explanation detailing exactly how the metric was derived.
   - 🔍 **Audit & Transparency Inspector**: Live collapsible view of the underlying JSON plan and execution latency.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- (Optional) [Ollama](https://ollama.com/) for local offline LLMs (e.g. `llama3`, `mistral`, `qwen2.5`) OR a free [Google Gemini API Key](https://aistudio.google.com/).

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone <repository-url>
cd "Conversational Data Anakytics Agent"

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the example environment file:
```bash
cp .env.example .env
```
Edit `.env` to configure your preferred LLM provider:
```ini
# Choose: "ollama", "gemini", or "mock"
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:latest

# If using Google Gemini:
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Supported Datasets & Formats

The application comes pre-configured with four benchmark datasets and supports arbitrary user file uploads:

| Dataset | Format | Domain | Rows × Cols | Sample Query |
| :--- | :--- | :--- | :--- | :--- |
| **Superstore Retail** | `.xls` | Sales & Logistics | 10,194 × 21 | *"Which category generated the highest profit?"* |
| **Telco Customer Churn** | `.csv` | Telecommunications | 7,043 × 21 | *"What is the average monthly charge by contract type?"* |
| **Online Shoppers** | `.csv` | E-commerce Web | 12,330 × 18 | *"What is the correlation between bounce rates and exit rates?"* |
| **Synthetic Business** | `.csv` | Enterprise Finance | 1,000 × 8 | *"Show monthly revenue trends over time"* |

---

## 🧪 Testing & Benchmark Evaluation

### Running Unit & Integration Tests
The project features a comprehensive test suite (70 tests) verifying loaders, cleaners, schema profilers, validators, analysis execution, chart generation, LLM planners, and Streamlit UI components:
```bash
pytest
```
To run tests with coverage reporting:
```bash
pytest --cov=src --cov-report=term-missing
```

### Running the 50-Question Benchmark
Evaluate accuracy across 50 real-world analytical questions (Easy, Medium, Hard):
```bash
python evaluation/evaluate.py
```
Output:
```
================================================================================
                      BENCHMARK EVALUATION SUMMARY
================================================================================
Total Questions Evaluated: 50
Total Passed:             50 (100.0%)
Total Failed:             0 (0.0%)

Results by Difficulty Tier:
  - Easy:       20/20 (100.0%)
  - Medium:     18/18 (100.0%)
  - Hard:       12/12 (100.0%)
================================================================================
```

---

## 🏗️ Project Structure

```
├── app.py                      # Main Streamlit web application entrypoint
├── requirements.txt            # Python package dependencies
├── pytest.ini                  # Pytest configuration & test paths
├── .env.example                # Example environment variables template
├── ARCHITECTURE.md             # In-depth architectural blueprint & contracts
├── datasets/                   # Benchmark datasets
│   ├── demo/                   # Superstore Retail (.xls)
│   ├── validation/             # Telco Churn & Online Shoppers (.csv)
│   └── synthetic/              # Synthetic Business transactions (.csv)
├── src/                        # Core application source code
│   ├── data/                   # Ingestion, cleaning & profiling
│   │   ├── loader.py           # Multi-format DataLoader (CSV, XLS, XLSX)
│   │   ├── cleaner.py          # Non-destructive data cleaner
│   │   └── profiler.py         # Dynamic schema extractor & LLM prompt builder
│   ├── analysis/               # Deterministic execution engine
│   │   ├── schemas.py          # Rigid Pydantic IR AnalysisPlan schema
│   │   ├── validator.py        # Pre-execution plan validation & fuzzy matcher
│   │   ├── statistics.py       # Outliers (IQR/Z-score) & Correlation routines
│   │   └── executor.py         # Deterministic Pandas engine (8 operations)
│   ├── llm/                    # LLM Planner & Provider Clients
│   │   ├── client.py           # Ollama, Gemini & Mock provider abstractions
│   │   ├── prompts.py          # Rigid JSON system instructions & few-shots
│   │   └── planner.py          # Plan extraction, parsing & error handling
│   ├── visualization/          # Plotly visualization engine
│   │   └── charts.py           # Multi-chart generator (Bar, Line, Scatter, Box, Pie)
│   └── assistant.py            # End-to-end orchestrator & 4-part response builder
├── tests/                      # Automated test suite (70 tests)
│   ├── test_loader.py          # DataLoader tests
│   ├── test_cleaner.py         # DataCleaner tests
│   ├── test_profiler.py        # Profiler tests
│   ├── test_validator.py       # PlanValidator tests
│   ├── test_analysis.py        # AnalysisExecutor engine tests
│   ├── test_planner.py         # LLM Planner tests
│   ├── test_visualization.py   # ChartGenerator tests
│   ├── test_assistant.py       # ConversationalAssistant pipeline tests
│   └── test_app.py             # Streamlit AppTest end-to-end UI tests
└── evaluation/                 # Benchmark evaluation harness
    ├── benchmark_questions.json# 50-question benchmark suite
    └── evaluate.py             # Automated accuracy evaluator
```

---

## ⚙️ Operations Reference

The Deterministic Engine supports 8 core operations:
1. `count`: Dataset, filtered, or unique cardinality calculations.
2. `aggregation`: Single/multi-metric column aggregations (`sum`, `mean`, `median`, `min`, `max`, `std`, `var`).
3. `groupby_aggregate`: Group-by operations with top-N ranking and multi-column sorting.
4. `trend`: Temporal resamplings (`day`, `week`, `month`, `quarter`, `year`) with auto-sorting.
5. `correlation`: Bivariate Pearson / Spearman statistical correlation with strength descriptions.
6. `outliers`: Outlier detection via Tukey's IQR ($1.5 \times IQR$) or Standard Deviation ($3\sigma$ Z-score).
7. `filter_preview`: Filtered segment inspection with column selections.
8. `unsupported`: Graceful rejection and suggestions for subjective/predictive questions.

---

## 🚢 Deployment

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Link your repository at [share.streamlit.io](https://share.streamlit.io/).
3. Set the Main file path to `app.py`.
4. Add any required environment secrets (e.g. `GEMINI_API_KEY`) under Advanced Settings -> Secrets.

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 📄 License
This project is licensed under the MIT License.
