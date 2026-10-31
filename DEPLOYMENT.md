# Deployment & Operations Guide — Conversational Data Analysis Assistant (HCL P_123)

This guide documents operational requirements, configuration, and production deployment procedures for the **Conversational Data Analysis Assistant**.

---

## 1. System Prerequisites

- **Python Runtime:** Python 3.10, 3.11, or 3.12 (tested on 3.12.8)
- **Package Manager:** `pip` or `uv`
- **Operating Systems:** Linux (Ubuntu/Debian/RHEL), macOS, or Windows
- **System Memory:** Minimum 2 GB RAM (4+ GB recommended when processing large datasets)
- **Disk Space:** ~500 MB for core dependencies and benchmark datasets

### Install Dependencies

```bash
pip install -r requirements.txt
```

*(Optional production WSGI servers: `pip install gunicorn` on Linux/macOS, or `pip install waitress` on Windows).*

---

## 2. Environment Variables & Configuration

Copy `.env.example` to `.env` in the project root:

```bash
cp .env.example .env
```

| Environment Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `production` | Deployment environment mode (`production` or `development`) |
| `FLASK_DEBUG` | `false` | Enables debug auto-reloading and debug tools. **Must be false in production**. |
| `SECRET_KEY` | `dev-secret-key...` | Cryptographic secret for signing session cookies and CSRF protection. |
| `PORT` | `5000` | Port for the web server to bind to. |
| `HOST` | `0.0.0.0` | Network interface to bind to (`0.0.0.0` listens on all interfaces). |
| `LLM_PROVIDER` | `ollama` | Active LLM client: `ollama`, `gemini`, `groq`, or `mock`. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint for local Ollama daemon. |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model identifier. |
| `GEMINI_API_KEY` | *(None)* | Google AI Studio Gemini API key. |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model identifier. |
| `GROQ_API_KEY` | *(None)* | Groq Cloud API key. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile`| Groq model identifier. |

> [!CAUTION]
> Never commit real API keys, secret credentials, or passwords into source control. Always inject them via environment variables or `.env` (which is excluded in `.gitignore`).

---

## 3. Server Startup Options

### Option A: Direct Production-Hardened Python Launch

Using `app_flask.py` (binds to `$HOST` and `$PORT` with `debug=False` by default):

```bash
python app_flask.py
```

### Option B: Production WSGI Server with Gunicorn (Linux / macOS)

For multi-worker concurrency and production resilience:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

Or using custom timeout for long analytical workloads:

```bash
gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:5000 wsgi:app
```

### Option C: Production WSGI Server with Waitress (Windows / Cross-Platform)

```bash
waitress-serve --host=0.0.0.0 --port=5000 wsgi:app
```

### Option D: Streamlit UI Alternative (Optional)

If preferred for lightweight local dashboarding:

```bash
streamlit run app.py --server.port 8501
```

---

## 4. File Upload Constraints & Security Hardening

- **Max Payload Size:** `16 MB` (`MAX_CONTENT_LENGTH = 16 * 1024 * 1024`). Files exceeding this limit are rejected automatically with HTTP 413.
- **Allowed Dataset Formats:** `.csv`, `.xls`, `.xlsx`.
- **Allowed Knowledge Formats:** `.txt`, `.md`, `.json`, `.csv`.
- **Filename Sanitization:** All incoming filenames are sanitized through `werkzeug.utils.secure_filename` to neutralize path traversal attacks (`../`, absolute paths, command injection).
- **Security Headers:** All HTTP responses include `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, and `X-XSS-Protection: 1; mode=block`.
- **Error Suppression:** Custom `404` and `500` error handlers prevent raw Python stack traces, file paths, or credentials from leaking to end users.

---

## 5. LLM Provider Configuration & Behavior

The system is 100% provider-agnostic and abstracts models behind `LLMClient`:

1. **Local Ollama (`ollama`):** Zero-cost, privacy-first local processing. Requires `ollama serve` running locally.
2. **Google Gemini (`gemini`):** Cloud API with free tier. Requires `GEMINI_API_KEY`.
3. **Groq Cloud (`groq`):** Cloud API with ultra-fast inference. Requires `GROQ_API_KEY`.
4. **Deterministic Mock (`mock`):** Built-in rule-based offline parser. Used for automated testing and air-gapped environments without LLM access.

When an API-backed provider is configured without its required key, the system fails clearly with a descriptive configuration error message rather than silently falling back to a paid service.

---

## 6. Dynamic Knowledge & RAG Behavior

- **Zero External Vector DB:** Embeddings and similarity search run locally in-memory using an optimized TF-IDF vector space model with sublinear term-frequency scaling and cosine similarity.
- **Dataset Isolation:** Every dataset maintains an isolated index. Retrievals for the active dataset never cross-contaminate with other datasets.
- **Automatic Invalidation:** Modifying or switching datasets, uploading new knowledge documents, or resetting knowledge immediately invalidates and rebuilds the active vector index.
- **Relevance Gating:** Direct knowledge questions enforce a minimum similarity threshold (`0.15`). If available documents lack the answer, the system states that information is not available rather than hallucinating.

---

## 7. Known Architectural Limitations

1. **In-Memory Session State:** The current version utilizes in-memory application state (`AppStateManager`). Uploaded custom datasets, uploaded knowledge files, and ephemeral vector indices reside in server process memory and will reset if the server process restarts.
2. **Single Active Dataset Per Process:** The in-memory state manager tracks one globally active dataset at a time.
3. **File Size Capacity:** Dataframes and Excel workbooks are loaded into pandas memory. Datasets should ideally be under 500,000 rows for optimal sub-second interactive analysis.
