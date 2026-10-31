"""Flask application factory and routes for the Conversational Data Analysis Assistant."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for, Response
from werkzeug.utils import secure_filename

from src.data.loader import EmptyFileError, UnsupportedFormatError, DataLoadError
from src.assistant import ConversationalAssistant, AssistantResponse
from src.llm.client import get_llm_client
from src.web.state import app_state, PRESET_DATASETS

ALLOWED_DATASET_EXTENSIONS = {".csv", ".xls", ".xlsx"}
ALLOWED_KNOWLEDGE_EXTENSIONS = {".txt", ".md", ".json", ".csv"}


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Creates and configures the Flask application."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key-hcl-p123")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max payload limit

    if test_config:
        app.config.update(test_config)

    @app.after_request
    def set_security_headers(response: Response) -> Response:
        """Sets defensive HTTP security headers."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

    @app.errorhandler(404)
    def handle_404(error):
        return render_template("404.html", state=app_state, active_page=""), 404

    @app.errorhandler(500)
    def handle_500(error):
        return render_template("500.html", state=app_state, active_page=""), 500

    @app.errorhandler(413)
    def handle_413(error):
        flash("Upload rejected: File size exceeds the maximum allowed limit of 16 MB.", "error")
        return redirect(request.referrer or url_for("dataset_page")), 413

    @app.route("/")
    def index():
        return redirect(url_for("ask_page"))

    @app.route("/ask")
    def ask_page():
        examples = []
        if app_state.dataset_name and app_state.dataset_name in PRESET_DATASETS:
            examples = PRESET_DATASETS[app_state.dataset_name]["examples"]

        chart_json = None
        plan_json_formatted = None
        if app_state.last_response:
            if app_state.last_response.chart_figure:
                chart_json = app_state.last_response.chart_figure.to_json()
            if app_state.last_response.plan:
                plan_json_formatted = json.dumps(
                    app_state.last_response.plan.model_dump(exclude_none=True),
                    indent=2,
                )

        return render_template(
            "ask.html",
            active_page="ask",
            state=app_state,
            examples=examples,
            response=app_state.last_response,
            chart_json=chart_json,
            plan_json_formatted=plan_json_formatted,
            current_query=request.args.get("q", ""),
        )

    @app.route("/ask/run", methods=["POST"])
    def run_query():
        question = request.form.get("question", "").strip()
        if not question:
            return redirect(url_for("ask_page"))

        if app_state.cleaned_df is None:
            flash("No dataset loaded. Please load a dataset first.", "error")
            return redirect(url_for("dataset_page"))

        # Run analysis through the existing pipeline with RAG router and retriever
        try:
            client = get_llm_client(provider=app_state.active_provider)
            assistant = ConversationalAssistant(
                llm_client=client,
                retriever=app_state.retriever,
                router=app_state.query_router,
            )
            response = assistant.analyze(
                question=question,
                df=app_state.cleaned_df,
                dataset_name=app_state.dataset_name,
            )
        except Exception as e:
            flash(f"Analysis Error: {str(e)}", "error")
            return redirect(url_for("ask_page", q=question))

        app_state.last_response = response
        app_state.query_history.append({
            "question": question,
            "answer": response.direct_answer,
            "timestamp": pd.Timestamp.now().isoformat(),
        })

        return redirect(url_for("ask_page", q=question))

    @app.route("/dashboard")
    def dashboard_page():
        index_status = None
        if app_state.dataset_name:
            app_state.sync_active_index()
            index_status = app_state.retriever.get_index_status(app_state.dataset_name)

        return render_template(
            "dashboard.html",
            active_page="dashboard",
            state=app_state,
            presets=PRESET_DATASETS,
            index_status=index_status,
        )

    @app.route("/dataset")
    def dataset_page():
        raw_preview_rows = []
        raw_preview_cols = []
        if app_state.cleaned_df is not None:
            raw_preview_cols = list(app_state.cleaned_df.columns)
            raw_preview_rows = app_state.cleaned_df.head(15).values.tolist()

        return render_template(
            "dataset.html",
            active_page="dataset",
            state=app_state,
            presets=PRESET_DATASETS,
            raw_preview_cols=raw_preview_cols,
            raw_preview_rows=raw_preview_rows,
        )

    @app.route("/dataset/switch", methods=["POST"])
    def switch_preset():
        preset_name = request.form.get("preset_name")
        if preset_name:
            if app_state.load_preset(preset_name):
                flash(f"Loaded benchmark dataset: {preset_name}", "success")
            else:
                flash(f"Failed to load preset: {preset_name}", "error")
        return redirect(url_for("dataset_page"))

    @app.route("/dataset/upload", methods=["POST"])
    def upload_dataset():
        if "file" not in request.files:
            flash("No file part in the upload request.", "error")
            return redirect(url_for("dataset_page"))

        uploaded_file = request.files["file"]
        if not uploaded_file or uploaded_file.filename == "":
            flash("No file selected for upload.", "error")
            return redirect(url_for("dataset_page"))

        raw_filename = uploaded_file.filename
        clean_filename = secure_filename(raw_filename)
        if not clean_filename:
            clean_filename = "uploaded_dataset" + Path(raw_filename).suffix.lower()

        ext = Path(clean_filename).suffix.lower()
        if ext not in ALLOWED_DATASET_EXTENSIONS:
            flash(
                f"Upload Error: Unsupported dataset format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_DATASET_EXTENSIONS))}",
                "error",
            )
            return redirect(url_for("dataset_page"))

        try:
            file_bytes = uploaded_file.read()
            app_state.load_uploaded_data(file_bytes, clean_filename, sheet_name=0)
            flash(
                f"Successfully ingested and profiled '{clean_filename}' ({len(app_state.cleaned_df):,} rows, {len(app_state.cleaned_df.columns)} columns).",
                "success",
            )
        except EmptyFileError as e:
            flash(f"Upload Error: {str(e)}", "error")
        except UnsupportedFormatError as e:
            flash(f"Format Error: {str(e)}", "error")
        except Exception as e:
            flash(f"Failed to process uploaded file '{clean_filename}': {str(e)}", "error")

        return redirect(url_for("dataset_page"))

    @app.route("/dataset/sheet", methods=["POST"])
    def switch_sheet():
        sheet_name = request.form.get("sheet_name")
        if not sheet_name:
            return redirect(url_for("dataset_page"))

        try:
            if app_state.is_uploaded:
                app_state.switch_uploaded_sheet(sheet_name)
                flash(f"Switched workbook sheet to '{sheet_name}'.", "success")
            elif app_state.dataset_name:
                app_state.load_preset(app_state.dataset_name, sheet_name=sheet_name)
                flash(f"Switched workbook sheet to '{sheet_name}'.", "success")
        except Exception as e:
            flash(f"Failed to switch sheet: {str(e)}", "error")

        return redirect(url_for("dataset_page"))

    @app.route("/visualizations")
    def visualizations_page():
        chart_json = None
        chart_type = "none"
        if app_state.last_response:
            if app_state.last_response.chart_figure:
                chart_json = app_state.last_response.chart_figure.to_json()
            if app_state.last_response.plan:
                chart_type = app_state.last_response.plan.chart.value

        return render_template(
            "visualizations.html",
            active_page="visualizations",
            state=app_state,
            chart_json=chart_json,
            chart_type=chart_type,
        )

    @app.route("/knowledge")
    def knowledge_page():
        artifact = app_state.current_knowledge
        index_status = None
        if app_state.dataset_name:
            app_state.sync_active_index()
            index_status = app_state.retriever.get_index_status(app_state.dataset_name)

        return render_template(
            "knowledge.html",
            active_page="knowledge",
            state=app_state,
            artifact=artifact,
            index_status=index_status,
            search_query="",
            search_results=[],
            routing_decision=None,
        )

    @app.route("/knowledge/search", methods=["POST"])
    def search_knowledge():
        query = request.form.get("query", "").strip()
        artifact = app_state.current_knowledge
        index_status = None
        search_results = []
        routing_decision = None

        if app_state.dataset_name:
            app_state.sync_active_index()
            index_status = app_state.retriever.get_index_status(app_state.dataset_name)
            if query:
                search_results = app_state.retriever.retrieve(
                    dataset_name=app_state.dataset_name,
                    query=query,
                    top_k=4,
                )
                routing_decision = app_state.query_router.route(query=query)

        return render_template(
            "knowledge.html",
            active_page="knowledge",
            state=app_state,
            artifact=artifact,
            index_status=index_status,
            search_query=query,
            search_results=search_results,
            routing_decision=routing_decision,
        )

    @app.route("/knowledge/upload", methods=["POST"])
    def upload_knowledge():
        if "file" not in request.files:
            flash("No file part in the knowledge upload request.", "error")
            return redirect(url_for("knowledge_page"))

        uploaded_file = request.files["file"]
        if not uploaded_file or uploaded_file.filename == "":
            flash("No file selected for knowledge upload.", "error")
            return redirect(url_for("knowledge_page"))

        raw_filename = uploaded_file.filename
        clean_filename = secure_filename(raw_filename)
        if not clean_filename:
            clean_filename = "uploaded_knowledge" + Path(raw_filename).suffix.lower()

        ext = Path(clean_filename).suffix.lower()
        if ext not in ALLOWED_KNOWLEDGE_EXTENSIONS:
            flash(
                f"Unsupported knowledge format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_KNOWLEDGE_EXTENSIONS))}",
                "error",
            )
            return redirect(url_for("knowledge_page"))

        try:
            content_bytes = uploaded_file.read()
            if not content_bytes or not content_bytes.strip():
                flash("Knowledge document is empty (0 bytes).", "error")
                return redirect(url_for("knowledge_page"))

            # Decode safely as text
            content_text = content_bytes.decode("utf-8", errors="replace")
            app_state.knowledge_manager.upload_supplied_knowledge(
                dataset_name=app_state.dataset_name or "active_dataset",
                content=content_text,
                filename=clean_filename,
                profile=app_state.profile,
                df=app_state.cleaned_df,
            )
            # Rebuild index with new knowledge
            app_state.sync_active_index(force=True)
            flash(
                f"Successfully attached user-supplied knowledge document '{clean_filename}' and rebuilt RAG index for '{app_state.dataset_name}'.",
                "success",
            )
        except Exception as e:
            flash(f"Failed to process knowledge document '{clean_filename}': {str(e)}", "error")

        return redirect(url_for("knowledge_page"))

    @app.route("/knowledge/download")
    def download_knowledge():
        artifact = app_state.current_knowledge
        if not artifact:
            flash("No active knowledge artifact to download.", "error")
            return redirect(url_for("knowledge_page"))

        text = artifact.get_full_text()
        safe_name = (app_state.dataset_name or "dataset").replace(" ", "_").replace("/", "_").replace(".xls", "").replace(".csv", "")
        filename = f"{safe_name}_knowledge_artifact.md"

        return Response(
            text,
            mimetype="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.route("/knowledge/reset", methods=["POST"])
    def reset_knowledge():
        if app_state.dataset_name and app_state.profile:
            app_state.knowledge_manager.reset_to_generated(
                dataset_name=app_state.dataset_name,
                profile=app_state.profile,
                df=app_state.cleaned_df,
            )
            app_state.sync_active_index(force=True)
            flash("Reset knowledge to auto-generated factual artifact and rebuilt RAG index.", "info")
        return redirect(url_for("knowledge_page"))

    @app.route("/pipeline")
    def pipeline_page():
        index_status = None
        if app_state.dataset_name:
            app_state.sync_active_index()
            index_status = app_state.retriever.get_index_status(app_state.dataset_name)

        return render_template(
            "pipeline.html",
            active_page="pipeline",
            state=app_state,
            index_status=index_status,
        )

    @app.route("/settings", methods=["GET", "POST"])
    def settings_page():
        if request.method == "POST":
            new_provider = request.form.get("provider", "ollama").strip().lower()
            valid_providers = {"ollama", "gemini", "groq", "mock"}
            if new_provider in valid_providers:
                app_state.active_provider = new_provider
                flash(f"Active LLM provider set to '{new_provider}'.", "success")
            else:
                flash(f"Invalid provider '{new_provider}'.", "error")
            return redirect(url_for("settings_page"))

        provider_configs = [
            {
                "id": "ollama",
                "name": "Local Ollama",
                "type": "Local Offline Server",
                "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                "endpoint": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "auth_status": "Not Required (Local)",
                "ready": True,
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "type": "Cloud API (Free Tier)",
                "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                "endpoint": "https://generativelanguage.googleapis.com",
                "auth_status": "Configured (via GEMINI_API_KEY)" if bool(os.getenv("GEMINI_API_KEY")) else "Key Missing (Set GEMINI_API_KEY)",
                "ready": bool(os.getenv("GEMINI_API_KEY")),
            },
            {
                "id": "groq",
                "name": "Groq Cloud",
                "type": "Cloud API (High Speed)",
                "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "endpoint": "https://api.groq.com/openai/v1",
                "auth_status": "Configured (via GROQ_API_KEY)" if bool(os.getenv("GROQ_API_KEY")) else "Key Missing (Set GROQ_API_KEY)",
                "ready": bool(os.getenv("GROQ_API_KEY")),
            },
            {
                "id": "mock",
                "name": "Deterministic Mock Client",
                "type": "Offline Test Engine",
                "model": "rule-based-mock",
                "endpoint": "In-Memory Local Engine",
                "auth_status": "Not Required (Local)",
                "ready": True,
            },
        ]

        env_info = {
            "environment": os.getenv("FLASK_ENV", "production"),
            "debug": os.getenv("FLASK_DEBUG", "False").lower() in {"true", "1", "yes"},
            "max_upload_mb": 16,
            "port": int(os.getenv("PORT", 5000)),
            "host": os.getenv("HOST", "0.0.0.0"),
            "session_persistence": "In-Memory Runtime State",
        }

        return render_template(
            "settings.html",
            active_page="settings",
            state=app_state,
            provider_configs=provider_configs,
            env_info=env_info,
        )

    @app.route("/reset")
    def reset_app_state():
        app_state.reset_state()
        flash("Application state has been reset to defaults.", "info")
        return redirect(url_for("ask_page"))

    return app
