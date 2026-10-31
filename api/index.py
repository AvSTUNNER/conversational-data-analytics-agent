"""WSGI entrypoint for Vercel Serverless Functions."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure working directory is project root so relative dataset paths resolve
os.chdir(str(PROJECT_ROOT))

try:
    from src.web.app import create_app
    app = create_app()
except Exception as e:
    import traceback
    err_msg = traceback.format_exc()
    from flask import Flask, Response
    app = Flask(__name__)
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def catch_all(path):
        return Response(f"<h1>Startup Error</h1><pre>{err_msg}</pre>", status=500, mimetype="text/html")
