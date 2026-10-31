"""WSGI entrypoint for Vercel Serverless Functions."""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure working directory is project root so relative paths resolve cleanly
os.chdir(str(PROJECT_ROOT))

from src.web.app import create_app

app = create_app()

@app.route("/api/index", defaults={"subpath": ""}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route("/api/index/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE"])
def vercel_entrypoint_router(subpath: str = ""):
    """Explicitly catches all Vercel-routed requests to /api/index and forwards to Flask views."""
    from flask import request

    target = request.args.get("__vercel_route__") or subpath
    if not target or target.strip("/") == "":
        target = "ask"
    else:
        target = target.lstrip("/")

    # Forward to the underlying Flask view cleanly
    with app.test_client() as client:
        resp = client.open(
            "/" + target,
            method=request.method,
            data=request.get_data(),
            headers=dict(request.headers),
            query_string=request.query_string.decode("utf-8", errors="ignore"),
        )
        return resp.get_data(), resp.status_code, list(resp.headers.items())

application = app
