"""Production entrypoint for Container & Local deployments (e.g. Gunicorn / Docker / Railway)."""
from __future__ import annotations

import os
from src.web.app import create_app

def get_wsgi_app():
    return create_app()

if __name__ == "__main__":
    app = get_wsgi_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() in {"true", "1", "yes"}
    app.run(host=host, port=port, debug=debug)
