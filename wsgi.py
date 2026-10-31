"""WSGI production entry point for Conversational Data Analysis Assistant (HCL P_123)."""
from __future__ import annotations

import os
from src.web.app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    app.run(host=host, port=port)
