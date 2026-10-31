"""Netlify Function handler for Conversational Data Analysis Assistant Flask App."""
import os
import sys
from pathlib import Path

# Set up project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(str(PROJECT_ROOT))

try:
    import serverless_wsgi
    from src.web.app import create_app
    app = create_app()

    def handler(event, context):
        return serverless_wsgi.handle_request(app, event, context)
except Exception as e:
    import traceback
    err_msg = traceback.format_exc()
    def handler(event, context):
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "text/html"},
            "body": f"<h1>Netlify Function Error</h1><pre>{err_msg}</pre>"
        }
