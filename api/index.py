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

flask_app = create_app()

class VercelPathFixer:
    """WSGI middleware ensuring proper PATH_INFO routing when Vercel rewrites to /api/index."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        
        # 1. Strip any vercel api prefixes
        prefixes = ["/api/index.py", "/api/index", "/api"]
        for prefix in prefixes:
            if path == prefix:
                path = "/"
                break
            elif path.startswith(prefix + "/"):
                path = path[len(prefix):]
                break

        if not path:
            path = "/"
            
        environ["PATH_INFO"] = path
        return self.wsgi_app(environ, start_response)

app = VercelPathFixer(flask_app)
application = app
