"""WSGI entrypoint for Vercel Serverless Functions."""
import os
import sys
import urllib.parse
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
    """WSGI middleware ensuring proper PATH_INFO routing when Vercel rewrites requests."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        qs = environ.get("QUERY_STRING", "")
        params = urllib.parse.parse_qs(qs)
        
        # 1. If Vercel passed __vercel_route__, restore the true requested route
        if "__vercel_route__" in params:
            route_val = params.pop("__vercel_route__")[0]
            if not route_val.startswith("/"):
                route_val = "/" + route_val
            environ["PATH_INFO"] = route_val
            # Reconstruct clean QUERY_STRING without internal routing parameter
            clean_pairs = []
            for k, vals in params.items():
                for v in vals:
                    clean_pairs.append(f"{urllib.parse.quote(k)}={urllib.parse.quote(v)}")
            environ["QUERY_STRING"] = "&".join(clean_pairs)
        else:
            # 2. Fallback prefix cleaning
            path = environ.get("PATH_INFO", "")
            prefixes = ["/api/index.py", "/api/index", "/api"]
            for prefix in prefixes:
                if path == prefix:
                    path = "/"
                    break
                elif path.startswith(prefix + "/"):
                    path = path[len(prefix):]
                    break
            environ["PATH_INFO"] = path or "/"

        return self.wsgi_app(environ, start_response)

app = VercelPathFixer(flask_app)
application = app
