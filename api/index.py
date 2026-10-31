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

# Vercel looks for a top-level WSGI variable named app or application
app = create_app()
application = app
