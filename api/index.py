import sys
import os
from pathlib import Path

# Add the project root directory to sys.path so 'app' imports resolve seamlessly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app

# Vercel WSGI / ASGI entrypoint handler
handler = app
