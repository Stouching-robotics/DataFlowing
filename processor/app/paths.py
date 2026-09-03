"""Project paths shared by the web server and frontend tooling.

Keeping the web tree in one place prevents individual routes and reload
scripts from making different assumptions about where templates and static
assets live.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
TEMPLATES_DIR = WEB_ROOT / "templates"
STATIC_DIR = WEB_ROOT / "static"
WORKFLOW_STUDIO_DIR = WEB_ROOT / "workflow-studio"
