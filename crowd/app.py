"""NiceGUI application setup for the crowd annotation platform.

Creates the app, sets the storage secret, and registers all page routes.
"""

import logging

from nicegui import app, ui

from crowd.settings import CrowdConfig
from crowd.db import CrowdDatabase
from crowd.pages.annotate import register_annotate_page
from crowd.pages.dashboard import register_dashboard_page
from crowd.pages.login import register_login_pages
from crowd.pages.reveal import register_reveal_page

logger = logging.getLogger(__name__)


def create_crowd_app(db: CrowdDatabase, config: CrowdConfig) -> None:
    """Configure the NiceGUI app and register all routes.

    Must be called before ui.run().
    """
    # Set storage secret for session cookies
    app.storage.secret = config.secret_key

    # Register page routes
    register_login_pages(db, config)
    register_dashboard_page(db)
    register_annotate_page(db, max_per_hour=config.max_annotations_per_user_per_hour)
    register_reveal_page(db)

    # Cleanup old login attempts on startup
    cleaned = db.cleanup_old_login_attempts(days=7)
    if cleaned:
        logger.info("Cleaned up %d old login attempts", cleaned)

    # Root redirects to dashboard (or login if not authenticated)
    @ui.page("/")
    def index() -> None:
        if app.storage.user.get("authenticated"):
            ui.navigate.to("/dashboard")
        else:
            ui.navigate.to("/login")

    logger.info(
        "Crowd annotation app configured — %d papers in database",
        db.get_paper_count(),
    )
