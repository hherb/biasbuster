"""Entry point for the crowd annotation platform.

Usage:
    CROWD_SECRET_KEY=your-secret uv run python -m crowd
    CROWD_SECRET_KEY=your-secret uv run python -m crowd --port 9090
"""

import argparse
import logging

from nicegui import ui

from biasbuster.crowd.app import create_crowd_app
from biasbuster.crowd.settings import load_config
from biasbuster.crowd.db import CrowdDatabase


def main() -> None:
    """Launch the crowd annotation web application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="BiasBuster Crowd Annotation Platform"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port to listen on (default: 9090, overrides CROWD_PORT env var)",
    )
    parser.add_argument(
        "--host", default=None,
        help="Host to bind to (default: 127.0.0.1, overrides CROWD_HOST env var)",
    )
    args = parser.parse_args()

    config = load_config()
    if args.port:
        config.port = args.port
    if args.host:
        config.host = args.host

    # Initialize database
    db = CrowdDatabase(config.db_path)
    db.initialize()

    paper_count = db.get_paper_count()
    if paper_count == 0:
        logging.warning(
            "No papers in crowd database. Run the export script first:\n"
            "  uv run python -m crowd.export_to_crowd "
            "--prod-db dataset/biasbuster.db "
            "--crowd-db %s --model deepseek --limit 200",
            config.db_path,
        )

    # Configure and run the app
    create_crowd_app(db, config)

    ui.run(
        title=config.title,
        host=config.host,
        port=config.port,
        reload=False,
        show=False,
    )


if __name__ == "__main__":
    main()
