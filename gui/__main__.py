"""Launch the BiasBuster Fine-Tuning Workbench.

Usage::

    uv run python -m gui
    uv run python -m gui --port 8080
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nicegui import ui

from gui.app import create_app
from gui.state import detect_platform, load_settings, default_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="BiasBuster Fine-Tuning Workbench")
    parser.add_argument("--port", type=int, default=8080, help="Web UI port")
    args = parser.parse_args()

    state = default_state()
    state["platform"] = detect_platform()
    state["project_dir"] = str(Path(__file__).resolve().parent.parent)
    state.update(load_settings())

    logger.info(
        "Platform: %s %s | backends: %s",
        state["platform"]["system"],
        state["platform"]["machine"],
        ", ".join(state["platform"]["backends"]) or "none",
    )

    create_app(state)
    ui.run(
        title="BiasBuster Workbench",
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
