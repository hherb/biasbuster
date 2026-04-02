"""Configuration for the crowd annotation platform.

All secrets are loaded from environment variables. The app will
refuse to start if CROWD_SECRET_KEY is not set.
"""

import os
from dataclasses import dataclass


@dataclass
class CrowdConfig:
    """Configuration for the crowd annotation web app."""

    # Database
    db_path: str = "dataset/crowd_annotations.db"

    # Session / security
    secret_key: str = ""  # MUST be set via CROWD_SECRET_KEY env var

    # Server
    host: str = "127.0.0.1"  # Bind to localhost; use reverse proxy for public
    port: int = 9090
    title: str = "BiasBuster Crowd Annotation"

    # Rate limiting
    max_failed_logins_per_ip: int = 5
    login_lockout_minutes: int = 15
    max_registrations_per_ip: int = 3
    registration_lockout_minutes: int = 60
    max_annotations_per_user_per_hour: int = 30

    # Assignment
    default_target_annotations: int = 3

    # Session timeout (seconds)
    session_timeout_seconds: int = 86400  # 24 hours


def load_config() -> CrowdConfig:
    """Load configuration from environment variables.

    Raises:
        SystemExit: If CROWD_SECRET_KEY is not set.
    """
    cfg = CrowdConfig()

    cfg.secret_key = os.environ.get("CROWD_SECRET_KEY", "")
    if not cfg.secret_key:
        raise SystemExit(
            "CROWD_SECRET_KEY environment variable is required. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )

    cfg.db_path = os.environ.get("CROWD_DB_PATH", cfg.db_path)
    cfg.host = os.environ.get("CROWD_HOST", cfg.host)
    cfg.port = int(os.environ.get("CROWD_PORT", str(cfg.port)))

    return cfg
