"""Authentication helpers for the crowd annotation platform.

Handles password hashing (argon2id), input validation, rate limiting,
and session-based auth guards for NiceGUI pages.
"""

import logging
import re
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import HashingError, VerifyMismatchError

from nicegui import app, ui

from biasbuster.crowd.settings import CrowdConfig
from biasbuster.crowd.db import CrowdDatabase

logger = logging.getLogger(__name__)

_ph = PasswordHasher()

# ── Password hashing ─────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Hash a password using argon2id."""
    return _ph.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a stored hash.

    Returns True if the password matches.
    """
    try:
        return _ph.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except HashingError:
        logger.exception("Error verifying password hash")
        return False


def needs_rehash(password_hash: str) -> bool:
    """Check if a password hash needs to be rehashed with updated parameters."""
    return _ph.check_needs_rehash(password_hash)


# ── Input validation ─────────────────────────────────────────────────

_USERNAME_RE = re.compile(r"[a-zA-Z0-9_]{3,30}$")
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def validate_registration(
    username: str,
    email: str,
    password: str,
    password_confirm: str,
) -> list[str]:
    """Validate registration inputs.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []

    # Username
    if not username or not _USERNAME_RE.fullmatch(username):
        errors.append(
            "Username must be 3-30 characters, alphanumeric and underscores only."
        )

    # Email
    if not email or not _EMAIL_RE.fullmatch(email):
        errors.append("Please enter a valid email address.")

    # Password strength
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    elif not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter.")
    elif not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter.")
    elif not re.search(r"[0-9]", password):
        errors.append("Password must contain at least one digit.")

    # Confirmation
    if password != password_confirm:
        errors.append("Passwords do not match.")

    return errors


def validate_text_input(text: str, max_length: int = 500) -> str:
    """Sanitize a text input: strip whitespace, enforce length limit."""
    return text.strip()[:max_length]


# ── Rate limiting ────────────────────────────────────────────────────


def check_rate_limit(
    db: CrowdDatabase,
    ip_address: str,
    action: str,
    config: CrowdConfig,
) -> Optional[str]:
    """Check if an IP is rate-limited for a given action.

    Returns an error message if rate-limited, or None if allowed.
    """
    if action == "login":
        count = db.count_recent_attempts(
            ip_address, action="login", minutes=config.login_lockout_minutes
        )
        if count >= config.max_failed_logins_per_ip:
            return "Too many failed login attempts. Please try again later."
    elif action == "register":
        count = db.count_recent_attempts(
            ip_address,
            action="register",
            minutes=config.registration_lockout_minutes,
        )
        if count >= config.max_registrations_per_ip:
            return "Too many registration attempts. Please try again later."

    return None


# ── Session helpers ──────────────────────────────────────────────────


def get_current_user() -> Optional[dict]:
    """Read the current user from NiceGUI session storage.

    Returns a dict with 'user_id', 'username' if authenticated, else None.
    """
    storage = app.storage.user
    if storage.get("authenticated"):
        return {
            "user_id": storage.get("user_id"),
            "username": storage.get("username"),
        }
    return None


def set_session(user_id: int, username: str) -> None:
    """Set session data after successful login."""
    app.storage.user.update({
        "authenticated": True,
        "user_id": user_id,
        "username": username,
    })


def clear_session() -> None:
    """Clear session data on logout."""
    app.storage.user.clear()


def require_auth() -> bool:
    """Check authentication and redirect to login if needed.

    Returns True if authenticated, False if redirected.
    Usage at the top of every protected @ui.page handler:
        if not require_auth():
            return
    """
    if not app.storage.user.get("authenticated"):
        ui.navigate.to("/login")
        return False
    return True
