"""Login and registration pages for the crowd annotation platform."""

import logging
import sqlite3

from nicegui import app, ui

from crowd.auth import (
    check_rate_limit,
    hash_password,
    needs_rehash,
    set_session,
    validate_registration,
    validate_text_input,
    verify_password,
)
from crowd.settings import CrowdConfig
from crowd.db import CrowdDatabase

logger = logging.getLogger(__name__)

EXPERTISE_OPTIONS = [
    "unknown",
    "student",
    "researcher",
    "clinician",
    "statistician",
]


def _get_client_ip() -> str:
    """Best-effort client IP from NiceGUI/Starlette request.

    Relies on the reverse proxy (nginx/Caddy) setting X-Forwarded-For
    or X-Real-IP headers. Takes the first IP from X-Forwarded-For to
    handle proxy chains (client, proxy1, proxy2).
    """
    try:
        xff = app.storage.request_headers.get("x-forwarded-for", "")
        if xff:
            # Take the leftmost (client) IP from the chain
            return xff.split(",")[0].strip()
        return app.storage.request_headers.get("x-real-ip", "unknown")
    except Exception:
        return "unknown"


def register_login_pages(db: CrowdDatabase, config: CrowdConfig) -> None:
    """Register the /login and /register page routes."""

    @ui.page("/login")
    def login_page() -> None:
        """Login page."""
        # Already logged in?
        if app.storage.user.get("authenticated"):
            ui.navigate.to("/dashboard")
            return

        with ui.column().classes(
            "absolute-center items-center w-full max-w-md gap-4"
        ):
            ui.label(config.title).classes("text-h4 text-weight-bold")
            ui.label("Sign in to start annotating").classes("text-subtitle1")

            error_label = ui.label("").classes("text-negative")
            error_label.set_visibility(False)

            username_input = ui.input("Username").classes("w-full").props(
                "outlined dense"
            )
            password_input = ui.input(
                "Password", password=True, password_toggle_button=True
            ).classes("w-full").props("outlined dense")

            async def do_login() -> None:
                username = validate_text_input(username_input.value, 30)
                password = password_input.value

                if not username or not password:
                    error_label.text = "Please fill in all fields."
                    error_label.set_visibility(True)
                    return

                ip = _get_client_ip()

                # Rate limit check
                rate_msg = check_rate_limit(db, ip, "login", config)
                if rate_msg:
                    error_label.text = rate_msg
                    error_label.set_visibility(True)
                    return

                # Verify credentials
                user = db.get_user_by_username(username)
                if user is None or not verify_password(password, user["password_hash"]):
                    db.record_attempt(ip, action="login", username=username, success=False)
                    error_label.text = "Invalid username or password."
                    error_label.set_visibility(True)
                    return

                if not user["is_active"]:
                    error_label.text = "Account is disabled."
                    error_label.set_visibility(True)
                    return

                # Success
                db.record_attempt(ip, action="login", username=username, success=True)
                db.update_last_login(user["id"])

                # Rehash if needed
                if needs_rehash(user["password_hash"]):
                    db.update_password_hash(user["id"], hash_password(password))

                set_session(user["id"], user["username"])
                ui.navigate.to("/dashboard")

            ui.button("Sign In", on_click=do_login).classes(
                "w-full"
            ).props("color=primary")

            # Enter key submits
            password_input.on("keydown.enter", do_login)

            ui.separator()
            ui.link("Create an account", "/register").classes("text-primary")

    @ui.page("/register")
    def register_page() -> None:
        """Registration page."""
        if app.storage.user.get("authenticated"):
            ui.navigate.to("/dashboard")
            return

        with ui.column().classes(
            "absolute-center items-center w-full max-w-md gap-4"
        ):
            ui.label("Create Account").classes("text-h4 text-weight-bold")

            error_label = ui.label("").classes("text-negative")
            error_label.set_visibility(False)

            username_input = ui.input("Username").classes("w-full").props(
                "outlined dense"
            )
            email_input = ui.input("Email").classes("w-full").props(
                "outlined dense"
            )
            display_name_input = ui.input("Display Name").classes("w-full").props(
                "outlined dense"
            )
            expertise_input = ui.select(
                EXPERTISE_OPTIONS,
                value="unknown",
                label="Expertise Level",
            ).classes("w-full")
            password_input = ui.input(
                "Password", password=True, password_toggle_button=True
            ).classes("w-full").props("outlined dense")
            confirm_input = ui.input(
                "Confirm Password", password=True, password_toggle_button=True
            ).classes("w-full").props("outlined dense")

            async def do_register() -> None:
                username = validate_text_input(username_input.value, 30)
                email = validate_text_input(email_input.value, 255)
                display_name = validate_text_input(display_name_input.value, 100)
                expertise = expertise_input.value or "unknown"
                password = password_input.value
                confirm = confirm_input.value

                # Validate inputs
                errors = validate_registration(username, email, password, confirm)
                if errors:
                    error_label.text = errors[0]
                    error_label.set_visibility(True)
                    return

                ip = _get_client_ip()

                # Rate limit check
                rate_msg = check_rate_limit(db, ip, "register", config)
                if rate_msg:
                    error_label.text = rate_msg
                    error_label.set_visibility(True)
                    return

                # Create user
                try:
                    user_id = db.create_user(
                        username=username,
                        email=email,
                        password_hash=hash_password(password),
                        display_name=display_name or username,
                        expertise_level=expertise,
                    )
                except sqlite3.IntegrityError:
                    db.record_attempt(ip, action="register", username=username, success=False)
                    error_label.text = "Username or email already taken."
                    error_label.set_visibility(True)
                    return

                db.record_attempt(ip, action="register", username=username, success=True)
                set_session(user_id, username)
                ui.navigate.to("/dashboard")

            ui.button("Create Account", on_click=do_register).classes(
                "w-full"
            ).props("color=primary")

            ui.separator()
            ui.link("Already have an account? Sign in", "/login").classes(
                "text-primary"
            )

    @ui.page("/logout")
    def logout_page() -> None:
        """Logout — clear session and redirect to login."""
        app.storage.user.clear()
        ui.navigate.to("/login")
