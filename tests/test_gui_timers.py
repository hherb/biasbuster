"""Tests for :mod:`biasbuster.gui.timers`.

These verify the ``managed_timer`` contract without a live NiceGUI server:
timer creation is deferred to a connect handler registered on the current
client, the timer is created with the caller's arguments when that handler
fires, and it is created only once so a browser auto-reconnect (which fires the
connect handler again) does not accumulate duplicate timers. Deferring creation
until connection is what prevents the "The parent slot of the element has been
deleted" tracebacks NiceGUI logs when it prunes phantom HTTP-only clients whose
timers never connected (zauberzeug/nicegui#5595).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from biasbuster.gui.timers import managed_timer


def test_managed_timer_defers_creation_to_connect_handler() -> None:
    """No timer is created eagerly; a single connect handler is registered."""
    with patch("biasbuster.gui.timers.ui.timer") as mk_timer, \
            patch("biasbuster.gui.timers.context") as mock_context:
        managed_timer(1.0, MagicMock())

    mk_timer.assert_not_called()
    mock_context.client.on_connect.assert_called_once()


def test_connect_handler_creates_timer_with_given_args() -> None:
    """Firing the connect handler creates the timer with the caller's args."""
    callback = MagicMock()
    with patch("biasbuster.gui.timers.ui.timer") as mk_timer, \
            patch("biasbuster.gui.timers.context") as mock_context:
        managed_timer(2.5, callback)
        handler = mock_context.client.on_connect.call_args.args[0]
        handler()

    mk_timer.assert_called_once_with(2.5, callback)


def test_connect_handler_is_idempotent_across_reconnects() -> None:
    """Repeated connects (browser auto-reconnect) create the timer only once."""
    with patch("biasbuster.gui.timers.ui.timer") as mk_timer, \
            patch("biasbuster.gui.timers.context") as mock_context:
        managed_timer(1.0, MagicMock())
        handler = mock_context.client.on_connect.call_args.args[0]
        handler()
        handler()
        handler()

    mk_timer.assert_called_once()
