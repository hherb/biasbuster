"""Client-lifecycle-aware timers for the workbench GUI.

NiceGUI's :func:`nicegui.ui.timer` starts a background task that first awaits
``client.connected()`` — and it does so with no timeout (see the ``_can_start``
override in ``nicegui/elements/timer.py``). A *phantom* client that fetches the
page over HTTP but never opens a websocket (a health check, ``curl``, a
link-preview bot, a browser prefetch) therefore leaves its timers blocked
forever. When NiceGUI later prunes that never-connected client, the timer's
one-time context setup dereferences a now-deleted parent slot and logs
``RuntimeError: The parent slot of the element has been deleted`` — one
traceback per orphaned timer, per phantom client. This is the upstream bug
tracked in zauberzeug/nicegui#5595 (only partially fixed).

The workbench registers per-client polling timers in ``training_tab.py``,
``export_tab.py`` and ``evaluation_tab.py`` (subprocess-completion polling and
live training-chart refresh), so every phantom hit to the page produces the
spam above.

:func:`managed_timer` avoids it by deferring timer creation until the client
has actually connected: a phantom client never reaches the connect handler, so
it never gets a timer to orphan, while a genuinely connected client gets an
ordinary polling timer that NiceGUI tears down with the client.
"""

from __future__ import annotations

from typing import Callable

from nicegui import context, ui


def managed_timer(interval: float, callback: Callable[[], None]) -> None:
    """Register a polling ``ui.timer`` that is created once its client connects.

    Behaves like :func:`nicegui.ui.timer` for a connected browser, but the
    timer is not created during page build. Instead it is created on the
    client's first connection, so phantom HTTP-only clients (which never
    connect) never spawn a timer — eliminating the
    "The parent slot of the element has been deleted" tracebacks NiceGUI
    otherwise logs when it prunes them (zauberzeug/nicegui#5595).

    The timer is created only on the *first* connection, so a browser
    auto-reconnect does not accumulate duplicate timers. No disconnect handler
    is needed: NiceGUI stops a connected client's timers when it tears the
    client down.

    Args:
        interval: Seconds between successive callback invocations.
        callback: Zero-argument callable invoked on each tick.
    """
    created = False

    def _start() -> None:
        nonlocal created
        if created:
            return
        created = True
        ui.timer(interval, callback)

    context.client.on_connect(_start)
