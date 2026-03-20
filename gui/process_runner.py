"""Async subprocess wrapper with output streaming for NiceGUI.

Provides a cross-platform ``ProcessRunner`` that launches long-running
commands via ``asyncio.create_subprocess_exec``, captures stdout/stderr
line-by-line, and exposes a polling-based completion interface safe for
NiceGUI's UI context.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Seconds to wait after SIGTERM before sending SIGKILL.
_GRACEFUL_SHUTDOWN_TIMEOUT = 5


class ProcessRunner:
    """Manages a single async subprocess with output streaming.

    Completion is detected by polling :attr:`finished_code` from a
    NiceGUI ``ui.timer`` rather than via callbacks, because NiceGUI
    does not allow element creation (``ui.timer``, ``ui.notify``, etc.)
    from background tasks.

    Attributes:
        status: One of ``"idle"``, ``"running"``, ``"completed"``,
            ``"failed"``, or ``"stopped"``.
        return_code: The process exit code once finished, else ``None``.
        output_lines: All captured stdout/stderr lines.
        finished_code: Set to the exit code (or ``None``) exactly once
            when the process finishes.  Call :meth:`consume_finished` from
            a UI-context timer to retrieve and clear it.
    """

    def __init__(self) -> None:
        self.status: str = "idle"
        self.return_code: int | None = None
        self.output_lines: list[str] = []
        self.finished_code: int | None | _Sentinel = _NOT_FINISHED
        self._process: asyncio.subprocess.Process | None = None
        self._read_task: asyncio.Task | None = None
        self._callbacks: list[Callable[[str], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_output(self, callback: Callable[[str], None]) -> None:
        """Register *callback* to be invoked for each new output line."""
        self._callbacks.append(callback)

    def consume_finished(self) -> tuple[bool, int | None]:
        """Check whether the process has just finished.

        Returns ``(True, exit_code)`` exactly once after the process
        ends, then ``(False, None)`` on subsequent calls until the next
        run.  Safe to call from a ``ui.timer`` callback.
        """
        if self.finished_code is not _NOT_FINISHED:
            code = self.finished_code
            self.finished_code = _NOT_FINISHED
            return True, code  # type: ignore[return-value]
        return False, None

    async def start(
        self,
        cmd: list[str],
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Launch *cmd* as an async subprocess.

        Args:
            cmd: Command and arguments.
            cwd: Working directory for the subprocess.
            env: Optional environment variable overrides.

        Raises:
            RuntimeError: If a process is already running.
        """
        if self.status == "running":
            raise RuntimeError("A process is already running")

        self._reset()
        self.status = "running"
        cmd_str = " ".join(cmd)
        logger.info("Starting subprocess: %s", cmd_str)
        self._emit(f"$ {cmd_str}\n")

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(cwd) if cwd else None,
                env=env,
            )
        except FileNotFoundError as exc:
            self.status = "failed"
            self._emit(f"ERROR: command not found — {exc}\n")
            self.finished_code = None
            return
        except OSError as exc:
            self.status = "failed"
            self._emit(f"ERROR: could not start process — {exc}\n")
            self.finished_code = None
            return

        self._read_task = asyncio.create_task(self._stream_output())

    async def stop(self) -> None:
        """Gracefully stop the running subprocess.

        Sends SIGTERM (or ``terminate()`` on Windows), waits up to
        ``_GRACEFUL_SHUTDOWN_TIMEOUT`` seconds, then sends SIGKILL if
        the process is still alive.
        """
        if self._process is None or self.status != "running":
            return

        self._emit("\n--- Stopping process ---\n")

        try:
            if sys.platform == "win32":
                self._process.terminate()
            else:
                self._process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass  # already exited

        try:
            await asyncio.wait_for(
                self._process.wait(), timeout=_GRACEFUL_SHUTDOWN_TIMEOUT
            )
        except asyncio.TimeoutError:
            self._emit("Process did not exit in time — killing.\n")
            try:
                self._process.kill()
            except ProcessLookupError:
                pass
            await self._process.wait()

        self.return_code = self._process.returncode
        self.status = "stopped"
        self._emit(f"Process stopped (exit code {self.return_code}).\n")
        self.finished_code = self.return_code

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the subprocess is currently active."""
        return self.status == "running"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._process = None
        self._read_task = None
        self.return_code = None
        self.output_lines.clear()
        self.finished_code = _NOT_FINISHED

    def _emit(self, line: str) -> None:
        """Store *line* and notify all registered callbacks."""
        self.output_lines.append(line)
        for cb in self._callbacks:
            try:
                cb(line)
            except Exception:
                logger.exception("Output callback error")

    async def _stream_output(self) -> None:
        """Read stdout line-by-line until EOF, then finalise status."""
        assert self._process is not None
        assert self._process.stdout is not None

        try:
            async for raw_line in self._process.stdout:
                text = raw_line.decode("utf-8", errors="replace")
                self._emit(text)
        except Exception:
            logger.exception("Error reading subprocess output")

        await self._process.wait()
        self.return_code = self._process.returncode

        if self.status == "running":
            if self.return_code == 0:
                self.status = "completed"
                self._emit(f"\nProcess completed successfully (exit code 0).\n")
            else:
                self.status = "failed"
                self._emit(
                    f"\nProcess failed (exit code {self.return_code}).\n"
                )
        self.finished_code = self.return_code


class _Sentinel:
    """Unique sentinel to distinguish 'not finished' from exit code None."""
    pass


_NOT_FINISHED = _Sentinel()
