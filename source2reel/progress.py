"""Small stderr progress display for long CLI operations."""
from __future__ import annotations

from contextlib import contextmanager
import sys
import threading
import time
from typing import TextIO


def _elapsed(seconds: float) -> str:
    minutes, seconds = divmod(int(max(0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes:02d}m {seconds:02d}s"


class Progress:
    def __init__(self, stream: TextIO | None = None, *, interval: float = 7.0):
        self.stream = stream if stream is not None else sys.stderr
        self.tty = self.stream.isatty()
        self.interval = interval
        self._lock = threading.Lock()

    def _write(self, message: str) -> None:
        with self._lock:
            self.stream.write(message)
            self.stream.flush()

    @contextmanager
    def step(self, label: str, *, heartbeat: bool = True):
        """Report a known step; animate elapsed time only on a TTY."""
        self._write(f"● {label}\n")
        started = time.monotonic()
        stop = threading.Event()
        thread = None
        if self.tty and heartbeat:
            def heartbeat() -> None:
                while not stop.wait(self.interval):
                    self._write(f"\r\033[K  Still working — elapsed {_elapsed(time.monotonic() - started)}")

            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
        try:
            yield
        except BaseException as exc:
            outcome = "Interrupted" if isinstance(exc, KeyboardInterrupt) else "Failed"
            raise
        else:
            outcome = "Complete"
        finally:
            stop.set()
            if thread is not None:
                thread.join()
                self._write("\r\033[K")
            self._write(f"{outcome}: {label}\n")

    def ready(self, label: str, path: object) -> None:
        self._write(f"{label}: {path}\n")

    def note(self, message: str) -> None:
        """Report an event such as bounded recovery without a periodic log line."""
        self._write(("\r\033[K" if self.tty else "") + f"{message}\n")


@contextmanager
def step(progress: Progress | None, label: str):
    """Allow the pipeline's progress display to remain optional to library callers."""
    if progress is None:
        yield
    else:
        with progress.step(label):
            yield
