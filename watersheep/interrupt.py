"""Ctrl+C handling that stops at the next safe point."""
from __future__ import annotations
import atexit
import os
import signal
import threading

STOP = threading.Event()
_hits = 0
_installed = False
_cleanups: list = []


class Interrupted(KeyboardInterrupt):
    """Raised by check() after a stop request."""


def _say(msg: str) -> None:
    # no logging here: the signal may hold its lock
    try:
        os.write(2, msg.encode("utf-8", "replace"))
    except Exception:
        pass


def run_cleanups() -> None:
    while _cleanups:
        fn = _cleanups.pop()
        try:
            fn()
        except Exception:
            pass


def _handler(signum, frame) -> None:
    global _hits
    _hits += 1
    if _hits == 1:
        STOP.set()
        _say("\n[stopping] saving progress at the next safe point - "
             "press Ctrl+C again to quit now\n")
    else:
        _say("\n[quit] leaving immediately; everything saved so far is intact\n")
        run_cleanups()
        os._exit(130)


def install() -> None:
    """Install the Ctrl+C handler (idempotent)."""
    global _installed
    if _installed:
        return
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass
    atexit.register(run_cleanups)
    _installed = True


def on_exit(fn) -> None:
    _cleanups.append(fn)


def forget(fn) -> None:
    try:
        _cleanups.remove(fn)
    except ValueError:
        pass


def stopping() -> bool:
    return STOP.is_set()


def check(where: str = "") -> None:
    if STOP.is_set():
        raise Interrupted(where or "stopped by user")


def sleep(seconds: float) -> bool:
    """Interruptible sleep; True if cut short."""
    return STOP.wait(seconds)


def reset() -> None:
    """Clear a pending stop request."""
    global _hits
    _hits = 0
    STOP.clear()
