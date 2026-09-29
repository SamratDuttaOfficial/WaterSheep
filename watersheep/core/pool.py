"""Interruptible parallel map."""
from __future__ import annotations
import queue
import threading
from typing import Callable, Iterable, Iterator

from . import interrupt
from .util import LOG

_END = object()


def pool_map(fn: Callable, items: Iterable, workers: int) -> Iterator:
    """Map `fn` over `items` on daemon threads, yielding results as they finish."""
    it = iter(items)
    lock = threading.Lock()
    done: queue.Queue = queue.Queue()
    halt = threading.Event()

    def next_item():
        with lock:
            try:
                return next(it)
            except StopIteration:
                return _END

    def loop():
        try:
            while not (halt.is_set() or interrupt.stopping()):
                item = next_item()
                if item is _END:
                    return
                try:
                    done.put(fn(item))
                except interrupt.Interrupted:
                    return
                except Exception as e:
                    LOG.debug("worker item failed: %r", e)
                    done.put(None)
        finally:
            done.put(_END)

    n = max(1, workers)
    threads = [threading.Thread(target=loop, daemon=True, name="w%d" % i) for i in range(n)]
    try:
        for t in threads:
            t.start()
        finished = 0
        while finished < n:
            try:
                r = done.get(timeout=0.5)
            except queue.Empty:
                if interrupt.stopping():
                    return
                continue
            if r is _END:
                finished += 1
                continue
            yield r
    finally:
        halt.set()
