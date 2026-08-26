"""A small progress bar with an ETA.

No dependency, and ASCII only: classic Command Prompt does not reliably render
block-drawing characters under its default code page, and a progress bar that
renders as mojibake is worse than none.
"""

from __future__ import annotations

import sys
import time


def _dur(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class Bar:
    """Usage:

        bar = Bar(total, "sampling")
        for ... :
            bar.advance(n)
        bar.done()
    """

    WIDTH = 28

    def __init__(self, total: int, label: str, note: str = ""):
        self.total = max(1, total)
        self.label = label
        self.note = note
        self.n = 0
        self.t0 = time.time()
        self._last = 0.0
        self._draw(force=True)

    def advance(self, k: int = 1, note: str | None = None) -> None:
        self.n = min(self.total, self.n + k)
        if note is not None:
            self.note = note
        self._draw()

    def _draw(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last < 0.15 and self.n < self.total:
            return
        self._last = now

        frac = self.n / self.total
        filled = int(round(frac * self.WIDTH))
        bar = "#" * filled + "-" * (self.WIDTH - filled)

        elapsed = now - self.t0
        if self.n > 0 and self.n < self.total:
            eta = f"eta {_dur(elapsed / self.n * (self.total - self.n))}"
        elif self.n >= self.total:
            eta = f"took {_dur(elapsed)}"
        else:
            eta = "eta --"

        line = (
            f"  {self.label:<22} [{bar}] "
            f"{self.n:>5}/{self.total:<5} {frac*100:>3.0f}%  {eta}"
        )
        if self.note:
            line += f"  {self.note}"
        # Pad to clear any longer previous line, then return to column 0.
        sys.stdout.write("\r" + line.ljust(110)[:110])
        sys.stdout.flush()

    def done(self, note: str | None = None) -> None:
        self.n = self.total
        if note is not None:
            self.note = note
        self._draw(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()


class Steps:
    """Tracks overall progress across the named stages of a run."""

    def __init__(self, names: list[str]):
        self.names = names
        self.i = 0
        self.t0 = time.time()

    def start(self, name: str) -> None:
        self.i += 1
        elapsed = time.time() - self.t0
        if self.i > 1:
            per = elapsed / (self.i - 1)
            eta = f"  overall eta {_dur(per * (len(self.names) - self.i + 1))}"
        else:
            eta = ""
        print(f"\n[{self.i}/{len(self.names)}] {name}{eta}", flush=True)

    def skipped(self) -> None:
        self.i += 1
