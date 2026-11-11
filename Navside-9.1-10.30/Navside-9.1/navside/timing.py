import os
import time


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "") == "1"


def timing_enabled() -> bool:
    return _env_flag("NAVSIDE_TIMING") or _env_flag("SRU_TIMING")


def timing_log(stage: str, seconds: float) -> None:
    if timing_enabled():
        print(f"[NavSide TIMING] {stage}: {seconds:.6f}s")


class RateMeter:
    def __init__(self, name: str, report_interval_sec: float = 2.0):
        self.name = name
        self.report_interval_sec = float(report_interval_sec)
        self.enabled = timing_enabled()
        self.count = 0
        self._last_report_t = time.perf_counter()

    def tick(self) -> None:
        if not self.enabled:
            return
        self.count += 1
        now = time.perf_counter()
        elapsed = now - self._last_report_t
        if elapsed < self.report_interval_sec:
            return
        hz = self.count / max(elapsed, 1e-9)
        print(f"[NavSide PERF] {self.name}: {hz:.1f} Hz over {elapsed:.2f}s")
        self.count = 0
        self._last_report_t = now
