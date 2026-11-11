#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import time


def _capture_stdout(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def _load_timing_module():
    timing_path = Path(__file__).resolve().parents[1] / "navside" / "timing.py"
    spec = importlib.util.spec_from_file_location("navside_timing_test", timing_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load timing module from {timing_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    timing_mod = _load_timing_module()
    old_nav = os.environ.pop("NAVSIDE_TIMING", None)
    old_sru = os.environ.pop("SRU_TIMING", None)
    try:
        assert not timing_mod.timing_enabled()
        assert _capture_stdout(lambda: timing_mod.timing_log("disabled_case", 0.001)) == ""

        os.environ["NAVSIDE_TIMING"] = "1"
        assert timing_mod.timing_enabled()
        timing_text = _capture_stdout(lambda: timing_mod.timing_log("enabled_case", 0.123456))
        assert "[NavSide TIMING] enabled_case: 0.123456s" in timing_text

        meter = timing_mod.RateMeter("encoder_trt", report_interval_sec=0.0)
        perf_text = _capture_stdout(meter.tick)
        assert "[NavSide PERF] encoder_trt:" in perf_text

        print("timing module smoke passed")
    finally:
        if old_nav is None:
            os.environ.pop("NAVSIDE_TIMING", None)
        else:
            os.environ["NAVSIDE_TIMING"] = old_nav
        if old_sru is None:
            os.environ.pop("SRU_TIMING", None)
        else:
            os.environ["SRU_TIMING"] = old_sru


if __name__ == "__main__":
    main()
