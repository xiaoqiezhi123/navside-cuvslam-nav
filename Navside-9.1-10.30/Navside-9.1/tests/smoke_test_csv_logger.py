#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nav_pkg = types.ModuleType("navside")
nav_pkg.__path__ = [str(ROOT / "navside")]
sys.modules["navside"] = nav_pkg

csv_logger_mod = _load_module("navside.csv_logger", ROOT / "navside" / "csv_logger.py")
_load_module("navside.image", ROOT / "navside" / "image.py")
depth_mod = _load_module("navside.depth", ROOT / "navside" / "depth.py")

CsvTickLogger = csv_logger_mod.CsvTickLogger
compute_depth_summary = depth_mod.compute_depth_summary


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = CsvTickLogger(tmp, mode="sim", enabled=True)
        assert logger.path is not None and logger.path.is_file()

        logger.write_tick(
            timestamp=123.456,
            control_mode="LOW_SPEED",
            lin_vel=np.array([0.1, 0.0, 0.0], dtype=np.float32),
            ang_vel=np.array([0.0, 0.0, 0.2], dtype=np.float32),
            gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            target_obs=np.array([0.5, 0.0, 0.0, 1.0], dtype=np.float32),
            depth_front_m=1.2,
            depth_valid_pct=80.0,
            raw_cmd=np.array([0.3, 0.0, 0.4], dtype=np.float32),
            final_cmd=np.array([0.3, 0.0, 0.4], dtype=np.float32),
            goal_dist=2.0,
            zero_reason="",
        )
        logger.write_tick(
            timestamp=123.856,
            control_mode="STANDBY",
            lin_vel=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ang_vel=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            target_obs=np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32),
            depth_front_m=0.0,
            depth_valid_pct=0.0,
            raw_cmd=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            final_cmd=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            goal_dist=0.5,
            zero_reason="standby",
        )
        logger.close()

        with logger.path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["control_mode"] == "LOW_SPEED"
        assert rows[0]["tick_seq"] == "1"
        assert rows[1]["tick_seq"] == "2"
        assert rows[0]["target_obs_log_dist"] == "1.0000"
        assert rows[0]["final_wz"] == "0.4000"
        assert rows[1]["zero_reason"] == "standby"

        front_dist, valid_pct = compute_depth_summary(
            np.full((4, 4), 2.0, dtype=np.float32)
        )
        assert front_dist == 2.0
        assert valid_pct == 100.0

    print("csv logger smoke passed")


if __name__ == "__main__":
    main()
