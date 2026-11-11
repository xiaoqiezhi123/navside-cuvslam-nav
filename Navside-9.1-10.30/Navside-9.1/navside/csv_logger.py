from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np


CSV_FIELDS = [
    "timestamp",
    "control_mode",
    "tick_seq",
    "lin_vel_x",
    "lin_vel_y",
    "lin_vel_z",
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "target_obs_x",
    "target_obs_y",
    "target_obs_z",
    "target_obs_log_dist",
    "depth_front_m",
    "depth_valid_pct",
    "raw_vx",
    "raw_wz",
    "final_vx",
    "final_wz",
    "goal_dist",
    "zero_reason",
]


def _format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _vector_value(value, index: int):
    if value is None:
        return ""
    arr = np.asarray(value).reshape(-1)
    if arr.size <= index:
        return ""
    return f"{float(arr[index]):.4f}"


def _scalar(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


class CsvTickLogger:
    """Per-run policy tick CSV logger.

    Creates one file per run and appends one row per policy tick.
    """

    def __init__(self, log_dir, mode: str = "sim", enabled: bool = True):
        self.enabled = bool(enabled)
        self.mode = str(mode).replace("/", "_").replace("\\", "_")
        self.path = None
        self._file = None
        self._writer = None
        self._next_seq = 1

        if not self.enabled:
            return

        try:
            base = Path(log_dir)
            base.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.mode}_{stamp}_{os.getpid()}.csv"
            self.path = base / filename
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=CSV_FIELDS,
                extrasaction="ignore",
            )
            self._writer.writeheader()
            self._file.flush()
            print(f"[CsvLogger] {self.path}")
        except Exception as exc:
            print(f"[CsvLogger] disabled: {exc}")
            self.close()

    def write_tick(
        self,
        timestamp=None,
        control_mode: str = "",
        lin_vel=None,
        ang_vel=None,
        gravity=None,
        target_obs=None,
        depth_front_m=None,
        depth_valid_pct=None,
        raw_cmd=None,
        final_cmd=None,
        goal_dist=None,
        zero_reason: str = "",
    ) -> None:
        if not self.enabled or self._writer is None:
            return

        now = time.time() if timestamp is None else float(timestamp)
        row = {
            "timestamp": _format_timestamp(now),
            "control_mode": str(control_mode),
            "tick_seq": self._next_seq,
            "lin_vel_x": _vector_value(lin_vel, 0),
            "lin_vel_y": _vector_value(lin_vel, 1),
            "lin_vel_z": _vector_value(lin_vel, 2),
            "ang_vel_x": _vector_value(ang_vel, 0),
            "ang_vel_y": _vector_value(ang_vel, 1),
            "ang_vel_z": _vector_value(ang_vel, 2),
            "gravity_x": _vector_value(gravity, 0),
            "gravity_y": _vector_value(gravity, 1),
            "gravity_z": _vector_value(gravity, 2),
            "target_obs_x": _vector_value(target_obs, 0),
            "target_obs_y": _vector_value(target_obs, 1),
            "target_obs_z": _vector_value(target_obs, 2),
            "target_obs_log_dist": _vector_value(target_obs, 3),
            "depth_front_m": _scalar(depth_front_m),
            "depth_valid_pct": _scalar(depth_valid_pct),
            "raw_vx": _vector_value(raw_cmd, 0),
            "raw_wz": _vector_value(raw_cmd, 2),
            "final_vx": _vector_value(final_cmd, 0),
            "final_wz": _vector_value(final_cmd, 2),
            "goal_dist": _scalar(goal_dist),
            "zero_reason": str(zero_reason),
        }

        try:
            self._writer.writerow(row)
            self._file.flush()
            self._next_seq += 1
        except Exception as exc:
            print(f"[CsvLogger] write error: {exc}")
            self.close()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._file = None
        self._writer = None


__all__ = ["CSV_FIELDS", "CsvTickLogger"]
