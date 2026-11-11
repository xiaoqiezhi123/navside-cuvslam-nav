#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

import numpy as np


def _reexec_into_repo_venv() -> None:
    if os.environ.get("NAVSIDE_SKIP_VENV_REEXEC") == "1":
        return

    script_root = Path(__file__).resolve().parents[1]
    workspace_root = script_root.parent
    candidates = (
        workspace_root / ".venv_navside" / "bin" / "python",
        script_root / ".venv_navside" / "bin" / "python",
        Path("/home/amov/nav_arm_mujoco/.venv_navside/bin/python"),
    )
    venv_python = None
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            venv_python = candidate
            break
    if venv_python is None:
        return

    current_prefix = Path(sys.prefix).resolve()
    venv_root = venv_python.parents[1].resolve()
    if current_prefix == venv_root:
        return

    os.environ["NAVSIDE_SKIP_VENV_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_into_repo_venv()

import pyrealsense2 as rs


FRONT_ROI_HEIGHT_RATIO = 0.25
FRONT_ROI_WIDTH_RATIO = 0.25


def _depth_stats(depth: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(depth) & (depth > 0.01)
    zero = depth == 0
    valid_pct = 100.0 * float(valid.sum()) / float(depth.size)
    zero_pct = 100.0 * float(zero.sum()) / float(depth.size)

    height, width = depth.shape[:2]
    roi_h = max(1, int(height * FRONT_ROI_HEIGHT_RATIO))
    roi_w = max(1, int(width * FRONT_ROI_WIDTH_RATIO))
    y0 = (height - roi_h) // 2
    x0 = (width - roi_w) // 2
    roi = depth[y0 : y0 + roi_h, x0 : x0 + roi_w]
    roi_valid = roi[np.isfinite(roi) & (roi > 0.0)]
    front = 0.0 if roi_valid.size == 0 else float(np.percentile(roi_valid, 10))
    return valid_pct, zero_pct, front


def _collect_stats(pipeline) -> tuple[float, float, float]:
    pcts = []
    zeros = []
    fronts = []
    for _ in range(5):
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        depth_frame = frames.get_depth_frame()
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001
        valid_pct, zero_pct, front = _depth_stats(depth)
        pcts.append(valid_pct)
        zeros.append(zero_pct)
        fronts.append(front)
    return (
        float(np.mean(pcts)),
        float(np.mean(zeros)),
        float(np.mean(fronts)),
    )


def main() -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    sensor = profile.get_device().first_depth_sensor()

    def get_opt(option):
        if sensor.supports(option):
            return sensor.get_option(option)
        return None

    def set_opt(option, value):
        if sensor.supports(option):
            sensor.set_option(option, value)

    def opt_name(option):
        try:
            return str(option)
        except Exception:
            return str(option.value)

    print("baseline options:")
    for option in (
        rs.option.visual_preset,
        rs.option.laser_power,
        rs.option.emitter_enabled,
        rs.option.enable_auto_exposure,
        rs.option.exposure,
        rs.option.gain,
    ):
        if sensor.supports(option):
            print(f"  {opt_name(option)} = {sensor.get_option(option)}")

    for _ in range(10):
        pipeline.wait_for_frames(timeout_ms=5000)
    valid_pct, zero_pct, front = _collect_stats(pipeline)
    print(
        f"baseline: valid_pct={valid_pct:.2f}% zero_pct={zero_pct:.2f}% "
        f"front={front:.3f}m"
    )

    preset_candidates = []
    for enum_name in ("rs400_visual_preset", "visual_preset", "l500_visual_preset"):
        preset_enum = getattr(rs, enum_name, None)
        if preset_enum is None:
            continue
        for name in dir(preset_enum):
            if name.startswith("_"):
                continue
            preset_candidates.append((f"{enum_name}.{name}", getattr(preset_enum, name)))

    if not preset_candidates and sensor.supports(rs.option.visual_preset):
        option_range = sensor.get_option_range(rs.option.visual_preset)
        start = int(option_range.min)
        end = int(option_range.max)
        step = max(1, int(option_range.step))
        preset_candidates = [
            (f"value_{value}", float(value))
            for value in range(start, end + 1, step)
        ]

    for preset_name, preset_value in preset_candidates:
        try:
            set_opt(rs.option.visual_preset, preset_value)
        except Exception as exc:
            print(f"preset {preset_name}: skip ({exc})")
            continue

        if sensor.supports(rs.option.laser_power):
            try:
                laser_range = sensor.get_option_range(rs.option.laser_power)
                set_opt(rs.option.laser_power, laser_range.max)
            except Exception as exc:
                print(f"preset {preset_name}: laser_power skip ({exc})")
        set_opt(rs.option.emitter_enabled, 1)

        for _ in range(10):
            pipeline.wait_for_frames(timeout_ms=5000)
        valid_pct, zero_pct, front = _collect_stats(pipeline)
        print(
            f"preset {preset_name}: valid_pct={valid_pct:.2f}% "
            f"zero_pct={zero_pct:.2f}% front={front:.3f}m "
            f"laser_power={get_opt(rs.option.laser_power)}"
        )

    custom_preset = None
    for enum_name in ("rs400_visual_preset", "visual_preset", "l500_visual_preset"):
        preset_enum = getattr(rs, enum_name, None)
        if preset_enum is not None and hasattr(preset_enum, "custom"):
            custom_preset = getattr(preset_enum, "custom")
            break
    if custom_preset is not None:
        try:
            set_opt(rs.option.visual_preset, custom_preset)
        except Exception:
            pass

    if sensor.supports(rs.option.enable_auto_exposure):
        print("exposure sweep (auto exposure off):")
        set_opt(rs.option.enable_auto_exposure, 0)
        for exposure in (100, 500, 1000, 3000, 10000, 20000, 33000):
            set_opt(rs.option.exposure, float(exposure))
            for _ in range(5):
                pipeline.wait_for_frames(timeout_ms=5000)
            valid_pct, zero_pct, front = _collect_stats(pipeline)
            print(
                f"  exposure={exposure}: valid_pct={valid_pct:.2f}% "
                f"zero_pct={zero_pct:.2f}% front={front:.3f}m"
            )
        set_opt(rs.option.enable_auto_exposure, 1)

    if sensor.supports(rs.option.laser_power):
        print("laser power sweep:")
        for laser_power in (0, 50, 100, 150, 250, 360):
            set_opt(rs.option.laser_power, float(laser_power))
            for _ in range(5):
                pipeline.wait_for_frames(timeout_ms=5000)
            valid_pct, zero_pct, front = _collect_stats(pipeline)
            print(
                f"  laser_power={laser_power}: valid_pct={valid_pct:.2f}% "
                f"zero_pct={zero_pct:.2f}% front={front:.3f}m"
            )
        set_opt(rs.option.laser_power, 150.0)

    if sensor.supports(rs.option.confidence_threshold):
        option_range = sensor.get_option_range(rs.option.confidence_threshold)
        start = int(option_range.min)
        end = int(option_range.max)
        print("confidence threshold sweep:")
        for confidence in range(start, end + 1):
            set_opt(rs.option.confidence_threshold, float(confidence))
            for _ in range(5):
                pipeline.wait_for_frames(timeout_ms=5000)
            valid_pct, zero_pct, front = _collect_stats(pipeline)
            print(
                f"  confidence={confidence}: valid_pct={valid_pct:.2f}% "
                f"zero_pct={zero_pct:.2f}% front={front:.3f}m"
            )

    pipeline.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
