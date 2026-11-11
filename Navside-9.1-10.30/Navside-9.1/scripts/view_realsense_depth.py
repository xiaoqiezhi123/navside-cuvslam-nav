#!/usr/bin/env python3
"""Open an Intel RealSense D455F camera and display its depth image.

Loads realsence_config.yaml for camera hardware options and post-processing
filters (same config used by the NavSide runtime).
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


def _reexec_into_repo_venv() -> None:
    """Run under the repo-local venv if it exists."""
    if os.environ.get("NAVSIDE_SKIP_VENV_REEXEC") == "1":
        return

    script_root = Path(__file__).resolve().parents[1]
    workspace_root = script_root.parent
    candidates = (
        workspace_root / ".venv_navside" / "bin" / "python",
        script_root / ".venv_navside" / "bin" / "python",
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "realsence_config.yaml"
GRAY_MIN_M = 0.2
GRAY_MAX_M = 3.0
DEFAULT_DEPTH_UNITS = 0.001


# ---------------------------------------------------------------------------
#  Config helpers
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    """Load a RealSense YAML config file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    return cfg


def coerce_option_value(name: str, value) -> float:
    """Convert a YAML value to the float that ``rs.option`` expects."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if name == "visual_preset" and isinstance(value, str):
        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        if not hasattr(rs.rs400_visual_preset, key):
            raise ValueError(f"Unknown visual_preset: {value!r}")
        return float(int(getattr(rs.rs400_visual_preset, key)))
    return float(value)


def apply_options(sensor, label: str, options: dict) -> None:
    """Apply a dict of {option_name: value} to a RealSense sensor."""
    if not isinstance(options, dict):
        return

    ordered_names = []
    if "visual_preset" in options:
        ordered_names.append("visual_preset")
    ordered_names.extend(name for name in options if name != "visual_preset")

    for name in ordered_names:
        value = options[name]
        option = getattr(rs.option, name, None)
        if option is None:
            print(f"WARN: unknown option {label}.{name}; skipped", file=sys.stderr)
            continue
        if not sensor.supports(option):
            print(f"WARN: {label}.{name} is not supported; skipped", file=sys.stderr)
            continue
        try:
            sensor.set_option(option, coerce_option_value(name, value))
        except Exception as exc:
            print(f"WARN: failed to set {label}.{name}={value!r}: {exc}", file=sys.stderr)


def apply_config(device, cfg: dict) -> None:
    """Apply depth-stream options from *cfg* to the RealSense device."""
    depth_options = (cfg.get("depth_stream") or {}).get("options") or {}
    for sensor in device.query_sensors():
        if sensor.is_depth_sensor():
            apply_options(sensor, "depth_stream", depth_options)


def build_post_processing_filters(cfg: dict) -> list:
    """Return an ordered list of ``(label, processing_block)`` tuples.

    Canonical D400 pipeline:
        disparity→ → spatial → temporal → disparity← → hole_filling
    """
    pp = cfg.get("post_processing") or {}
    if not pp.get("enabled", False):
        return []

    filters: list = []
    spat = pp.get("spatial") or {}
    temp = pp.get("temporal") or {}
    hf_cfg = pp.get("hole_filling") or {}

    use_disparity = spat.get("enabled") or temp.get("enabled")

    if use_disparity:
        filters.append(("disparity→", rs.disparity_transform(True)))

    if spat.get("enabled", False):
        f = rs.spatial_filter()
        f.set_option(rs.option.filter_magnitude, 2)
        f.set_option(rs.option.filter_smooth_alpha, float(spat.get("smooth_alpha", 0.5)))
        f.set_option(rs.option.filter_smooth_delta, float(spat.get("smooth_delta", 20)))
        f.set_option(rs.option.holes_fill, int(spat.get("holes_fill", 0)))
        filters.append(("spatial", f))

    if temp.get("enabled", False):
        f = rs.temporal_filter()
        f.set_option(rs.option.filter_smooth_alpha, float(temp.get("smooth_alpha", 0.4)))
        f.set_option(rs.option.filter_smooth_delta, float(temp.get("smooth_delta", 20)))
        filters.append(("temporal", f))

    if use_disparity:
        filters.append(("disparity←", rs.disparity_transform(False)))

    if hf_cfg.get("enabled", False):
        f = rs.hole_filling_filter()
        mode = int(hf_cfg.get("holes_fill", 1))
        f.set_option(rs.option.holes_fill, 1 if mode == 1 else 2)
        filters.append(("hole_filling", f))

    return filters


# ---------------------------------------------------------------------------
#  Display
# ---------------------------------------------------------------------------


def make_gray_depth(depth_m: np.ndarray) -> np.ndarray:
    """Map depth in meters to grayscale, marking invalid pixels in red."""
    gray = np.clip((GRAY_MAX_M - depth_m) / (GRAY_MAX_M - GRAY_MIN_M), 0.0, 1.0)
    gray = (gray * 255.0).astype(np.uint8)

    invalid = depth_m == 0.0
    gray[invalid] = 0
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    display[invalid] = (0, 0, 255)
    return display


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display live depth from a RealSense D455F camera."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="path to RealSense YAML config file (default: config/realsence_config.yaml)",
    )
    parser.add_argument("--width", type=int, default=None, help="depth stream width (overrides config)")
    parser.add_argument("--height", type=int, default=None, help="depth stream height (overrides config)")
    parser.add_argument("--fps", type=int, default=None, help="depth stream frame rate (overrides config)")
    parser.add_argument(
        "--gray",
        action="store_true",
        help="show depth as grayscale; invalid depth is red",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # --- load config ---
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 1

    depth_cfg = cfg.get("depth_stream") or {}

    depth_width = args.width if args.width is not None else depth_cfg.get("width", 640)
    depth_height = args.height if args.height is not None else depth_cfg.get("height", 480)
    depth_fps = args.fps if args.fps is not None else depth_cfg.get("fps", 30)

    # --- start pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)

    colorizer = rs.colorizer()

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        print(f"Failed to start camera: {exc}", file=sys.stderr)
        print(
            "Make sure the D455F is connected and the RealSense udev rules are installed.",
            file=sys.stderr,
        )
        return 1

    device = profile.get_device()
    apply_config(device, cfg)

    depth_sensor = device.first_depth_sensor()
    if depth_sensor.supports(rs.option.depth_units):
        depth_units = depth_sensor.get_option(rs.option.depth_units)
    else:
        depth_units = DEFAULT_DEPTH_UNITS

    print(f"Loaded config: {args.config}")
    print(f"Started camera: {device.get_info(rs.camera_info.name)}")
    print(f"Depth stream: {depth_width}x{depth_height} @ {depth_fps} fps")

    pp_filters = build_post_processing_filters(cfg)
    if pp_filters:
        labels = " -> ".join(label for label, _ in pp_filters)
        print(f"Post-processing: {labels}")
    else:
        print("Post-processing: disabled")

    print("Press Q or Esc to quit.")

    # --- main loop ---
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            # apply post-processing filters
            for _label, pp in pp_filters:
                depth_frame = pp.process(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_m = depth_image.astype(np.float32) * depth_units

            if args.gray:
                display = make_gray_depth(depth_m)
            else:
                display = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # Overlay: center-pixel distance and valid-pixel ratio
            h, w = depth_image.shape
            center_depth = depth_image[h // 2, w // 2]
            if center_depth == 0:
                distance_text = "center: invalid"
            else:
                distance_text = f"center: {center_depth * depth_units:.3f} m"

            valid_ratio = float(np.count_nonzero(depth_image)) / depth_image.size * 100.0
            valid_text = f"valid: {valid_ratio:.1f}%"

            cv2.putText(
                display,
                distance_text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                valid_text,
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("D455F Depth", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
