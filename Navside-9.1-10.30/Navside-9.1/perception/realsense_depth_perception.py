from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .zed_depth_perception import DepthPerceptionOutput


def depth_frame_to_meters(depth_frame, depth_scale: float = 0.001) -> np.ndarray:
    """Convert a RealSense Z16 depth frame to float32 meters."""
    data = depth_frame.get_data()
    if data is None:
        raise RuntimeError("RealSense depth frame data is empty")
    depth = np.asanyarray(data)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32) * float(depth_scale)


# ---------------------------------------------------------------------------
#  YAML config helpers (ported from view_d455f_depth.py)
# ---------------------------------------------------------------------------


def _load_realsense_config(path: str) -> Optional[dict]:
    """Load a RealSense camera YAML config file, returning None on failure."""
    p = Path(path)
    if not p.is_file():
        print(f"[RealSense] config file not found: {path}, using hardcoded defaults", file=sys.stderr)
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        print(f"[RealSense] failed to load config {path}: {exc}, using hardcoded defaults", file=sys.stderr)
        return None
    if not isinstance(cfg, dict):
        print(f"[RealSense] config root must be a mapping, using hardcoded defaults", file=sys.stderr)
        return None
    return cfg


def _coerce_option_value(name: str, value) -> float:
    """Convert a YAML value to the float that ``rs.option`` expects."""
    import pyrealsense2 as rs

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if name == "visual_preset" and isinstance(value, str):
        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        if not hasattr(rs.rs400_visual_preset, key):
            raise ValueError(f"Unknown visual_preset: {value!r}")
        return float(int(getattr(rs.rs400_visual_preset, key)))
    return float(value)


def _apply_options(sensor, label: str, options: dict) -> None:
    """Apply a dict of {option_name: value} to a RealSense sensor."""
    import pyrealsense2 as rs

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
            sensor.set_option(option, _coerce_option_value(name, value))
        except Exception as exc:
            print(f"WARN: failed to set {label}.{name}={value!r}: {exc}", file=sys.stderr)


def _apply_config(device, cfg: dict) -> None:
    """Apply depth-stream options from *cfg* to the RealSense device."""
    depth_options = (cfg.get("depth_stream") or {}).get("options") or {}

    for sensor in device.query_sensors():
        if sensor.is_depth_sensor():
            _apply_options(sensor, "depth_stream", depth_options)


def _build_post_processing_filters(cfg: dict) -> list:
    """Return an ordered list of ``(label, processing_block)`` tuples.

    Canonical D400 pipeline:
        disparity→ → spatial → temporal → disparity← → hole_filling
    """
    import pyrealsense2 as rs

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
#  RealSenseDepthPerception
# ---------------------------------------------------------------------------


class RealSenseDepthPerception:
    """RealSense depth camera grabber with the same interface as ZED."""

    RAW_HEIGHT = 720
    RAW_WIDTH = 1280
    DEFAULT_FPS = 30
    MIN_DEPTH_M = 0.10
    MAX_DEPTH_M = 10.0

    def __init__(
        self,
        width: int = RAW_WIDTH,
        height: int = RAW_HEIGHT,
        fps: int = DEFAULT_FPS,
        serial: Optional[str] = None,
        front_roi_height_ratio: float = 0.25,
        front_roi_width_ratio: float = 0.25,
        frame_timeout_s: float = 2.0,
        realsense_config_path: str = "",
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial = serial
        self.front_roi_height_ratio = front_roi_height_ratio
        self.front_roi_width_ratio = front_roi_width_ratio
        self.frame_timeout_s = float(frame_timeout_s)
        self.realsense_config_path = realsense_config_path

        self._rs = None
        self._pipeline = None
        self._profile = None
        self._depth_scale = 0.001
        self._pp_filters: list = []          # post-processing filters

    def start(self) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        if self._pipeline is not None:
            self.close()
        self._open_camera()

    def read(self) -> DepthPerceptionOutput:
        try:
            depth_raw = self._retrieve_depth()
        except Exception as exc:
            try:
                self._reopen_once()
            except Exception as reopen_exc:
                return DepthPerceptionOutput(
                    success=False,
                    depth_input=None,
                    depth_feature=None,
                    front_distance_m=0.0,
                    error=f"{exc}; reopen failed: {reopen_exc}",
                )
            return DepthPerceptionOutput(
                success=False,
                depth_input=None,
                depth_feature=None,
                front_distance_m=0.0,
                error=str(exc),
            )

        front_distance_m = self._compute_front_distance(depth_raw)
        return DepthPerceptionOutput(
            success=True,
            depth_input=depth_raw.astype(np.float32, copy=False),
            depth_feature=None,
            front_distance_m=front_distance_m,
        )

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._profile = None
        self._pp_filters = []

    def _open_camera(self) -> None:
        if self._rs is None:
            raise RuntimeError("pyrealsense2 is not initialized")

        rs = self._rs

        # --- load YAML config (optional) ---
        rs_cfg: Optional[dict] = None
        if self.realsense_config_path:
            rs_cfg = _load_realsense_config(self.realsense_config_path)

        # --- resolve stream resolution: YAML overrides constructor args ---
        depth_cfg = (rs_cfg or {}).get("depth_stream") or {}
        stream_width = depth_cfg.get("width", self.width)
        stream_height = depth_cfg.get("height", self.height)
        stream_fps = depth_cfg.get("fps", self.fps)

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.depth,
            int(stream_width),
            int(stream_height),
            rs.format.z16,
            int(stream_fps),
        )

        profile = pipeline.start(config)

        # --- apply hardware options from YAML ---
        device = profile.get_device()
        if rs_cfg is not None:
            _apply_config(device, rs_cfg)
            print(f"[RealSense] loaded config: {self.realsense_config_path}")
        else:
            # fallback: hardcoded recommended settings
            depth_sensor = device.first_depth_sensor()
            self._apply_recommended_depth_settings(depth_sensor)

        # --- read actual depth units from the sensor ---
        depth_sensor = device.first_depth_sensor()
        try:
            if depth_sensor.supports(rs.option.depth_units):
                self._depth_scale = float(depth_sensor.get_option(rs.option.depth_units))
        except Exception:
            self._depth_scale = 0.001

        # --- build post-processing filter chain ---
        if rs_cfg is not None:
            self._pp_filters = _build_post_processing_filters(rs_cfg)
            if self._pp_filters:
                labels = " -> ".join(label for label, _ in self._pp_filters)
                print(f"[RealSense] post-processing: {labels}")

        self._pipeline = pipeline
        self._profile = profile

    def _apply_recommended_depth_settings(self, depth_sensor) -> None:
        """Hardcoded fallback when no YAML config is provided."""
        rs = self._rs
        if rs is None:
            return

        preset_enum = getattr(rs, "rs400_visual_preset", None)
        default_preset = getattr(preset_enum, "default", None) if preset_enum else None
        if default_preset is not None:
            try:
                depth_sensor.set_option(
                    rs.option.visual_preset, float(default_preset)
                )
            except Exception:
                pass

        if depth_sensor.supports(rs.option.laser_power):
            try:
                laser_range = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power, laser_range.max)
            except Exception:
                pass

        if depth_sensor.supports(rs.option.emitter_enabled):
            try:
                depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
            except Exception:
                pass

        if depth_sensor.supports(rs.option.enable_auto_exposure):
            try:
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
            except Exception:
                pass

    def _retrieve_depth(self) -> np.ndarray:
        if self._rs is None or self._pipeline is None:
            raise RuntimeError("RealSense camera is not started")

        frames = self._wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError("RealSense depth frame is not available")

        # --- apply post-processing filters ---
        for _label, pp in self._pp_filters:
            depth_frame = pp.process(depth_frame)

        depth = depth_frame_to_meters(depth_frame, self._depth_scale)
        if depth.shape != (self.height, self.width):
            raise RuntimeError(
                f"unexpected RealSense depth shape: {depth.shape}, "
                f"expected {(self.height, self.width)}"
            )
        return depth

    def _wait_for_frames(self):
        if self._pipeline is None:
            raise RuntimeError("RealSense camera is not started")

        timeout_ms = int(self.frame_timeout_s * 1000.0)
        try:
            return self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except TypeError:
            return self._pipeline.wait_for_frames()
        except RuntimeError as exc:
            raise TimeoutError(f"RealSense frame timeout: {exc}") from exc

    def _compute_front_distance(self, depth: np.ndarray) -> float:
        height, width = depth.shape[:2]
        roi_h = max(1, int(height * self.front_roi_height_ratio))
        roi_w = max(1, int(width * self.front_roi_width_ratio))
        y0 = (height - roi_h) // 2
        x0 = (width - roi_w) // 2
        roi = depth[y0 : y0 + roi_h, x0 : x0 + roi_w]
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if valid.size == 0:
            return 0.0
        return float(np.percentile(valid, 10))

    def _reopen_once(self) -> None:
        self.close()
        self._open_camera()
