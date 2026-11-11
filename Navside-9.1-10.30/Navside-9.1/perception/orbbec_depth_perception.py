from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from .zed_depth_perception import DepthPerceptionOutput


class OrbbecDepthPerception:
    """Orbbec (Gemini 336L) depth grabber — same interface as ZED/RealSense.

    Depth stream is Y16 (raw uint16, in millimeters). Hole filling is applied
    by default (the Gemini 336L benefits strongly from it). The other Orbbec
    filters are opt-in: the generic NoiseRemovalFilter is destructive on this
    camera (kills most valid pixels) and stays off by default.
    """

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
        front_roi_height_ratio: float = 0.25,
        front_roi_width_ratio: float = 0.25,
        frame_timeout_ms: int = 2000,
        enable_hole_filling: bool = True,
        enable_noise_removal: bool = False,
        enable_spatial_filter: bool = False,
        enable_temporal_filter: bool = False,
        serial: Optional[str] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.front_roi_height_ratio = front_roi_height_ratio
        self.front_roi_width_ratio = front_roi_width_ratio
        self.frame_timeout_ms = int(frame_timeout_ms)

        self.enable_hole_filling = enable_hole_filling
        self.enable_noise_removal = enable_noise_removal
        self.enable_spatial_filter = enable_spatial_filter
        self.enable_temporal_filter = enable_temporal_filter

        # 双相机方案：按序列号绑定「相机 B」（深度感知，激光开）。留空 = 第一台（单相机/旧行为）。
        # 优先用显式入参，否则读环境变量 NAVSIDE_CAMERA_SERIAL。
        self.serial = serial or os.environ.get("NAVSIDE_CAMERA_SERIAL", "") or None

        self._ob = None
        self._pipeline = None
        self._filters: list = []

    def start(self) -> None:
        import pyorbbecsdk as ob

        self._ob = ob
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
                    success=False, depth_input=None, depth_feature=None,
                    front_distance_m=0.0,
                    error=f"{exc}; reopen failed: {reopen_exc}",
                )
            return DepthPerceptionOutput(
                success=False, depth_input=None, depth_feature=None,
                front_distance_m=0.0, error=str(exc),
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
        for f in self._filters:
            try:
                f.reset()
            except Exception:
                pass
        self._filters = []
        self._pipeline = None

    def _open_camera(self) -> None:
        if self._ob is None:
            raise RuntimeError("pyorbbecsdk is not initialized")

        ob = self._ob
        # 保持 Context 引用：2.1.2 SDK 的 DeviceList 持有 deviceMgr 裸指针，
        # 临时 ob.Context() 一旦被 GC 就悬空，导致 get_device_by_serial_number
        # 抛 "NULL pointer passed for argument deviceMgr"。
        self._ctx = ob.Context()
        devices = self._ctx.query_devices()
        if self.serial:
            device = devices.get_device_by_serial_number(self.serial)
            if device is None:
                raise RuntimeError(
                    f"未找到序列号 {self.serial} 的相机（当前已连接 {devices.get_count()} 台）"
                )
        else:
            device = devices.get_device_by_index(0)
            if device is None:
                raise RuntimeError("未找到任何 Orbbec 相机")
        pipeline = ob.Pipeline(device)

        # 激光功率取 5：满功率(等级 6)会让 Gemini 336L 的 896mA 电流在弱供电 hub 下
        # 压降 → USB 掉线(setXu failed / 内核日志 USB disconnect 重枚举)。
        # 功率 1~5 实测稳定，其中 5 的深度覆盖率最高(78% 有效像素、最远 7.5m)。
        # 部分固件不支持写该属性时忽略。
        try:
            device.set_int_property(
                ob.OBPropertyID.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT, 5
            )
        except Exception:
            pass

        config = ob.Config()

        profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_video_stream_profile(
            self.width, self.height, ob.OBFormat.Y16, self.fps
        )
        if depth_profile is None:
            raise RuntimeError(
                f"no Orbbec depth profile {self.width}x{self.height} "
                f"@ {self.fps}fps Y16 available"
            )

        config.enable_stream(depth_profile)
        pipeline.start(config)
        self._pipeline = pipeline
        self._filters = self._build_filters()

    def _build_filters(self) -> list:
        """Build the depth post-processing filter chain.

        Order: noise removal -> hole filling -> spatial -> temporal.
        Each filter is best-effort: if the SDK variant lacks one, we skip it
        rather than failing the whole backend.
        """
        ob = self._ob
        filters: list = []

        if self.enable_noise_removal:
            # 注意: 此滤波在 Gemini 336L 上很激进(min_diff 过小会误删大量有效深度),
            # 默认关闭。若确实要开,请用更大的 min_diff 并在真实场景下调参。
            try:
                f = ob.FilterFactory.create_filter("NoiseRemovalFilter")
                f.set_config_value("max_size", 5)
                f.set_config_value("min_diff", 50)
                filters.append(f)
            except Exception as exc:
                print(f"[Orbbec] NoiseRemovalFilter unavailable: {exc}", file=sys.stderr)

        if self.enable_hole_filling:
            try:
                f = ob.FilterFactory.create_filter("HoleFillingFilter")
                f.set_config_value("hole_filling_mode", 2)  # 填洞模式
                filters.append(f)
            except Exception as exc:
                print(f"[Orbbec] HoleFillingFilter unavailable: {exc}", file=sys.stderr)

        if self.enable_spatial_filter:
            try:
                f = ob.FilterFactory.create_filter("SpatialModerateFilter")
                f.set_config_value("magnitude", 2)     # 平滑强度 1-3
                filters.append(f)
            except Exception as exc:
                print(f"[Orbbec] SpatialModerateFilter unavailable: {exc}", file=sys.stderr)

        if self.enable_temporal_filter:
            try:
                f = ob.FilterFactory.create_filter("TemporalFilter")
                f.set_config_value("diff_scale", 0.5)  # 帧间差异缩放
                f.set_config_value("weight", 0.5)      # 历史帧权重(越小越平滑)
                filters.append(f)
            except Exception as exc:
                print(f"[Orbbec] TemporalFilter unavailable: {exc}", file=sys.stderr)

        if filters:
            print(f"[Orbbec] depth filter chain: {[f.get_name() for f in filters]}")
        return filters

    def _retrieve_depth(self) -> np.ndarray:
        if self._ob is None or self._pipeline is None:
            raise RuntimeError("Orbbec camera is not started")

        frames = self._pipeline.wait_for_frames(self.frame_timeout_ms)
        if frames is None:
            raise TimeoutError("Orbbec frame timeout")

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError("Orbbec depth frame is not available")

        # --- apply post-processing filter chain ---
        frame = depth_frame
        for f in self._filters:
            frame = f.process(frame)

        height = frame.get_height()
        width = frame.get_width()
        raw = np.frombuffer(frame.get_data(), dtype=np.uint16)
        expected = height * width
        if raw.size < expected:
            raise RuntimeError(f"Orbbec depth data size {raw.size} < {expected}")
        depth = raw[:expected].reshape((height, width)).astype(np.float32)

        # Y16: raw value * depth_scale = millimeters (default scale is 1.0)
        depth_scale = float(frame.get_depth_scale())
        depth_m = depth * depth_scale / 1000.0

        # 0 and 65535 are "no depth" sentinels
        depth_m[(depth == 0) | (depth == 65535)] = 0.0
        return depth_m

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
