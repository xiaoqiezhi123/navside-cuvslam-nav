from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthPerceptionOutput:
    success: bool
    depth_input: Optional[np.ndarray]
    depth_feature: Optional[np.ndarray]
    front_distance_m: float
    error: str = ""


class ZedDepthPerception:
    """ZED camera depth grabber — returns raw depth, no VAE encoding.

    Preprocessing and VAE encoding are handled by the adapter
    (SruNavAdapter.depth_preprocess) so that MuJoCo simulation and
    the real robot use identical preprocessing.
    """
    RAW_HEIGHT = 720
    RAW_WIDTH = 1280
    MIN_DEPTH_M = 0.10
    MAX_DEPTH_M = 10.0

    def __init__(
        self,
        front_roi_height_ratio: float = 0.25,
        front_roi_width_ratio: float = 0.25,
    ) -> None:
        self.front_roi_height_ratio = front_roi_height_ratio
        self.front_roi_width_ratio = front_roi_width_ratio

        self._sl = None
        self._zed = None
        self._runtime_params = None
        self._depth_mat = None

    def start(self) -> None:
        import pyzed.sl as sl
        self._sl = sl
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
        if self._zed is not None:
            self._zed.close()
        self._zed = None
        self._runtime_params = None
        self._depth_mat = None

    def _open_camera(self) -> None:
        if self._sl is None:
            raise RuntimeError("pyzed is not initialized")

        sl = self._sl
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

        zed = sl.Camera()
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"failed to open ZED camera: {status}")

        self._zed = zed
        self._runtime_params = sl.RuntimeParameters()
        self._depth_mat = sl.Mat()

    def _retrieve_depth(self) -> np.ndarray:
        if self._sl is None or self._zed is None or self._runtime_params is None:
            raise RuntimeError("ZED camera is not started")
        if self._depth_mat is None:
            raise RuntimeError("ZED depth buffer is not initialized")

        status = self._zed.grab(self._runtime_params)
        if status != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"failed to grab ZED frame: {status}")

        status = self._zed.retrieve_measure(self._depth_mat, self._sl.MEASURE.DEPTH)
        if status != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"failed to retrieve ZED depth: {status}")

        depth = self._depth_mat.get_data()
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.shape != (self.RAW_HEIGHT, self.RAW_WIDTH):
            raise RuntimeError(
                f"unexpected ZED depth shape: {depth.shape}, "
                f"expected {(self.RAW_HEIGHT, self.RAW_WIDTH)}"
            )
        return depth

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
