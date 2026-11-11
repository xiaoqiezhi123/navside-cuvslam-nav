from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception import create_depth_perception
from perception.realsense_depth_perception import RealSenseDepthPerception
from perception.realsense_depth_perception import depth_frame_to_meters
from navside.adapter import SruNavAdapter


def test_depth_conversion() -> None:
    raw = np.array(
        [
            [1000, 0],
            [2000, 65535],
        ],
        dtype=np.uint16,
    )
    frame = SimpleNamespace(get_data=lambda: raw)
    meters = depth_frame_to_meters(frame, depth_scale=0.001)
    assert meters.shape == (2, 2)
    assert meters.dtype == np.float32
    np.testing.assert_allclose(
        meters,
        raw.astype(np.float32) * 0.001,
    )


def test_factory_backends() -> None:
    realsense = create_depth_perception("realsense")
    zed = create_depth_perception("zed")
    assert isinstance(realsense, RealSenseDepthPerception)
    assert zed.__class__.__name__ == "ZedDepthPerception"


def test_realsense_crop_shape() -> None:
    depth = np.zeros((720, 1280), dtype=np.float32)
    cropped = SruNavAdapter._center_crop_depth(
        None, depth, target_width=1152, target_height=720
    )
    assert cropped.shape == (720, 1152)


if __name__ == "__main__":
    test_depth_conversion()
    test_factory_backends()
    test_realsense_crop_shape()
    print("realsense depth smoke passed")
