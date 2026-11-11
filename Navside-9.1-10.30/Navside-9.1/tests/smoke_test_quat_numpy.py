#!/usr/bin/env python3
"""NumPy-only quaternion rotation smoke test."""

from __future__ import annotations

import numpy as np

from navside.adapter import SruNavAdapter


def assert_close(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-5) -> None:
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rot_identity = SruNavAdapter._rotation_matrix_from_quat_wxyz(identity)
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert_close(rot_identity @ vec, vec)

    yaw_90 = np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)], dtype=np.float32)
    rot_yaw_90 = SruNavAdapter._rotation_matrix_from_quat_wxyz(yaw_90)
    rotated = rot_yaw_90 @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert_close(rotated, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-4)

    print("quaternion numpy smoke passed")


if __name__ == "__main__":
    main()
