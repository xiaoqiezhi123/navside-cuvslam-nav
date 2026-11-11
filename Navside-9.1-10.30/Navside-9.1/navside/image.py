from __future__ import annotations

import numpy as np


def resize_image(image: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    """Resize a 2D or 3D image with bilinear interpolation."""
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(f"resize_image expects 2D or 3D input, got shape={arr.shape}")

    in_height, in_width = arr.shape[:2]
    if in_height == out_height and in_width == out_width:
        return arr.copy()

    if in_height < 1 or in_width < 1:
        raise ValueError(f"resize_image got empty input shape={arr.shape}")

    arr_f = arr.astype(np.float32, copy=False)
    y = np.linspace(0.0, in_height - 1.0, out_height, dtype=np.float32)
    x = np.linspace(0.0, in_width - 1.0, out_width, dtype=np.float32)

    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_height - 1)
    x1 = np.minimum(x0 + 1, in_width - 1)

    wy = (y - y0).reshape(out_height, 1)
    wx = (x - x0).reshape(1, out_width)

    if arr.ndim == 2:
        top_left = arr_f[y0[:, None], x0[None, :]]
        top_right = arr_f[y0[:, None], x1[None, :]]
        bottom_left = arr_f[y1[:, None], x0[None, :]]
        bottom_right = arr_f[y1[:, None], x1[None, :]]
        out = (
            (1.0 - wy) * (1.0 - wx) * top_left
            + (1.0 - wy) * wx * top_right
            + wy * (1.0 - wx) * bottom_left
            + wy * wx * bottom_right
        )
        return out.astype(np.float32)

    top_left = arr_f[y0[:, None], x0[None, :], :]
    top_right = arr_f[y0[:, None], x1[None, :], :]
    bottom_left = arr_f[y1[:, None], x0[None, :], :]
    bottom_right = arr_f[y1[:, None], x1[None, :], :]
    out = (
        (1.0 - wy)[..., None] * (1.0 - wx)[..., None] * top_left
        + (1.0 - wy)[..., None] * wx[..., None] * top_right
        + wy[..., None] * (1.0 - wx)[..., None] * bottom_left
        + wy[..., None] * wx[..., None] * bottom_right
    )
    return out.astype(np.float32)


def rgb_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"depth_to_uint8 expects a 2D depth image, got shape={arr.shape}")

    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    valid = arr[finite]
    dmin = float(valid.min())
    dmax = float(valid.max())
    if abs(dmax - dmin) < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)

    norm = np.clip((arr - dmin) / (dmax - dmin), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def depth_to_rgb_uint8(depth: np.ndarray) -> np.ndarray:
    gray = depth_to_uint8(depth)
    return np.repeat(gray[:, :, None], 3, axis=2)
