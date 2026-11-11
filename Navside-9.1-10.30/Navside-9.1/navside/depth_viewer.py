from __future__ import annotations

import time
from typing import Optional

import numpy as np


MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 10.0


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    height, width = arr.shape[:2]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    invalid = (
        ~np.isfinite(arr)
        | (arr <= 0.01)
        | (arr > MAX_DEPTH_M)
    )
    clipped = np.clip(arr, MIN_DEPTH_M, MAX_DEPTH_M)
    norm = np.log(clipped / MIN_DEPTH_M) / np.log(
        MAX_DEPTH_M / MIN_DEPTH_M
    )
    gray = ((1.0 - norm) * 255.0).astype(np.uint8)
    rgb[..., 0] = gray
    rgb[..., 1] = gray
    rgb[..., 2] = gray
    rgb[invalid] = (255, 0, 255)
    return rgb


def _depth_to_rgb_raw(depth: np.ndarray) -> np.ndarray:
    """Linear adaptive depth-to-RGB, no fixed range clipping.

    Near = white, far = black, invalid = red.
    """
    arr = np.asarray(depth, dtype=np.float32)
    height, width = arr.shape[:2]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    valid_mask = np.isfinite(arr) & (arr > 0.0)
    if np.any(valid_mask):
        valid = arr[valid_mask]
        dmin = float(valid.min())
        dmax = float(valid.max())
        span = dmax - dmin
        if span < 1e-6:
            norm = np.zeros_like(arr)
        else:
            norm = np.clip((arr - dmin) / span, 0.0, 1.0)
    else:
        norm = np.zeros_like(arr)

    gray = ((1.0 - norm) * 255.0).astype(np.uint8)
    rgb[..., 0] = gray
    rgb[..., 1] = gray
    rgb[..., 2] = gray
    rgb[~valid_mask] = (255, 0, 0)
    return rgb


def _ppm_bytes(rgb: np.ndarray) -> bytes:
    height, width = rgb.shape[:2]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + rgb.tobytes()


class DepthViewer:
    """Tkinter viewer for raw RealSense depth, without OpenCV or Pillow."""

    def __init__(self, display_scale: float = 0.5, show_raw: bool = False) -> None:
        import tkinter as tk

        self._tk = tk
        self._scale = float(display_scale)
        self._show_raw = bool(show_raw)
        self._photo = None
        self._photo_raw = None
        self._last_fps_time = time.perf_counter()
        self._fps_frame_count = 0
        self._fps = 0.0

        self._root = tk.Tk()
        title = "RealSense Depth"
        if self._show_raw:
            title += " — processed (L) + raw (R)"
        self._root.title(title)

        if self._show_raw:
            frame = tk.Frame(self._root)
            frame.pack(fill="both", expand=True)
            self._canvas = tk.Canvas(
                frame, bg="black", highlightthickness=0,
            )
            self._canvas.pack(side="left", fill="both", expand=True)
            self._canvas_raw = tk.Canvas(
                frame, bg="#1a1a1a", highlightthickness=0,
            )
            self._canvas_raw.pack(side="right", fill="both", expand=True)
        else:
            self._canvas = tk.Canvas(
                self._root, bg="black", highlightthickness=0,
            )
            self._canvas.pack()
            self._canvas_raw = None

        self._root.update()

    def update(
        self,
        depth: Optional[np.ndarray],
        crop_width: int,
        crop_height: int,
        front_m: float,
        valid_pct: float,
        backend: str = "realsense",
    ) -> None:
        if depth is None:
            self._update_status(
                "no depth frame", crop_width, crop_height, front_m, valid_pct, backend
            )
            return

        rgb = _depth_to_rgb(depth)
        photo = self._tk.PhotoImage(data=_ppm_bytes(rgb))
        if self._scale < 1.0:
            subsample = max(1, int(round(1.0 / self._scale)))
            photo = photo.subsample(subsample, subsample)

        height, width = rgb.shape[:2]
        scale = self._scale if self._scale >= 1.0 else 1.0 / max(
            1, int(round(1.0 / self._scale))
        )
        display_w = max(1, int(round(width * scale)))
        display_h = max(1, int(round(height * scale)))
        self._canvas.configure(width=display_w, height=display_h)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=photo)

        x0 = max(0, (width - crop_width) // 2) * scale
        y0 = max(0, (height - crop_height) // 2) * scale
        x1 = min(width, x0 + crop_width * scale)
        y1 = min(height, y0 + crop_height * scale)
        self._canvas.create_rectangle(
            x0, y0, x1, y1, outline="#00ff00", width=2
        )

        self._photo = photo
        status_text = self._update_status(
            f"raw={width}x{height} crop={crop_width}x{crop_height}",
            crop_width,
            crop_height,
            front_m,
            valid_pct,
            backend,
        )
        self._canvas.create_text(
            8,
            8,
            anchor="nw",
            fill="#00ff00",
            text=status_text,
        )

        if self._show_raw and self._canvas_raw is not None:
            rgb_raw = _depth_to_rgb_raw(depth)
            photo_raw = self._tk.PhotoImage(data=_ppm_bytes(rgb_raw))
            if self._scale < 1.0:
                subsample = max(1, int(round(1.0 / self._scale)))
                photo_raw = photo_raw.subsample(subsample, subsample)
            raw_h, raw_w = rgb_raw.shape[:2]
            raw_dw = max(1, int(round(raw_w * scale)))
            raw_dh = max(1, int(round(raw_h * scale)))
            self._canvas_raw.configure(width=raw_dw, height=raw_dh)
            self._canvas_raw.delete("all")
            self._canvas_raw.create_image(
                0, 0, anchor="nw", image=photo_raw,
            )
            self._canvas_raw.create_text(
                8, 8, anchor="nw", fill="#ff4444",
                text=f"RAW linear | {raw_w}x{raw_h}",
            )
            self._photo_raw = photo_raw

        self._root.update()

    def update_error(self, message: str) -> None:
        self._canvas.delete("all")
        self._canvas.create_text(
            10,
            10,
            anchor="nw",
            fill="white",
            text=f"RealSense error: {message}",
        )
        self._root.update()

    def is_open(self) -> bool:
        try:
            return bool(self._root.winfo_exists())
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._root.destroy()
        except Exception:
            pass

    def _update_status(
        self,
        geometry: str,
        crop_width: int,
        crop_height: int,
        front_m: float,
        valid_pct: float,
        backend: str,
    ) -> str:
        now = time.perf_counter()
        self._fps_frame_count += 1
        if now - self._last_fps_time >= 1.0:
            self._fps = self._fps_frame_count / max(now - self._last_fps_time, 1e-6)
            self._fps_frame_count = 0
            self._last_fps_time = now

        text = (
            f"{backend} | {geometry} | {self._fps:.1f} FPS | "
            f"valid={valid_pct:.1f}% | front={front_m:.2f}m | "
            f"crop={crop_width}x{crop_height}"
        )
        self._root.title(text[:180])
        return text
