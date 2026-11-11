#!/usr/bin/env python3
"""Orbbec Gemini 336L 深度可视化。

三种模式:
  python view_orbbec_depth.py                  # 实时窗口 (cv2.imshow, 需显示环境/ssh -X)
  python view_orbbec_depth.py --save <dir>     # 存 PNG 热力图 (无显示也能跑)
  python view_orbbec_depth.py --terminal       # 终端 ANSI 热力图 (SSH 下立刻看)

依赖: 项目 .venv_navside (cv2 4.8 + numpy + pyorbbecsdk2)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception import create_depth_perception  # noqa: E402


def depth_to_heatmap(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """深度(米) -> JET 热力图 BGR 图。"""
    import cv2

    d = np.clip(depth_m.astype(np.float32), min_m, max_m)
    norm = (d - min_m) / max(max_m - min_m, 1e-6)
    gray = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def valid_pct(depth_m: np.ndarray) -> float:
    v = depth_m[(depth_m > 0) & np.isfinite(depth_m)]
    return v.size / depth_m.size * 100.0


def overlay_text(img: np.ndarray, front_m: float, valid: float) -> np.ndarray:
    import cv2

    cv2.putText(img, f"front={front_m:.2f}m", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"valid={valid:.1f}%", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# 终端 ANSI 模式
# ---------------------------------------------------------------------------
def _jet_rgb(t: float) -> tuple[int, int, int]:
    """标准 jet 颜色映射 (t in [0,1]) -> (r,g,b) 0..255。"""
    t = min(max(t, 0.0), 1.0)
    r = int(np.clip(1.5 - abs(4 * t - 3), 0, 1) * 255)
    g = int(np.clip(1.5 - abs(4 * t - 2), 0, 1) * 255)
    b = int(np.clip(1.5 - abs(4 * t - 1), 0, 1) * 255)
    return r, g, b


def render_terminal(depth_m: np.ndarray, min_m: float, max_m: float,
                    cols: int = 96, rows: int = 30) -> str:
    """把深度图降采样成 (rows, cols) 的 ANSI 24bit 色块字符串。"""
    H, W = depth_m.shape
    d = np.clip(depth_m, min_m, max_m)
    yb = np.linspace(0, H, rows + 1).astype(int)
    xb = np.linspace(0, W, cols + 1).astype(int)
    lines = []
    for r in range(rows):
        parts = []
        for c in range(cols):
            block = d[yb[r]:yb[r + 1], xb[c]:xb[c + 1]]
            valid = block[(block > 0) & np.isfinite(block)]
            if valid.size == 0:
                parts.append("\033[40m  \033[0m")  # 黑色 = 无深度
            else:
                t = (float(np.median(valid)) - min_m) / max(max_m - min_m, 1e-6)
                rr, gg, bb = _jet_rgb(t)
                parts.append(f"\033[48;2;{rr};{gg};{bb}m  \033[0m")
        lines.append("".join(parts))
    return "\n".join(lines)


def run_live(cam, args) -> None:
    import cv2

    print("[view] 实时窗口模式 (按 q 退出)。若无显示请改用 --save 或 --terminal。")
    try:
        while True:
            out = cam.read()
            if out.success:
                heat = depth_to_heatmap(out.depth_input, args.min, args.max)
                overlay_text(heat, out.front_distance_m, valid_pct(out.depth_input))
                cv2.imshow("Gemini 336L Depth", heat)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(f"[view] read fail: {out.error}")
                time.sleep(0.1)
    except Exception as exc:
        print(f"[view] 显示失败 (无 X 显示?): {exc}")
        print("[view] 建议: ssh -X 重连后跑, 或改用 --save <dir> / --terminal")
    finally:
        cv2.destroyAllWindows()


def run_save(cam, args) -> None:
    import cv2

    outdir = Path(args.save)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[view] 存 PNG 到 {outdir}/ (Ctrl+C 停止)")
    n = 0
    try:
        while args.frames == 0 or n < args.frames:
            out = cam.read()
            if not out.success:
                print(f"[view] read fail: {out.error}")
                time.sleep(0.1)
                continue
            heat = depth_to_heatmap(out.depth_input, args.min, args.max)
            overlay_text(heat, out.front_distance_m, valid_pct(out.depth_input))
            path = outdir / f"depth_{n:05d}.png"
            cv2.imwrite(str(path), heat)
            n += 1
            if n % 10 == 0:
                print(f"[view] 已存 {n} 帧 -> {path}")
    except KeyboardInterrupt:
        pass
    print(f"[view] 共保存 {n} 帧")


def run_terminal(cam, args) -> None:
    print("[view] 终端 ANSI 热力图 (Ctrl+C 退出)")
    try:
        while True:
            out = cam.read()
            if out.success:
                frame = render_terminal(out.depth_input, args.min, args.max,
                                        args.cols, args.rows)
                print(f"\033[Hfront={out.front_distance_m:.2f}m  "
                      f"valid={valid_pct(out.depth_input):.1f}%   "
                      f"(min={args.min}m max={args.max}m)        \n" + frame, flush=True)
            else:
                print(f"\033[H[view] read fail: {out.error}", flush=True)
                time.sleep(0.2)
            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\033[H[view] 退出")
        sys.stdout.write("\033[0m\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbbec Gemini 336L 深度可视化")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="实时窗口 (cv2.imshow)")
    mode.add_argument("--save", default=None, help="存 PNG 热力图到指定目录")
    mode.add_argument("--terminal", action="store_true", help="终端 ANSI 热力图")
    parser.add_argument("--min", type=float, default=0.1, help="深度显示下限(米)")
    parser.add_argument("--max", type=float, default=6.0, help="深度显示上限(米)")
    parser.add_argument("--frames", type=int, default=0, help="--save 模式帧数(0=不限)")
    parser.add_argument("--cols", type=int, default=96, help="--terminal 列数")
    parser.add_argument("--rows", type=int, default=30, help="--terminal 行数")
    args = parser.parse_args()

    cam = create_depth_perception("orbbec")
    cam.start()
    print("[view] Orbbec 深度相机已启动")

    try:
        if args.save:
            run_save(cam, args)
        elif args.terminal:
            run_terminal(cam, args)
        else:
            run_live(cam, args)
    finally:
        cam.close()
        print("[view] done")


if __name__ == "__main__":
    main()
