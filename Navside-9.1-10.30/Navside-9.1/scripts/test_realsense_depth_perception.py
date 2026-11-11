#!/usr/bin/env python3
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.realsense_depth_perception import RealSenseDepthPerception


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RealSense depth backend.")
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    camera = RealSenseDepthPerception()
    try:
        camera.start()
    except Exception as exc:
        print(f"[RealSense Test] camera start failed: {exc}", file=sys.stderr)
        sys.exit(1)

    start = time.perf_counter()
    try:
        for i in range(max(args.frames, 1)):
            out = camera.read()
            if not out.success:
                print(f"[RealSense Test] frame {i} failed: {out.error}", file=sys.stderr)
                sys.exit(1)
            depth = out.depth_input
            valid_mask = np.isfinite(depth) & (depth > 0.01)
            valid_pct = 100.0 * float(valid_mask.sum()) / float(depth.size)
            print(
                f"frame {i}: shape={depth.shape} "
                f"valid_pct={valid_pct:.2f}% front={out.front_distance_m:.3f}m"
            )
        elapsed = time.perf_counter() - start
        print(f"[RealSense Test] read_fps={args.frames / max(elapsed, 1e-6):.2f}")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
