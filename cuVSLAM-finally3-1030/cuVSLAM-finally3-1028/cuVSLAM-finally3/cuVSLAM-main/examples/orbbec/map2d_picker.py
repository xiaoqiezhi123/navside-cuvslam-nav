# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""把 cuVSLAM 建图录制的 3D landmarks 压成 2D 俯视地图，交互式选点输出全局坐标。

数据来源：run_vio.py / run_vio_tasknav.py 运行时落盘的 .rrd（Rerun 记录），
其中 ``world/map_landmarks`` 是 SLAM 稀疏点云（全局系），``world/trajectory_slam``
是平滑轨迹。本脚本读 .rrd 取最后一次记录的数据，投影到 XZ 水平面（传统 2D SLAM
地图样式），在图上点击选点，输出该点附近的 (x, y, z) 全局坐标——可直接粘贴到
run_vio_tasknav.py 的 TASK_POINTS 列表。

坐标系：cuVSLAM 世界系为 OpenCV 约定（+X 右、+Y 下、+Z 前）。竖直轴是 Y（向下），
水平面是 XZ。图中横轴 = X（右），纵轴 = Z（前，上方为前进方向）。

用法：
  ./run_map2d_picker.sh                         # 读 orbbec/vio_mapping_tracking.rrd
  ./run_map2d_picker.sh --rrd 其它.rrd           # 指定 .rrd
  ./run_map2d_picker.sh --y 0.0                  # 选点高度固定（不从附近点取中位数）
  ./run_map2d_picker.sh --save 任务点.txt         # 选点结果同时追加写入文件

操作：左键点选（可连续点多个）；右键 / 中键 / 回车结束选点。
"""

import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")  # load_recording 0.32 起标记 deprecated，仍可用
from rerun.recording import load_recording  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("TkAgg")  # 物理桌面交互选点
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_RRD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vio_mapping_tracking.rrd")
PICK_RADIUS_M = 0.5  # 取该半径内 landmarks 的高度中位数作为选点高度


def extract_from_rrd(path: str):
    """从 .rrd 提取 landmarks (N,3) 与轨迹 strips（各取最后一次记录）。

    Returns:
        (landmarks, strips): landmarks 为 (N,3) float ndarray（可为空）；
        strips 为 list of (M,3) ndarray（轨迹折线，可为空列表）。
    """
    rec = load_recording(path)
    landmark_rows = []
    strips = []
    for chunk in rec.chunks():
        entity = str(chunk.entity_path)
        if chunk.num_rows == 0:
            continue
        rb = chunk.to_record_batch()
        if entity == "/world/map_landmarks" and "Points3D:positions" in rb.column_names:
            # 每行 = 一个时间点的全部 landmarks；取最后一行（最完整）。
            rows = rb.column("Points3D:positions").to_pylist()
            if rows:
                landmark_rows.append(np.asarray(rows[-1], dtype=float))
        elif entity == "/world/trajectory_slam" and "LineStrips3D:strips" in rb.column_names:
            rows = rb.column("LineStrips3D:strips").to_pylist()
            if rows:
                for strip in rows[-1]:
                    strips.append(np.asarray(strip, dtype=float))
    landmarks = (
        np.vstack(landmark_rows) if landmark_rows else np.zeros((0, 3), dtype=float)
    )
    return landmarks, strips


def pick_height(points: np.ndarray, x: float, z: float, fixed_y=None) -> float:
    """点击 (x, z) 处的高度：优先取附近 landmarks 的 Y 中位数，否则最近点高度。

    Args:
        points: (N,3) 参考点云（landmarks，或轨迹点）
        x, z: 点击位置（XZ 平面）
        fixed_y: 非 None 时直接返回该固定高度
    """
    if fixed_y is not None:
        return float(fixed_y)
    if len(points) == 0:
        return 0.0
    d = np.hypot(points[:, 0] - x, points[:, 2] - z)
    nearby = points[d <= PICK_RADIUS_M, 1]
    if len(nearby):
        return float(np.median(nearby))
    return float(points[int(np.argmin(d)), 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="cuVSLAM 3D landmarks -> 2D 俯视地图选点")
    parser.add_argument("--rrd", default=DEFAULT_RRD, help=f"Rerun 记录文件（默认 {DEFAULT_RRD}）")
    parser.add_argument("--y", type=float, default=None, help="选点高度固定为该值（默认取附近 landmarks 的 Y 中位数）")
    parser.add_argument("--save", default=None, help="选点结果追加写入的文件（TASK_POINTS 格式）")
    args = parser.parse_args()

    if not os.path.isfile(args.rrd):
        raise SystemExit(f"找不到 .rrd 文件: {args.rrd}\n"
                         f"请先跑一次 run_vio.py --mode map（ENABLE_MAPPING_VISUALIZATION=True）"
                         f"生成 landmarks 记录。")

    if os.path.getsize(args.rrd) < 10_000:
        print(f"警告：{args.rrd} 只有 {os.path.getsize(args.rrd)} 字节，疑似空记录"
              f"（上次运行被 kill / 未正常 Ctrl+C 退出，Rerun 缓冲未落盘）。\n"
              f"请重新跑一次建图并按 Ctrl+C 正常退出后再试。")

    landmarks, strips = extract_from_rrd(args.rrd)
    print(f"landmarks: {len(landmarks)} 点 | 轨迹折线: {len(strips)} 段")
    if len(landmarks) == 0 and not strips:
        raise SystemExit("该 .rrd 里既无 landmarks 也无轨迹（建图时可视化建图数据未启用或未记录到地图点）。")

    # 高度参考点：优先 landmarks，否则用轨迹点。
    ref_points = landmarks if len(landmarks) else np.vstack(strips)

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.canvas.manager.set_window_title("cuVSLAM 2D map picker (XZ top view)")
    if len(landmarks):
        ax.scatter(landmarks[:, 0], landmarks[:, 2], s=1.5, c="tab:blue",
                   label=f"SLAM landmarks ({len(landmarks)})")
    for i, s in enumerate(strips):
        ax.plot(s[:, 0], s[:, 2], c="tab:red", lw=1.5, label="trajectory_slam" if i == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel("X (right, m)")
    ax.set_ylabel("Z (forward, m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("左键: 选点（可连续多个）| 右键/中键/回车: 结束选点\n"
                 "滚轮缩放，按住左键拖动平移")

    print("在地图上选点（右键结束）...")
    picks = plt.ginput(n=-1, timeout=0, show_clicks=True)  # (x, z) 列表
    plt.close(fig)

    if not picks:
        print("未选任何点。")
        return

    print("\n" + "=" * 62)
    print(f"选点结果（共 {len(picks)} 个，TASK_POINTS 格式可直接粘贴）：")
    lines = []
    for i, (x, z) in enumerate(picks, 1):
        y = pick_height(ref_points, x, z, fixed_y=args.y)
        line = f"    ({x:.3f}, {y:.3f}, {z:.3f}),  # 任务点 {i}"
        lines.append(line.rstrip(","))
        print(line)
    print("=" * 62)

    if args.save:
        with open(args.save, "a") as f:
            f.write("\n".join(lines) + "\n")
        print(f"已追加写入: {args.save}")


if __name__ == "__main__":
    main()
