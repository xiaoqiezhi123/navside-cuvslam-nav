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

"""锚定层模块：地图管理、last-pose、三级 localize、分辨率缩放、stdin 读取。

从 A* 回廊版 `run_vio_tasknav.py` 提取的纯协议 / 无污染部分，供 finally 基座叠加使用。
不 import pyorbbecsdk；`cuvslam` 仅用于类型与 `localize_in_map` 调用；PIL 仅用于软件
降采样（分辨率匹配实验）。这是「只添加、不改基座」的一层。
"""

import os
import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import cuvslam as vslam
from camera_utils import extrinsic_to_pose, process_ir_frame as _process_ir_frame_native

# ---- 地图管理 ----
# 地图默认取本模块所在目录（脚本目录下的 <name> 子文件夹，与 A* 版约定一致）。
MAPS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAP_NAME = "orbbec_map"


def map_dir_path(name: str) -> str:
    """地图名 → 地图文件夹路径（<MAPS_DIR>/<name>）。"""
    return os.path.join(MAPS_DIR, name)


def last_pose_path(name: str) -> str:
    """地图名 → 该地图的 last-pose 文件路径（<MAPS_DIR>/<name>_last_pose.txt）。"""
    return os.path.join(MAPS_DIR, name + "_last_pose.txt")


def list_maps() -> List[str]:
    """列出脚本目录下已存在的地图名（含 data.mdb 的 LMDB 目录，按名称排序）。"""
    if not os.path.isdir(MAPS_DIR):
        return []
    return sorted(
        name for name in os.listdir(MAPS_DIR)
        if os.path.isfile(os.path.join(MAPS_DIR, name, "data.mdb"))
    )


def load_last_pose(name: str = DEFAULT_MAP_NAME) -> Optional[vslam.Pose]:
    """读取 <MAPS_DIR>/<name>_last_pose.txt，用作重定位初始 guess；缺失/损坏返回 None。"""
    return load_last_pose_from_path(last_pose_path(name))


def load_last_pose_from_path(path: str) -> Optional[vslam.Pose]:
    """读取任意路径的 last-pose 文件（7 个 float：x y z qx qy qz qw）；缺失/损坏返回 None。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            vals = f.read().split()
        tx, ty, tz, qx, qy, qz, qw = (float(v) for v in vals)
        return vslam.Pose(translation=[tx, ty, tz], rotation=[qx, qy, qz, qw])
    except Exception as e:
        print(f"Warning: failed to load last pose from {path}: {e}")
        return None


def save_last_pose(pose: vslam.Pose, name: str = DEFAULT_MAP_NAME) -> None:
    """把全局位姿落盘到 <MAPS_DIR>/<name>_last_pose.txt，供下次 localize 作初始 guess。"""
    path = last_pose_path(name)
    os.makedirs(MAPS_DIR, exist_ok=True)
    try:
        t, r = pose.translation, pose.rotation
        with open(path, "w") as f:
            f.write(f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f}\n")
        print(f"Saved last pose to {path}")
    except Exception as e:
        print(f"Warning: failed to save last pose: {e}")


# ---- 分辨率缩放（软件降采样：匹配锚定查询帧与建图 landmarks 的分辨率）----

def resize_gray(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """把 uint8 灰度图缩放到 ``size=(width, height)``（已是该尺寸则原样返回）。"""
    if img.shape[1] == size[0] and img.shape[0] == size[1]:
        return img
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:  # 老版本 Pillow 没有 Resampling 枚举
        resample = Image.BILINEAR
    return np.asarray(Image.fromarray(img).resize(size, resample=resample))


def process_ir_frame_scaled(ir_frame, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """调用基座 `camera_utils.process_ir_frame`，再软件缩放到 ``target_size``（None = 不缩放）。"""
    gray = _process_ir_frame_native(ir_frame)
    if gray is None:
        return None
    if target_size is not None:
        return resize_gray(gray, target_size)
    return gray


def build_stereo_rig_scaled(stereo_params: dict, target_size: Optional[Tuple[int, int]] = None) -> vslam.Rig:
    """由标定参数构建双目 rig；``target_size`` 非空时按轴线性缩放内参（软降采样对应）。"""
    def make_camera(intrinsic, extrinsic=None):
        cam = vslam.Camera()
        cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
        if target_size is not None:
            sx = target_size[0] / intrinsic.width
            sy = target_size[1] / intrinsic.height
            cam.focal = (intrinsic.fx * sx, intrinsic.fy * sy)
            cam.principal = (intrinsic.cx * sx, intrinsic.cy * sy)
            cam.size = (target_size[0], target_size[1])
        else:
            cam.focal = (intrinsic.fx, intrinsic.fy)
            cam.principal = (intrinsic.cx, intrinsic.cy)
            cam.size = (intrinsic.width, intrinsic.height)
        if extrinsic is not None:
            cam.rig_from_camera = extrinsic_to_pose(extrinsic)
        return cam

    rig = vslam.Rig()
    rig.cameras = [
        make_camera(stereo_params['left_intrinsic']),
        make_camera(stereo_params['right_intrinsic'], stereo_params['right_extrinsic']),
    ]
    return rig


# ---- 三级 localize 锚定（由粗到细搜索，沿用 A* 版定稿参数）----
LOCALIZE_COARSE_H_RADIUS = 25.0
LOCALIZE_COARSE_V_RADIUS = 0.2
LOCALIZE_COARSE_H_STEP = 1.0
LOCALIZE_COARSE_V_STEP = 0.2
LOCALIZE_COARSE_A_STEP = 0.15
LOCALIZE_MID_H_RADIUS = 4.0
LOCALIZE_MID_V_RADIUS = 0.1
LOCALIZE_MID_H_STEP = 0.5
LOCALIZE_MID_V_STEP = 0.1
LOCALIZE_MID_A_STEP = 0.06
LOCALIZE_FINE_H_RADIUS = 1.0
LOCALIZE_FINE_V_RADIUS = 0.1
LOCALIZE_FINE_H_STEP = 0.2
LOCALIZE_FINE_V_STEP = 0.1
LOCALIZE_FINE_A_STEP = 0.03
LOCALIZE_GUESS_TRANSLATION = (0.0, 0.0, 0.0)
LOCALIZE_GUESS_ROTATION = (0.0, 0.0, 0.0, 1.0)


def cold_start_guess() -> vslam.Pose:
    """冷启动先验：原点平移 + 单位姿态（粗层 25m 全图搜索覆盖）。"""
    return vslam.Pose(rotation=list(LOCALIZE_GUESS_ROTATION), translation=list(LOCALIZE_GUESS_TRANSLATION))


def format_pose(pose: vslam.Pose) -> str:
    t = pose.translation
    return f"xyz=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})"


def localize_anchor(
    tracker,
    map_dir: str,
    timestamp_ns: int,
    images,
    guess: vslam.Pose,
    coarse_h_radius_override: Optional[float] = None,
    progress_cb=None,
    cancel_event: Optional[threading.Event] = None,
    timeout_s: float = 180.0,
):
    """由粗到细三级 ``localize_in_map``，返回 ``(pose, status)``。

    status ∈ {"ok", "fail", "timeout", "cancelled"}：
      - "ok"：得到位姿（含后级失败回退前级位姿的降级成功）；
      - "fail"：首级搜索失败（无匹配）；
      - "timeout"：首级搜索在 ``timeout_s`` 内未返回结果；
      - "cancelled"：等待期间 ``cancel_event`` 被置位（用户取消）。

    async 模式下 localize_in_map 排队到 SLAM 后台线程执行，finish_cb 由官方绑定层
    包 ``gil_scoped_acquire`` 回调；本函数只在调用线程等待回调完成。每级之间串行
    等待；后级失败回退前级位姿，首级失败返回 (None, status)。
    """
    if progress_cb is None:
        progress_cb = print
    images = list(images)

    def run_pass(guess_pose, h_radius, v_radius, h_step, v_step, a_step, label):
        settings = vslam.Tracker.SlamLocalizationSettings(
            horizontal_search_radius=h_radius,
            vertical_search_radius=v_radius,
            horizontal_step=h_step,
            vertical_step=v_step,
            angular_step_rads=a_step,
        )
        out = {"pose": None, "error": None}
        done = threading.Event()

        def finish_cb(pose, error_message):
            out["pose"] = pose
            out["error"] = error_message
            done.set()

        progress_cb(f"[localize] {label}: H_radius={h_radius}m, h_step={h_step}m, a_step={a_step}rad ...")
        # start_cb 绑定要求 Callable（传 None 会 TypeError）。
        tracker.localize_in_map(map_dir, timestamp_ns, guess_pose, images, settings, lambda: None, finish_cb)

        # 轮询等待回调，期间响应取消 / 超时（首次全图搜索可能较慢）。
        waited = 0.0
        while not done.is_set():
            if cancel_event is not None and cancel_event.is_set():
                progress_cb(f"[localize] {label}: cancelled by user")
                return None, "cancelled"
            if waited >= timeout_s:
                progress_cb(f"[localize] {label}: timeout waiting for result")
                return None, "timeout"
            time.sleep(0.1)
            waited += 0.1

        if out["pose"] is None:
            progress_cb(f"[localize] {label}: failed ({out['error']})")
            return None, "fail"
        return out["pose"], "ok"

    coarse_h = coarse_h_radius_override if coarse_h_radius_override is not None else LOCALIZE_COARSE_H_RADIUS
    stages = [
        ("coarse", coarse_h, LOCALIZE_COARSE_V_RADIUS,
         LOCALIZE_COARSE_H_STEP, LOCALIZE_COARSE_V_STEP, LOCALIZE_COARSE_A_STEP),
        ("mid", LOCALIZE_MID_H_RADIUS, LOCALIZE_MID_V_RADIUS,
         LOCALIZE_MID_H_STEP, LOCALIZE_MID_V_STEP, LOCALIZE_MID_A_STEP),
        ("fine", LOCALIZE_FINE_H_RADIUS, LOCALIZE_FINE_V_RADIUS,
         LOCALIZE_FINE_H_STEP, LOCALIZE_FINE_V_STEP, LOCALIZE_FINE_A_STEP),
    ]

    current = guess
    last_pose: Optional[vslam.Pose] = None
    for label, h_radius, v_radius, h_step, v_step, a_step in stages:
        pose, status = run_pass(current, h_radius, v_radius, h_step, v_step, a_step, label)
        if status in ("cancelled", "timeout", "fail"):
            if last_pose is not None:
                progress_cb(f"[localize] {label} {status}; using previous-stage pose: {format_pose(last_pose)}")
                return last_pose, "ok"
            return None, status
        last_pose = pose
        current = pose

    progress_cb(f"[localize] Localized: {format_pose(last_pose)}")
    return last_pose, "ok"


# ---- stdin 指令读取 ----
def stdin_reader(cmd_queue: queue.Queue, stop_event: threading.Event,
                 cancel_event: Optional[threading.Event] = None) -> None:
    """非阻塞读 stdin 行，放入命令队列；主循环轮询消费。

    若提供 ``cancel_event``，读到 cancel 命令时立即置位（供主线程阻塞在 localize
    时实时取消），且不放入队列（由主循环的锚定状态机统一产出 ``anchor=cancel``）。
    """
    import sys
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if line:
            cmd = line.strip().lower()
            if cmd:
                if cancel_event is not None and cmd in ("cancel", "c"):
                    cancel_event.set()
                    continue
                cmd_queue.put(cmd)
        else:  # EOF（如 stdin 被重定向）
            break
