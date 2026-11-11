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

"""位姿数学与坐标系变换（从 A* 回廊版 foxglove_odom_server.py / run_vio_tasknav.py 提取）。

只含「纯数学 + cuVSLAM Pose 结构」的无污染部分：
  - opencv_pose_to_zup：cuVSLAM OpenCV 世界系（+X右 +Y下 +Z前）→ 下游 Z-up 机体系
    （REP-103：+X前 +Y左 +Z上），供 UDP 位姿出口与 [SCHED] pose 事件使用。
  - quat_mul / quat_rotate / pose_compose / pose_inverse：段内 odom 帧间增量合成
    （全局位姿 = 锚定基线 ⊗ odom 增量）。

四元数约定：cuVSLAM Pose.rotation 为 (x, y, z, w)，Hamilton 乘法。
"""

import numpy as np

import cuvslam as vslam

# OpenCV（相机系）→ 机体系 base_link（REP-103）的固定旋转矩阵：
#   x_body =  z_cam（前）
#   y_body = -x_cam（右）
#   z_body = -y_cam（下）
_OPENCV_TO_BODY_R = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def quaternion_xyzw_to_matrix(q) -> np.ndarray:
    """单位四元数 (x, y, z, w) → 旋转矩阵（Hamilton 约定，v' = R v）。"""
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 单位四元数 (x, y, z, w)。"""
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def opencv_pose_to_zup(position_xyz, quaternion_xyzw):
    """cuVSLAM 相机位姿（OpenCV 世界）→ 下游机体位姿（Z-up 世界）。

    Args:
        position_xyz: (3,) 相机在 OpenCV 世界下的位置 (x右, y下, z前)。
        quaternion_xyzw: (4,) 相机朝向 (x, y, z, w)。

    Returns:
        (position_zup (3,), quaternion_zup_xyzw (4,))
    """
    p = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    R_cw = quaternion_xyzw_to_matrix(quaternion_xyzw)
    # 位置：向量随世界系约定变化，直接左乘（共轭只作用于旋转）。
    p_zup = _OPENCV_TO_BODY_R @ p
    # 朝向：同时改变世界系约定与机体系约定 → 旋转矩阵做共轭 R_BC · R_cw · R_BC^T。
    R_bw_z = _OPENCV_TO_BODY_R @ R_cw @ _OPENCV_TO_BODY_R.T
    q_zup = matrix_to_quaternion_xyzw(R_bw_z)
    return p_zup, q_zup


def quat_mul(q1, q2) -> np.ndarray:
    """四元数乘法（x, y, z, w 顺序，Hamilton 约定）。"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=float)


def quat_rotate(q, v) -> np.ndarray:
    """用四元数 q 旋转向量 v（q 为单位四元数）。"""
    qv = np.asarray(q[:3], dtype=float)
    qw = float(q[3])
    t = 2.0 * np.cross(qv, v)
    return np.asarray(v, dtype=float) + qw * t + np.cross(qv, t)


def pose_compose(p1: vslam.Pose, p2: vslam.Pose) -> vslam.Pose:
    """位姿合成 p1 ⊗ p2（先 p2 后 p1）。"""
    q = quat_mul(p1.rotation, p2.rotation)
    t = np.asarray(p1.translation, dtype=float) + quat_rotate(p1.rotation, p2.translation)
    return vslam.Pose(rotation=q.tolist(), translation=t.tolist())


def pose_inverse(p: vslam.Pose) -> vslam.Pose:
    """位姿逆。"""
    q = np.asarray(p.rotation, dtype=float)
    q_inv = np.array([-q[0], -q[1], -q[2], q[3]])
    t = np.asarray(p.translation, dtype=float)
    return vslam.Pose(rotation=q_inv.tolist(), translation=(-quat_rotate(q_inv, t)).tolist())
