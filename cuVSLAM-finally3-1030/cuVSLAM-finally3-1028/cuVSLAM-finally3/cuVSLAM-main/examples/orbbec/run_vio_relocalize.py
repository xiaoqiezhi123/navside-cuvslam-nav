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

"""加载已保存的 SLAM 地图，对实时 Orbbec 相机重定位，并在 Rerun 中显示结果。

前置条件：先用 run_vio_finally.py（SAVE_RELOCALIZATION_MAP=True）录制并保存地图，
生成 LMDB 文件夹 vio_relocalization_map。

本脚本流程：
  1. 构建与录制时完全一致的双目惯性 rig + tracker；
  2. 通过 localize_in_map 加载 LMDB 地图并重定位当前相机位姿；
  3. 把地图里的「点云 + 原轨迹」一次性显示出来作为参照；
  4. 用黄线实时显示本次运行估计的轨迹，用红点显示重定位后的当前位置；
  5. 可选：每隔 RELOCALIZE_INTERVAL_S 秒重新 localize_in_map 一次（用当前位姿作猜测）。

本脚本不再收集 / 保存点云，只与 LMDB 地图中的点云做比对（定位）。
"""

import os
import queue
import signal
import threading
import time
from typing import List

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from pyorbbecsdk import Config, Pipeline

import cuvslam as vslam

from camera_utils import get_stereo_calibration
from run_vio_finally import (
    IMAGE_JITTER_THRESHOLD_NS,
    IMU_JITTER_THRESHOLD_NS,
    IMU_QUEUE_MAX_SIZE,
    LOOP_CLOSURE_THROTTLING_MS,
    NUM_DESIRED_TRACKS,
    PLANAR_CONSTRAINTS,
    SHOW_GRAVITY,
    SLAM_SYNC_MODE,
    STEREO_SENSORS,
    ThreadWithTimestamp,
    camera_thread,
    configure_ir_streams,
    disable_ir_emitter,
    get_orbbec_vio_rig,
    imu_thread,
)

# 重定位目标地图目录（run_vio_finally.py 保存的 LMDB 地图，内部含 data.mdb）。
MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vio_relocalization_map")

# 初始位姿猜测（用于首次重定位，缩小搜索范围）。单位：米 + 四元数 [x, y, z, w]。
# 相机需从地图原点附近启动；若实际起点不同，请修改这里。
GUESS_TRANSLATION = [0.0, 0.0, 0.0]
GUESS_ROTATION = [0.0, 0.0, 0.0, 1.0]

# 重定位网格搜索参数（搜索半径与步长）。
# 搜索量 ≈ (2·水平半径/水平步长) × (2·垂直半径/垂直步长) × (2·水平半径/水平步长) × (2π/角度步长)。
# 步长太细会生成数百万个候选位姿、跑很久；下面用较粗的步长，搜索更快。
LOCALIZATION_SETTINGS = vslam.Tracker.SlamLocalizationSettings(
    horizontal_search_radius=8.0,
    vertical_search_radius=2.0,
    horizontal_step=1.0,
    vertical_step=0.5,
    angular_step_rads=0.05,
)

# 周期性重定位间隔（秒）。
#   > 0 = 每隔该秒数重新 localize_in_map 一次（会重新加载地图，可能短暂卡顿）；
#   = 0 = 只在启动时定位一次，之后依靠 SLAM 回环对已加载地图持续校正（更轻量，推荐）。
RELOCALIZE_INTERVAL_S = 10.0

# 等待首次重定位完成的最长时间（秒），超时只提示、不中断。
LOCALIZATION_TIMEOUT_S = 30.0

# --- 显示颜色 / 半径 ---
MAP_LANDMARK_COLOR = [128, 128, 255]       # 原地图点云（浅蓝）
ORIGINAL_TRAJECTORY_COLOR = [0, 64, 255]   # 原地图轨迹（蓝）
CURRENT_TRAJECTORY_COLOR = [255, 255, 0]   # 当前运行估计轨迹（黄）
CURRENT_POSITION_COLOR = [255, 0, 0]       # 当前重定位位置（红点）
CURRENT_POSITION_RADIUS = 0.2


def main() -> None:
    """加载已保存地图，重定位实时相机，并显示地图 + 黄线轨迹 + 红点位置。"""
    if not os.path.exists(MAP_DIR):
        raise RuntimeError(
            f"Map folder not found: {MAP_DIR}. "
            "Run run_vio_finally.py with SAVE_RELOCALIZATION_MAP=True first."
        )

    # --- IR 流水线（与录制时一致）---
    ir_pipeline = Pipeline()
    ir_config = Config()
    sensor_types = {
        ir_pipeline.get_device().get_sensor_list().get_type_by_index(i)
        for i in range(ir_pipeline.get_device().get_sensor_list().get_count())
    }
    if not STEREO_SENSORS.issubset(sensor_types):
        raise RuntimeError("Device does not support dual IR sensors (LEFT_IR + RIGHT_IR required)")

    disable_ir_emitter(ir_pipeline)
    configure_ir_streams(ir_pipeline, ir_config)
    try:
        ir_pipeline.enable_frame_sync()
    except Exception as e:
        print(f"Warning: Could not enable frame sync: {e}")
    ir_pipeline.start(ir_config)

    print("Getting stereo calibration...")
    stereo_params = get_stereo_calibration(ir_pipeline)
    rig = get_orbbec_vio_rig(stereo_params)

    # --- tracker（配置与录制时一致）---
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=True,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        debug_imu_mode=False,
        odometry_mode=vslam.Tracker.OdometryMode.Inertial,
        rectified_stereo_camera=True,
        use_gpu=True,
    )
    slam_cfg = vslam.Tracker.SlamConfig(
        sync_mode=SLAM_SYNC_MODE,
        planar_constraints=PLANAR_CONSTRAINTS,
        throttling_time_ms=LOOP_CLOSURE_THROTTLING_MS,
        use_gpu=True,
    )
    tracker = vslam.Tracker(rig, cfg, slam_cfg)
    internals = vslam.Tracker.Internals()
    internals.num_desired_tracks = NUM_DESIRED_TRACKS

    # --- IMU 流水线（与录制时一致）---
    imu_pipeline = Pipeline()
    imu_config = Config()
    imu_config.enable_accel_stream()
    imu_config.enable_gyro_stream()

    # --- Rerun 视图（3D 地图 + 黄线轨迹 + 红点位置）---
    rr.init("cuVSLAM Relocalization", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Spatial3DView(origin="world", name="Map + relocalization"),
        ),
        make_active=True,
    )

    # --- 重定位状态 / 回调 ---
    localization_complete = threading.Event()
    has_localized = [False]
    localize_in_flight = False
    initial_localized = False
    last_localize_time = 0.0
    localization_timeout_warned = False
    last_heartbeat = 0.0
    map_logged = False

    def localization_start_cb() -> None:
        print("Localization started...")

    def localization_finish_cb(pose, error_message) -> None:
        if error_message:
            print(f"Localization failed: {error_message}")
        else:
            has_localized[0] = True
            print(f"Localization succeeded. Pose translation: {pose.translation}")
        localization_complete.set()

    def start_localization(timestamp: int, images, guess_pose: vslam.Pose) -> None:
        localization_complete.clear()
        tracker.localize_in_map(
            MAP_DIR,
            timestamp,
            guess_pose,
            list(images),
            LOCALIZATION_SETTINGS,
            localization_start_cb,
            localization_finish_cb,
        )

    # --- 队列 / 线程（复用 run_vio_finally 的采集线程）---
    q = queue.Queue()
    imu_queue = queue.Queue(maxsize=IMU_QUEUE_MAX_SIZE)
    thread_with_timestamp = ThreadWithTimestamp(IMAGE_JITTER_THRESHOLD_NS, IMU_JITTER_THRESHOLD_NS)

    imu_pipeline.start(imu_config)
    imu_thread_obj = threading.Thread(
        target=imu_thread, args=(imu_queue, thread_with_timestamp, imu_pipeline), daemon=True
    )
    camera_thread_obj = threading.Thread(
        target=camera_thread,
        args=(tracker, q, imu_queue, thread_with_timestamp, ir_pipeline, SHOW_GRAVITY, internals),
        daemon=True,
    )
    imu_thread_obj.start()
    camera_thread_obj.start()

    frame_id = 0
    current_trajectory: List[np.ndarray] = []

    print("Starting relocalization against saved map...")
    print(f"Map: {MAP_DIR}")
    print("Keep the camera still until localization completes. Press Ctrl+C to stop.")

    try:
        while True:
            try:
                (
                    timestamp,
                    odom_pose,
                    images,
                    _observations,
                    _gravity,
                    slam_pose,
                    slam_poses,
                    map_landmarks,
                    _loop_closure_poses,
                ) = q.get(timeout=1.0)
            except queue.Empty:
                continue

            if odom_pose is None:
                continue

            # 当前位姿（优先用 SLAM 位姿：重定位成功后它处于地图坐标系，与地图点一致）。
            viz_pose = slam_pose if slam_pose is not None else odom_pose
            xyz = viz_pose.translation

            # 重定位完成时清掉 in-flight 标记。
            if localize_in_flight and localization_complete.is_set():
                localize_in_flight = False

            # 触发重定位：首次用配置的猜测位姿；之后周期性地用「当前位姿」作为猜测。
            if not localize_in_flight:
                should_localize = False
                guess_pose = None
                if not initial_localized:
                    should_localize = True
                    guess_pose = vslam.Pose(rotation=GUESS_ROTATION, translation=GUESS_TRANSLATION)
                    initial_localized = True
                elif RELOCALIZE_INTERVAL_S > 0 and (time.time() - last_localize_time) >= RELOCALIZE_INTERVAL_S:
                    should_localize = True
                    guess_pose = viz_pose
                if should_localize:
                    start_localization(timestamp, images, guess_pose)
                    localize_in_flight = True
                    last_localize_time = time.time()
                    last_heartbeat = time.time()

            # 第一步：定位成功后，先把地图点云 + 原蓝轨迹一次性画出来作为参照。
            # 点云与原轨迹只有在 localize_in_map 成功把地图换进 SLAM 状态之后才能读到，
            # 所以「加载点云/轨迹」和「重定位」是同一件事，无法在定位之前单独先加载。
            if has_localized[0] and not map_logged:
                if map_landmarks is not None and map_landmarks.landmarks:
                    rr.log(
                        "world/map_landmarks",
                        rr.Points3D(
                            [l.coords for l in map_landmarks.landmarks],
                            colors=[MAP_LANDMARK_COLOR],
                        ),
                        static=True,
                    )
                    if slam_poses is not None:
                        rr.log(
                            "world/original_trajectory",
                            rr.LineStrips3D(
                                [p.pose.translation for p in slam_poses],
                                colors=[ORIGINAL_TRAJECTORY_COLOR],
                            ),
                            static=True,
                        )
                    map_logged = True

            # 第二步：始终画当前黄线 + 红点（定位期间也画，避免 Rerun 空白）。
            # 定位成功前位姿在里程计坐标系；成功后换到地图坐标系，红点自然贴到蓝轨迹附近。
            rr.set_time("frame", sequence=frame_id)
            current_trajectory.append(np.asarray(xyz))
            rr.log(
                "world/current_trajectory",
                rr.LineStrips3D(current_trajectory, colors=[CURRENT_TRAJECTORY_COLOR]),
            )
            rr.log(
                "world/current_position",
                rr.Points3D(
                    positions=[[float(xyz[0]), float(xyz[1]), float(xyz[2])]],
                    colors=[CURRENT_POSITION_COLOR],
                    radii=[CURRENT_POSITION_RADIUS],
                ),
            )
            rr.log("world/rig", rr.Transform3D(translation=xyz, quaternion=viz_pose.rotation))

            # 定位进行中：每 5 秒打印一次心跳，提示网格搜索还没完成（不是卡死）。
            if localize_in_flight and not localization_complete.is_set():
                if time.time() - last_heartbeat >= 5.0:
                    print("Localization still in progress (grid search)...")
                    last_heartbeat = time.time()

            # 首次定位超时提示（只提示一次）。
            if (
                initial_localized
                and not has_localized[0]
                and not localization_timeout_warned
                and (time.time() - last_localize_time) > LOCALIZATION_TIMEOUT_S
            ):
                print(f"Localization did not complete within {LOCALIZATION_TIMEOUT_S}s; continuing without map.")
                localization_timeout_warned = True

            frame_id += 1

    except KeyboardInterrupt:
        print("Stopping relocalization...")
    finally:
        # 清理期间忽略 SIGINT，避免第二次 Ctrl+C 中断流水线关闭导致 std::terminate。
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        imu_pipeline.stop()
        ir_pipeline.stop()
        print("Pipelines stopped.")


if __name__ == "__main__":
    main()
