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

# =============================================================================
# run_vio_tasknav_finally.py —— 分段重定位任务导航（finally 稳定基座 + 锚定层）
#
# 与 A* 回廊版 run_vio_tasknav.py 对应的 finally 版入口，只「添加、不改基座」：
#   - finally 基座 run_vio_finally.py 的稳定内核（官方 17.0.0 wheel、原生参数、
#     无 IMU 噪声膨胀、回环节流 1s）保持不变；
#   - 叠加 A* 的分段重定位协议：双分辨率双 tracker + 阶段切换重配 IR 流 +
#     odom 增量⊗锚定基线 + UDP 位姿出口 + [SCHED] 事件通道。
#
# 双分辨率（本版关键差异）：
#   - 行进 tracker：640×400@30 原生（finally 内核分辨率）。
#   - 重定位 tracker：848×480@15 原生（与参考地图 landmarks 分辨率一致）。
#   两个分辨率 FOV 不同（640×400=16:10 满传感器，848×480=16:9 裁上下），
#   因此「软降采样」无法在两者间等价换算，必须阶段切换重配 IR 流。
#
# 数据流：
#   点 1：stdin "localize" → 切 848×480@15 → 三级 localize 锚定 → 切回 640×400@30
#         → 行进（odom 增量⊗锚定基线）→ UDP 8082 实时发全局坐标
#   点 2：再次 "localize" → 重复上述流程（漂移被清零）
#
# 终端指令：localize / l | cancel / c | pause / p | resume / r | quit / q
# =============================================================================

import argparse
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from pyorbbecsdk import (
    Config,
    Context,
    OBFormat,
    OBFrameType,
    OBPermissionType,
    OBPropertyID,
    OBSensorType,
    Pipeline,
)

import cuvslam as vslam

# 复用 finally 基座的无状态内核函数（不修改 run_vio_finally.py，只 import）。
from run_vio_finally import (
    IMU_ACCELEROMETER_NOISE_DENSITY,
    IMU_ACCELEROMETER_RANDOM_WALK,
    IMU_FREQUENCY,
    IMU_GYROSCOPE_NOISE_DENSITY,
    IMU_GYROSCOPE_RANDOM_WALK,
    IMU_JITTER_THRESHOLD_NS,
    IMU_QUEUE_MAX_SIZE,
    NUM_DESIRED_TRACKS,
    ImuSample,
    ThreadWithTimestamp,
    disable_ir_emitter,
    get_orbbec_vio_rig,
    get_stereo_calibration,
    process_ir_frame,
    register_imu_until,
)

from anchor_layer import (
    cold_start_guess,
    format_pose,
    localize_anchor,
    map_dir_path,
    save_last_pose,
    stdin_reader,
)
from pose_transform import pose_compose, pose_inverse
from udp_pose_sender import UdpPoseSender
from vio_handoff import AnchorHandoff, TrackerImuBuffer, wait_while_paused
from visualizer_mapping import MappingVisualizer

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------

# 行进 finally 分辨率（基座原生）。
TRAVEL_RESOLUTION = (640, 400)
TRAVEL_FPS = 30
# 重定位 / 参考地图分辨率（10261 建图时的原生分辨率）。
LOCALIZE_RESOLUTION = (848, 480)
LOCALIZE_FPS = 15

# 行进期 SLAM：开 = finally 内核后台线程 + 1s 节流（默认）；--travel-pure-vio 关。
USE_SLAM_MODE = True
SLAM_SYNC_MODE = False  # 行进 async（后台 SLAM 线程）；定位 tracker 固定 sync
PLANAR_CONSTRAINTS = True
LOOP_CLOSURE_THROTTLING_MS = 1000  # 回环节流（finally 稳定值，不是 A* 的 1h）

ANCHOR_WARMUP_FRAMES = 10  # 锚定前预热帧数（保留 A* 机制，决策 4）
ANCHOR_RETRY_COOLDOWN_S = 10.0  # 锚定结束冷却，挡背靠背 localize 的 GIL 竞态

# 位姿出口：odom 增量⊗锚定基线（决策 2）。
USE_ODOM_FOR_POSE_OUTPUT = True
SEND_POSE_BEFORE_ANCHOR = True  # 锚定前也发本地系位姿（与 A* 一致）

POSE_UDP_OUTPUT_ENABLED = True
POSE_UDP_HOST = "127.0.0.1"
POSE_UDP_PORT = 8082
POSE_PRINT_INTERVAL_S = 0.1  # [SCHED] pose / 实时坐标打印节流（10Hz）
POSE_MAX_RESULT_AGE_S = 0.5  # 结果太旧不当作新位姿发送

# 阶段切换重配 IR 流后等待流稳定（秒）。
STREAM_SWITCH_WAIT_S = 1.0
# 锚定预热 / 相机挂起超时。
ANCHOR_PREPARE_TIMEOUT_S = 30.0

# Rerun 可视化（参考 finally 版 MappingVisualizer；--no-viz 可关闭）。
SHOW_GRAVITY = False
ENABLE_MAPPING_VISUALIZATION = False  # 行进 tracker 的地图是本地系，默认不显示
SAVE_RECORDING = False
NUM_VIZ_CAMERAS = 1

REF_MAP_NAME = os.environ.get("CUVSLAM_REF_MAP", "10261")
CAMERA_A_SERIAL = os.environ.get("CUVSLAM_CAMERA_SERIAL", "")
SLAM_SCHED_FILE = os.environ.get("SLAM_SCHED_FILE", "")

STEREO_SENSORS = {OBSensorType.LEFT_IR_SENSOR, OBSensorType.RIGHT_IR_SENSOR}


def sched_emit(text: str) -> None:
    """[SCHED] 低频事件：stdout（先换行断掉 \r 位置行）+ SLAM_SCHED_FILE。"""
    print("\n" + text, flush=True)
    if SLAM_SCHED_FILE:
        try:
            with open(SLAM_SCHED_FILE, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass


def sched_emit_file(text: str) -> None:
    """[SCHED] 高频事件：只追加写 SLAM_SCHED_FILE，不刷 stdout（避免终端刷屏）。"""
    if SLAM_SCHED_FILE:
        try:
            with open(SLAM_SCHED_FILE, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 相机 / 流 辅助
# ---------------------------------------------------------------------------

def open_camera_pipeline(serial: str = "") -> Pipeline:
    """按序列号打开相机；空串 = 枚举到的第一台（单相机兼容）。

    双相机方案里 cuVSLAM 只打开「相机 A」（SLAM 被动双目，激光关）。串号用
    enumerate_devices.py 读出后填入，或用 CUVSLAM_CAMERA_SERIAL 覆盖。
    """
    if not serial:
        return Pipeline()
    ctx = Context()  # 保持引用，防止临时 Context 被 GC 后 deviceMgr 悬空
    device = None
    last_err: Optional[Exception] = None
    for _attempt in range(15):
        devices = ctx.query_devices()
        try:
            device = devices.get_device_by_serial_number(serial)
        except Exception as exc:
            last_err = exc
            device = None
        if device is None:
            try:
                for i in range(devices.get_count()):
                    if devices.get_device_serial_number_by_index(i) == serial:
                        device = devices.get_device_by_index(i)
                        break
            except Exception as exc:
                last_err = exc
        if device is not None:
            break
        time.sleep(1.0)
    if device is None:
        raise RuntimeError(
            f"未找到序列号 '{serial}' 的相机（当前已连接 {devices.get_count()} 台；"
            f"最后错误: {last_err}）。请重插该相机 USB（或重启）后重试。"
        )
    return Pipeline(device)


def configure_ir_streams_at(pipeline: Pipeline, config: Config, resolution, fps) -> None:
    """在 pipeline 配置中启用 LEFT_IR + RIGHT_IR 流（指定分辨率/帧率）。"""
    for sensor_type in [OBSensorType.LEFT_IR_SENSOR, OBSensorType.RIGHT_IR_SENSOR]:
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        profile = None
        for fmt in [OBFormat.Y8, OBFormat.Y16]:
            try:
                profile = profile_list.get_video_stream_profile(resolution[0], resolution[1], fmt, fps)
                break
            except Exception as e:
                print(f"Warning: Failed to get stream profile for {sensor_type} "
                      f"@ {resolution[0]}x{resolution[1]}@{fps}: {e}")
                continue
        if profile is None:
            profile = profile_list.get_default_video_stream_profile()
            print(f"Using default profile for {sensor_type}: "
                  f"{profile.get_width()}x{profile.get_height()} @ {profile.get_fps()} FPS")
        config.enable_stream(profile)


def reconfigure_ir_pipeline(ir_pipeline: Pipeline, resolution, fps) -> dict:
    """停→重配→启 IR 流，返回新流下的双目标定参数。

    复用同一个 Pipeline 对象（不重新枚举 USB 设备），只改流配置。
    """
    try:
        ir_pipeline.stop()
    except Exception as e:
        print(f"Warning: ir_pipeline.stop() failed: {e}")
    time.sleep(STREAM_SWITCH_WAIT_S)
    config = Config()
    configure_ir_streams_at(ir_pipeline, config, resolution, fps)
    try:
        ir_pipeline.enable_frame_sync()
    except Exception as e:
        print(f"Warning: Could not enable frame sync: {e}")
    ir_pipeline.start(config)
    print(f"IR stream reconfigured to {resolution[0]}x{resolution[1]}@{fps}; reading calibration...")
    return get_stereo_calibration(ir_pipeline)


# ---------------------------------------------------------------------------
# tracker 工厂（每次新建 OdometryConfig，避免 SLAM tracker 污染后续配置）
# ---------------------------------------------------------------------------

def _odometry_settings() -> dict:
    return dict(
        async_sba=True,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        debug_imu_mode=False,
        odometry_mode=vslam.Tracker.OdometryMode.Inertial,
        rectified_stereo_camera=True,
        use_gpu=True,
    )


def make_travel_tracker(rig: vslam.Rig, slam: bool = True) -> vslam.Tracker:
    """行进 tracker：640×400 原生，SLAM 可选（后台线程 + 1s 节流）。"""
    cfg = vslam.Tracker.OdometryConfig(**_odometry_settings())
    scfg = (
        vslam.Tracker.SlamConfig(
            sync_mode=SLAM_SYNC_MODE,
            planar_constraints=PLANAR_CONSTRAINTS,
            throttling_time_ms=LOOP_CLOSURE_THROTTLING_MS,
            use_gpu=True,
        )
        if slam
        else None
    )
    return vslam.Tracker(rig, cfg, scfg)


def make_localize_tracker(rig: vslam.Rig) -> vslam.Tracker:
    """定位 tracker：848×480 原生，SLAM 开 + sync（localize_in_map 要求，防 GIL 竞态）。"""
    cfg = vslam.Tracker.OdometryConfig(**_odometry_settings())
    scfg = vslam.Tracker.SlamConfig(
        sync_mode=True,
        planar_constraints=PLANAR_CONSTRAINTS,
        throttling_time_ms=LOOP_CLOSURE_THROTTLING_MS,
        use_gpu=True,
    )
    return vslam.Tracker(rig, cfg, scfg)


# ---------------------------------------------------------------------------
# 采集线程
# ---------------------------------------------------------------------------

def imu_thread(
    imu_queue: queue.Queue,
    thread_with_timestamp: ThreadWithTimestamp,
    imu_pipe: Pipeline,
    stop_event: threading.Event,
    pause_event: threading.Event,
    handoff: AnchorHandoff,
) -> None:
    """IMU 采集线程（finally 内核 + A* 暂停协议）。

    任务点 localize 期间 pause_event 置位：线程挂起不取帧，避免定位回调与
    pyorbbecsdk wait_for_frames 并发；挂起后丢弃旧陀螺仪，防止与新加速度帧错配。
    """
    high_rate_threshold = thread_with_timestamp.high_rate_threshold_ns
    prev_timestamp = None
    drop_count = 0
    queue_drop_count = 0
    last_gyro = None

    try:
        while not stop_event.is_set():
            if wait_while_paused(pause_event, stop_event, handoff.imu_parked):
                last_gyro = None  # 不把挂起前的陀螺仪配给挂起后的加速度帧
                continue
            frames = imu_pipe.wait_for_frames(100)
            if frames is None:
                continue

            gyro_frame = frames.get_frame(OBFrameType.GYRO_FRAME)
            if gyro_frame is not None:
                gyro_frame = gyro_frame.as_gyro_frame()
                if gyro_frame is not None:
                    last_gyro = (gyro_frame.get_x(), gyro_frame.get_y(), gyro_frame.get_z())

            accel_frame = frames.get_frame(OBFrameType.ACCEL_FRAME)
            if accel_frame is not None:
                accel_frame = accel_frame.as_accel_frame()
            if accel_frame is None or last_gyro is None:
                continue

            current_timestamp = int(accel_frame.get_timestamp_us() * 1000)

            if prev_timestamp is not None:
                timestamp_diff = current_timestamp - prev_timestamp
                if timestamp_diff < 0:
                    # 时间戳回绕/重置：重新对齐基准
                    prev_timestamp = current_timestamp
                    continue
                if timestamp_diff > high_rate_threshold:
                    drop_count += 1
                    if drop_count % 100 == 1:
                        print(f"\nWarning: IMU drops detected ({drop_count} total, "
                              f"last gap: {timestamp_diff/1e6:.2f} ms)")
            prev_timestamp = current_timestamp
            thread_with_timestamp.prev_high_rate_timestamp = current_timestamp

            sample = ImuSample(
                timestamp_ns=current_timestamp,
                linear_accelerations=(accel_frame.get_x(), accel_frame.get_y(), accel_frame.get_z()),
                angular_velocities=last_gyro,
            )
            try:
                imu_queue.put_nowait(sample)
            except queue.Full:
                queue_drop_count += 1
                try:
                    imu_queue.get_nowait()
                except queue.Empty:
                    pass
                imu_queue.put_nowait(sample)
                if queue_drop_count % 100 == 1:
                    print(f"\nWarning: IMU queue overflow ({queue_drop_count} dropped samples)")
    except Exception as e:
        print(f"\nIMU thread error: {e}")
        if not stop_event.is_set():
            handoff.fail(f'IMU thread error: {e}')


def camera_thread(
    tracker: vslam.Tracker,
    result_queue: queue.Queue,
    imu_queue: queue.Queue,
    thread_with_timestamp: ThreadWithTimestamp,
    ir_pipe: Pipeline,
    internals: vslam.Tracker.Internals,
    stop_event: threading.Event,
    pause_event: threading.Event,
    handoff: AnchorHandoff,
    travel_slam: bool,
) -> None:
    """相机线程：持有行进 VIO 与临时定位 tracker，串行注册 IMU / Track。

    阶段切换（行进 640×400 ↔ 定位 848×480）都在本线程内完成：
      1. 收到 localize 请求 → 切 848×480 流 → 建定位 tracker → 预热 → park；
      2. 主线程用预热好的定位 tracker 执行三级 localize（两采集线程已挂起）；
      3. release → 切回 640×400 流 → 重建行进 tracker → complete。
    """
    pending_imu: Deque[ImuSample] = deque()
    last_tracker_timestamp: Optional[int] = None
    tracker_generation = 0
    camera_drop_count = 0
    low_rate_threshold_ns = (1000.0 / TRAVEL_FPS + 5) * 1e6  # 行进相机帧间隔容差

    try:
        while not stop_event.is_set():
            # ---- 任务点锚定交接 ----
            if handoff.requested.is_set():
                warm = 0
                # 切到定位分辨率并建定位 tracker。
                try:
                    stereo_params = reconfigure_ir_pipeline(ir_pipe, LOCALIZE_RESOLUTION, LOCALIZE_FPS)
                    localize_rig = get_orbbec_vio_rig(stereo_params)
                    anchor_trk = make_localize_tracker(localize_rig)
                except Exception as e:
                    handoff.fail(f'定位流切换失败: {e}')
                    break
                handoff.data['anchor_tracker'] = anchor_trk
                anchor_imu = TrackerImuBuffer()
                warm_deadline = time.monotonic() + ANCHOR_PREPARE_TIMEOUT_S
                # 预热：凑齐带新鲜 IMU 的有效帧（缺陷 4：取消检查）。
                while (warm < ANCHOR_WARMUP_FRAMES and not stop_event.is_set()
                       and not handoff.cancelled.is_set() and time.monotonic() < warm_deadline):
                    frames = ir_pipe.wait_for_frames(100)
                    if frames is None:
                        continue
                    left_frame = frames.get_frame(OBFrameType.LEFT_IR_FRAME)
                    right_frame = frames.get_frame(OBFrameType.RIGHT_IR_FRAME)
                    if left_frame is None or right_frame is None:
                        continue
                    ts = int(left_frame.get_timestamp_us() * 1000)
                    left_img = process_ir_frame(left_frame)
                    right_img = process_ir_frame(right_frame)
                    if left_img is None or right_img is None:
                        continue
                    if not anchor_imu.prepare(ts, imu_queue, stop_event, handoff.cancelled):
                        continue
                    anchor_imu.last_timestamp = register_imu_until(
                        anchor_trk, imu_queue, anchor_imu.pending, ts, anchor_imu.last_timestamp,
                    )
                    odom_est, _sl = anchor_trk.track(ts, (left_img, right_img), internals=internals)
                    anchor_imu.tracked(ts)  # 缺陷 3：跟踪后更新时间戳为最后图像时间戳
                    if odom_est is not None and odom_est.world_from_rig is not None:
                        handoff.data['images'] = (left_img, right_img)
                        handoff.data['timestamp_ns'] = ts
                        handoff.data['n_obs'] = len(anchor_trk.get_last_observations(0))
                        warm += 1
                handoff.data['warm_ok'] = (warm >= ANCHOR_WARMUP_FRAMES
                                           and not handoff.cancelled.is_set())
                if not handoff.data['warm_ok']:
                    handoff.data['anchor_error'] = (
                        'cancel' if handoff.cancelled.is_set() else 'timeout'
                    )
                if stop_event.is_set() or not handoff.park():
                    break

                # 释放临时定位 tracker 的引用，再重建行进 tracker（缺陷 2：原子交接）。
                handoff.data['anchor_tracker'] = None
                anchor_trk = None
                anchor_imu = None

                # 切回行进分辨率并重建行进 tracker。
                try:
                    stereo_params = reconfigure_ir_pipeline(ir_pipe, TRAVEL_RESOLUTION, TRAVEL_FPS)
                    travel_rig = get_orbbec_vio_rig(stereo_params)
                    tracker = make_travel_tracker(travel_rig, slam=travel_slam)
                except Exception as e:
                    handoff.fail(f'行进流切换失败: {e}')
                    break

                pending_imu = deque()  # 清空旧 tracker 的 IMU 缓存
                last_tracker_timestamp = None
                tracker_generation = handoff.generation
                thread_with_timestamp.prev_low_rate_timestamp = None
                handoff.complete(tracker)
                continue

            # ---- 调度器 pause / resume ----
            if wait_while_paused(pause_event, stop_event):
                continue

            # ---- 行进 track 循环 ----
            frames = ir_pipe.wait_for_frames(100)
            if frames is None:
                continue

            left_frame = frames.get_frame(OBFrameType.LEFT_IR_FRAME)
            right_frame = frames.get_frame(OBFrameType.RIGHT_IR_FRAME)
            if left_frame is None or right_frame is None:
                continue

            current_timestamp = int(left_frame.get_timestamp_us() * 1000)

            if thread_with_timestamp.prev_low_rate_timestamp is not None:
                timestamp_diff = current_timestamp - thread_with_timestamp.prev_low_rate_timestamp
                if timestamp_diff > low_rate_threshold_ns:
                    camera_drop_count += 1
                    if camera_drop_count % 100 == 1:
                        print(f"\nWarning: Camera stream drop: gap {timestamp_diff/1e6:.2f} ms "
                              f"({camera_drop_count} total)")
            thread_with_timestamp.prev_low_rate_timestamp = current_timestamp

            left_img = process_ir_frame(left_frame)
            right_img = process_ir_frame(right_frame)
            if left_img is None or right_img is None:
                print("\nWarning: Failed to convert IR frames; skipping frame pair")
                continue
            images = (left_img, right_img)

            # 全新 tracker：丢弃重建前遗留的陈旧 IMU（保留 ~200ms 初始化窗口）。
            if last_tracker_timestamp is None:
                last_tracker_timestamp = current_timestamp - 200_000_000
            last_tracker_timestamp = register_imu_until(
                tracker, imu_queue, pending_imu, current_timestamp, last_tracker_timestamp,
            )
            odom_pose_estimate, slam_pose = tracker.track(current_timestamp, images, internals=internals)
            last_tracker_timestamp = current_timestamp

            odom_pose_with_cov = (
                odom_pose_estimate.world_from_rig if odom_pose_estimate is not None else None
            )
            if odom_pose_with_cov is None:
                print(f"\nTracking failed at frame {current_timestamp}")
                continue

            observations = tracker.get_last_observations(0)
            gravity = tracker.get_last_gravity() if SHOW_GRAVITY else None
            handoff.data['images'] = images
            handoff.data['timestamp_ns'] = current_timestamp
            handoff.data['n_obs'] = len(observations)

            result_queue.put([
                current_timestamp,
                odom_pose_with_cov.pose,
                slam_pose,
                tracker_generation,
                images,
                observations,
                gravity,
            ])
            thread_with_timestamp.last_low_rate_timestamp = current_timestamp
    except Exception as e:
        print(f"\nCamera thread error: {e}")
        if not stop_event.is_set():
            handoff.fail(f'Camera thread error: {e}')


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="分段重定位任务导航（finally 基座 + 锚定层）")
    parser.add_argument("--ref-map", default=REF_MAP_NAME,
                        help="参考地图名（脚本目录下的 <name> 子文件夹，只读）")
    parser.add_argument("--no-udp", action="store_true", help="关闭 UDP 位姿出口")
    parser.add_argument("--travel-pure-vio", action="store_true",
                        help="行进期关闭 SLAM（纯 VIO 前端，回退开关）")
    parser.add_argument("--cooldown", type=float, default=ANCHOR_RETRY_COOLDOWN_S,
                        help="锚定结束冷却（秒）")
    parser.add_argument("--no-viz", action="store_true", help="关闭 Rerun 可视化（无显示环境）")
    parser.add_argument("--scheduler", action="store_true",
                        help="调度器模式：锚定完成后保持采集暂停，由外部调度器显式 resume 恢复 VIO 与位姿输出")
    args = parser.parse_args()

    map_name: str = args.ref_map
    travel_slam: bool = not args.travel_pure_vio
    map_path = map_dir_path(map_name)
    if not os.path.isdir(map_path):
        raise RuntimeError(f"参考地图文件夹不存在: {map_path}，请先建图")
    print(f"参考地图: {map_name} ({map_path}) —— 仅用于任务点锚定，退出时不写回。")

    # --- IR 流水线（LEFT_IR + RIGHT_IR），初始行进分辨率 ---
    ir_pipeline = open_camera_pipeline(CAMERA_A_SERIAL)
    ir_config = Config()

    sensor_types = {
        ir_pipeline.get_device().get_sensor_list().get_type_by_index(i)
        for i in range(ir_pipeline.get_device().get_sensor_list().get_count())
    }
    if not STEREO_SENSORS.issubset(sensor_types):
        raise RuntimeError("Device does not support dual IR sensors (LEFT_IR + RIGHT_IR required)")

    disable_ir_emitter(ir_pipeline)
    configure_ir_streams_at(ir_pipeline, ir_config, TRAVEL_RESOLUTION, TRAVEL_FPS)
    try:
        ir_pipeline.enable_frame_sync()
    except Exception as e:
        print(f"Warning: Could not enable frame sync: {e}")
    ir_pipeline.start(ir_config)

    print("Getting stereo calibration...")
    stereo_params = get_stereo_calibration(ir_pipeline)
    travel_rig = get_orbbec_vio_rig(stereo_params)
    tracker = make_travel_tracker(travel_rig, slam=travel_slam)

    internals = vslam.Tracker.Internals()
    internals.num_desired_tracks = NUM_DESIRED_TRACKS

    # --- IMU 流水线（ACCEL + GYRO），与视频分开 ---
    imu_pipeline = open_camera_pipeline(CAMERA_A_SERIAL)
    imu_config = Config()
    imu_config.enable_accel_stream()
    imu_config.enable_gyro_stream()

    result_queue: queue.Queue = queue.Queue()
    imu_queue: queue.Queue = queue.Queue(maxsize=IMU_QUEUE_MAX_SIZE)
    thread_with_timestamp = ThreadWithTimestamp(
        (1000.0 / TRAVEL_FPS + 5) * 1e6, IMU_JITTER_THRESHOLD_NS,
    )

    pose_sender = None
    if POSE_UDP_OUTPUT_ENABLED and not args.no_udp:
        pose_sender = UdpPoseSender(host=POSE_UDP_HOST, port=POSE_UDP_PORT)

    # Rerun 可视化器（在启动流水线前创建，避免 fd 继承，参考 finally）。
    visualizer = None
    if not args.no_viz:
        try:
            visualizer = MappingVisualizer(
                num_viz_cameras=NUM_VIZ_CAMERAS,
                image_size=TRAVEL_RESOLUTION,
                show_gravity=SHOW_GRAVITY,
                show_mapping=ENABLE_MAPPING_VISUALIZATION,
                spawn=True,
            )
        except Exception as e:
            print(f"Warning: Rerun visualizer init failed ({e}); continuing without visualization.")
            visualizer = None

    imu_pipeline.start(imu_config)

    stop_event = threading.Event()
    pause_event = threading.Event()
    handoff = AnchorHandoff(pause_event, stop_event)

    imu_thread_obj = threading.Thread(
        target=imu_thread,
        args=(imu_queue, thread_with_timestamp, imu_pipeline, stop_event, pause_event, handoff),
        daemon=True,
    )
    camera_thread_obj = threading.Thread(
        target=camera_thread,
        args=(tracker, result_queue, imu_queue, thread_with_timestamp, ir_pipeline,
              internals, stop_event, pause_event, handoff, travel_slam),
        daemon=True,
    )
    imu_thread_obj.start()
    camera_thread_obj.start()

    cmd_queue: queue.Queue = queue.Queue()
    stdin_thread = threading.Thread(target=stdin_reader, args=(cmd_queue, stop_event, handoff.cancelled), daemon=True)
    stdin_thread.start()

    # ---- 状态 ----
    anchored = False
    pub_pose: Optional[vslam.Pose] = None
    last_odom: Optional[vslam.Pose] = None
    odom_baseline_pending = False
    last_global_pose: Optional[vslam.Pose] = None
    localize_busy = False
    localize_req_time = 0.0
    paused_by_cmd = False
    last_anchor_ok_time = 0.0
    last_pose_generation = 0
    last_sched_pose_emit_m = 0.0
    last_pose_print = 0.0
    frame_id = 0
    trajectory_global: List[np.ndarray] = []

    def request_anchor(reason: str) -> None:
        nonlocal localize_busy, localize_req_time
        if localize_busy:
            return
        if time.monotonic() - last_anchor_ok_time < args.cooldown:
            print(f"\n距上次锚定结束不足 {args.cooldown:.0f}s，忽略 localize（防竞态）。")
            return
        localize_busy = True
        localize_req_time = time.monotonic()
        handoff.start()
        print("\n" + reason)
        sched_emit("[SCHED] anchor=busy")

    def choose_guess() -> Optional[vslam.Pose]:
        """锚定初始 guess：已锚定 → localize 前一帧的全局位姿；否则建图起点（原点 + 单位姿态）。"""
        if anchored and pub_pose is not None:
            guess = vslam.Pose(rotation=list(pub_pose.rotation),
                               translation=list(pub_pose.translation))
            print(f"\n[localize] initial guess = 前一帧全局位姿 {format_pose(guess)}")
            return guess
        print("\n[localize] initial guess = 建图起点（原点 + 单位姿态，粗层 25m 搜索覆盖）")
        return cold_start_guess()

    def drain_result_queue() -> None:
        while True:
            try:
                result_queue.get_nowait()
            except queue.Empty:
                return

    print("Starting task navigation (dual-resolution: travel 640x400@30 / localize 848x480@15)...")
    print(f"Reference map: {map_name}  ({map_path})")
    print(f"Travel SLAM: {'ON' if travel_slam else 'OFF(纯 VIO)'} | "
          f"UDP: {'ON' if pose_sender is not None else 'OFF'} -> {POSE_UDP_HOST}:{POSE_UDP_PORT}")
    print("=" * 62)
    print("终端指令：localize / l = 锚定 | cancel / c = 取消锚定 | pause / p = 暂停 | resume / r = 恢复 | quit / q = 退出")
    print("=" * 62)

    # 起点也要重定位：启动后自动执行一次任务点锚定（保持机器人静止）。
    request_anchor("起点锚定：程序启动，先重定位。请保持机器人静止。")

    try:
        while True:
            # ---- 终端指令 ----
            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break
                if cmd in ("localize", "l"):
                    request_anchor("任务点锚定开始：请保持机器人静止。")
                elif cmd in ("pause", "p"):
                    if localize_busy:
                        print("锚定流程进行中，忽略 pause。")
                    else:
                        paused_by_cmd = True
                        pause_event.set()
                        sched_emit("[SCHED] paused=1")
                elif cmd in ("resume", "r"):
                    if localize_busy:
                        print("锚定流程进行中，忽略 resume。")
                    else:
                        paused_by_cmd = False
                        pause_event.clear()
                        sched_emit("[SCHED] paused=0")
                elif cmd in ("quit", "q", "exit"):
                    raise KeyboardInterrupt

            # ---- 锚定流程状态机 ----
            if stop_event.is_set():
                raise RuntimeError(handoff.data.get('worker_error') or '采集线程已停止')
            if localize_busy and not handoff.ready.is_set():
                # 等待相机线程挂起并交付锚定帧；超时则取消。
                if (time.monotonic() - localize_req_time > ANCHOR_PREPARE_TIMEOUT_S
                        and not handoff.cancelled.is_set()):
                    print('定位预热超时，正在取消；等待采集线程安全交接后再报告失败。')
                    handoff.cancel()

            if localize_busy and handoff.ready.is_set():
                # 两采集线程已挂起；此刻执行同步 localize 是安全的。
                drain_result_queue()
                holder = handoff.data
                pose = None
                status = "fail"

                if handoff.cancelled.is_set():
                    status = "cancel"
                elif not holder.get('warm_ok'):
                    status = holder.get('anchor_error') or "fail"
                    print(holder.get('anchor_error') or '锚定帧不可用。')
                else:
                    guess = choose_guess()
                    try:
                        pose, status = localize_anchor(
                            holder['anchor_tracker'],
                            map_dir_path(map_name),
                            holder['timestamp_ns'],
                            holder['images'],
                            guess,
                            progress_cb=print,
                            cancel_event=handoff.cancelled,
                        )
                    except Exception as e:
                        print(f"localize_anchor 异常: {e}")
                        pose, status = None, "fail"

                # 恢复相机线程（切回行进流、重建行进 tracker）。
                handoff.finish()
                localize_busy = False
                last_anchor_ok_time = time.monotonic()

                # 锚定后首帧只重置 odom 基线，不累积增量。
                drain_result_queue()
                last_pose_generation = handoff.generation
                odom_baseline_pending = True
                last_odom = None

                if status == "ok" and pose is not None:
                    anchored = True
                    pub_pose = pose
                    last_global_pose = pose
                    save_last_pose(pose, map_name)
                    print(f"锚定成功：本段起点位姿 {format_pose(pose)}")
                    _t, _r = pose.translation, pose.rotation
                    sched_emit(f"[SCHED] anchor=ok pose=({_t[0]:.4f},{_t[1]:.4f},{_t[2]:.4f},"
                               f"{_r[0]:.4f},{_r[1]:.4f},{_r[2]:.4f},{_r[3]:.4f})")
                elif status == "cancel":
                    print("锚定已取消（用户取消）。")
                    sched_emit("[SCHED] anchor=fail")
                elif status == "timeout":
                    print("锚定超时：请调整机器人位姿后再次 localize。")
                    sched_emit("[SCHED] anchor=fail")
                else:
                    print("锚定失败：请调整机器人位姿后再次 localize。")
                    sched_emit("[SCHED] anchor=fail")

                if not paused_by_cmd and not args.scheduler:
                    pause_event.clear()

            # ---- 结果消费 ----
            try:
                result = result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # 丢弃积压旧帧，只保留最新一帧（odom 增量是相对变换，跳帧不影响累积；参考 finally）。
            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break
            if localize_busy:
                continue  # 锚定准备期不消费旧结果

            timestamp, odom_pose, slam_pose, generation, images, observations, gravity = result

            if generation != last_pose_generation:
                if generation < last_pose_generation:
                    continue  # 旧 tracker 结果，丢弃
                last_pose_generation = generation
                odom_baseline_pending = True
                last_odom = None

            if odom_pose is None:
                continue

            # ---- 段内纯 VIO 出口位姿（odom 帧间增量累积）----
            if USE_ODOM_FOR_POSE_OUTPUT and anchored:
                if odom_baseline_pending:
                    last_odom = odom_pose
                    odom_baseline_pending = False
                elif last_odom is not None and pub_pose is not None:
                    delta = pose_compose(pose_inverse(last_odom), odom_pose)
                    pub_pose = pose_compose(pub_pose, delta)
                    last_odom = odom_pose

            if pub_pose is not None:
                last_global_pose = pub_pose
            elif slam_pose is not None:
                last_global_pose = slam_pose

            # ---- 位姿出口：UDP + [SCHED] pose + 实时坐标打印 ----
            if (anchored or SEND_POSE_BEFORE_ANCHOR) and not paused_by_cmd and not localize_busy:
                out_pose = None
                if USE_ODOM_FOR_POSE_OUTPUT and anchored and pub_pose is not None:
                    out_pose = pub_pose
                elif slam_pose is not None:
                    out_pose = slam_pose
                if out_pose is not None:
                    now_m = time.monotonic()
                    if pose_sender is not None:
                        pose_sender.send_pose(out_pose.translation, out_pose.rotation)
                    if anchored and now_m - last_sched_pose_emit_m >= POSE_PRINT_INTERVAL_S:
                        last_sched_pose_emit_m = now_m
                        _t, _r = out_pose.translation, out_pose.rotation
                        sched_emit_file(f"[SCHED] pose=({_t[0]:.4f},{_t[1]:.4f},{_t[2]:.4f},"
                                        f"{_r[0]:.4f},{_r[1]:.4f},{_r[2]:.4f},{_r[3]:.4f}) "
                                        f"t={time.time():.3f} source_timestamp_ns={timestamp}")
                    if now_m - last_pose_print >= POSE_PRINT_INTERVAL_S:
                        last_pose_print = now_m
                        prefix = "[odom ]" if (USE_ODOM_FOR_POSE_OUTPUT and anchored) else \
                            ("[global]" if anchored else "[local ]")
                        print(f"\r{prefix} {format_pose(out_pose)}          ", end="", flush=True)

            # ---- Rerun 可视化（参考 finally 版）----
            if visualizer is not None:
                frame_id += 1
                # 轨迹只在锚定后累加（全局系），避免把本地系段混进地图轨迹。
                if anchored and pub_pose is not None:
                    viz_pose = pub_pose
                    trajectory_global.append(np.asarray(viz_pose.translation, dtype=np.float64))
                elif slam_pose is not None:
                    viz_pose = slam_pose
                else:
                    viz_pose = odom_pose
                if viz_pose is not None:
                    visualizer.visualize_frame(
                        frame_id=frame_id,
                        images=[images[0]],
                        pose=viz_pose,
                        observations_main_cam=[observations],
                        trajectory=trajectory_global,
                        timestamp=timestamp,
                        gravity=gravity,
                    )

    except KeyboardInterrupt:
        print("Stopping task navigation...")
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_event.set()
        handoff.release.set()  # 解除相机线程可能挂起的交接等待
        pause_event.clear()
        imu_pipeline.stop()
        ir_pipeline.stop()
        imu_thread_obj.join(timeout=5.0)
        camera_thread_obj.join(timeout=5.0)
        if imu_thread_obj.is_alive() or camera_thread_obj.is_alive():
            print("WARNING: 采集线程未在 5s 内退出（仍为 daemon，将随进程结束）")
        if pose_sender is not None:
            pose_sender.close()
        if anchored and last_global_pose is not None:
            save_last_pose(last_global_pose, map_name)
        print("Pipelines stopped. Clean shutdown complete.")


if __name__ == "__main__":
    main()
