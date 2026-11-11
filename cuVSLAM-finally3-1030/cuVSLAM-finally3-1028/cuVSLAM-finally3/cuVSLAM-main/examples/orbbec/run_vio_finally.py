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

import gc
import os
import queue
import signal
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Optional

import numpy as np

from pyorbbecsdk import (
    Config,
    OBFormat,
    OBFrameType,
    OBPermissionType,
    OBPropertyID,
    OBSensorType,
    Pipeline,
)

import cuvslam as vslam
from camera_utils import get_orbbec_stereo_rig, get_stereo_calibration, process_ir_frame
from visualizer_mapping import MappingVisualizer

# Gemini 336L 支持的 IR 分辨率与帧率（LEFT_IR 与 RIGHT_IR 一致）。
# 由 query_device_info.py 枚举得到；除注明外均为 Y8 格式：
#     1280x800: 5/10/15/30 fps    （另有 25 fps，格式为 Y12/Y16）
#     1280x720: 5/10/15/30 fps
#      848x480: 5/10/15/30/60 fps
#      640x480: 5/10/15/30/60/90 fps
#      640x400: 5/10/15/30 fps    （另有 25 fps，格式为 Y12/Y16）
#      640x360: 5/10/15/30/60/90 fps
#      480x270: 5/10/15/30/60/90 fps
#      424x266: 5/10/15/30 fps
#      424x240: 5/10/15/30/60/90 fps
#      848x100: 100 fps           （高速条带模式）
RESOLUTION = (640, 400)
FPS = 30

# IMU 采样频率（Gemini 336L 内置 IMU 约以 200 Hz 输出）
IMU_FREQUENCY = 200

# IMU 噪声参数（沿用 RealSense BMI055 的默认值；Orbbec SDK 未暴露这些参数）。
# 可用 kalibr 标定以获得更高精度：https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
IMU_GYROSCOPE_NOISE_DENSITY = 6.0673370376614875e-03
IMU_GYROSCOPE_RANDOM_WALK = 3.6211951458325785e-05
IMU_ACCELEROMETER_NOISE_DENSITY = 3.3621979208052800e-02
IMU_ACCELEROMETER_RANDOM_WALK = 9.8256589971851467e-04

FRAME_PERIOD_MS = 1000 / FPS
IMAGE_JITTER_THRESHOLD_NS = (FRAME_PERIOD_MS + 5) * 1e6  # 相机帧间隔容差
IMU_JITTER_THRESHOLD_NS = 20 * 1e6  # IMU 采样间隔容差（~200 Hz -> ~5 ms）
IMU_QUEUE_MAX_SIZE = IMU_FREQUENCY * 5  # 最多缓冲约 5 s 的 IMU 数据

SHOW_GRAVITY = False  # 可视化估计的重力向量

USE_SLAM_MODE = True  # 启用回环检测 + 位姿图优化
SLAM_SYNC_MODE = False  # False = SLAM 在后台线程运行（推荐）
PLANAR_CONSTRAINTS = True  # 将 SLAM 位姿约束到平面（平面运动时设为 True）
LOOP_CLOSURE_THROTTLING_MS = 1000  # 回环节流间隔（ms）；0 = 不限制

# 是否在 Rerun 里实时显示建图内容（稀疏地图点 / 位姿图 / 回环位置）。
# True  = 显示建图；False = 仅显示轨迹与 2D 特征（与 run_vio_slam.py 一致），并跳过建图数据读取。
ENABLE_MAPPING_VISUALIZATION = False
# 建图数据（地图点 / 位姿图）的刷新节流间隔（ms），避免每帧读取全量 landmarks。
MAP_UPDATE_INTERVAL_MS = 200

# 是否把 Rerun 录制（点云 / 轨迹 / 回环）保存为 .rrd 文件。
# True  = 从开始就落盘到 vio_mapping_tracking.rrd；False = 只在 viewer 中实时显示，不保存。
SAVE_RECORDING = False

# 是否保存用于重定位的 SLAM 地图（LMDB 数据库，供后续 tracker.localize_in_map 重定位使用）。
# True  = 程序结束（Ctrl+C）时把地图保存到 RELOCALIZATION_MAP_DIR；False = 不保存。
# 注意：需要 USE_SLAM_MODE = True 才有地图可保存；每次运行会覆盖该目录内容。
SAVE_RELOCALIZATION_MAP = False
# 重定位地图的保存目录（LMDB 文件夹，内部含 data.mdb）。
RELOCALIZATION_MAP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vio_relocalization_map"
)

NUM_DESIRED_TRACKS = 700  # 每帧期望的特征跟踪数量

STEREO_SENSORS = {OBSensorType.LEFT_IR_SENSOR, OBSensorType.RIGHT_IR_SENSOR}


@dataclass
class ImuSample:
    """注册进 cuVSLAM 之前缓存的原始 IMU 采样。"""

    timestamp_ns: int
    linear_accelerations: tuple
    angular_velocities: tuple


class ThreadWithTimestamp:
    """用于在相机线程与 IMU 线程之间共享时间戳的辅助类。"""

    def __init__(self, low_rate_threshold_ns: int, high_rate_threshold_ns: int) -> None:
        self.prev_low_rate_timestamp: Optional[int] = None
        self.prev_high_rate_timestamp: Optional[int] = None
        self.low_rate_threshold_ns = low_rate_threshold_ns
        self.high_rate_threshold_ns = high_rate_threshold_ns
        self.last_low_rate_timestamp: Optional[int] = None


def disable_ir_emitter(pipeline: Pipeline) -> None:
    """关闭 IR 激光发射器，用于被动双目跟踪。

    IR 发射器会投射散斑图案，干扰立体匹配。
    被动双目（以及双目惯性）跟踪必须将其关闭。

    Args:
        pipeline: Orbbec pipeline
    """
    device = pipeline.get_device()
    emitter_disabled = False

    # 尝试 OB_PROP_LASER_BOOL（最常见）
    try:
        if device.is_property_supported(
            OBPropertyID.OB_PROP_LASER_BOOL, OBPermissionType.PERMISSION_READ_WRITE
        ):
            device.set_bool_property(OBPropertyID.OB_PROP_LASER_BOOL, False)
            current = device.get_bool_property(OBPropertyID.OB_PROP_LASER_BOOL)
            if not current:
                print("IR emitter disabled (OB_PROP_LASER_BOOL)")
                emitter_disabled = True
            else:
                print("Warning: Failed to disable laser via OB_PROP_LASER_BOOL")
    except Exception as e:
        print(f"OB_PROP_LASER_BOOL not available: {e}")

    # 尝试 OB_PROP_LASER_CONTROL_INT（0=关，1=开，2=自动）
    if not emitter_disabled:
        try:
            if device.is_property_supported(
                OBPropertyID.OB_PROP_LASER_CONTROL_INT, OBPermissionType.PERMISSION_READ_WRITE
            ):
                device.set_int_property(OBPropertyID.OB_PROP_LASER_CONTROL_INT, 0)
                print("IR emitter disabled (OB_PROP_LASER_CONTROL_INT)")
                emitter_disabled = True
        except Exception as e:
            print(f"OB_PROP_LASER_CONTROL_INT not available: {e}")

    # 尝试 OB_PROP_LDP_BOOL（激光二极管功率）
    if not emitter_disabled:
        try:
            if device.is_property_supported(
                OBPropertyID.OB_PROP_LDP_BOOL, OBPermissionType.PERMISSION_READ_WRITE
            ):
                device.set_bool_property(OBPropertyID.OB_PROP_LDP_BOOL, False)
                print("IR emitter disabled (OB_PROP_LDP_BOOL)")
                emitter_disabled = True
        except Exception as e:
            print(f"OB_PROP_LDP_BOOL not available: {e}")

    if not emitter_disabled:
        print("Warning: Could not disable IR emitter - device may not support this feature")


def get_orbbec_imu(
    frequency: int = IMU_FREQUENCY,
    gyroscope_noise_density: float = IMU_GYROSCOPE_NOISE_DENSITY,
    gyroscope_random_walk: float = IMU_GYROSCOPE_RANDOM_WALK,
    accelerometer_noise_density: float = IMU_ACCELEROMETER_NOISE_DENSITY,
    accelerometer_random_walk: float = IMU_ACCELEROMETER_RANDOM_WALK,
) -> vslam.ImuCalibration:
    """为 Orbbec 内置 IMU 构建 IMU 标定对象。

    Orbbec SDK 未暴露 IMU<->相机外参和噪声模型，因此：
    - rig_from_imu 为占位值（单位旋转、零平移）；请用 kalibr 标定。
    - 噪声密度 / 随机游走沿用 RealSense BMI055 的默认值。
    """
    imu = vslam.ImuCalibration()
    # 单位四元数（x, y, z, w）+ 零平移。
    imu.rig_from_imu = vslam.Pose(rotation=[0.0, 0.0, 0.0, 1.0], translation=[0.0, 0.0, 0.0])
    imu.gyroscope_noise_density = gyroscope_noise_density
    imu.gyroscope_random_walk = gyroscope_random_walk
    imu.accelerometer_noise_density = accelerometer_noise_density
    imu.accelerometer_random_walk = accelerometer_random_walk
    imu.frequency = frequency
    return imu


def get_orbbec_vio_rig(
    stereo_params: dict,
    gyroscope_noise_density: float = IMU_GYROSCOPE_NOISE_DENSITY,
    gyroscope_random_walk: float = IMU_GYROSCOPE_RANDOM_WALK,
    accelerometer_noise_density: float = IMU_ACCELEROMETER_NOISE_DENSITY,
    accelerometer_random_walk: float = IMU_ACCELEROMETER_RANDOM_WALK,
) -> vslam.Rig:
    """根据 Orbbec 双目参数构建 VIO 配置（2 个相机 + 1 个 IMU）。"""
    rig = get_orbbec_stereo_rig(stereo_params)
    rig.imus = [
        get_orbbec_imu(
            gyroscope_noise_density=gyroscope_noise_density,
            gyroscope_random_walk=gyroscope_random_walk,
            accelerometer_noise_density=accelerometer_noise_density,
            accelerometer_random_walk=accelerometer_random_walk,
        )
    ]
    return rig


def imu_thread(
    imu_queue: queue.Queue,
    thread_with_timestamp: ThreadWithTimestamp,
    imu_pipe: Pipeline,
    stop_event: threading.Event,
) -> None:
    """IMU 采集线程：读取加速度计 + 陀螺仪并写入 ImuSample 队列。

    Orbbec IMU 将加速度计和陀螺仪暴露为两个独立传感器，但由同一个物理 IMU 采样。
    SDK 可能把二者放在同一个 FrameSet 中，也可能分开返回，因此我们保留最新的陀螺仪值，
    每来一帧新的加速度计数据就输出一个 ImuSample（以加速度计作为时钟）。
    """
    high_rate_threshold = thread_with_timestamp.high_rate_threshold_ns
    prev_timestamp = None
    drop_count = 0
    queue_drop_count = 0
    last_gyro = None  # 最新的 (gx, gy, gz)

    try:
        while not stop_event.is_set():
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

            current_timestamp_us = int(accel_frame.get_timestamp_us())
            current_timestamp = current_timestamp_us * 1000

            # 时间戳间隔检查
            if prev_timestamp is not None:
                timestamp_diff = current_timestamp - prev_timestamp
                if timestamp_diff < 0:
                    # 时间戳回绕/重置：重新对齐基准，避免 prev_timestamp 卡在回绕前的大值上。
                    print(
                        f"Warning: IMU timestamp reset/wrap detected: "
                        f"prev ts_us={prev_timestamp // 1000}, current ts_us={current_timestamp_us}"
                    )
                    prev_timestamp = current_timestamp
                    continue
                if timestamp_diff > high_rate_threshold:
                    drop_count += 1
                    if drop_count % 100 == 1:
                        print(
                            f"Warning: IMU drops detected ({drop_count} total, "
                            f"last gap: {timestamp_diff/1e6:.2f} ms; "
                            f"prev ts_us={prev_timestamp // 1000}, current ts_us={current_timestamp_us})"
                        )
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
                # 丢弃最旧的采样，避免采集线程被阻塞。
                try:
                    imu_queue.get_nowait()
                except queue.Empty:
                    pass
                imu_queue.put_nowait(sample)
                if queue_drop_count % 100 == 1:
                    print(f"Warning: IMU queue overflow ({queue_drop_count} dropped samples)")
    except Exception as e:
        print(f"IMU thread error: {e}")


def register_imu_until(
    tracker: vslam.Tracker,
    imu_queue: queue.Queue,
    pending_imu: Deque[ImuSample],
    timestamp_ns: int,
    last_tracker_timestamp_ns: Optional[int],
) -> Optional[int]:
    """把缓存的 IMU 采样注册到当前图像时间戳为止。

    cuVSLAM 要求 Track 与 register_imu_measurement 的调用按时间戳顺序进行。
    IMU 采集在独立线程中运行；所有 tracker 调用都由相机线程汇总到这里。
    """
    while True:
        try:
            pending_imu.append(imu_queue.get_nowait())
        except queue.Empty:
            break

    while pending_imu and pending_imu[0].timestamp_ns <= timestamp_ns:
        sample = pending_imu.popleft()
        if last_tracker_timestamp_ns is not None and sample.timestamp_ns < last_tracker_timestamp_ns:
            continue
        imu_measurement = vslam.ImuMeasurement()
        imu_measurement.timestamp_ns = sample.timestamp_ns
        imu_measurement.linear_accelerations = sample.linear_accelerations
        imu_measurement.angular_velocities = sample.angular_velocities
        tracker.register_imu_measurement(0, imu_measurement)
        last_tracker_timestamp_ns = sample.timestamp_ns
    return last_tracker_timestamp_ns


def camera_thread(
    tracker: vslam.Tracker,
    result_queue: queue.Queue,
    imu_queue: queue.Queue,
    thread_with_timestamp: ThreadWithTimestamp,
    ir_pipe: Pipeline,
    show_gravity: bool,
    internals: vslam.Tracker.Internals,
    stop_event: threading.Event,
) -> None:
    """相机处理线程：读取双目 IR，注册 IMU，并进行跟踪。"""
    pending_imu: Deque[ImuSample] = deque()
    last_tracker_timestamp: Optional[int] = None
    last_map_update_ns = 0
    last_trajectory_update_ns = 0
    camera_drop_count = 0

    try:
        while not stop_event.is_set():
            frames = ir_pipe.wait_for_frames(100)
            if frames is None:
                continue

            left_frame = frames.get_frame(OBFrameType.LEFT_IR_FRAME)
            right_frame = frames.get_frame(OBFrameType.RIGHT_IR_FRAME)
            if left_frame is None or right_frame is None:
                continue

            current_timestamp = int(left_frame.get_timestamp_us() * 1000)

            # 相机流间隔检查
            if thread_with_timestamp.prev_low_rate_timestamp is not None:
                timestamp_diff = current_timestamp - thread_with_timestamp.prev_low_rate_timestamp
                if timestamp_diff > thread_with_timestamp.low_rate_threshold_ns:
                    camera_drop_count += 1
                    if camera_drop_count % 100 == 1:
                        prev_ts = thread_with_timestamp.prev_low_rate_timestamp
                        print(
                            f"Warning: Camera stream message drop: timestamp gap "
                            f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                            f"{thread_with_timestamp.low_rate_threshold_ns/1e6:.2f} ms "
                            f"({camera_drop_count} total); "
                            f"prev frame ts={prev_ts} ns ({prev_ts/1e6:.2f} ms), "
                            f"current frame ts={current_timestamp} ns ({current_timestamp/1e6:.2f} ms)"
                        )
            thread_with_timestamp.prev_low_rate_timestamp = current_timestamp

            left_img = process_ir_frame(left_frame)
            right_img = process_ir_frame(right_frame)
            if left_img is None or right_img is None:
                print("Warning: Failed to convert IR frames; skipping frame pair")
                continue

            images = (left_img, right_img)

            last_tracker_timestamp = register_imu_until(
                tracker, imu_queue, pending_imu, current_timestamp, last_tracker_timestamp
            )
            odom_pose_estimate, slam_pose = tracker.track(current_timestamp, images, internals=internals)
            last_tracker_timestamp = current_timestamp

            odom_pose_with_cov = odom_pose_estimate.world_from_rig
            if odom_pose_with_cov is None:
                print(f"Tracking failed at frame {current_timestamp}")
                continue

            observations = tracker.get_last_observations(0)
            gravity = tracker.get_last_gravity() if show_gravity else None

            # 读取 SLAM 平滑轨迹（回环后回溯矫正历史位姿），按节流间隔刷新。
            # 需与 track 同线程调用（满足 Slam 单线程要求）。
            slam_poses = None
            if slam_pose is not None and (
                current_timestamp - last_trajectory_update_ns
            ) >= MAP_UPDATE_INTERVAL_MS * 1e6:
                slam_poses = tracker.get_all_slam_poses()
                last_trajectory_update_ns = current_timestamp

            # 读取建图数据（与 track 同线程，满足 Slam 单线程要求），按节流间隔刷新。
            map_landmarks = None
            loop_closure_poses = None
            if ENABLE_MAPPING_VISUALIZATION and (
                current_timestamp - last_map_update_ns
            ) >= MAP_UPDATE_INTERVAL_MS * 1e6:
                map_landmarks = tracker.get_slam_landmarks(vslam.Tracker.SlamDataLayer.Map)
                loop_closure_poses = tracker.get_loop_closure_poses()
                last_map_update_ns = current_timestamp

            result_queue.put(
                [
                    current_timestamp,
                    odom_pose_with_cov.pose,
                    images,
                    observations,
                    gravity,
                    slam_pose,
                    slam_poses,
                    map_landmarks,
                    loop_closure_poses,
                ]
            )
            thread_with_timestamp.last_low_rate_timestamp = current_timestamp
    except Exception as e:
        print(f"Camera thread error: {e}")


def configure_ir_streams(pipeline: Pipeline, config: Config) -> None:
    """在 pipeline 的配置中启用 LEFT_IR + RIGHT_IR 流。"""
    for sensor_type in [OBSensorType.LEFT_IR_SENSOR, OBSensorType.RIGHT_IR_SENSOR]:
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        profile = None
        for fmt in [OBFormat.Y8, OBFormat.Y16]:
            try:
                profile = profile_list.get_video_stream_profile(RESOLUTION[0], RESOLUTION[1], fmt, FPS)
                break
            except Exception as e:
                print(f"Warning: Failed to get stream profile for {sensor_type}: {e}")
                continue
        if profile is None:
            profile = profile_list.get_default_video_stream_profile()
            print(
                f"Using default profile for {sensor_type}: "
                f"{profile.get_width()}x{profile.get_height()} @ {profile.get_fps()} FPS"
            )
        config.enable_stream(profile)


def save_relocalization_map(tracker: vslam.Tracker, map_dir: str, timeout_s: float = 30.0) -> None:
    """把当前 SLAM 地图保存为 LMDB 数据库（供后续 localize_in_map 重定位使用）。

    ``save_map`` 是异步的：回调参数 ``success=True`` 表示保存完成。这里轮询等待回调，
    最长等待 ``timeout_s`` 秒。
    """
    save_done = threading.Event()
    save_result = [False]

    def on_save(success: bool) -> None:
        save_result[0] = success
        save_done.set()

    tracker.save_map(map_dir, on_save)

    waited = 0.0
    while not save_done.wait(timeout=0.1):
        waited += 0.1
        if waited >= timeout_s:
            print(f"WARNING: Map save did not complete within {timeout_s}s")
            return
        if int(waited * 10) % 50 == 0:
            print(f"  waiting for map save... {waited:.1f}s")

    if save_result[0]:
        print(f"Relocalization map saved to: {map_dir}")
    else:
        print("WARNING: Map save failed (callback returned False)")


def main() -> None:
    """使用 Orbbec Gemini 336L 进行双目惯性 VIO + SLAM 跟踪，可实时可视化建图。"""
    # --- IR 流水线（LEFT_IR + RIGHT_IR）---
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

    # 双目标定（从已在运行的 IR 流水线读取）。
    print("Getting stereo calibration...")
    stereo_params = get_stereo_calibration(ir_pipeline)

    # 构建 VIO 配置（2 个相机 + 1 个 IMU）。
    rig = get_orbbec_vio_rig(stereo_params)

    # 配置 tracker 用于双目惯性 VIO。
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=True,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        debug_imu_mode=False,
        odometry_mode=vslam.Tracker.OdometryMode.Inertial,
        rectified_stereo_camera=True,
        use_gpu=True,
    )

    slam_cfg = (
        vslam.Tracker.SlamConfig(
            sync_mode=SLAM_SYNC_MODE,
            planar_constraints=PLANAR_CONSTRAINTS,
            throttling_time_ms=LOOP_CLOSURE_THROTTLING_MS,
            use_gpu=True,
        )
        if USE_SLAM_MODE
        else None
    )

    tracker = vslam.Tracker(rig, cfg, slam_cfg)

    internals = vslam.Tracker.Internals()
    internals.num_desired_tracks = NUM_DESIRED_TRACKS

    # --- IMU 流水线（ACCEL + GYRO），按 SDK 建议与视频分开 ---
    imu_pipeline = Pipeline()
    imu_config = Config()
    # 使用设备默认的量程 / 采样率（与 SDK 的 07_imu.py 示例一致）。
    imu_config.enable_accel_stream()
    imu_config.enable_gyro_stream()

    # 队列与可视化器（在流水线之前启动 Rerun，避免 fd 继承）。
    q = queue.Queue()
    imu_queue = queue.Queue(maxsize=IMU_QUEUE_MAX_SIZE)
    visualizer = MappingVisualizer(
        image_size=RESOLUTION,
        show_gravity=SHOW_GRAVITY,
        show_mapping=ENABLE_MAPPING_VISUALIZATION,
        spawn=True,
        save_path=(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"vio_mapping_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rrd",
            )
            if SAVE_RECORDING
            else None
        ),
    )
    thread_with_timestamp = ThreadWithTimestamp(IMAGE_JITTER_THRESHOLD_NS, IMU_JITTER_THRESHOLD_NS)

    imu_pipeline.start(imu_config)

    # 用于通知采集线程退出，以便在程序结束时释放 tracker / pipeline 占用的内存。
    stop_event = threading.Event()
    imu_thread_obj = threading.Thread(
        target=imu_thread,
        args=(imu_queue, thread_with_timestamp, imu_pipeline, stop_event),
        daemon=True,
    )
    camera_thread_obj = threading.Thread(
        target=camera_thread,
        args=(tracker, q, imu_queue, thread_with_timestamp, ir_pipeline, SHOW_GRAVITY, internals, stop_event),
        daemon=True,
    )
    imu_thread_obj.start()
    camera_thread_obj.start()

    frame_id = 0
    trajectory_slam: List[np.ndarray] = []

    print("Starting VIO+SLAM tracking with cuvslam...")
    print(f"Mapping visualization: {'ON' if ENABLE_MAPPING_VISUALIZATION else 'OFF'}")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # 取出一帧（最多阻塞 1s）
            try:
                result = q.get(timeout=1.0)
            except queue.Empty:
                continue

            # 丢弃队列里积压的旧帧，只保留最新一帧用于显示。
            # 否则可视化速度跟不上相机帧率时，显示会越来越滞后。
            while True:
                try:
                    result = q.get_nowait()
                except queue.Empty:
                    break

            (
                timestamp,
                odom_pose,
                images,
                observations,
                gravity,
                slam_pose,
                slam_poses,
                map_landmarks,
                loop_closure_poses,
            ) = result

            if odom_pose is None:
                continue

            frame_id += 1
            # 用 SLAM 平滑轨迹（回环后回溯矫正）替换逐帧追加，避免回环跳变。
            if slam_poses is not None:
                trajectory_slam = [p.pose.translation for p in slam_poses]

            # 一旦 SLAM 优化结果可用，优先使用它。
            viz_pose = slam_pose if slam_pose is not None else odom_pose

            visualizer.visualize_frame(
                frame_id=frame_id,
                images=[images[0]],
                pose=viz_pose,
                observations_main_cam=[observations],
                trajectory=trajectory_slam,
                timestamp=timestamp,
                gravity=gravity,
                map_landmarks=map_landmarks,
                loop_closure_poses=loop_closure_poses,
            )

    except KeyboardInterrupt:
        print("Stopping VIO+SLAM tracking...")
    finally:
        # 清理期间忽略 SIGINT，避免第二次 Ctrl+C 中断流水线关闭导致 std::terminate。
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # 先保存重定位地图（LMDB），再关闭流水线，确保进程退出前地图已写盘。
        if SAVE_RELOCALIZATION_MAP and USE_SLAM_MODE:
            save_relocalization_map(tracker, RELOCALIZATION_MAP_DIR)
        # 停止采集线程：它们持有 tracker / pipeline 的引用，必须先退出，
        # 否则 tracker 无法被回收，GPU 显存不会释放。
        stop_event.set()
        imu_thread_obj.join(timeout=2.0)
        camera_thread_obj.join(timeout=2.0)
        imu_pipeline.stop()
        ir_pipeline.stop()
        # 显式释放 cuVSLAM tracker（C++ Odometry/Slam 实例）占用的 CPU/GPU 内存。
        del tracker
        gc.collect()
        print("Pipelines stopped, memory released.")


if __name__ == "__main__":
    main()
