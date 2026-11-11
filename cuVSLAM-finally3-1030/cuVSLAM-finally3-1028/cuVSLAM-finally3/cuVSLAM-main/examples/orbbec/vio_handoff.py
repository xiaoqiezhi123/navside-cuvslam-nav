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

"""Camera-owned tracker lifecycle and ordered IMU buffers. No SDK imports.

从 A* 回廊版 vio_handoff.py 移植，并落实迁移方案 §六 的 6 个缺陷修复：
  1. 锚定请求前真正解除已有暂停（start() 先 clear pause，让 IMU 恢复供预热取新鲜数）；
  2. 行进 tracker 替换在相机线程侧通过共享容器（data/anchor_tracker + generation）原子交接；
  3. 预热跟踪后更新 tracker 时间戳为最后图像时间戳（TrackerImuBuffer.tracked）；
  4. 预热循环加请求取消检查（cancelled Event）；
  5. last-pose 自动重试同样走冷却（由主循环 request_anchor 的冷却闸统一控制）；
  6. 恢复/取消/重试交接状态机明确化（start/park/finish/complete/fail/cancel 六态闭环）。

相对 A* 版：去掉 use_lastpose_guess（guess 由主循环直接决定，不经过握手容器）；
本版本用于「阶段切换重配 IR 流」的双分辨率场景，握手事件含义不变。
"""

from collections import deque
import queue
import threading
import time


def wait_while_paused(pause, stop, acknowledgement=None):
    """已置位的 pause Event 本身不能作休眠等待，需用 stop.wait() 提供可中断睡眠。"""
    if not pause.is_set():
        return False
    try:
        while pause.is_set() and not stop.is_set():
            if acknowledgement is not None:
                acknowledgement.set()
            stop.wait(.02)
    finally:
        if acknowledgement is not None:
            acknowledgement.clear()
    return True


class AnchorHandoff:
    """一次只处理一个锚定请求；ready 表示相机与 IMU 均已挂起。

    主线程调用 start()，仅在 ready 后才执行 localize，随后 finish()。
    相机线程预热临时定位 tracker，调用 park() 挂起，release 后重建行进 VIO tracker，
    再 complete()。新的请求不能越过这次重建。
    """

    def __init__(self, pause, stop):
        self.pause, self.stop = pause, stop
        self.requested = threading.Event()
        self.ready = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.imu_parked = threading.Event()
        self.cancelled = threading.Event()
        self.generation = 0
        self.data = dict(images=None, timestamp_ns=None, anchor_tracker=None,
                         warm_ok=False, anchor_error=None, worker_error=None)

    def start(self):
        if self.requested.is_set() or self.stop.is_set():
            raise RuntimeError('上一轮 tracker 交接尚未完成或采集已停止')
        self.ready.clear()
        self.release.clear()
        self.finished.clear()
        self.cancelled.clear()
        self.imu_parked.clear()
        self.generation += 1
        self.data.update(images=None, timestamp_ns=None, anchor_tracker=None,
                         warm_ok=False, anchor_error=None, worker_error=None)
        # 发布请求前先解除已有暂停：相机线程进入预热、IMU 恢复采集，取到新鲜 IMU。
        self.requested.set()
        self.pause.clear()

    def cancel(self):
        """请求取消：相机线程预热循环会检测到并提前安全交接（缺陷 4/6）。"""
        self.cancelled.set()

    def park(self, timeout_s=3.0):
        self.pause.set()
        deadline = time.monotonic() + timeout_s
        while not self.imu_parked.wait(.02):
            if self.stop.is_set() or time.monotonic() >= deadline:
                raise RuntimeError('IMU 线程未确认挂起，禁止进入 localize')
        self.ready.set()
        while not self.release.wait(.02):
            if self.stop.is_set():
                return False
        self.release.clear()
        return not self.stop.is_set()

    def finish(self, timeout_s=15.0):
        if not self.ready.is_set() or not self.imu_parked.is_set():
            raise RuntimeError('定位结束时采集握手状态不一致')
        self.release.set()
        deadline = time.monotonic() + timeout_s
        while not self.finished.wait(.02):
            if self.stop.is_set() or time.monotonic() >= deadline:
                raise RuntimeError('相机线程未完成新行进 tracker 交接')
        if self.data.get('worker_error'):
            raise RuntimeError(self.data['worker_error'])
        self.ready.clear()

    def complete(self, tracker):
        self.data.update(vio_generation=self.generation, vio_tracker_id=id(tracker))
        self.requested.clear()
        self.finished.set()

    def fail(self, error):
        self.data['worker_error'] = str(error)
        self.stop.set()
        self.finished.set()


class TrackerImuBuffer:
    """状态属于单个 tracker，绝不在行进 VIO 与定位 tracker 之间共享。"""

    def __init__(self):
        self.pending = deque()
        self.last_timestamp = None
        self.last_image = None
        self.discarded = 0
        self.reason = None

    def prepare(self, timestamp_ns, imu_queue, stop, cancelled=None, timeout_s=.15,
                max_lag_ns=20_000_000, max_ahead_ns=250_000_000):
        self.reason = None
        if stop.is_set() or (cancelled is not None and cancelled.is_set()):
            self.reason = 'cancelled'
            return False
        if self.last_image is not None and timestamp_ns <= self.last_image:
            self.reason = 'image_timestamp_not_increasing'
            return False
        # 全新 tracker 不得摄入长暂停期间遗留的 IMU：保留一个短暂的当前初始化窗口，
        # 之后每次 Track 推进上界。
        if self.last_image is None:
            self.last_timestamp = timestamp_ns - 200_000_000
        deadline = time.monotonic() + timeout_s
        while True:
            while True:
                try:
                    self.pending.append(imu_queue.get_nowait())
                except queue.Empty:
                    break
            while self.pending and self.pending[0].timestamp_ns < self.last_timestamp:
                self.pending.popleft()
                self.discarded += 1
            before = [s.timestamp_ns for s in self.pending if s.timestamp_ns <= timestamp_ns]
            newest = self.pending[-1].timestamp_ns if self.pending else None
            if newest is not None and newest - timestamp_ns > max_ahead_ns:
                self.reason = 'image_behind_imu'
                return False
            if before and 0 <= timestamp_ns - before[-1] <= max_lag_ns:
                return True
            if stop.is_set() or (cancelled is not None and cancelled.is_set()):
                self.reason = 'cancelled'
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.reason = 'imu_not_fresh'
                return False
            try:
                self.pending.append(imu_queue.get(timeout=min(.01, remaining)))
            except queue.Empty:
                pass

    def tracked(self, timestamp_ns):
        # 即使一次 Track 没有得到有效位姿，也消费了图像时间戳。
        self.last_timestamp = timestamp_ns
        self.last_image = timestamp_ns
