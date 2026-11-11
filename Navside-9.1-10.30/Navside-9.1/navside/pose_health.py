"""Pose freshness and continuity checks; no I/O, clock or robot side effects."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PoseRecoveryConfig:
    timeout_s: float = 5.0
    stable_s: float = 0.6
    min_samples: int = 3
    max_sample_gap_s: float = 0.3
    max_position_jump_m: float = 1.0
    max_rotation_jump_deg: float = 60.0
    stable_position_span_m: float = 0.15
    stable_rotation_span_deg: float = 10.0

    def __post_init__(self):
        for name in ('timeout_s', 'stable_s', 'max_sample_gap_s',
                     'max_position_jump_m', 'max_rotation_jump_deg',
                     'stable_position_span_m', 'stable_rotation_span_deg'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'policy.pose_recovery.{name} 必须是有限正数')
        if type(self.min_samples) is not int or self.min_samples < 2:
            raise ValueError('policy.pose_recovery.min_samples 必须是 >=2 的整数')
        if (self.stable_s >= self.timeout_s or self.max_rotation_jump_deg > 180
                or self.stable_rotation_span_deg > 180):
            raise ValueError('位姿恢复要求 stable_s < timeout_s，转角阈值 <=180 度')


@dataclass(frozen=True)
class PoseSample:
    sequence: int
    received: float
    position: np.ndarray
    quaternion: np.ndarray


def inspect_pose(packet, max_age_s, now):
    """Distinguish missing/stale packets from malformed data, using receipt time."""
    if packet is None:
        return 'missing', None, None
    try:
        seq, received = packet.pose_sequence, float(packet.received_monotonic)
        if (isinstance(seq, bool) or not isinstance(seq, (int, np.integer)) or seq < 0
                or not math.isfinite(received) or received < 0 or received > now):
            return 'invalid', None, None
        arrays = [(packet.robot_pos_w, (3,)), (packet.robot_quat_wxyz, (4,)),
                  (packet.linear_vel_b, (3,)), (packet.angular_vel_b, (3,))]
        if not all(np.asarray(v).shape == shape and np.isfinite(v).all() for v, shape in arrays):
            return 'invalid', None, None
        quat = np.asarray(packet.robot_quat_wxyz, dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if not math.isfinite(norm) or not 0.9 <= norm <= 1.1:
            return 'invalid', None, None
        if seq == 0:
            return 'missing', None, None
        age = now - received
        sample = PoseSample(int(seq), received, np.asarray(packet.robot_pos_w, dtype=np.float64).copy(), quat / norm)
        return ('fresh' if age <= max_age_s else 'stale'), sample, age
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 'invalid', None, None


def pose_difference(a, b):
    distance = float(np.linalg.norm(a.position - b.position))
    # q and -q represent the same orientation.
    angle = math.degrees(2 * math.acos(float(np.clip(abs(np.dot(a.quaternion, b.quaternion)), 0, 1))))
    return distance, angle


class PoseHealthMonitor:
    def __init__(self, max_age_s=1.0, config=None):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('policy.state_max_age_s 必须是有限正数')
        self.max_age_s = max_age_s
        self.config = config or PoseRecoveryConfig()
        self.reset()

    def reset(self):
        self.last = None
        self.baseline = None
        self.wait_started = None
        self.deadline = None
        self.stable_first = None
        self.stable_last = None
        self.stable_samples = 0
        self.stable_poses = []
        self.wait_position_limit = self.config.max_position_jump_m

    def update(self, packet, now, *, force_wait_reason=None, max_linear_speed_mps=0.0):
        kind, sample, age = inspect_pose(packet, self.max_age_s, now)
        details = dict(pose_age_s=round(age, 4) if age is not None else 'none',
                       pose_sequence=sample.sequence if sample else 0)
        if self.wait_started is not None:
            details['wait_s'] = round(now - self.wait_started, 4)
        if kind == 'invalid':
            return 'error', 'pose_invalid_requires_localize', details
        # A packet that arrives after the deadline must not release a latched stop.
        if self.deadline is not None and now >= self.deadline:
            return 'error', 'pose_timeout_requires_localize', details
        is_new = sample is not None and (self.last is None or sample.sequence != self.last.sequence)
        gap = sample.received - self.last.received if is_new and self.last else 0.0
        entering_wait = self.wait_started is None and (kind != 'fresh' or gap > self.max_age_s or force_wait_reason)
        if entering_wait:
            self.wait_started = now
            self.baseline = self.last
            self.deadline = now + self.config.timeout_s
            if self.last and (kind != 'fresh' or gap > self.max_age_s):
                self.deadline = min(self.deadline, self.last.received + self.max_age_s + self.config.timeout_s)
            if ((age is not None and age >= self.max_age_s + self.config.timeout_s)
                    or now >= self.deadline):
                return 'error', 'pose_timeout_requires_localize', details
            # The robot may move between the last receipt and the first zero command.
            # Freeze this allowance NOW; waiting longer must never increase it.
            elapsed_before_zero = max(0., now-self.last.received) if self.last else 0.
            speed = float(max_linear_speed_mps)
            if not math.isfinite(speed) or speed < 0:
                return 'error', 'pose_invalid_speed_bound_requires_localize', details
            self.wait_position_limit = self.config.max_position_jump_m + speed*elapsed_before_zero
            details.update(wait_s=0.0, before_zero_s=round(elapsed_before_zero, 4),
                           motion_allowance_m=round(speed*elapsed_before_zero, 4))
        if self.wait_started is not None:
            details['wait_position_limit_m'] = round(self.wait_position_limit, 4)
        if sample is not None and self.last is not None:
            if (sample.sequence < self.last.sequence or sample.received < self.last.received
                    or (sample.sequence == self.last.sequence and sample.received != self.last.received)
                    or (is_new and sample.received <= self.last.received)):
                return 'error', 'pose_sequence_reset_requires_localize', details
            for reference in (self.last, self.baseline):
                if reference is None:
                    continue
                distance, angle = pose_difference(sample, reference)
                is_baseline = reference is self.baseline
                limit = self.wait_position_limit if is_baseline else self.config.max_position_jump_m
                if distance > limit or angle > self.config.max_rotation_jump_deg:
                    details.update(position_jump_m=round(distance, 4), rotation_jump_deg=round(angle, 3),
                        jump_reference='pre_wait_pose' if is_baseline else 'previous_sample',
                        reference_sequence=reference.sequence,
                        sample_dt_s=round(sample.received-reference.received, 4),
                        position_limit_m=round(limit, 4))
                    return 'error', 'pose_jump_requires_localize', details
            if not is_new and (not np.array_equal(sample.position, self.last.position)
                               or pose_difference(sample, self.last)[1] > 1e-4):
                return 'error', 'pose_invalid_requires_localize', details
        if entering_wait:
            return 'wait', force_wait_reason or ('pose_gap' if kind == 'fresh' else 'pose_'+kind), details
        if self.wait_started is not None:
            # Recovery must comprise distinct, recent receipts, not repeated cache reads.
            recent = kind == 'fresh' and age <= self.config.max_sample_gap_s
            if not recent:
                self.stable_first = self.stable_last = None
                self.stable_samples = 0
                self.stable_poses = []
            elif is_new:
                if self.stable_last is None or sample.received - self.stable_last > self.config.max_sample_gap_s:
                    self.stable_poses = []
                # Zero velocity is already commanded: do not resume while delayed
                # frames are catching up or the stationary robot's pose is drifting.
                if any(distance > self.config.stable_position_span_m or angle > self.config.stable_rotation_span_deg
                       for distance, angle in (pose_difference(sample, p) for p in self.stable_poses)):
                    self.stable_poses = []
                self.stable_poses.append(sample)
                self.stable_first = self.stable_poses[0].received
                self.stable_last = sample.received
                self.stable_samples = len(self.stable_poses)
            if is_new:
                self.last = sample
            details['stable_samples'] = self.stable_samples
            if (recent and self.stable_samples >= self.config.min_samples
                    and self.stable_last - self.stable_first >= self.config.stable_s - 1e-9):
                self.wait_started = self.deadline = self.baseline = None
                self.stable_first = self.stable_last = None
                self.stable_samples = 0
                self.stable_poses = []
                return 'resume', 'pose_stable', details
            return 'wait', 'pose_stabilizing', details
        if is_new:
            self.last = sample
        return 'ok', 'pose_valid', details
