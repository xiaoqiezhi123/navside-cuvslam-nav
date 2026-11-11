import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation
# from scipy.spatial.transform import Rotation as R  # replaced by manual quaternion

from .image import resize_image
from .state import SruRobotState
from .timing import timing_log, RateMeter


STATE_DIM = 16
DEPTH_EMBEDDING_DIM = 2560
PATH_WAYPOINTS = 15
PATH_OBS_DIM = 60
LSTM_HIDDEN_DIM = 512
DEFAULT_POLICY_SCALE = np.array([1.5, 1.0, 1.0], dtype=np.float32)
ZED_MINI_CROP_WIDTH = 1728
ZED_MINI_CROP_HEIGHT = 1080
ENCODER_INPUT_WIDTH = 64
ENCODER_INPUT_HEIGHT = 40


class SruNavAdapter:
    """Host-side SRU navigation adapter for NavSide."""

    def __init__(
        self,
        encoder_path: str,
        policy_path: str,
        dry_run_hz: float = 5.0,
        min_depth: float = 0.25,
        max_depth: float = 10.0,
        crop_width: int = ZED_MINI_CROP_WIDTH,
        crop_height: int = ZED_MINI_CROP_HEIGHT,
        policy_scale: np.ndarray = DEFAULT_POLICY_SCALE,
        verbose: bool = True,
        *,
        policy_mode: str = "legacy",
        over_range_value: float = 6.0,
    ):
        import onnxruntime as ort

        self.encoder_path = encoder_path
        if policy_mode not in ('legacy', 'path_aware'):
            raise ValueError('Unknown policy.mode: '+str(policy_mode))
        self.policy_mode = policy_mode
        self.over_range_value = float(over_range_value)
        if (not np.isfinite(dry_run_hz) or dry_run_hz <= 0
                or not np.isfinite(self.over_range_value)
                or crop_width <= 0 or crop_height <= 0):
            raise ValueError('无效的推理频率或深度配置')
        self.obs_dim = STATE_DIM + DEPTH_EMBEDDING_DIM + (PATH_OBS_DIM if policy_mode == 'path_aware' else 0)
        self.policy_path = policy_path
        self.dry_run_interval = 1.0 / dry_run_hz
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_crop_width = int(crop_width)
        self.depth_crop_height = int(crop_height)
        self.policy_scale = np.asarray(policy_scale, dtype=np.float32)
        self.verbose = verbose
        self.last_policy_time: Optional[float] = None
        self.last_action = np.zeros(3, dtype=np.float32)
        self.h_state = np.zeros((1, 1, LSTM_HIDDEN_DIM), dtype=np.float32)
        self.c_state = np.zeros((1, 1, LSTM_HIDDEN_DIM), dtype=np.float32)
        self.last_depth_preprocess_info: Dict[str, np.ndarray] = {}
        self._prev_depth_feature: Optional[np.ndarray] = None

        self._check_model_path(self.encoder_path)
        self._check_model_path(self.policy_path)

        try:
            ort.preload_dlls(directory="")
        except Exception as e:
            print(f"[SRU] preload_dlls warning: {e}")

        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.encoder_session = ort.InferenceSession(
            self.encoder_path, sess_options=sess_options, providers=providers
        )
        self.policy_session = ort.InferenceSession(
            self.policy_path, sess_options=sess_options, providers=providers
        )

        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_name = self.encoder_session.get_outputs()[0].name
        self.policy_output_names = [out.name for out in self.policy_session.get_outputs()]
        self._validate_models()

        print(
            "[SRU] adapter ready | providers={} encoder={} policy={}".format(
                providers, self.encoder_path, self.policy_path
            )
        )
        self._timing_logged_first_encode_depth = False
        self._timing_logged_first_run_policy = False
        self.encoder_rate_meter = RateMeter("encoder_ort")
        self.policy_rate_meter = RateMeter("policy_ort")

    def _check_model_path(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SRU model file not found: {path}")

    def _validate_models(self):
        def check(node, suffix):
            if node.type != 'tensor(float)' or list(node.shape)[-len(suffix):] != suffix:
                raise ValueError(f'模型接口不匹配: {node.name} {node.type} {node.shape}, expected *{suffix}')
        inputs = {v.name: v for v in self.policy_session.get_inputs()}
        outputs = {v.name: v for v in self.policy_session.get_outputs()}
        if set(inputs) != {'obs', 'h_in', 'c_in'} or set(outputs) != {'actions', 'h_out', 'c_out'}:
            raise ValueError('策略模型输入输出名称不匹配')
        check(inputs['obs'], [self.obs_dim])
        check(outputs['actions'], [3])
        if len(inputs['obs'].shape) != 2 or len(outputs['actions'].shape) != 2:
            raise ValueError('策略 obs/actions 必须为二维 batch 输入输出')
        for node in [inputs['h_in'], inputs['c_in'], outputs['h_out'], outputs['c_out']]:
            check(node, [512])
            if len(node.shape) != 3 or node.shape[0] != 1:
                raise ValueError('循环状态形状必须为 [1,batch,512]')
        enc_in, enc_out = self.encoder_session.get_inputs(), self.encoder_session.get_outputs()
        if len(enc_in) != 1 or len(enc_out) != 1:
            raise ValueError('VAE 必须为单输入、单输出')
        check(enc_in[0], [1,40,64])
        check(enc_out[0], [64,5,8])
        if len(enc_in[0].shape) != 4 or len(enc_out[0].shape) != 4:
            raise ValueError('编码器输入输出必须为四维')
        self.policy_output_names = ['actions', 'h_out', 'c_out']

    def should_tick(self, now: float) -> bool:
        if self.last_policy_time is None:
            return True
        return (now - self.last_policy_time) >= self.dry_run_interval

    def reset_recurrent_state(self) -> None:
        self.last_policy_time = None
        self.last_action = np.zeros(3, dtype=np.float32)
        self._prev_depth_feature = None
        self.h_state = np.zeros((1, 1, LSTM_HIDDEN_DIM), dtype=np.float32)
        self.c_state = np.zeros((1, 1, LSTM_HIDDEN_DIM), dtype=np.float32)

    def step(
        self,
        depth_img: np.ndarray,
        state: SruRobotState,
        target_pos_w: np.ndarray,
        timestamp: Optional[float] = None,
        *,
        path_w: np.ndarray | None = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        now = time.time() if timestamp is None else timestamp
        if not self.should_tick(now):
            return None
        path_obs = None
        if self.policy_mode == 'path_aware':
            path_obs = self.build_path_observation(path_w, state.robot_pos_w, state.robot_quat_wxyz)

        dt = None if self.last_policy_time is None else now - self.last_policy_time
        self.last_policy_time = now

        depth_feature = self.depth_preprocess(depth_img)

        # -- Depth feature stats --
        depth_feat_mean = float(np.mean(depth_feature))
        depth_feat_std = float(np.std(depth_feature))
        if self._prev_depth_feature is not None:
            depth_feat_delta = float(np.linalg.norm(depth_feature - self._prev_depth_feature))
        else:
            depth_feat_delta = 0.0
        self._prev_depth_feature = depth_feature.copy()
        target_position, target_vec_b = self.build_target_position(
            target_pos_w, state.robot_pos_w, state.robot_quat_wxyz
        )

        state_input = np.concatenate(
            [
                state.linear_vel_b.astype(np.float32),
                state.angular_vel_b.astype(np.float32),
                state.projected_gravity_b.astype(np.float32),
                self.last_action.astype(np.float32),
                target_position.astype(np.float32),
            ]
        )
        parts = [state_input, depth_feature] if path_obs is None else [state_input, path_obs, depth_feature]
        obs = np.concatenate(parts)[np.newaxis].astype(np.float32)
        if obs.shape != (1, self.obs_dim) or not np.isfinite(obs).all():
            raise ValueError('策略观测维度错误或存在非有限值')

        policy_t0 = time.perf_counter()
        outputs = self.policy_session.run(
            self.policy_output_names,
            {
                "obs": obs,
                "h_in": self.h_state,
                "c_in": self.c_state,
            },
        )
        raw_action, self.h_state, self.c_state = outputs
        if not all(np.isfinite(v).all() for v in outputs):
            self.reset_recurrent_state()
            raise ValueError('策略输出或循环状态存在非有限值')
        if not self._timing_logged_first_run_policy:
            timing_log("adapter_first_run_policy", time.perf_counter() - policy_t0)
            self._timing_logged_first_run_policy = True
        self.policy_rate_meter.tick()
        raw_action = raw_action.squeeze(0).astype(np.float32)
        cmd_vel = (np.tanh(raw_action) * self.policy_scale).astype(np.float32)
        self.last_action = raw_action.copy()

        diag = {
            "timestamp": np.array([now], dtype=np.float64),
            "dt": np.array([-1.0 if dt is None else dt], dtype=np.float64),
            "depth_shape": np.array(depth_img.shape, dtype=np.int32),
            "depth_minmax": np.array([np.nanmin(depth_img), np.nanmax(depth_img)], dtype=np.float32),
            "depth_crop_shape": self.last_depth_preprocess_info.get(
                "crop_shape", np.array([-1, -1], dtype=np.int32)
            ),
            "depth_processed_shape": self.last_depth_preprocess_info.get(
                "processed_shape", np.array([-1, -1, -1, -1], dtype=np.int32)
            ),
            "depth_processed_minmax": self.last_depth_preprocess_info.get(
                "processed_minmax", np.array([np.nan, np.nan], dtype=np.float32)
            ),
            "depth_feature_shape": np.array(depth_feature.shape, dtype=np.int32),
            "depth_feature_mean": np.array([depth_feat_mean], dtype=np.float32),
            "depth_feature_std": np.array([depth_feat_std], dtype=np.float32),
            "depth_feature_delta": np.array([depth_feat_delta], dtype=np.float32),
            "linear_vel_b": state.linear_vel_b.astype(np.float32),
            "angular_vel_b": state.angular_vel_b.astype(np.float32),
            "projected_gravity_b": state.projected_gravity_b.astype(np.float32),
            "robot_pos_w": state.robot_pos_w.astype(np.float32),
            "robot_quat_wxyz": state.robot_quat_wxyz.astype(np.float32),
            "target_position": target_position.astype(np.float32),
            "target_vec_b": target_vec_b.astype(np.float32),
            "obs_shape": np.array(obs.shape, dtype=np.int32),
            "raw_action": raw_action,
            "cmd_vel": cmd_vel,
            "zero_reason": None,
        }
        if self.verbose:
            self.print_diagnostics(diag)
        if path_obs is not None:
            diag['path_w'] = np.asarray(path_w, dtype=np.float32).copy()
            diag['path_obs'] = path_obs
        return diag

    @staticmethod
    def build_path_observation(path_w, robot_pos_w, robot_quat_wxyz):
        path = np.asarray(path_w, dtype=np.float32)
        pos = np.asarray(robot_pos_w, dtype=np.float32)
        quat = np.asarray(robot_quat_wxyz, dtype=np.float32)
        if (path.shape != (15,3) or pos.shape != (3,) or quat.shape != (4,)
                or not all(np.isfinite(v).all() for v in (path,pos,quat))
                or np.linalg.norm(quat) < 1e-8):
            raise ValueError('路径模式要求有限的 15×3 全局点及有效机器人位姿')
        rot = Rotation.from_quat(quat[[1,2,3,0]])
        vec = rot.inv().apply(path-pos).astype(np.float32)
        distances = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
        logdist = np.log(distances+1.0)
        return np.concatenate([vec/distances, logdist/max(float(logdist.max()),1e-9)], axis=1).flatten().astype(np.float32)

    def build_control_command(
        self,
        diag: Dict[str, np.ndarray],
        vx_max: float = 0.7,
        wz_max: float = 0.9,
        walk_threshold: float = 0.3,
    ) -> Dict[str, np.ndarray]:
        zero_reason = diag.get("zero_reason")
        raw_cmd = diag.get("cmd_vel")

        if raw_cmd is None:
            zero_reason = "cmd_missing"
            raw_cmd = np.zeros(3, dtype=np.float32)
        else:
            raw_cmd = np.asarray(raw_cmd, dtype=np.float32).reshape(3)

        if zero_reason is None and not np.all(np.isfinite(raw_cmd)):
            zero_reason = "cmd_nan_or_inf"

        if zero_reason is None:
            final_cmd = np.array(
                [
                    np.clip(float(raw_cmd[0]), 0.0, vx_max),
                    0.0,
                    np.clip(float(raw_cmd[2]), -wz_max, wz_max),
                ],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(final_cmd)):
                zero_reason = "final_cmd_nan_or_inf"
        else:
            final_cmd = np.zeros(3, dtype=np.float32)


        above_walk_threshold = bool(final_cmd[0] > walk_threshold)
        should_send = zero_reason is None

        return {
            "raw_cmd": raw_cmd.astype(np.float32),
            "final_cmd": final_cmd.astype(np.float32),
            "above_walk_threshold": above_walk_threshold,
            "should_send": should_send,
            "zero_reason": zero_reason or "",
            "walk_threshold": float(walk_threshold),
            "vx_max": float(vx_max),
            "wz_max": float(wz_max),
        }

    def depth_preprocess(self, depth_img: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_img, dtype=np.float32).copy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=self.max_depth * 2.0, neginf=0.0)
        depth[depth > self.max_depth] = self.over_range_value
        depth[depth < self.min_depth] = 0.0
        if self.policy_mode == 'path_aware':
            if depth.ndim != 2 or depth.shape[1] < self.depth_crop_width or depth.shape[0] < self.depth_crop_height:
                raise ValueError('深度图小于配置裁剪范围')
            depth = self._center_crop_depth(depth, self.depth_crop_width, self.depth_crop_height)
        else:
            depth = self._center_crop_depth(depth, ZED_MINI_CROP_WIDTH, ZED_MINI_CROP_HEIGHT)
        depth_resized = resize_image(depth, ENCODER_INPUT_WIDTH, ENCODER_INPUT_HEIGHT)
        depth_tensor = depth_resized[np.newaxis, np.newaxis, :, :].astype(np.float32)
        self.last_depth_preprocess_info = {
            "crop_shape": np.array(depth.shape, dtype=np.int32),
            "processed_shape": np.array(depth_tensor.shape, dtype=np.int32),
            "processed_minmax": np.array(
                [np.nanmin(depth_resized), np.nanmax(depth_resized)],
                dtype=np.float32,
            ),
        }
        encode_t0 = time.perf_counter()
        vae_output = self.encoder_session.run(
            [self.encoder_output_name], {self.encoder_input_name: depth_tensor}
        )[0]
        if not self._timing_logged_first_encode_depth:
            timing_log("adapter_first_encode_depth", time.perf_counter() - encode_t0)
            self._timing_logged_first_encode_depth = True
        self.encoder_rate_meter.tick()
        return vae_output.flatten().astype(np.float32)

    def _center_crop_depth(self, depth: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        height, width = depth.shape[:2]
        if width >= target_width and height >= target_height:
            x0 = (width - target_width) // 2
            y0 = (height - target_height) // 2
            return depth[y0 : y0 + target_height, x0 : x0 + target_width]
        # Image smaller than the configured crop size: skip the crop and let
        # the caller resize directly.
        return depth

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    def build_target_position(
        self,
        target_pos_w: np.ndarray,
        robot_pos_w: np.ndarray,
        robot_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_pos_w = np.asarray(target_pos_w, dtype=np.float32).reshape(3)
        robot_pos_w = np.asarray(robot_pos_w, dtype=np.float32).reshape(3)
        robot_quat_wxyz = np.asarray(robot_quat_wxyz, dtype=np.float32).reshape(4)
        if getattr(self, 'policy_mode', 'legacy') == 'path_aware':
            # Match the new adapter's scipy Rotation normalization for goal and path.
            robot_quat_wxyz = robot_quat_wxyz / np.linalg.norm(robot_quat_wxyz)

        # Manual quaternion rotation (world -> body), matching bridge.py convention.
        # Uses: v_body = conj(q) * [0, v_world] * q  where q = [w, x, y, z]
        target_vec_w = target_pos_w - robot_pos_w
        q_conj = self._quat_conj(robot_quat_wxyz)
        v_quat = np.array([0.0, target_vec_w[0], target_vec_w[1], target_vec_w[2]])
        result = self._quat_multiply(self._quat_multiply(q_conj, v_quat), robot_quat_wxyz)
        target_vec_b = result[1:].astype(np.float32)

        dist = float(np.linalg.norm(target_vec_b) + 1e-6)
        target_dir_b = target_vec_b / dist
        target_position = np.concatenate(
            [target_dir_b, np.array([np.log(dist + 1.0)], dtype=np.float32)]
        )
        return target_position.astype(np.float32), target_vec_b.astype(np.float32)

    def print_diagnostics(self, diag: Dict[str, np.ndarray]) -> None:
        dt = diag["dt"][0]
        dt_text = "first" if dt < 0 else f"{dt:.3f}s"
        print(
            "[SRU DRY] tick={:.3f} dt={} depth shape={} min/max={:.3f}/{:.3f} "
            "crop shape={} processed shape={} processed min/max={:.3f}/{:.3f} "
            "depth_feature shape={} obs shape={}".format(
                diag["timestamp"][0],
                dt_text,
                tuple(diag["depth_shape"].tolist()),
                diag["depth_minmax"][0],
                diag["depth_minmax"][1],
                tuple(diag["depth_crop_shape"].tolist()),
                tuple(diag["depth_processed_shape"].tolist()),
                diag["depth_processed_minmax"][0],
                diag["depth_processed_minmax"][1],
                tuple(diag["depth_feature_shape"].tolist()),
                tuple(diag["obs_shape"].tolist()),
            )
        )
        print(
            "[SRU DRY] linear_vel_b={} angular_vel_b={} projected_gravity_b={}".format(
                np.array2string(diag["linear_vel_b"], precision=4),
                np.array2string(diag["angular_vel_b"], precision=4),
                np.array2string(diag["projected_gravity_b"], precision=4),
            )
        )
        print(
            "[SRU DRY] robot_pos_w={} robot_quat_wxyz={} target_position={}".format(
                np.array2string(diag["robot_pos_w"], precision=4),
                np.array2string(diag["robot_quat_wxyz"], precision=4),
                np.array2string(diag["target_position"], precision=4),
            )
        )
        print(
            "[SRU DRY] raw_action={} postprocessed cmd_vel={}".format(
                np.array2string(diag["raw_action"], precision=4),
                np.array2string(diag["cmd_vel"], precision=4),
            )
        )
