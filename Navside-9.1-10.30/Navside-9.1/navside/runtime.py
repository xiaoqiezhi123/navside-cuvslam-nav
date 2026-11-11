import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .adapter import SruNavAdapter
from .state import SruRobotState
from .timing import timing_log
from .pose_health import PoseRecoveryConfig


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "nav.yaml"


@dataclass
class NavSideConfig:
    encoder_path: str
    policy_path: str
    dry_run_hz: float = 5.0
    vx_max: float = 1
    wz_max: float = 1.5
    walk_threshold: float = 0.3
    goal_pos_tolerance: float = 0.08
    min_depth: float = 0.25
    max_depth: float = 10.0
    depth_backend: str = "realsense"
    realsense_config_path: str = ""
    depth_crop_width: int = 1728
    depth_crop_height: int = 1080
    verbose_sru: bool = True
    default_goal_w: tuple = (5.0, 1.0, 0.0)
    csv_enabled: bool = True
    csv_dir: str = "logs"
    policy_mode: str = "legacy"
    over_range_value: float = 6.0
    state_max_age_s: float = 1.0
    pose_recovery: PoseRecoveryConfig = field(default_factory=PoseRecoveryConfig)


def load_nav_config(config_path: str) -> NavSideConfig:
    t0 = time.perf_counter()
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_cfg = data.get("models", {})
    control_cfg = data.get("control", {})
    depth_cfg = data.get("depth", {})
    log_cfg = data.get("logging", {})
    goal_cfg = data.get("goal", {})
    base_dir = Path(config_path).resolve().parent
    config = NavSideConfig(
        encoder_path=str(base_dir / model_cfg.get("encoder_path", "../models/vae_encoder.onnx")),
        policy_path=str(base_dir / model_cfg.get("policy_path", "../models/policy.onnx")),
        dry_run_hz=float(control_cfg.get("dry_run_hz", 5.0)),
        vx_max=float(control_cfg.get("vx_max", 0.45)),
        wz_max=float(control_cfg.get("wz_max", 0.4)),
        walk_threshold=float(control_cfg.get("walk_threshold", 0.3)),
        goal_pos_tolerance=float(control_cfg.get("goal_pos_tolerance", 0.08)),
        min_depth=float(depth_cfg.get("min_depth", 0.25)),
        max_depth=float(depth_cfg.get("max_depth", 10.0)),
        depth_backend=str(depth_cfg.get("backend", "realsense")).lower(),
        realsense_config_path=str(base_dir / depth_cfg.get("realsense_config", "realsence_config.yaml")) if depth_cfg.get("realsense_config") else "",
        depth_crop_width=int(depth_cfg.get("crop_width", 1728)),
        depth_crop_height=int(depth_cfg.get("crop_height", 1080)),
        verbose_sru=bool(log_cfg.get("verbose_sru", True)),
        default_goal_w=tuple(goal_cfg.get("default_goal_w", [5.0, 1.0, 0.0])),
        csv_enabled=bool(log_cfg.get("csv_enabled", True)),
        csv_dir=str(log_cfg.get("csv_dir", "logs")),
        policy_mode=str(data.get('policy', {}).get('mode', 'legacy')),
        over_range_value=float(depth_cfg.get('over_range_value', 6.0)),
        state_max_age_s=float(data.get('policy', {}).get('state_max_age_s', 1.0)),
        pose_recovery=PoseRecoveryConfig(**data.get('policy', {}).get('pose_recovery', {})),
    )
    timing_log("runtime_load_nav_config", time.perf_counter() - t0)
    return config


class NavSideApp:
    def __init__(self, config: NavSideConfig):
        t0 = time.perf_counter()
        self.config = config
        if not np.isfinite(config.state_max_age_s) or config.state_max_age_s <= 0:
            raise ValueError('policy.state_max_age_s 必须是有限正数')
        adapter_t0 = time.perf_counter()
        self.adapter = SruNavAdapter(
            encoder_path=config.encoder_path,
            policy_path=config.policy_path,
            dry_run_hz=config.dry_run_hz,
            min_depth=config.min_depth,
            max_depth=config.max_depth,
            crop_width=config.depth_crop_width,
            crop_height=config.depth_crop_height,
            verbose=config.verbose_sru,
            policy_mode=config.policy_mode,
            over_range_value=config.over_range_value,
        )
        timing_log("runtime_adapter_construct", time.perf_counter() - adapter_t0)
        self.last_final_cmd = np.zeros(3, dtype=np.float32)
        timing_log("runtime_app_init_total", time.perf_counter() - t0)

    @classmethod
    def from_config(cls, config_path: str):
        return cls(load_nav_config(config_path))

    def default_goal(self) -> np.ndarray:
        return np.asarray(self.config.default_goal_w, dtype=np.float32)

    def default_state(self) -> SruRobotState:
        return SruRobotState(
            linear_vel_b=np.zeros(3, dtype=np.float32),
            angular_vel_b=np.zeros(3, dtype=np.float32),
            projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            robot_pos_w=np.array([0.0, 0.0, 0.695], dtype=np.float32),
            robot_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def step(
        self,
        depth_img: np.ndarray,
        state: SruRobotState,
        target_pos_w: np.ndarray,
        timestamp: float | None = None,
        vx_max: float | None = None,
        wz_max: float | None = None,
        walk_threshold: float | None = None,
        print_control: bool = True,
        *,
        path_w: np.ndarray | None = None,
    ):
        diag = self.adapter.step(
            depth_img=depth_img,
            state=state,
            target_pos_w=target_pos_w,
            timestamp=timestamp,
            path_w=path_w,
        )
        if diag is None:
            return None

        vx_limit = self.config.vx_max if vx_max is None else vx_max
        wz_limit = self.config.wz_max if wz_max is None else wz_max
        walk_limit = self.config.walk_threshold if walk_threshold is None else walk_threshold

        control_info = self.adapter.build_control_command(
            diag,
            vx_max=vx_limit,
            wz_max=wz_limit,
            walk_threshold=walk_limit,
        )

        goal_dist = float(np.linalg.norm(np.asarray(diag["target_vec_b"], dtype=np.float32)))
        if goal_dist <= self.config.goal_pos_tolerance:
            control_info["final_cmd"] = np.zeros(3, dtype=np.float32)
            control_info["should_send"] = True
            control_info["zero_reason"] = "goal_reached"
            control_info["above_walk_threshold"] = False

        self.last_final_cmd = np.asarray(control_info["final_cmd"], dtype=np.float32).copy()
        if print_control:
            print(
                "[NavSide] control raw={} final={} walk_threshold={} above_walk_threshold={} zero_reason={} goal_dist={:.4f}".format(
                    np.array2string(control_info["raw_cmd"], precision=4),
                    np.array2string(control_info["final_cmd"], precision=4),
                    control_info["walk_threshold"],
                    control_info["above_walk_threshold"],
                    control_info["zero_reason"],
                    goal_dist,
                )
            )

        return {
            "diag": diag,
            "control": control_info,
            "goal_dist": goal_dist,
        }


def parse_goal(goal_text):
    if goal_text is None:
        return None
    parts = [p.strip() for p in goal_text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError('--goal must be "x,y,z"')
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)


def build_parser():
    parser = argparse.ArgumentParser(description="SRU-only NavSide launcher")
    parser.add_argument("--sim-control", action="store_true", help="Run MuJoCo simulation mode (start GUI).")
    parser.add_argument("--real", action="store_true", help="Run real robot mode (headless depth camera + UDP).")
    parser.add_argument(
        "--use-real-state",
        action="store_true",
        help="In --sim-control mode, get state from Foxglove WS (real robot) instead of UDP/MuJoCo. Depth still comes from MuJoCo rendering.",
    )
    parser.add_argument("--deploy-config", default=None, help="Path to nav_deploy.yaml for RobotComm (default: <config_dir>/nav_deploy.yaml).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to NavSide config YAML.")
    parser.add_argument("--goal", default=None, help='Optional goal override as "x,y,z".')
    parser.add_argument("--spawn", default=None, help='Optional robot spawn override as "x,y,z".')
    parser.add_argument("--spawn-yaw", type=float, default=None, help="Optional robot spawn yaw override in radians.")
    parser.add_argument(
        "--ignore-udp-state",
        action="store_true",
        help="Do not mirror incoming robot-side UDP state into the NavSide MuJoCo viewer.",
    )
    parser.add_argument("--mujoco-xml", default=None, help="Optional MuJoCo XML override for sim mode.")
    parser.add_argument("--camera-name", default=None, help="Optional MuJoCo camera name override for sim mode.")
    parser.add_argument("--verbose-sru", action="store_true", help="Print full SRU per-tick diagnostics in sim mode.")
    parser.add_argument("--summary-hz", type=float, default=None, help="Summary log frequency in sim mode.")
    parser.add_argument("--csv-dir", default=None, help="Override logging.csv_dir for per-run CSV logs.")
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Open a live RealSense depth viewer in real mode.",
    )
    return parser


def main():
    t0 = time.perf_counter()
    args = build_parser().parse_args()
    if args.real:
        from .real import run as run_real
        timing_log("runtime_main_pre_real", time.perf_counter() - t0)
        run_real(args)
    elif args.sim_control:
        from .sim import run as run_mujoco_sim
        timing_log("runtime_main_pre_sim", time.perf_counter() - t0)
        run_mujoco_sim(args)
    else:
        raise SystemExit("Specify --sim-control (simulation) or --real (real robot) to select mode.")


if __name__ == "__main__":
    main()
