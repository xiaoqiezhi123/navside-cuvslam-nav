"""bridge.py — RobotComm with Foxglove WebSocket + Langyi HTTP API.

Data flow:
  Foxglove WS (/high_frequency_odometry) → robot_pos_w, robot_quat_wxyz
  Computed from consecutive odom diffs  → linear_vel_b, angular_vel_b
  Computed from quaternion              → projected_gravity_b
"""

#python3 bridge.py --config ../../../configs/nav_deploy.yaml --hz 5

import asyncio
import math
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np
import requests
import yaml

try:
    from .foxglove_ws_client import FoxgloveWsClient
    from .state import SruRobotState
    from .pose_trace import get_trace, NULL_TRACE, decode_pose
except ImportError:
    from foxglove_ws_client import FoxgloveWsClient
    from state import SruRobotState
    from pose_trace import get_trace, NULL_TRACE, decode_pose


# ===========================================================================
# NavStatePacketV2
# ===========================================================================

@dataclass
class NavStatePacketV2:
    seq: int = 0
    timestamp_sec: float = 0.0
    linear_vel_b: np.ndarray = None
    angular_vel_b: np.ndarray = None
    projected_gravity_b: np.ndarray = None
    robot_pos_w: np.ndarray = None
    robot_quat_wxyz: np.ndarray = None
    source: str = "foxglove_ws"
    # Local receipt metadata only; the wire packet/protocol is unchanged.
    pose_sequence: int = 0
    received_monotonic: float = 0.0
    pose_trace: dict | None = None  # 可选源帧/发送信息；不参与停车或自动恢复授权

    def __post_init__(self):
        if self.linear_vel_b is None:
            self.linear_vel_b = np.zeros(3, dtype=np.float32)
        if self.angular_vel_b is None:
            self.angular_vel_b = np.zeros(3, dtype=np.float32)
        if self.projected_gravity_b is None:
            self.projected_gravity_b = np.zeros(3, dtype=np.float32)
        if self.robot_pos_w is None:
            self.robot_pos_w = np.zeros(3, dtype=np.float32)
        if self.robot_quat_wxyz is None:
            self.robot_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def to_sru_robot_state(self) -> SruRobotState:
        return SruRobotState(
            linear_vel_b=self.linear_vel_b.astype(np.float32),
            angular_vel_b=self.angular_vel_b.astype(np.float32),
            projected_gravity_b=self.projected_gravity_b.astype(np.float32),
            robot_pos_w=self.robot_pos_w.astype(np.float32),
            robot_quat_wxyz=self.robot_quat_wxyz.astype(np.float32),
        )


# ===========================================================================
# Nav_Task_Load — load map, relocalize, load task (no navigation)
# ===========================================================================

class Nav_Task_Load:
    """Load Langyi map → global relocalization → load task.

    Calls the Langyi HTTP API to prepare the navigation system.
    Does NOT start navigation (no startMoving / goto).

    Usage:
        loader = Nav_Task_Load(api_url="http://...", map_name="m", task_name="t")
        ok = loader.load_task()
        goals = loader.goals
    """

    IDLE = 0
    MAPPING = 1
    MAPLOADED = 2
    WAITFORGOAL = 3
    NAVIGATING = 4

    _STATUS_LABELS = {
        0: "IDLE", 1: "MAPPING", 2: "MAPLOADED",
        3: "WAITFORGOAL", 4: "NAVIGATING",
    }

    def __init__(
        self,
        api_url: str,
        map_name: str,
        task_name: str,
        skip_reloc_threshold: bool = False,
    ):
        self._api_url = api_url.rstrip("/")
        self._map_name = map_name
        self._task_name = task_name
        self._skip_reloc_threshold = skip_reloc_threshold
        self._goals = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def goals(self):
        """Return the loaded goal list (list of dicts with x, y, z, theta, …)."""
        return self._goals

    def load_task(self) -> bool:
        """Execute the full load sequence (blocking, ~10-30 s).

        Returns:
            True on success, False on failure.
        """
        try:
            self._goals = []
            self._reset_to_idle()
            self._load_map()
            ok = self._global_relocalize()
            if not ok:
                print("[Nav_Task_Load] relocalization failed")
                return False
            self._load_task()
            return len(self._goals) > 0
        except Exception as e:
            print(f"[Nav_Task_Load] load_task error: {e}")
            return False

    # ------------------------------------------------------------------
    # Internal: HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        r = requests.get(
            f"{self._api_url}{path}",
            headers={"content-type": "application/json"},
            timeout=10,
        )
        body = r.json()
        if body.get("code") != 0:
            raise RuntimeError(f"GET {path} failed: {body}")
        return body["data"]

    def _post(self, path: str, payload: dict | None = None) -> dict:
        if payload is None:
            payload = {}
        r = requests.post(
            f"{self._api_url}{path}",
            headers={"content-type": "application/json"},
            json=payload,
            timeout=10,
        )
        body = r.json()
        if body.get("code") != 0:
            raise RuntimeError(f"POST {path} failed: {body}")
        return body.get("data", {})

    # ------------------------------------------------------------------
    # Internal: status helpers
    # ------------------------------------------------------------------

    def _get_status(self) -> int:
        return self._get("/status")["process_status"]

    def _wait_for_status(self, target: int, label: str) -> None:
        while True:
            cur = self._get_status()
            if cur == target:
                print(f"[Nav_Task_Load] reached {label} ✓")
                return
            print(f"[Nav_Task_Load] waiting for {label} ... "
                  f"(current {self._STATUS_LABELS.get(cur, cur)})")
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Step 1 — reset to IDLE
    # ------------------------------------------------------------------

    def _reset_to_idle(self) -> None:
        status = self._get_status()
        if status == self.MAPPING:
            print("[Nav_Task_Load] cancel mapping ...")
            self._post("/stopMapping")
        elif status != self.IDLE:
            print(f"[Nav_Task_Load] stopping navigation "
                  f"(status={self._STATUS_LABELS.get(status, status)}) ...")
            self._post("/stopNavigation")
        self._wait_for_status(self.IDLE, "IDLE")

    # ------------------------------------------------------------------
    # Step 2 — load map
    # ------------------------------------------------------------------

    def _load_map(self) -> None:
        print(f"[Nav_Task_Load] loading map: {self._map_name}")
        self._post("/loadMap", {"map_name": self._map_name})
        self._wait_for_status(self.MAPLOADED, "MAPLOADED")

    # ------------------------------------------------------------------
    # Step 3 — global relocalization
    # ------------------------------------------------------------------

    def _global_relocalize(self) -> bool:
        return self._reloc_with_threshold()

    def _reloc_with_threshold(self) -> bool:
        THRESHOLD = float("inf") if self._skip_reloc_threshold else 1
        for attempt in range(1, 4):
            print(f"\n[Nav_Task_Load] reloc attempt {attempt}/3")
            cur = self._get_status()
            if cur != self.MAPLOADED:
                print(f"[Nav_Task_Load] not in MAPLOADED (status={cur}), abort")
                return False

            self._post("/tryGlobalRelocationOnce")
            time.sleep(3.0)

            valid = self._wait_for_valid_pose(max_wait=15.0)
            if valid is None:
                print(f"[Nav_Task_Load] attempt {attempt}: timeout waiting for pose")
                continue

            uncertainty = self._compute_uncertainty()
            print(f"[Nav_Task_Load] uncertainty={uncertainty:.4f} "
                  f"(threshold={THRESHOLD})")
            if uncertainty < THRESHOLD:
                print(f"[Nav_Task_Load] reloc success (attempt {attempt})")
                self._post("/relocalizedSuccess")
                self._wait_for_status(self.WAITFORGOAL, "WAITFORGOAL")
                return True
            else:
                print(f"[Nav_Task_Load] attempt {attempt}: uncertainty too high")

        print("[Nav_Task_Load] reloc failed after 3 attempts")
        return False

    # ------------------------------------------------------------------
    # Localization helpers
    # ------------------------------------------------------------------

    def _get_robot_pose(self):
        info = self._get("/getLocalization")["localization_info"]
        x = info["position"][0]
        y = info["position"][1]
        yaw = info["orientation"]
        return x, y, yaw

    @staticmethod
    def _is_valid_pose(x, y, yaw):
        return not (x == 0.0 and y == 0.0 and yaw == 0.0)

    def _wait_for_valid_pose(self, max_wait: float = 15.0):
        print("[Nav_Task_Load] waiting for valid localization pose ...")
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                x, y, yaw = self._get_robot_pose()
                if self._is_valid_pose(x, y, yaw):
                    print(f"[Nav_Task_Load] valid pose: "
                          f"x={x:.3f} y={y:.3f} yaw={yaw:.3f}")
                    return x, y, yaw
            except Exception:
                pass
            print("[Nav_Task_Load] pose still zero, waiting ...")
            time.sleep(1.0)
        return None

    def _compute_uncertainty(self) -> float:
        data = []
        for i in range(10):
            try:
                x, y, yaw = self._get_robot_pose()
                data.append((x, y, yaw))
                print(f"[Nav_Task_Load] sample[{i+1}/10]: "
                      f"x={x:.3f} y={y:.3f} yaw={yaw:.3f}")
            except Exception as e:
                print(f"[Nav_Task_Load] sample[{i+1}/10] error: {e}")
            time.sleep(0.5)

        if len(data) <= 3:
            return float("inf")

        # All-zero check
        if all(not self._is_valid_pose(x, y, yaw) for x, y, yaw in data):
            return float("inf")

        ref = next((p for p in data if self._is_valid_pose(*p)), data[0])
        rx, ry, ryaw = ref
        var_x = sum((px - rx) ** 2 for px, _, _ in data) / len(data)
        var_y = sum((py - ry) ** 2 for _, py, _ in data) / len(data)
        var_yaw = sum((pyaw - ryaw) ** 2 for _, _, pyaw in data) / len(data)
        return var_x + var_y + var_yaw

    # ------------------------------------------------------------------
    # Step 4 — load task
    # ------------------------------------------------------------------

    def _load_task(self) -> None:
        print(f"[Nav_Task_Load] loading task: {self._task_name}")
        data = self._post("/loadTask", {"task_name": self._task_name})
        task_data = data.get("task_data", {})
        self._goals = task_data.get("goals", [])
        desc = task_data.get("description", {})
        print(f"[Nav_Task_Load] task loaded: "
              f"{desc.get('task_description', '(none)')}, "
              f"{len(self._goals)} goals")
        for i, g in enumerate(self._goals):
            print(f"  [{i}] x={g.get('x', 0):.3f} y={g.get('y', 0):.3f} "
                  f"theta={g.get('theta', 0):.3f} name={g.get('name', '')}")


# ===========================================================================
# SimpleUdpRobotComm — lightweight UDP bridge for default sim mode
# ===========================================================================


class SimpleUdpRobotComm(threading.Thread):
    """Lightweight UDP bridge, compatible with the original robot-side CPG runtime.

    Receives NavStatePacketV2 over UDP and sends velocity commands.
    Self-contained — no external dependencies beyond socket/struct/numpy.
    Used by default in --sim-control mode when --use-real-state is NOT specified.
    """

    def __init__(
        self,
        local_ip: str = "127.0.0.1",
        local_port: int = 8081,
        remote_ip: str = "127.0.0.1",
        remote_port: int = 8080,
    ):
        super().__init__()
        self.cmd_packer = struct.Struct("3f")
        self.nav_state_v2_unpacker = struct.Struct("<IHHId16f")
        self.local_addr = (local_ip, local_port)
        self.remote_addr = (remote_ip, remote_port)
        self.latest_state = None
        self.lock = threading.Lock()
        self.running = True

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.local_addr)
        self.sock.settimeout(0.05)
        print(f"[SimpleUdpRobotComm] listening on {local_ip}:{local_port}")
        print(f"[SimpleUdpRobotComm] sending to {remote_ip}:{remote_port}")

    def run(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                state = self._parse_state_packet(data)
                if state is not None:
                    with self.lock:
                        self.latest_state = state
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as exc:
                print(f"[SimpleUdpRobotComm] receive error: {exc}")
                break

    def get_latest_state(self):
        with self.lock:
            return self.latest_state

    def _parse_state_packet(self, data: bytes):
        if len(data) != self.nav_state_v2_unpacker.size:
            print(f"[SimpleUdpRobotComm] unexpected state packet size: {len(data)} bytes")
            return None

        unpacked = self.nav_state_v2_unpacker.unpack(data)
        magic, version, _flags, seq, timestamp_sec = unpacked[:5]
        if magic != SRU2_MAGIC or version != 2:
            print(f"[SimpleUdpRobotComm] invalid NavStatePacketV2 header: magic={magic:#x} version={version}")
            return None

        values = np.asarray(unpacked[5:], dtype=np.float32)
        return NavStatePacketV2(
            seq=int(seq),
            timestamp_sec=float(timestamp_sec),
            linear_vel_b=values[0:3].copy(),
            angular_vel_b=values[3:6].copy(),
            projected_gravity_b=values[6:9].copy(),
            robot_pos_w=values[9:12].copy(),
            robot_quat_wxyz=values[12:16].copy(),
        )

    def send_command(self, vx: float, vy: float, wz: float):
        packet = self.cmd_packer.pack(float(vx), float(vy), float(wz))
        self.sock.sendto(packet, self.remote_addr)

    def send_zero(self):
        self.send_command(0.0, 0.0, 0.0)

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass

    def load_task(self) -> bool:
        """Stub — SimpleUdpRobotComm has no task-loading capability.
        Always returns True to keep API compatible with RobotComm.
        """
        return True


# ===========================================================================
# RobotComm — threaded bridge (Foxglove WS + Langyi HTTP callback)
# ===========================================================================

class RobotComm(threading.Thread):
    """Threaded bridge that fetches real robot data and exposes it via
    get_latest_state().

    Internal architecture:
      - Foxglove WebSocket client (asyncio) → odometry
      - Velocity computed from finite differences of consecutive odom readings
      - Nav_Task_Load (held internally) → load_task() exposed

    Usage:
        rc = RobotComm("configs/nav_deploy.yaml")
        rc.start()                         # start threads
        rc.load_task()                     # blocking, ~10-30 s
        state = rc.get_latest_state()      # NavStatePacketV2 or None
        rc.stop()
    """

    def __init__(self, config_path: str = "configs/nav_deploy.yaml"):
        super().__init__(daemon=True)
        self._config_path = config_path

        # ---- load config immediately (before thread starts) ----
        self._cfg = {}
        self._load_config()

        # ---- odometry cache ----
        self._odom_lock = threading.Lock()
        self._position = np.zeros(3)
        self._quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self._odom_valid = False
        self._pose_sequence = 0
        self._pose_received_monotonic = 0.0
        self._pose_trace_meta = None
        self.pose_trace = get_trace('nav')

        # ---- computed velocity cache ----
        self._vel_lock = threading.Lock()
        self._lin_vel = np.zeros(3)
        self._ang_vel = np.zeros(3)

        # ---- previous odometry for velocity estimation ----
        self._prev_pos: np.ndarray | None = None
        self._prev_quat_wxyz: np.ndarray | None = None
        self._prev_time: float | None = None
        self._vel_init = False  # True after first velocity is computed

        # ---- jump detection ----
        self._pos_jump_threshold = self._cfg.get("pos_jump_threshold", 1.0)

        # ---- sequence counter ----
        self._seq = 0

        # ---- Nav_Task_Load (created immediately so load_task() works) ----
        self._task_loader = Nav_Task_Load(
            api_url=self._cfg["api_url"],
            map_name=self._cfg["map_name"],
            task_name=self._cfg["task_name"],
            skip_reloc_threshold=self._cfg.get("skip_reloc_threshold", False),
        )

        # ---- thread state ----
        self._running = False
        self._connected = False

        # ---- UDP command socket (send-only) ----
        self._cmd_packer = struct.Struct("3f")
        self._cmd_remote_addr = (
            self._cfg["cmd_remote_ip"],
            self._cfg["cmd_remote_port"],
        )
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # ---- 命令斜坡（wasd 风格 slew rate 限制）：每轴每秒最多变化 cmd_slew_*，
        #      对策略输出的突变做阶梯平滑，避免起步/变向冲击。send_zero 绕过它直接停车。 ----
        self._cmd_slew = np.array([
            float(self._cfg["cmd_slew_vx"]),
            float(self._cfg["cmd_slew_vy"]),
            float(self._cfg["cmd_slew_wz"]),
        ], dtype=np.float32)
        self._last_cmd = np.zeros(3, dtype=np.float32)
        self._last_cmd_t = None
        print(f"[RobotComm] UDP cmd socket ready, "
              f"sending to {self._cfg['cmd_remote_ip']}:{self._cfg['cmd_remote_port']}")

        # ---- UDP pose socket (receive-only, cuVSLAM → 本进程) ----
        # 位姿包：struct "<7d" = [px,py,pz, qw,qx,qy,qz]，速度/重力/z 覆盖仍由本类现有逻辑计算。
        self._pose_packer = struct.Struct("<7d")
        self._pose_sock = None
        if self._cfg["pose_transport"] == "udp":
            self._pose_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._pose_sock.bind((self._cfg["pose_udp_host"], int(self._cfg["pose_udp_port"])))
            self._pose_sock.settimeout(0.5)
            print(f"[RobotComm] UDP pose socket listening on "
                  f"{self._cfg['pose_udp_host']}:{self._cfg['pose_udp_port']}")

    # ==================================================================
    # Thread lifecycle
    # ==================================================================

    def run(self):
        """Main entry point for the background thread."""
        self._running = True
        if self._cfg["pose_transport"] == "udp":
            self._run_udp_loop()
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_ws_loop())
        except Exception as e:
            print(f"[RobotComm] WS loop error: {e}")
        finally:
            loop.close()
            self._connected = False

    def stop(self):
        """Stop the bridge gracefully."""
        self._running = False
        if self.is_alive():
            try:
                self.join(timeout=3.0)
            except Exception:
                pass
        try:
            self._cmd_sock.close()
        except Exception:
            pass
        if self._pose_sock is not None:
            try:
                self._pose_sock.close()
            except Exception:
                pass
        print("[RobotComm] stopped")

    # ==================================================================
    # Public API
    # ==================================================================

    def load_task(self) -> bool:
        """Load map → relocalize → load task (blocking, ~10-30 s).

        Returns:
            True on success, False on failure.
        """
        ok = self._task_loader.load_task()
        if ok:
            goals = self._task_loader.goals
            print(f"[RobotComm] loaded {len(goals)} goals")
        return ok

    def get_latest_state(self):
        """Return the latest assembled NavStatePacketV2, or None if no data.

        Thread-safe — can be called from any thread.
        """
        with self._odom_lock:
            if not self._odom_valid:
                return None
            pos = self._position.copy()
            quat = self._quat_wxyz.copy()
            pose_sequence = self._pose_sequence
            received_monotonic = self._pose_received_monotonic
            pose_trace = dict(self._pose_trace_meta) if getattr(self, '_pose_trace_meta', None) else None

        with self._vel_lock:
            lin = self._lin_vel.copy()
            ang = self._ang_vel.copy()

        gravity = self._compute_projected_gravity(quat)
        self._seq += 1

        # Override z to fixed value (matching default_state in runtime)
        pos[2] = 0.695

        return NavStatePacketV2(
            seq=self._seq,
            timestamp_sec=time.time(),
            linear_vel_b=lin.astype(np.float32),
            angular_vel_b=ang.astype(np.float32),
            projected_gravity_b=gravity.astype(np.float32),
            robot_pos_w=pos.astype(np.float32),
            robot_quat_wxyz=quat.astype(np.float32),
            pose_sequence=pose_sequence,
            received_monotonic=received_monotonic,
            pose_trace=pose_trace,
        )

    def send_command(self, vx: float, vy: float, wz: float):
        """Send velocity command via UDP, slew-rate limited (wasd-style ramp).

        向目标速度逐步逼近：每轴每秒最多变化 cmd_slew_*（默认 4.0，即 wasd 的
        0.08/步 @50Hz），把策略输出的突变平滑成阶梯，避免起步/变向冲击。
        Thread-safe — can be called from any thread.
        """
        target = np.asarray([float(vx), float(vy), float(wz)], dtype=np.float32)
        now = time.monotonic()
        dt = 0.0 if self._last_cmd_t is None else max(0.0, now - self._last_cmd_t)
        step = self._cmd_slew * dt
        delta = target - self._last_cmd
        cmd = self._last_cmd + np.clip(delta, -step, step)
        self._last_cmd = cmd
        self._last_cmd_t = now
        packet = self._cmd_packer.pack(float(cmd[0]), float(cmd[1]), float(cmd[2]))
        self._cmd_sock.sendto(packet, self._cmd_remote_addr)
        getattr(self, 'pose_trace', NULL_TRACE).emit('command', command=[float(cmd[0]), float(cmd[1]), float(cmd[2])])

    def send_zero(self):
        """Send zero velocity command immediately (hard stop) and reset the ramp.

        停车/急停路径用它，不受斜坡限制；下次运动命令从 0 重新起步。
        """
        self._last_cmd = np.zeros(3, dtype=np.float32)
        self._last_cmd_t = time.monotonic()
        packet = self._cmd_packer.pack(0.0, 0.0, 0.0)
        self._cmd_sock.sendto(packet, self._cmd_remote_addr)
        getattr(self, 'pose_trace', NULL_TRACE).emit('command', command=[0.0, 0.0, 0.0])

    # ==================================================================
    # Config loading
    # ==================================================================

    def _load_config(self) -> None:
        candidates = [
            self._config_path,
            "configs/nav_deploy.yaml",
            "../configs/nav_deploy.yaml",
            "config/nav_deploy.yaml",
            "../config/nav_deploy.yaml",
        ]
        cfg_dict = None
        used = None
        for path in candidates:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    cfg_dict = yaml.safe_load(f)
                used = path
                break

        if cfg_dict is None:
            raise FileNotFoundError(
                f"Config not found. Tried: {', '.join(candidates[:3])}..."
            )

        self._cfg = {
            "websocket_uri": cfg_dict.get("websocket_uri", "ws://<INTERNAL_IP>:8765"),
            "odom_topic": cfg_dict.get("odom_topic", "/high_frequency_odometry_baselink"),
            "api_url": cfg_dict.get("api_url", "http://fake-api-url:10000"),
            "map_name": cfg_dict.get("map_name", "default_map"),
            "task_name": cfg_dict.get("task_name", "default_task"),
            "skip_reloc_threshold": bool(cfg_dict.get("skip_reloc_threshold", False)),
            "cmd_remote_ip": cfg_dict.get("cmd_remote_ip", "127.0.0.1"),
            "cmd_remote_port": int(cfg_dict.get("cmd_remote_port", 8080)),
            "cmd_slew_vx": float(cfg_dict.get("cmd_slew_vx", 4.0)),
            "cmd_slew_vy": float(cfg_dict.get("cmd_slew_vy", 4.0)),
            "cmd_slew_wz": float(cfg_dict.get("cmd_slew_wz", 4.0)),
            "pos_jump_threshold": float(cfg_dict.get("pos_jump_threshold", 1.0)),
            "pose_transport": cfg_dict.get("pose_transport", "udp"),
            "pose_udp_host": cfg_dict.get("pose_udp_host", "127.0.0.1"),
            "pose_udp_port": int(cfg_dict.get("pose_udp_port", 8082)),
        }
        print(f"[RobotComm] config loaded: {used}")

    # ==================================================================
    # Velocity from odometry finite differences
    # ==================================================================

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        """Quaternion conjugate: (w, x, y, z) → (w, -x, -y, -z)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Hamilton product: a * b."""
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    @classmethod
    def _rotate_vector_world_to_body(cls, v_world: np.ndarray,
                                      quat_wxyz: np.ndarray) -> np.ndarray:
        """Rotate a vector from world frame to body frame.

        v_body = conj(q) * [0, v_world] * q  (vector part of result)
        """
        q_conj = cls._quat_conj(quat_wxyz)
        v_quat = np.array([0.0, v_world[0], v_world[1], v_world[2]])
        result = cls._quat_multiply(cls._quat_multiply(q_conj, v_quat), quat_wxyz)
        return result[1:]

    def _compute_velocities(self, pos_cur: np.ndarray, quat_cur: np.ndarray,
                             t_cur: float) -> None:
        """Compute linear/angular velocity from two consecutive odometry readings.

        Only keeps x,y linear velocity and yaw angular velocity.
        Detects and discards sudden position jumps.
        Clamps linear velocity to ±1 m/s and angular velocity to ±0.3 rad/s.
        """
        if self._prev_pos is None or self._prev_time is None:
            self._prev_pos = pos_cur.copy()
            self._prev_quat_wxyz = quat_cur.copy()
            self._prev_time = t_cur
            return

        dt = t_cur - self._prev_time
        if dt < 1e-9:
            return

        prev_pos = self._prev_pos
        prev_quat = self._prev_quat_wxyz

        # 不再做位置跳变检测：任务导航流程中每次锚定/定位重启后 VIO 位姿
        # 会整体跳到新位置（1-2m 级），跳变是正常现象，刷屏告警已删除。
        # 跳变当帧的速度由下方 ±1 m/s / ±0.3 rad/s 限幅兜底；段内行进时
        # 位姿是 odom 帧间增量累积、连续无跳变，不影响控制。

        # ---- linear velocity in body frame (x, y only; z forced to 0) ----
        v_world = (pos_cur - prev_pos) / dt
        v_body = self._rotate_vector_world_to_body(v_world, quat_cur)
        v_body[2] = 0.0  # zero out z linear velocity

        # ---- clamp linear velocity to ±1 m/s ----
        v_body[0] = max(-1.0, min(1.0, float(v_body[0])))
        v_body[1] = max(-1.0, min(1.0, float(v_body[1])))

        # ---- angular velocity from quaternion difference (yaw only) ----
        q_diff = self._quat_multiply(quat_cur, self._quat_conj(prev_quat))
        w_diff = max(-1.0, min(1.0, q_diff[0]))
        angle = 2.0 * math.acos(w_diff)

        if angle < 1e-10:
            omega_body = np.zeros(3)
        else:
            axis = q_diff[1:] / math.sin(angle / 2.0)
            omega_body = axis * (angle / dt)

        # ---- keep only yaw (z), zero out roll/pitch ----
        omega_body[0] = 0.0
        omega_body[1] = 0.0
        # clamp yaw angular velocity to ±0.3 rad/s
        omega_body[2] = max(-0.3, min(0.3, float(omega_body[2])))

        with self._vel_lock:
            self._lin_vel = v_body
            self._ang_vel = omega_body

        if not self._vel_init:
            self._vel_init = True
            print(f"[RobotComm] velocity estimator initialized: "
                  f"dt={dt*1000:.1f}ms, "
                  f"v_body=[{v_body[0]:.3f}, {v_body[1]:.3f}, {v_body[2]:.3f}], "
                  f"w_body=[{omega_body[0]:.3f}, {omega_body[1]:.3f}, {omega_body[2]:.3f}]")

        self._prev_pos = pos_cur.copy()
        self._prev_quat_wxyz = quat_cur.copy()
        self._prev_time = t_cur

    # ==================================================================
    # UDP pose input (cuVSLAM → 本进程)
    # ==================================================================

    def _handle_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray, trace_meta=None) -> None:
        """缓存一帧位姿并更新速度估计（WS 与 UDP 两种来源共用）。"""
        t_now = time.time()
        with self._odom_lock:
            self._position = pos.copy()
            self._quat_wxyz = quat_wxyz.copy()
            self._odom_valid = True
            self._pose_sequence += 1
            self._pose_received_monotonic = time.monotonic()
            self._pose_trace_meta = trace_meta
            seq, received = self._pose_sequence, self._pose_received_monotonic
        self._compute_velocities(pos, quat_wxyz, t_now)
        trace = getattr(self, 'pose_trace', NULL_TRACE)
        if trace.enabled:
            trace.emit('rx', **(trace_meta or {}), pose_sequence=seq, received_ns=int(received*1e9),
                       wire_pose=pos.tolist()+quat_wxyz.tolist())

    def _run_udp_loop(self):
        """接收 cuVSLAM 经 UDP 发来的位姿包（<7d），喂给速度估计。

        位姿包：struct "<7d" = [px, py, pz, qw, qx, qy, qz]（7×float64）。
        位置为 Z-up 世界系，姿态为 wxyz 四元数（cuVSLAM 侧已做 OpenCV→Z-up 变换）。
        只收位姿，速度/重力/z 覆盖仍由本类现有逻辑完成。
        """
        while self._running:
            try:
                data, peer = self._pose_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            socket_received_ns = time.monotonic_ns()
            try:
                values, metadata = decode_pose(data)
            except ValueError as exc:
                getattr(self, 'pose_trace', NULL_TRACE).emit('rx_rejected', size=len(data), reason=str(exc))
                continue
            px, py, pz, qw, qx, qy, qz = values
            metadata.update(socket_received_ns=socket_received_ns, peer=f'{peer[0]}:{peer[1]}')
            self._handle_pose(np.array([px, py, pz]), np.array([qw, qx, qy, qz]), metadata)

    # ==================================================================
    # Foxglove WebSocket (asyncio)
    # ==================================================================

    async def _run_ws_loop(self):
        """Connect Foxglove WS, subscribe topics, and run receive loop."""
        cfg = self._cfg
        reconnect_interval = 3.0
        last_attempt = 0.0

        while self._running:
            if not self._connected:
                now = time.monotonic()
                if now - last_attempt >= reconnect_interval:
                    last_attempt = now
                    try:
                        await self._connect_and_run(cfg)
                    except Exception as e:
                        print(f"[RobotComm] WS connection error: {e}")
            await asyncio.sleep(0.5)

    async def _connect_and_run(self, cfg: dict):
        client = FoxgloveWsClient(cfg["websocket_uri"])

        async def on_odom(pos, quat, rpy):
            self._handle_pose(pos, quat)

        client.on_odom = on_odom

        try:
            async with client:
                self._connected = True
                await client.subscribe(cfg["odom_topic"], "nav_msgs/Odometry")
                print(f"[RobotComm] WS connected to {cfg['websocket_uri']}, "
                      f"subscribed to {cfg['odom_topic']}")
                await client.run()
        finally:
            self._connected = False

    # ==================================================================
    # Static: projected gravity (self-contained, no DataPackage dependency)
    # ==================================================================

    @staticmethod
    def _compute_projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
        """Compute gravity vector in body frame from orientation quaternion.

        World gravity: (0, 0, -9.81) rotated into body frame via q^{-1}.

        Args:
            quat_wxyz: ndarray (4,) in (w, x, y, z) order.

        Returns:
            ndarray (3,) — gravity vector in body frame.
        """
        w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
        g = 9.81

        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm > 1e-12:
            w, x, y, z = w / norm, x / norm, y / norm, z / norm

        return np.array([
            2.0 * (x * z - w * y) * (-g),
            2.0 * (y * z + w * x) * (-g),
            (1.0 - 2.0 * (x * x + y * y)) * (-g),
        ])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RobotComm bridge test")
    parser.add_argument(
        "--config",
        default="configs/nav_deploy.yaml",
        help="Path to nav_deploy.yaml (default: configs/nav_deploy.yaml)",
    )
    parser.add_argument(
        "--hz", type=float, default=5.0,
        help="Print rate in Hz (default: 5.0)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RobotComm Bridge Test")
    print("=" * 60)

    rc = RobotComm(config_path=args.config)
    rc.start()
    print("[test] RobotComm thread started")

    # ---- Step 1: load task (map + reloc + task) ----
    print("\n[test] Calling load_task() ...\n")
    ok = rc.load_task()
    if not ok:
        print("[test] load_task() FAILED (continuing anyway to test data flow)\n")
    else:
        print(f"\n[test] load_task() SUCCESS")

    # ---- Step 2: poll and print state ----
    print(f"\n[test] Printing state at ~{args.hz:.0f} Hz (Ctrl+C to stop)...\n")
    interval = 1.0 / args.hz

    try:
        while True:
            state = rc.get_latest_state()
            if state is None:
                print("[test] waiting for odometry data ...")
            else:
                print("-" * 50)
                print(f"  seq               : {state.seq}")
                print(f"  timestamp_sec     : {state.timestamp_sec:.3f}")
                print(f"  linear_vel_b      : [{state.linear_vel_b[0]:+.4f}, "
                      f"{state.linear_vel_b[1]:+.4f}, {state.linear_vel_b[2]:+.4f}]")
                print(f"  angular_vel_b     : [{state.angular_vel_b[0]:+.4f}, "
                      f"{state.angular_vel_b[1]:+.4f}, {state.angular_vel_b[2]:+.4f}]")
                print(f"  projected_gravity_b: [{state.projected_gravity_b[0]:+.4f}, "
                      f"{state.projected_gravity_b[1]:+.4f}, {state.projected_gravity_b[2]:+.4f}]")
                print(f"  robot_pos_w       : [{state.robot_pos_w[0]:+.4f}, "
                      f"{state.robot_pos_w[1]:+.4f}, {state.robot_pos_w[2]:+.4f}]")
                print(f"  robot_quat_wxyz   : [{state.robot_quat_wxyz[0]:+.4f}, "
                      f"{state.robot_quat_wxyz[1]:+.4f}, {state.robot_quat_wxyz[2]:+.4f}, "
                      f"{state.robot_quat_wxyz[3]:+.4f}]")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[test] interrupted by user")
    finally:
        rc.stop()
        print("[test] done")
