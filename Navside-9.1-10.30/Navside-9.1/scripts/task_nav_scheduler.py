#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""task_nav_scheduler.py —— 任务导航调度器（一键分段导航）

编排两个子进程，把一条大路径导航切成多个小段：

    SLAM 子进程 = run_orbbec_tasknav.sh --scheduler（相机 B，纯 VIO 位姿源）
    NAV  子进程 = run_nav.py --real（相机 A 深度，SRU 推理 + 控制；
                 独立交互终端窗口运行，面板按键 A/S/D/F/G 最高优先级）

状态机：

    启动前：在审核后的规划图选取目标，地图身份校验通过后才启动子进程。
    BOOT ──SLAM 就绪──→ WAIT_STARTUP_ANCHOR（cuvslam 首帧自动 localize）
      └─ anchor=busy → ANCHORING
          └─ anchor=ok → PLANNING（真实重定位起点 → 第一个目标）
              └─ A* 成功 → SLAM 面板显示 15 个 (X,0,Z)
                  → WAIT_PATH_READY → WAIT_PATH_RUNNING → SEGMENT
              └─ 失败 → PLAN_FAILED：停车；localize 后重试同一目标
    SEGMENT ──cuVSLAM 实时位姿距任务点 ≤ arrive_tolerance_m 且持续
      arrive_confirm_s（VIO 到达判定；NavSide goal_reached 作备份）──→
      → NAV "A"（停推理）+ SLAM "pause"（关 VIO）→ ARRIVED，提示输入 localize
    ARRIVED ──用户 localize → 转发 SLAM → ANCHORING
      └─ anchor=ok 且偏差 ≤ anchor_verify_tolerance_m → 推进任务点 → 下一段 PLANNING
      └─ anchor=fail / 超时 / 偏差超限 → 回 ARRIVED 重试
    崩溃 → 停车/取消旧规划 → 重启定位 → PLANNING（同一未完成目标，不推进）
    人工中途停车 → PAUSED → 用户 localize → 同一目标重新规划/交付
    legacy 配置保留原 goal/S 接口。
    force 只能跳过成功锚定的偏差校验，不能替代定位或跳过 A*。
    全部任务点完成 → DONE：两端优雅退出。

通信协议：
- SLAM 子进程：调度器 → stdin 写命令；[SCHED] 事件经 SLAM_SCHED_FILE 文件通道
  （cuvslam 双写 stdout+文件；含每 0.1s 的 pose=(...) t=.. 实时位姿流，调度器
  据此做 VIO 到达判定；[SCHED] 行不写日志，SLAM 观察窗口显示原地刷新的
  位姿面板和当前段 A* 参考点。管道曾出现 anchor=ok 行丢失，文件通道留痕可查）。
- NavSide：跑在独立交互终端窗口，与调度器走两个文件通道——
  NAVSIDE_CMD_FILE（调度器追加命令，NavSide 低优先级应用；交互终端按键
  A/S/D/F/G 永远是最高优先级、随时可停车）和 NAVSIDE_SCHED_FILE（NavSide
  追加 [SCHED] 状态行，调度器 tail 解析；退出时写 process=exit）。
用户命令在本终端输入：l(重定位) / force / status / send <nav|slam> <命令> / quit。

显示布局：SLAM 打印及当前段 15 个 XYZ 写入 logs/task_nav/slam_*.log 并弹观察窗口 tail
（gnome-terminal/xterm，无显示器则只落盘）；NavSide 有自己的交互终端窗口
（原始模式面板 + 键盘）；调度器本终端只显示 [TASK] 状态信息与提示。

坐标约定：任务点写在 cuVSLAM 参考地图全局系（OpenCV：+X 右、+Y 下、+Z 前），
下发 NavSide 的 goal 需变换到 Z-up（+X 前、+Y 左、+Z 上）：
goal_zup = (t_z, -t_x, robot_height_z)；z 必须为机器人高度，NavSide 到达判定
是 3D 距离（bridge.py 硬编码机器人 z=0.695），z 不匹配则 goal_reached 永不触发。
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml

from task_nav_astar import (SegmentAStarPlanner, validate_anchor_pose,
                            load_task_points, reference_lines, REFERENCE_Y_M)

# 不加行首锚定：子进程若打印 input() 提示符等前缀，[SCHED] 会拼在同一行里。
SCHED_RE = re.compile(r"\[SCHED\]\s*(.*)$")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
# 与上游 pose_panel_text 的清屏前缀一致；每次重绘同步显示当前参考点。
SLAM_PANEL_PREFIX = "\033[H\033[J"

STATE_BOOT = "BOOT"
STATE_WAIT_STARTUP_ANCHOR = "WAIT_STARTUP_ANCHOR"
STATE_ANCHORING = "ANCHORING"
STATE_SEGMENT = "SEGMENT"
STATE_ARRIVED = "ARRIVED"
STATE_PLANNING = "PLANNING"
STATE_PLAN_FAILED = "PLAN_FAILED"
STATE_RESTART_WAIT = "RESTART_WAIT"
STATE_WAIT_PATH_READY = 'WAIT_PATH_READY'
STATE_WAIT_PATH_RUNNING = 'WAIT_PATH_RUNNING'
STATE_WAIT_POSE = 'WAIT_POSE'
STATE_PAUSED = 'PAUSED'
STATE_DONE = "DONE"

SLAM_RESTART_DELAY_S = 10.0  # SLAM 崩溃后重启前的等待（秒）
# 立即重开会踩 Orbbec USB 栈的释放/枚举竞态：标定取帧失败、甚至相机从
# 总线消失（实测多次快速开关后只剩一台相机）。留 10s 让相机干净复位。


class ChildProc:
    """一个受管子进程：stdin 收命令，stdout 被 tee + 解析。"""

    def __init__(self, tag: str, cmd: list, cwd: str, env: dict):
        self.tag = tag  # "SLAM" / "NAV"
        self.cmd = list(cmd)
        self.cwd = cwd
        self.env = env
        self.proc: subprocess.Popen | None = None
        self._stdin_lock = threading.Lock()
        # 调度器主线程整体替换字符串，日志线程每帧读取一份完整快照。
        # 内容仅用于 SLAM 观察窗口，不进入 SLAM/SRU 命令或事件通道。
        self.reference_panel_text = ""

    def start(self) -> None:
        merged_env = dict(os.environ)
        merged_env.update({k: v for k, v in self.env.items() if v})
        # stdout 接管道后 Python 默认块缓冲，[SCHED] 事件会延迟刷出；
        # 强制行缓冲，保证调度器实时收到状态行。
        merged_env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            env=merged_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # 独立进程组：终端 Ctrl+C 只打到调度器，由调度器统一编排两端退出。
            start_new_session=True,
        )

    def send(self, text: str) -> bool:
        """向子进程 stdin 写一行命令。"""
        if self.proc is None or self.proc.poll() is not None or self.proc.stdin is None:
            return False
        try:
            with self._stdin_lock:
                self.proc.stdin.write(text.strip() + "\n")
                self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def interrupt(self) -> None:
        """SIGINT（NavSide 的 finally 清理依赖 KeyboardInterrupt）。"""
        if self.alive():
            try:
                self.proc.send_signal(signal.SIGINT)
            except OSError:
                pass


def reader_loop(child: ChildProc, events: queue.Queue, tag_lower: str, log_path: Path,
                generation: int = 0) -> None:
    """把子进程 stdout 写入日志文件，SLAM 面板重绘时插入当前段参考点。

    [SCHED] 事件不再从管道解析——cuvslam 侧把 [SCHED] 行双写到
    SLAM_SCHED_FILE（sched_file_loop 读取），管道只承担日志 + 退出检测。
    """
    assert child.proc is not None and child.proc.stdout is not None
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        for raw in child.proc.stdout:
            line = raw.rstrip("\n")
            if SCHED_RE.search(line):
                continue  # 机器行不写日志（保持位姿面板干净），事件走文件通道
            if tag_lower == "slam" and line.startswith(SLAM_PANEL_PREFIX):
                panel = child.reference_panel_text
                if panel:
                    # 不能只追加一次：上游每 0.1s 清屏会把普通打印覆盖。
                    line = SLAM_PANEL_PREFIX + panel + "\n" + line[len(SLAM_PANEL_PREFIX):]
            log_file.write(line + "\n")
            if tag_lower == "slam" and "Starting task navigation" in line:
                events.put((tag_lower, {"__ready__": "1", "__generation__": str(generation)}))
    events.put((tag_lower, {"__exit__": "1", "__generation__": str(generation)}))


def open_viewer_window(title: str, log_path: Path) -> bool:
    """在独立图形终端窗口里 tail 日志，方便观察子进程原始打印。

    只在有显示器时可用（与 run_orbbec_tasknav.sh 相同的 DISPLAY 回退逻辑）；
    失败时返回 False，由调用方提示日志文件路径。
    """
    if not os.environ.get("DISPLAY"):
        gdm_xauth = Path(f"/run/user/{os.getuid()}/gdm/Xauthority")
        if gdm_xauth.is_file():
            os.environ["XAUTHORITY"] = str(gdm_xauth)
        os.environ["DISPLAY"] = ":0"
    try:
        term = shutil.which("gnome-terminal") or shutil.which("xterm")
        if not term:
            return False
        if term.endswith("gnome-terminal"):
            cmd = [term, "--title", title, "--", "bash", "-c",
                   'tail -n +1 -f "$0"', str(log_path)]
        else:
            cmd = [term, "-T", title, "-e", "tail", "-n", "+1", "-f", str(log_path)]
        subprocess.Popen(
            cmd, start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except OSError:
        return False


def sched_file_loop(path: str, events: queue.Queue, stop_event: threading.Event,
                    tag: str = "nav", generation: int = 0) -> None:
    """tail 子进程的 [SCHED] 状态文件，解析成事件投递。

    两端通用：NavSide 跑在独立交互终端窗口（stdout 归终端），cuvslam 的
    [SCHED] 行也双写到文件（管道曾出现事件行丢失，文件通道留痕可查）。
    process=exit 行 → __exit__ 事件。
    """
    offset = 0
    while not stop_event.is_set():
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(offset)
                while not stop_event.is_set():
                    line = f.readline()
                    if line:
                        offset = f.tell()
                        line = line.rstrip("\n")
                        m = SCHED_RE.search(line)
                        if not m:
                            continue
                        body = m.group(1)
                        if body.startswith("note="):
                            # 诊断文本可含空格/中文，KV 解析会碎——整体取 note 值
                            kv = {"note": body[5:]}
                        else:
                            kv = dict(KV_RE.findall(body))
                        if "process" in kv and kv["process"] == "exit":
                            kv = {"__exit__": "1"}
                        if tag == 'slam':
                            kv['__generation__'] = str(generation)
                        kv['__received_at__'] = str(time.time())
                        events.put((tag, kv))
                    else:
                        time.sleep(0.05)
        except OSError:
            time.sleep(0.2)


class NavSideChild:
    """NavSide 子进程：跑在独立交互终端窗口（原始面板 + 键盘），控制走文件通道。

    调度器对它不持有 stdin/stdout 管道：
    - 命令 → 追加写 NAVSIDE_CMD_FILE（NavSide 低优先级应用；终端按键最高优先级）
    - 状态 ← tail NAVSIDE_SCHED_FILE（[SCHED] 行）
    - 退出 ← 状态文件里的 process=exit 行
    """

    tag = "NAV"

    def __init__(self, cmd_file: str, sched_file: str):
        self.cmd_file = cmd_file
        self.sched_file = sched_file


def user_input_loop(user_queue: queue.Queue, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        cmd = line.strip().lower()
        if cmd:
            user_queue.put(cmd)


def parse_pose(text: str):
    """把 "[SCHED] ... pose=(x,y,z,qx,qy,qz,qw)" 的值解析成 7 元组。"""
    try:
        return validate_anchor_pose(text.strip("()").split(","))
    except (ValueError, AttributeError):
        return None


class TaskNavScheduler:
    def __init__(self, cfg: dict, args):
        self.cfg = cfg
        self.args = args
        self.events: queue.Queue = queue.Queue()
        self.user_queue: queue.Queue = queue.Queue()
        self.slam: ChildProc | None = None
        self.nav: ChildProc | None = None
        self.state = STATE_BOOT
        self.idx = 0                # 当前目标段（= 任务点下标）
        self.segment_t0 = 0.0       # 本段出发时刻（monotonic）
        self.anchor_t0 = 0.0        # 本次锚定开始时刻
        self.startup_anchor_t0 = 0.0  # 启动锚定等待起点
        self.anchor_startup_phase = False  # ANCHORING 是否发生在启动阶段
        self.nav_state: dict = {}
        self.slam_state: dict = {}
        self.nav_cmd_file = ""      # 调度器 → NavSide 指令文件
        self.nav_sched_file = ""    # NavSide → 调度器状态文件
        self.slam_sched_file = ""   # cuvSlam → 调度器事件文件（[SCHED] 双写通道）
        self.nav_exited = False     # 状态文件收到 process=exit
        self.startup_anchor_failed = False  # 启动自动锚定是否已失败过（决定是否转发手动 localize）
        self.slam_restart_pending = False  # cuvslam 意外退出，等待自动重启
        self.slam_restart_at = 0.0         # 计划重启时刻（延迟重启，等相机 USB 重新枚举）
        self.resume_after_anchor = False   # 重启后锚定成功 → 重新规划当前目标，不推进
        self.slam_restarts = 0             # cuvslam 重启总次数（日志/文件命名用）
        self.slam_crashes = 0              # cuvslam 意外崩溃次数（>3 放弃重启）
        self.shutting_down = False         # shutdown() 已开始（不再触发重启逻辑）
        self.log_dir: Path | None = None   # 日志目录（重启时复用）
        self._slam_file_stop: threading.Event | None = None  # 当前 SLAM 事件 tailer 的停止标志
        self._nav_file_stop: threading.Event | None = None   # 关停等待结束后才停止读取退出确认
        self.arrive_hold_t0 = 0.0   # VIO 到达确认计时（位姿首次进容差时刻；0=未进容差）
        self.arrival_prompted = False
        self.done = False
        self.stop_event = threading.Event()
        self.exit_code = 0
        self.planner = None
        self.anchor_pose = None     # 只接收当次 anchor=ok，绝不用滚动 VIO 替代
        self.anchor_sequence = 0
        self.reanchor_current = False   # PLAN_FAILED 重定位重试，不推进目标
        self.plan_generation = 0
        self.plan_cancel = threading.Event()
        self.plan_results = queue.Queue()
        self.plan_output_dir = None
        self.current_route = None
        self.last_plan_error = None
        self.segment_ready_at = 0.0  # 过滤发出本段命令前已排队的到达位姿
        self.sru_enabled = cfg.get('sru', {}).get('enabled', True)
        self.path_aware = bool(cfg.get('path_aware', {}).get('enabled', False))
        self.session_id = uuid.uuid4().hex
        self.path_revision = 0
        self.path_segment_id = None
        self.path_directory = None
        self.path_deadline = 0.0
        self.nav_manual_token = 0
        self.authorized_resume_token = None

    # ------------------------------------------------------------------ utils
    def log(self, msg: str) -> None:
        print(f"[TASK] {msg}", flush=True)

    def nav_send(self, text: str) -> bool:
        """向 NavSide 指令文件追加一行（NavSide 低优先级应用；终端按键最高优先级）。"""
        if not self.sru_enabled:
            return False
        try:
            with open(self.nav_cmd_file, "a", encoding="utf-8") as f:
                f.write(text.strip() + "\n")
            return True
        except OSError:
            return False

    def send(self, child, text: str) -> bool:
        if not self.sru_enabled and (child is None or getattr(child, 'tag', '') == 'NAV'):
            self.log('SRU 已关闭，未发送 NavSide 命令')
            return False
        if child is None:
            self.log(f"发送失败：子进程未就绪，命令 {text}")
            return False
        if isinstance(child, NavSideChild):
            ok = self.nav_send(text)
        else:
            ok = child.send(text)
        display = 'segment_load <目标+15点，详见 segment_delivery.json>' if text.startswith('segment_load ') else text
        self.log(f"→ {child.tag}: {display}" + ("" if ok else "  （发送失败）"))
        return ok

    def stop_nav(self) -> bool:
        """No NavSide process or command channel exists in scheduler-only mode."""
        return not self.sru_enabled or self.send(self.nav, 'A')

    # ------------------------------------------------------------------ 动作
    def start_segment(self, i: int) -> None:
        """Plan first; publish the result to the SLAM panel before resuming."""
        self._cancel_plan()
        self.idx = i
        self.state = STATE_PLANNING
        self.last_plan_error = None
        if self.anchor_pose is None:
            self._plan_failed('缺少本次有效重定位位姿，请 localize 后重试')
            return
        if self.planner is None or self.plan_output_dir is None:
            self._plan_failed('A* 或规划输出目录未初始化')
            return
        # Stop SRU while planning; scheduler-only mode has no motion controller.
        if not self.stop_nav():
            self._plan_failed('无法发送停车指令')
            return
        self.send(self.slam, 'pause')
        generation, cancel = self.plan_generation, self.plan_cancel
        anchor, sequence = self.anchor_pose, self.anchor_sequence
        target = tuple(self.cfg['task_points'][i])
        self.log(f"A* 第 {i+1} 段：本次重定位 XZ=({anchor[0]:.6f},{anchor[2]:.6f}) "
                 f"→ 地图选点 XZ=({target[0]:.6f},{target[2]:.6f})；后台规划中")

        def work():
            try:
                result = self.planner.plan(anchor, target, i, anchor_sequence=sequence, cancel=cancel)
                result['scheduler']['sru_enabled'] = self.sru_enabled
                if cancel.is_set():
                    return
                stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                directory = self.plan_output_dir/f'segment_{i+1:03d}_anchor_{sequence:03d}_{stamp}'
                self.planner.export(result, directory)
                outcome = (result, directory)
            except Exception as exc:
                outcome = exc
            self.plan_results.put((generation, i, sequence, outcome))
        threading.Thread(target=work, daemon=True, name=f'astar-segment-{i+1}').start()

    def _cancel_plan(self):
        self.plan_cancel.set()
        self.plan_cancel = threading.Event()
        self.plan_generation += 1
        self.current_route = None
        self.path_segment_id = None
        self.path_deadline = 0.0
        self.path_directory = None
        if self.slam is not None:
            self.slam.reference_panel_text = ""

    def _plan_failed(self, reason):
        self._cancel_plan()
        self.last_plan_error = str(reason)
        self.state = STATE_PLAN_FAILED
        self.stop_nav()
        self.send(self.slam, 'pause')
        self.log(f'第 {self.idx+1} 段暂停：{reason}。不推进目标、不下发行进指令。')
        self.log('调整机器人到可定位、可通行位置后 localize 重试当前目标；'
                 '如需修改目标，请 quit 后重新在地图选点。force 不能绕过规划失败。')
        if self.plan_output_dir is not None:
            try:
                self.plan_output_dir.mkdir(parents=True, exist_ok=True)
                with (self.plan_output_dir/'failures.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
                        segment_index=self.idx, error=str(reason), anchor_pose=self.anchor_pose,
                        target=self.cfg['task_points'][self.idx]), ensure_ascii=False)+'\n')
            except OSError as exc:
                self.log(f'规划错误日志写入失败：{exc}')

    def poll_planning(self):
        while True:
            try:
                generation, i, sequence, outcome = self.plan_results.get_nowait()
            except queue.Empty:
                return
            if (self.done or self.state != STATE_PLANNING or generation != self.plan_generation
                    or i != self.idx or sequence != self.anchor_sequence):
                continue
            if isinstance(outcome, Exception):
                self._plan_failed(outcome)
                continue
            result, directory = outcome
            self.current_route = result
            self.path_directory = directory
            self.log(f"A* 成功：第 {i+1} 段，15 个参考点，打印 Y 固定 {REFERENCE_Y_M:g}m；"
                     f"起点{'包含' if result['includes_start'] else '不包含'}，终点{'包含' if result['includes_goal'] else '不包含'}。")
            self.log(f'原始重定位位姿 (X,Y,Z,qx,qy,qz,qw)={self.anchor_pose}')
            if self.slam is not None:
                self.slam.reference_panel_text = '\n'.join([
                    f'=== A* 第 {i+1} 段 | 定位 #{sequence} | 15 个参考点 (m) ===',
                    *reference_lines(result),
                    (f'打印 Y={REFERENCE_Y_M:g}；SRU OFF；仅定位/规划，移动由独立遥控完成。' if not self.sru_enabled
                     else f'打印 Y={REFERENCE_Y_M:g}；模型路径 Z=0.5；5 Hz PathAware。' if self.path_aware
                     else f'X/Z 为地图坐标；打印 Y={REFERENCE_Y_M:g}；尚未接入 SRU。'),
                ])
            self.log('15 个参考点 XYZ 显示在 SLAM 终端，随位姿面板持续刷新。')
            self.log(f'参考点 XYZ / JSON / CSV 已保存：{directory}')
            self._resume_segment(i)

    def _resume_segment(self, i: int) -> None:
        if self.done or self.state != STATE_PLANNING:
            return
        if not self.sru_enabled:
            if not self.send(self.slam, 'resume'):
                self._plan_failed('恢复 VIO 失败')
                return
            self.segment_ready_at = time.time()
            self.segment_t0 = time.monotonic()
            self.arrive_hold_t0 = 0.0
            self.state = STATE_SEGMENT
            self.log(f'SRU OFF：第 {i+1} 段规划完成，VIO 已恢复；可检查 15 点或独立遥控移动。'
                     '到达由真实 VIO 判定，停稳后 localize 确认下一段；pause 可暂停并重试当前段。')
            return
        if self.path_aware:
            self._load_path_segment(i)
            return
        tx, ty, tz = self.cfg["task_points"][i]
        h = float(self.cfg["robot_height_z"])
        gx, gy, gz = tz, -tx, h  # OpenCV → Z-up
        self.log(
            f"开始第 {i + 1}/{len(self.cfg['task_points'])} 段："
            f"任务点(cuvslam)=({tx},{ty},{tz}) → goal(Z-up)=({gx:.3f},{gy:.3f},{gz:.3f})"
        )
        if not self.send(self.slam, "resume"):
            self._plan_failed('恢复 VIO 失败')
            return
        time.sleep(0.5)
        if self.done:
            self.send(self.nav, 'A')
            self.send(self.slam, 'pause')
            return
        if not self.send(self.nav, f"goal {gx:.3f} {gy:.3f} {gz:.3f}"):
            self._plan_failed('发送原有单目标 goal 失败')
            return
        time.sleep(0.2)
        if self.done:
            self.send(self.nav, 'A')
            self.send(self.slam, 'pause')
            return
        if not self.send(self.nav, "S"):
            self._plan_failed('恢复原有导航推理失败')
            return
        self.segment_ready_at = time.time()
        self.segment_t0 = time.monotonic()
        self.arrive_hold_t0 = 0.0  # 新段重新开始到达确认计时
        self.state = STATE_SEGMENT

    def _load_path_segment(self, i):
        if not self.sru_enabled:
            return
        route = self.current_route
        if route is None or not route['includes_start'] or not route['includes_goal']:
            self._plan_failed('PathAware 必须有包含起终点的 15 点')
            return
        self.path_revision += 1
        self.path_segment_id = f'{self.session_id}:{self.path_revision}'
        tx, _, tz = self.cfg['task_points'][i]
        payload = dict(schema_version=1, session_id=self.session_id, revision=self.path_revision,
            segment_id=self.path_segment_id, frame='navside_zup', goal_w=[tz,-tx,.695],
            path_w=[[z,-x,.5] for x,z in route['references_xy']],
            resume_token=self.authorized_resume_token)
        self.state = STATE_WAIT_PATH_READY
        self.path_deadline = time.monotonic()+float(self.cfg.get('path_aware', {}).get('ready_timeout_s', 60))
        try:
            (self.path_directory/'segment_delivery.json').write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        except (OSError, ValueError) as exc:
            self._plan_failed(f'段数据保存失败：{exc}')
            return
        if not self.send(self.nav, 'segment_load '+json.dumps(payload, separators=(',', ':'), allow_nan=False)):
            self._plan_failed('下发整段路径失败')
            return
        self.log(f'等待下游加载确认：{self.path_segment_id}')

    def _handle_path_event(self, kv):
        if kv.get('session_id') != self.session_id:
            return
        phase = kv.get('path')
        if phase == 'paused':
            try:
                token = int(kv.get('manual_token', '0'))
            except ValueError:
                return
            if token <= self.nav_manual_token:
                return
            self.nav_manual_token = token
            self.authorized_resume_token = None
            self._pause_current('人工停车')
            return
        if kv.get('segment_id') != self.path_segment_id or self.path_segment_id is None:
            return
        if phase not in ('ready', 'running', 'waiting_pose', 'pose_resumed', 'error', 'rejected'):
            return
        try:
            with (self.path_directory/'delivery_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(), **kv))+'\n')
            if phase == 'ready' and self.state == STATE_WAIT_PATH_READY:
                self.current_route['scheduler'].update(sent_to_sru=True,
                    nav_segment_id=self.path_segment_id, model_path_z_m=.5)
                temporary = self.path_directory/'route.json.tmp'
                temporary.write_text(json.dumps(self.current_route, ensure_ascii=False, indent=2,
                                               allow_nan=False)+'\n', encoding='utf-8')
                temporary.replace(self.path_directory/'route.json')
        except OSError as exc:
            self._plan_failed(f'接收确认记录失败：{exc}')
            return
        if phase in ('error', 'rejected'):
            reason = '下游拒绝/停止当前段：'+kv.get('reason', phase)
            if kv.get('reason') == 'pose_jump_requires_localize':
                reference = {'pre_wait_pose': '等待前位姿', 'previous_sample': '上一新采样'}.get(
                    kv.get('jump_reference'), '未记录')
                reason += (f"；对比={reference}，位移={kv.get('position_jump_m', '?')}m"
                           f" / 上限={kv.get('position_limit_m', '?')}m，"
                           f"转角={kv.get('rotation_jump_deg', '?')}°")
            self._plan_failed(reason)
            return
        if phase == 'ready' and self.state == STATE_WAIT_PATH_READY:
            self.state = STATE_WAIT_PATH_RUNNING
            self.path_deadline = time.monotonic()+float(self.cfg.get('path_aware', {}).get('running_timeout_s', 10))
            if not self.send(self.slam, 'resume'):
                self._plan_failed('恢复 VIO 失败')
                return
            if not self.send(self.nav, 'segment_start '+self.path_segment_id):
                self._plan_failed('发送本段启动请求失败')
                return
            self.log('路径已加载；等待下游收到恢复后的新位姿并确认运行')
        elif phase == 'running' and self.state == STATE_WAIT_PATH_RUNNING:
            self.state = STATE_SEGMENT
            self.path_deadline = 0.0
            self.segment_ready_at = time.time()
            self.segment_t0 = time.monotonic()
            self.arrive_hold_t0 = 0.0
            self.authorized_resume_token = None
            self.log(f'PathAware 已确认运行：第 {self.idx+1} 段，{self.path_segment_id}')
        elif phase == 'waiting_pose' and self.state in (STATE_SEGMENT, STATE_WAIT_PATH_RUNNING):
            self.state = STATE_WAIT_POSE
            self.path_deadline = 0.0
            self.arrive_hold_t0 = 0.0
            self.nav_state.clear()
            self.log(f'第 {self.idx+1} 段等待位姿：已发零速，保留当前目标和 15 点路径；'
                     f"原因={kv.get('reason')}，包龄={kv.get('pose_age_s', 'none')}s，"
                     f"等待前后位置偏移上限={kv.get('wait_position_limit_m', '?')}m。")
        elif phase == 'pose_resumed' and self.state == STATE_WAIT_POSE:
            self.state = STATE_SEGMENT
            self.segment_ready_at = time.time()
            self.segment_t0 = time.monotonic()
            self.arrive_hold_t0 = 0.0
            self.nav_state.clear()
            self.log(f'第 {self.idx+1} 段位姿连续有效，自动继续原目标和原路径；'
                     f"等待={kv.get('wait_s', 'none')}s，新采样数={kv.get('stable_samples', 'none')}。")

    def poll_path_timeout(self):
        if (self.state in (STATE_WAIT_PATH_READY, STATE_WAIT_PATH_RUNNING)
                and time.monotonic() >= self.path_deadline):
            self._plan_failed('下游路径加载/启动确认超时')

    def _pause_current(self, reason):
        arrived = self.state == STATE_ARRIVED
        self._cancel_plan()
        self.anchor_pose = None
        self.reanchor_current = False
        self.stop_nav()
        self.send(self.slam, 'pause')
        self.state = STATE_ARRIVED if arrived else STATE_PAUSED
        self.log(f'{reason}：' + ('保持停止。' if self.sru_enabled else 'VIO 已暂停，请用独立遥控停车并保持静止。')
                 + '输入 localize 后重新定位；'
                 + ('确认到点并规划下一段。' if arrived else '继续未完成的当前目标。'))

    def arrive(self) -> None:
        if self.state != STATE_SEGMENT:
            return  # 防双触发（VIO 判定与 NavSide goal_reached 可能同时到达）
        self._cancel_plan()
        self.anchor_pose = None
        # 顺序要求：先停推理（等价于按 A 键：STANDBY + 清 LSTM + 发零速），
        # 再关 VIO——关 VIO 后机器人状态源冻结，必须先让 SRU 停稳。
        self.log(f"第 {self.idx + 1} 段到达：" + ('先停推理，再关 VIO，机器人停止' if self.sru_enabled
                 else 'SRU OFF，暂停 VIO；请通过独立遥控停车，停稳后 localize'))
        self.stop_nav()
        if self.sru_enabled:
            time.sleep(0.5)            # 留足 NavSide 文件通道处理时间
        self.send(self.slam, "pause")  # 关 VIO 里程计（挂起采集 + 停发位姿）
        # 到达提示：调度器终端 + SLAM 面板消息行各一份。
        self.send(self.slam, f"msg 已到达任务点 {self.idx + 1}/{len(self.cfg['task_points'])}："
                             "保持静止，输入 localize；以实际定位位置规划下一段")
        self.state = STATE_ARRIVED
        self.arrival_prompted = False

    def advance(self, reason: str) -> None:
        self.log(f"任务点 {self.idx + 1}/{len(self.cfg['task_points'])} 完成（{reason}），推进")
        self.idx += 1
        if self.idx >= len(self.cfg["task_points"]):
            self.log("全部任务点完成！")
            self.done = True
            self.exit_code = 0
        else:
            self.start_segment(self.idx)

    # ------------------------------------------------------------- 锚定结果
    def _anchor_deviation(self, pose) -> tuple:
        """锚定位姿距当前/下一任务点的最小偏差（米）与对应任务点名。

        定位时机器人可能在当前任务点处（真机：到达后车停在此），也可能
        已被推到下一任务点处（台架测试）——两者都接受，取较近的一个。
        """
        pts = self.cfg["task_points"]
        best = None
        for i in (self.idx, self.idx + 1):
            if i >= len(pts):
                continue
            p = pts[i]
            d = float(((pose[0] - p[0]) ** 2 + (pose[2] - p[2]) ** 2) ** 0.5)
            if best is None or d < best[0]:
                best = (d, f"任务点 {i + 1}")
        return best

    def on_anchor_ok(self, pose_text) -> None:
        pose = parse_pose(pose_text) if pose_text else None
        if pose is None:
            self.on_anchor_fail('重定位成功事件缺少有效的 7 维有限位姿')
            return
        self.anchor_pose = pose
        self.anchor_sequence += 1
        if self.resume_after_anchor or self.reanchor_current:
            self.log(f'重定位成功，从新位置重新规划未完成的目标 {self.idx+1}，不推进任务点')
            self.resume_after_anchor = self.reanchor_current = False
            self.startup_anchor_failed = False
            self.start_segment(self.idx)
            return
        if self.anchor_startup_phase:
            self.startup_anchor_failed = False
            self.log('启动重定位成功，以实际重定位位置作为第一段 A* 起点')
            self.start_segment(0)
            return
        err, target_name = self._anchor_deviation(pose)
        tol = float(self.cfg["anchor_verify_tolerance_m"])
        self.log(f"锚定位姿 (x={pose[0]:.3f}, z={pose[2]:.3f}) 距{target_name} 偏差 {err:.2f}m")
        if err <= tol:
            self.advance(f"锚定偏差 {err:.2f}m ≤ {tol}m")
        else:
            self.log(f"偏差超限（>{tol}m）：不推进。请重试 localize，或输入 force 强制推进。")
            self.state = STATE_ARRIVED
            self.arrival_prompted = False

    def on_anchor_fail(self, reason: str) -> None:
        self.anchor_pose = None
        self._cancel_plan()
        self.log(f"锚定失败/超时（{reason}）")
        if self.reanchor_current:
            self.state = STATE_PLAN_FAILED
            self.log('当前段仍暂停，请 localize 重试；目标不推进')
        elif self.anchor_startup_phase:
            self.startup_anchor_failed = True  # 此后允许转发手动 localize 重试
            self.log("启动锚定未完成：请重新调整机器人位姿（移动/转向）后输入 localize 重试。")
            self.state = STATE_WAIT_STARTUP_ANCHOR
            self.startup_anchor_t0 = time.monotonic()  # 重计超时
        else:
            self.state = STATE_ARRIVED
            self.arrival_prompted = False
            self.log("请重新调整机器人位姿（移动/转向）后再次输入 localize 重定位"
                     "。缺少本次有效重定位时 force 不会推进。")

    # ------------------------------------------------------------- 到达判定
    def _check_vio_arrival(self, kv: dict) -> None:
        """SEGMENT 到达判定：cuVSLAM 实时位姿距任务点 XZ 距离 ≤ 容差，
        持续 arrive_confirm_s 后判定到达（不再依赖 NavSide 的机器人状态）。"""
        pose = parse_pose(kv.get("pose")) if kv.get("pose") else None
        if pose is None:
            return
        try:
            pose_t = float(kv.get("t", "0"))
        except ValueError:
            pose_t = 0.0
        if not math.isfinite(pose_t) or pose_t < self.segment_ready_at or not 0 <= time.time() - pose_t <= 1.0:
            return  # 位姿过期（流中断 >1s），不做判定
        tx, ty, tz = self.cfg["task_points"][self.idx]
        dist = float(((pose[0] - tx) ** 2 + (pose[2] - tz) ** 2) ** 0.5)
        tol = float(self.cfg.get("arrive_tolerance_m", 0.5))
        confirm_s = float(self.cfg.get("arrive_confirm_s", 1.0))
        if dist <= tol:
            if self.arrive_hold_t0 == 0.0:
                self.arrive_hold_t0 = time.monotonic()  # 首次进容差，开始确认计时
            elif time.monotonic() - self.arrive_hold_t0 >= confirm_s:
                self.log(f"VIO 判定到达：位姿距任务点 {dist:.2f}m ≤ {tol}m（确认 {confirm_s}s）")
                self.arrive()
        else:
            self.arrive_hold_t0 = 0.0

    # --------------------------------------------------------------- 事件
    def handle_sched(self, tag: str, kv: dict) -> None:
        if tag == 'nav' and not self.sru_enabled:
            return
        if tag == 'nav' and self.path_aware and 'path' in kv:
            self._handle_path_event(kv)
            return
        if tag == 'slam':
            # Events left in the queue by a crashed instance must never provide
            # the next segment's anchor or trigger another restart.
            if kv.get('__generation__', str(self.slam_restarts)) != str(self.slam_restarts):
                return
            if self.slam_restart_pending:
                return
        if "__exit__" in kv:
            if tag == "slam" and not self.shutting_down:
                self._cancel_plan()
                self.anchor_pose = None
                self.stop_nav()
                self.state = STATE_RESTART_WAIT
                if self.slam_crashes < 3:
                    self.slam_crashes += 1
                    self.slam_restart_pending = True
                    self.slam_restart_at = time.monotonic() + SLAM_RESTART_DELAY_S
                    self.log(f"WARNING: SLAM 进程意外退出（第 {self.slam_crashes} 次），"
                             f"{SLAM_RESTART_DELAY_S:.0f}s 后自动重启（等相机 USB 重新枚举）...")
                    return
                self.log("ERROR: SLAM 进程反复崩溃，放弃自动重启")
            self.log(f"ERROR: {tag.upper()} 进程已退出")
            if tag == "nav":
                self.nav_exited = True
            self.exit_code = 1
            self.done = True
            return
        if "__ready__" in kv:
            if tag == 'slam' and self.state == STATE_BOOT:
                self.state = STATE_WAIT_STARTUP_ANCHOR
                self.startup_anchor_t0 = time.monotonic()
                self.log('等待启动自动重定位；成功后先规划第一段 A*')
            return
        if "note" in kv:
            self.log(f"SLAM: {kv['note']}")  # cuvslam 诊断（定位失败原因/埋点计时）
            return

        if tag == "slam":
            self.slam_state.update(kv)
            if "pose" in kv and 'anchor' not in kv and self.state == STATE_SEGMENT:
                self._check_vio_arrival(kv)
            if "anchor" in kv:
                a = kv["anchor"]
                if self.path_aware and self.state in (STATE_SEGMENT, STATE_WAIT_POSE, STATE_WAIT_PATH_RUNNING):
                    self.anchor_pose = None
                    self._plan_failed('行进/等待期间发生重定位，坐标系连续性未确认，要求 localize')
                    return
                if a == "busy":
                    if self.state in (STATE_BOOT, STATE_WAIT_STARTUP_ANCHOR, STATE_ARRIVED, STATE_PLAN_FAILED):
                        self.anchor_startup_phase = self.state in (STATE_BOOT, STATE_WAIT_STARTUP_ANCHOR)
                        if self.state == STATE_PLAN_FAILED:
                            self.reanchor_current = True
                        self.anchor_pose = None
                        self._cancel_plan()
                        self.anchor_t0 = time.monotonic()
                        self.state = STATE_ANCHORING
                        self.log("锚定流程开始...")
                elif a == "ok":
                    if self.state == STATE_ANCHORING:
                        self.on_anchor_ok(kv.get("pose"))
                elif a == "fail":
                    if self.state == STATE_ANCHORING:
                        self.on_anchor_fail("cuvslam 返回失败")
        elif tag == "nav":
            if self.path_aware and (kv.get('session_id') != self.session_id
                    or kv.get('segment_id') != self.path_segment_id):
                return
            prev_mode = self.nav_state.get("mode")
            self.nav_state.update(kv)
            # Status received before this segment's start may still be queued
            # during the resume/goal delays. It cannot mark the new goal reached.
            received = kv.get('__received_at__')
            if (self.state == STATE_SEGMENT and received is not None
                    and float(received) < self.segment_ready_at and kv.get('mode') != 'EMERGENCY'):
                return
            if self.state == STATE_PLANNING and kv.get('mode') == 'EMERGENCY':
                self.anchor_pose = None
                self._plan_failed('规划期间收到急停，已取消本次结果')
                return
            if self.state == STATE_SEGMENT:
                new_mode = kv.get("mode")
                # 用户终端按键接管（最高优先级）：行进中 mode 变为
                # STANDBY/EMERGENCY = 有人按了停车键。关 VIO、确认停车
                # （"A" 同时清除 NavSide 侧的用户接管标志，交还调度器控制权）、
                # 进入 PAUSED，localize 后继续未完成目标。
                if (
                    not self.path_aware
                    and
                    new_mode in ("STANDBY", "EMERGENCY")
                    and prev_mode in ("LOW_SPEED", "MEDIUM_SPEED")
                ):
                    self.log(
                        f"检测到终端按键接管（{prev_mode} → {new_mode}）："
                        "机器人已停，进入 ARRIVED"
                    )
                    self.send(self.nav, "A")  # 先确认停推理（清接管标志 + 清 LSTM）
                    time.sleep(0.1)
                    self.send(self.slam, "pause")  # 再关 VIO
                    self._cancel_plan()
                    self.anchor_pose = None
                    self.send(self.slam, "msg 检测到终端按键接管：机器人已停。"
                                          "静止后输入 localize，以新定位继续当前目标")
                    self.state = STATE_PAUSED
                    self.arrival_prompted = False
                    return
                if kv.get("zero_reason") == "goal_reached":
                    self.arrive()
                elif kv.get("zero_reason") == "emergency":
                    self.log("警告：NavSide 进入 EMERGENCY（有人按下急停），机器人已刹停")

    # ------------------------------------------------------------- 用户命令
    def handle_user(self, cmd: str) -> None:
        if cmd in ("quit", "q", "exit"):
            self.log("用户请求退出")
            self.done = True
            self._cancel_plan()
            return
        if cmd in ("help", "h"):
            print(
                "指令：l / localize = 到点后确认下一段；PAUSED/PLAN_FAILED 中重定位继续当前目标\n"
                "      force = 仅跳过本次成功锚定的任务点偏差校验，不能绕过 A* 失败\n"
                "      status / s = 打印当前状态\n"
                "      pause = 暂停当前段；localize 后重试（SRU OFF 时需独立遥控停车）\n"
                "      send <nav|slam> <命令> = 向子进程透传命令（调试用）\n"
                "      quit / q = 优雅退出"
            )
            return
        if cmd in ("status", "s"):
            self.log(
                f"state={self.state} 任务点={self.idx + 1}/{len(self.cfg['task_points'])} "
                f"anchor_seq={self.anchor_sequence} astar_error={self.last_plan_error} "
                f"SRU={'ON' if self.sru_enabled else 'OFF'} "
                f"nav={self.nav_state} slam={self.slam_state}"
            )
            return
        if cmd == 'pause':
            if self.state in (STATE_SEGMENT, STATE_PLANNING, STATE_WAIT_PATH_READY, STATE_WAIT_PATH_RUNNING, STATE_WAIT_POSE):
                self._pause_current('用户暂停当前段')
            else:
                self.log(f'当前状态 {self.state}，无需暂停行进；status 可查看状态')
            return
        if cmd.startswith("send "):
            parts = cmd.split(None, 2)
            if len(parts) == 3 and parts[1] in ("nav", "slam"):
                child = self.nav if parts[1] == "nav" else self.slam
                self.send(child, parts[2])
            else:
                self.log("用法：send <nav|slam> <命令>")
            return
        if cmd in ("localize", "l"):
            if self.state in (STATE_ARRIVED, STATE_WAIT_STARTUP_ANCHOR, STATE_PLAN_FAILED, STATE_PAUSED):
                # 启动阶段首帧后 cuvslam 会自动锚定；在它失败之前无需手动定位。
                if self.state == STATE_WAIT_STARTUP_ANCHOR and not self.startup_anchor_failed:
                    self.log("启动自动锚定尚未失败，无需手动 localize"
                             "（首帧后 cuvslam 会自动锚定，失败后会自动回到此状态）。")
                    return
                # 直接转发（进程内 localize，不重启相机）。「定位=重启」方案
                # 实测反被拖垮：重启循环反复开关相机，Orbbec USB 栈失稳
                # （标定取帧失败 → 相机从总线消失）；且绑定层 GIL 崩溃并非
                # 「进程内第二次」独有（新进程首次锚定也崩过）。若真崩了，
                # 由带延迟的自动重启兜底（resume_after_anchor 续接任务点）。
                if self.state in (STATE_PLAN_FAILED, STATE_PAUSED):
                    self.reanchor_current = True
                if self.path_aware and self.nav_manual_token:
                    self.authorized_resume_token = self.nav_manual_token
                if self.state == STATE_PAUSED:
                    self.state = STATE_ANCHORING
                    self.anchor_startup_phase = False
                    self.anchor_t0 = time.monotonic()
                # A failed/missing reply must not leave a previous successful
                # anchor available to force while the new request is pending.
                self.anchor_pose = None
                self.send(self.slam, "localize")
            elif self.state == STATE_ANCHORING:
                self.log("锚定流程进行中，请稍候...")
            else:
                self.log(f'当前状态 {self.state}，请等待规划/重启完成；行进到点后会提示重定位。')
            return
        if cmd == "force":
            if self.state == STATE_ARRIVED:
                if self.anchor_pose is None:
                    self.log('force 被拒绝：必须先取得本次有效重定位位姿，不能用旧定位或 VIO 代替')
                else:
                    self.log("强制推进（仅跳过偏差校验，下一段仍需 A* 成功）...")
                    self.advance("force 强制推进")
            else:
                self.log("force 仅在 ARRIVED 且有本次成功重定位时有效，不能绕过规划失败。")
            return
        self.log(f"未知指令：{cmd}（help 查看帮助）")

    # ------------------------------------------------------------- 崩溃重启
    def _restart_slam(self) -> None:
        """cuvslam 意外退出后自动重启（绑定层 GIL 竞态崩溃的兜底）。

        重启后首帧在当前位置自动锚定；resume_after_anchor 保持当前目标，
        以新定位重算 A*，不调用 advance。重启实例加 --no-viz。
        """
        cfg = self.cfg
        self.slam_restarts += 1
        n = self.slam_restarts
        self._cancel_plan()
        self.anchor_pose = None
        self.reanchor_current = False
        self.resume_after_anchor = True
        self.startup_anchor_failed = False
        ts = time.strftime("%H%M%S")
        assert self.log_dir is not None
        slam_log = self.log_dir / f"slam_r{n}_{ts}.log"
        self.slam_sched_file = str(self.log_dir / f"slam_sched_r{n}_{ts}.log")
        Path(self.slam_sched_file).write_text("", encoding="utf-8")
        slam_cmd = list(cfg["cuvslam_cmd"]) + ["--ref-map", str(cfg["ref_map"]), "--scheduler"]
        if "--no-viz" not in slam_cmd:
            slam_cmd.append("--no-viz")  # 重启实例不弹 Rerun 窗口（避免与残留 viewer 冲突）
        slam_env = dict(cfg["env"].get("cuvslam", {}))
        slam_env["SLAM_SCHED_FILE"] = self.slam_sched_file
        self.slam = ChildProc("SLAM", slam_cmd, cfg["cuvslam_repo"], slam_env)
        self.slam.start()
        threading.Thread(
            target=reader_loop, args=(self.slam, self.events, "slam", slam_log, n), daemon=True
        ).start()
        if self._slam_file_stop is not None:
            self._slam_file_stop.set()  # 旧事件 tailer 退出
        stop = threading.Event()
        self._slam_file_stop = stop
        threading.Thread(
            target=sched_file_loop,
            args=(self.slam_sched_file, self.events, stop, "slam", n),
            daemon=True,
        ).start()
        if open_viewer_window(f"cuvslam 重启#{n} (SLAM 位姿)", slam_log):
            self.log(f"SLAM 已重启（#{n}）：日志 {slam_log.name}")
        else:
            self.log(f"SLAM 已重启（#{n}）：日志仅落盘 {slam_log.name}")
        self.resume_after_anchor = True
        self.state = STATE_WAIT_STARTUP_ANCHOR
        self.startup_anchor_t0 = time.monotonic()
        self.log("等待重启后的启动自动锚定（在机器人当前位置锚定）...")

    # --------------------------------------------------------------- 关停
    def shutdown(self, code: int) -> None:
        if self.stop_event.is_set():
            return
        self.shutting_down = True  # 关停期间 SLAM 退出不再触发自动重启
        self._cancel_plan()
        self.stop_event.set()
        self.exit_code = code
        self.log("正在关闭两端进程...")
        if self.slam is not None and self.slam.alive():
            self.slam.send("q")  # cuvslam 优雅退出（KeyboardInterrupt → finally）
        # NavSide：经指令文件发 quit → 主循环优雅退出（finally 发零）→ 状态
        # 文件写 process=exit。等待期间继续消费事件队列以收到该确认。
        if self.nav is not None and not self.nav_exited:
            self.nav_send("quit")
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            while True:
                try:
                    tag, kv = self.events.get_nowait()
                except queue.Empty:
                    break
                if tag == "nav" and "__exit__" in kv:
                    self.nav_exited = True
            if not (self.slam is not None and self.slam.alive()) and (not self.sru_enabled or self.nav_exited):
                break
            time.sleep(0.2)
        if self.slam is not None and self.slam.alive():
            self.log("SLAM 未在时限内退出，强制终止")
            try:
                self.slam.proc.terminate()
            except OSError:
                pass
            try:
                self.slam.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                try:
                    self.slam.proc.kill()
                    self.slam.proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.log('SLAM 强制终止后仍未确认进程退出')
                except OSError:
                    pass
        # 输入线程可先停止；状态读取必须活到资源清理确认之后。
        if self._nav_file_stop is not None:
            self._nav_file_stop.set()
        if self._slam_file_stop is not None:
            self._slam_file_stop.set()
        if self.sru_enabled and not self.nav_exited:
            self.log("未收到 NavSide 退出确认（其终端窗口应显示 '=== NavSide 已退出 ==='）")
        if ((self.slam is not None and self.slam.alive())
                or (self.sru_enabled and self.nav is not None and not self.nav_exited)):
            self.exit_code = code or 1
            self.log(f'调度器退出（exit={self.exit_code}），子进程尚未全部确认退出；'
                     '请检查原终端和相机占用后再启动下一轮。')
        else:
            self.log(f"全部退出（exit={code}）")

    # ------------------------------------------------- NavSide 交互终端窗口
    def _spawn_nav_terminal(self, nav_cmd: list, repo: str, extra_env: dict) -> None:
        """在独立 gnome-terminal 里运行 NavSide 真身：原始面板 + 键盘交互。

        stdin/stdout 归该终端所有（用户按键 A/S/D/F/G 最高优先级、随时可停车）；
        调度器与 NavSide 的机器通信走两个文件通道（NAVSIDE_CMD_FILE /
        NAVSIDE_SCHED_FILE）。窗口在 NavSide 退出后保持打开，方便看收尾输出。
        """
        if not self.sru_enabled:
            return
        if not os.environ.get("DISPLAY"):
            gdm_xauth = Path(f"/run/user/{os.getuid()}/gdm/Xauthority")
            if gdm_xauth.is_file():
                os.environ["XAUTHORITY"] = str(gdm_xauth)
            os.environ["DISPLAY"] = ":0"
        exports = [
            f"export NAVSIDE_CMD_FILE={shlex.quote(self.nav_cmd_file)}",
            f"export NAVSIDE_SCHED_FILE={shlex.quote(self.nav_sched_file)}",
            f"export NAVSIDE_SESSION_ID={shlex.quote(self.session_id)}",
            "export PYTHONUNBUFFERED=1",
        ]
        # gnome-terminal may reuse a server with an older environment.
        for key in ('NAV_POSE_DIAG_DIR', 'NAV_POSE_DIAG_MODULE'):
            exports.append(f"export {key}={shlex.quote(os.environ.get(key, ''))}")
        for k, v in extra_env.items():
            if v:
                exports.append(f"export {k}={shlex.quote(str(v))}")
        script = (
            "cd " + shlex.quote(repo)
            + " && " + " && ".join(exports)
            + " && " + " ".join(shlex.quote(a) for a in nav_cmd)
            + "; echo; echo '=== NavSide 已退出 ==='; read -p '按回车关闭窗口' x"
        )
        term = shutil.which("gnome-terminal") or shutil.which("xterm")
        if term is None:
            self.log("ERROR: 找不到图形终端（gnome-terminal/xterm），无法启动 NavSide 交互窗口")
            self.exit_code = 1
            self.done = True
            return
        if term.endswith("gnome-terminal"):
            cmd = [term, "--title", "NavSide (SRU 推理)", "--", "bash", "-c", script]
        else:
            cmd = [term, "-T", "NavSide (SRU 推理)", "-e", "bash", "-c", script]
        try:
            subprocess.Popen(
                cmd, start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.log("已打开 NavSide 交互终端窗口（面板按键最高优先级，可随时停车）")
        except OSError as exc:
            self.log(f"ERROR: NavSide 终端窗口启动失败：{exc}")
            self.exit_code = 1
            self.done = True

    def prepare_planning(self):
        """Pick/validate goals before launching any camera or navigation process."""
        if getattr(self.args, 'check_only', False):
            self.log('检查模式：只校验已保存的任务点，不打开地图窗口。'
                     '重新选点请运行 ./run_task_nav.sh --pick-only')
        self.planner = SegmentAStarPlanner(self.cfg)
        if self.path_aware and (not self.planner.include_start or not self.planner.include_goal):
            raise ValueError('PathAware 配置必须同时包含起点和终点')
        if self.sru_enabled:
            nav_command = self.cfg['navside_cmd']
            if '--config' not in nav_command:
                raise ValueError('请显式指定 NavSide --config')
            nav_config = Path(nav_command[nav_command.index('--config')+1])
            if not nav_config.is_absolute():
                nav_config = Path(self.cfg['navside_repo'])/nav_config
            nav_settings = yaml.safe_load(nav_config.read_text(encoding='utf-8'))
            mode = nav_settings.get('policy', {}).get('mode', 'legacy')
            if (mode == 'path_aware') != self.path_aware:
                raise ValueError('调度器 path_aware.enabled 与 NavSide policy.mode 不一致')
            if self.path_aware:
                if float(nav_settings['control']['dry_run_hz']) != 5.0:
                    raise ValueError('当前 PathAware 部署约定为 5 Hz')
                if float(self.cfg['robot_height_z']) != .695:
                    raise ValueError('当前真实机器人和目标高度必须为 0.695')
                self.log('PathAware：15 点含起终点，模型路径高度 0.5，mix 编码器，5 Hz')
        else:
            self.log('SRU OFF：仅运行调度器、SLAM 和 A*；不加载下游配置/模型，不启动深度相机或运动控制。')
        supplied_file = getattr(self.args, 'task_points_file', None)
        path = Path(supplied_file or self.cfg.get('task_points_file', 'config/task_points_selected.json')).expanduser()
        if not path.is_absolute():
            path = Path(self.cfg['navside_repo'])/path
        self.log(f"A* 地图：{self.planner.pm.directory}；定位数据库指纹已匹配")
        if self.planner.center_enabled:
            center = self.planner.center_settings
            self.log(f'A* 居中偏好 ON：衰减距离 {center["decay_length_m"]:g}m，权重 {center["cost_weight"]:g}，'
                     f'简化容差 {center["simplification_tolerance_m"]:g}m，净空损失上限 {center["max_clearance_loss_m"]:g}m')
        if supplied_file or getattr(self.args, 'check_only', False):
            if not path.is_file():
                raise ValueError(f'尚无地图选点文件：{path}；请先运行 ./run_task_nav.sh --pick-only')
            points = load_task_points(self.planner, path)
        else:
            initial = None
            if path.is_file():
                try:
                    initial = load_task_points(self.planner, path)
                except (ValueError, OSError) as exc:
                    self.log(f'旧选点未载入，请重新在地图选点：{exc}')
            from task_point_picker import pick_task_points
            points = pick_task_points(self.planner, path, initial)
            if points is None:
                self.log('已取消选点，未启动相机或导航')
                return False
        self.cfg['task_points'] = [tuple(p) for p in points]
        self.log(f'已载入地图选取的 {len(points)} 个目标：{path}')
        for i, (x, y, z) in enumerate(points, 1):
            self.log(f'目标 {i}: X={x:.6f} Y={y:.6f} Z={z:.6f}')
        self.log('真实起点由启动/每段重定位提供；'+('SRU OFF，15 点只显示和保存' if not self.sru_enabled
                 else '15 点通过确认协议交给 PathAware SRU'
                 if self.path_aware else '15 点只打印保存，SRU 接收原单目标 goal'))
        return True

    # --------------------------------------------------------------- 主循环
    def run(self) -> int:
        try:
            if not self.prepare_planning():
                return 0
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            self.log(f'地图/选点准备失败：{exc}；未启动相机或导航')
            return 2
        if getattr(self.args, 'pick_only', False) or getattr(self.args, 'check_only', False):
            self.log(('地图和已保存任务点检查通过' if getattr(self.args, 'check_only', False)
                      else '地图选点已保存')+'，未启动相机或导航')
            return 0
        cfg = self.cfg
        points = cfg["task_points"]

        # --- 子进程命令 ---
        slam_cmd = list(cfg["cuvslam_cmd"]) + ["--ref-map", str(cfg["ref_map"]), "--scheduler"]
        nav_cmd = []
        if self.sru_enabled:
            nav_python = Path(cfg["navside_repo"]) / ".venv_navside" / "bin" / "python"
            if not nav_python.is_file():
                nav_python = Path(sys.executable)
            nav_cmd = [str(nav_python), *cfg["navside_cmd"]]

        self.log("任务导航调度器启动")
        self.log(f"  参考地图: {cfg['ref_map']}  任务点数: {len(points)}")
        # 双相机防呆：两端串号为空或相同都会导致相机互相抢占（uvc_open failed -6）。
        cuv_serial = cfg["env"].get("cuvslam", {}).get("CUVSLAM_CAMERA_SERIAL", "")
        nav_serial = cfg["env"].get("navside", {}).get("NAVSIDE_CAMERA_SERIAL", "")
        self.log(f"  SRU={'ON' if self.sru_enabled else 'OFF'}；相机(SLAM)={cuv_serial or '(第一台)'}")
        if self.sru_enabled:
            self.log(f"  相机(NavSide深度)={nav_serial or '(第一台)'}")
        if self.sru_enabled and (not cuv_serial or not nav_serial):
            self.log("警告：未配置相机串号的一端会打开枚举到的第一台设备，"
                     "双相机时可能与另一端抢占同一台相机")
        elif self.sru_enabled and cuv_serial == nav_serial:
            self.log(f"警告：两端相机串号相同（{cuv_serial}），会互相抢占设备")
        self.log(f"  SLAM: {' '.join(slam_cmd)}")
        if self.sru_enabled:
            self.log(f"  NAV : {' '.join(nav_cmd)}   （独立交互终端窗口，按键最高优先级）")
        else:
            self.log('  SRU OFF：由独立遥控负责移动和停车，调度器只根据真实 VIO 判断到达。')
        self.log(f"  到达判定: VIO 位姿距任务点 ≤ {cfg.get('arrive_tolerance_m', 0.5)}m "
                 f"持续 {cfg.get('arrive_confirm_s', 1.0)}s"
                 f"（--arrive-tolerance 可单独调整）")

        # SLAM 原始打印 → 日志文件 + 观察窗口；NavSide 跑独立交互终端窗口，
        # 控制走文件通道（NAVSIDE_CMD_FILE / NAVSIDE_SCHED_FILE）。
        # 调度器终端只显示 [TASK] 信息。
        log_dir = Path(__file__).resolve().parents[1] / "logs" / "task_nav"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        ts = time.strftime("%Y%m%d_%H%M%S")
        output = Path(cfg.get('astar', {}).get('output_directory', 'logs/task_nav/astar'))
        if not output.is_absolute():
            output = Path(cfg['navside_repo'])/output
        self.plan_output_dir = output/('run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
        self.plan_output_dir.mkdir(parents=True, exist_ok=False)
        (self.plan_output_dir/'session.json').write_text(json.dumps(dict(
            map_id=self.planner.pm.meta['map_id'], map_version=self.planner.pm.meta['map_version'],
            grid_sha256=self.planner.pm.meta['grid_sha256'],
            source_db_sha256=self.planner.pm.meta['source_db_sha256'], task_points_xyz=points,
            reference_y_m=REFERENCE_Y_M, model_path_z_m=.5 if self.path_aware else None,
            session_id=self.session_id, sru_enabled=self.sru_enabled,
            sru_reference_input_connected=self.sru_enabled and self.path_aware), ensure_ascii=False, indent=2)+'\n')
        self.log(f'各段 A* 参考点及错误日志：{self.plan_output_dir}')
        slam_log = log_dir / f"slam_{ts}.log"
        self.slam_sched_file = str(log_dir / f"slam_sched_{ts}.log")
        # 每次运行截断重建：两端 tailer 从头读，不会重放陈旧指令/状态。
        if self.sru_enabled:
            self.nav_cmd_file = str(log_dir / f"nav_cmd_{ts}.log")
            self.nav_sched_file = str(log_dir / f"nav_sched_{ts}.log")
            Path(self.nav_cmd_file).write_text("", encoding="utf-8")
            Path(self.nav_sched_file).write_text("", encoding="utf-8")
        Path(self.slam_sched_file).write_text("", encoding="utf-8")
        self.log(f"  SLAM 日志: {slam_log}")
        self.log(f"  SLAM 事件通道: {self.slam_sched_file}（[SCHED] 双写留痕）")
        if self.sru_enabled:
            self.log(f"  NAV  指令/状态通道: {self.nav_cmd_file} / {self.nav_sched_file}")

        # cuvslam 侧注入 SLAM_SCHED_FILE：[SCHED] 事件双写（管道 + 文件），
        # 调度器从文件通道读事件——管道曾出现 anchor=ok 行丢失的问题。
        slam_env = dict(cfg["env"].get("cuvslam", {}))
        slam_env["SLAM_SCHED_FILE"] = self.slam_sched_file
        self.slam = ChildProc("SLAM", slam_cmd, cfg["cuvslam_repo"], slam_env)
        self.slam.start()
        if self.sru_enabled:
            self.nav = NavSideChild(self.nav_cmd_file, self.nav_sched_file)
            self._spawn_nav_terminal(nav_cmd, cfg["navside_repo"], cfg["env"].get("navside", {}))
        if self.done:  # NavSide 终端窗口启动失败（无窗口可退，直接收尾 SLAM）
            self.nav_exited = True
            self.shutdown(self.exit_code)
            return self.exit_code

        threading.Thread(
            target=reader_loop, args=(self.slam, self.events, "slam", slam_log), daemon=True
        ).start()
        stop = threading.Event()
        self._slam_file_stop = stop
        threading.Thread(
            target=sched_file_loop,
            args=(self.slam_sched_file, self.events, stop, "slam"),
            daemon=True,
        ).start()
        if self.sru_enabled:
            self._nav_file_stop = threading.Event()
            threading.Thread(
                target=sched_file_loop,
                args=(self.nav_sched_file, self.events, self._nav_file_stop, "nav"),
                daemon=True,
            ).start()

        # SLAM 原始打印的观察窗口（与手动开终端跑一致）；NavSide 有自己的
        # 交互终端窗口。无显示器（SSH 无 X 且无本地 GDM 会话）时仅落盘日志。
        if open_viewer_window("cuvslam (SLAM 位姿)", slam_log):
            self.log("已打开 cuvSlam 观察窗口")
        else:
            self.log(f"警告：无法打开图形终端，SLAM 打印仅记录在 {slam_log}")
        threading.Thread(
            target=user_input_loop, args=(self.user_queue, self.stop_event), daemon=True
        ).start()

        def _sig_handler(signum, frame):
            self.log(f"收到信号 {signum}，准备优雅退出")
            self.done = True

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        boot_t0 = time.monotonic()
        slam_ready = False
        nav_ready = not self.sru_enabled
        self.startup_anchor_t0 = 0.0

        while not self.done:
            now = time.monotonic()

            # ---- 子进程事件 ----
            while True:
                try:
                    tag, kv = self.events.get_nowait()
                except queue.Empty:
                    break
                if "__ready__" in kv:
                    if tag == 'slam' and kv.get('__generation__', str(self.slam_restarts)) == str(self.slam_restarts):
                        slam_ready = True
                        self.log("SLAM 已就绪")
                    self.handle_sched(tag, kv)
                    continue
                # NAV 就绪 = 收到第一条带 mode 字段的 [SCHED]（主循环已开始跑）
                if tag == "nav" and "mode" in kv and not nav_ready:
                    nav_ready = True
                    self.log("NAV 已就绪（主循环运行中）")
                self.handle_sched(tag, kv)

            # ---- 就绪检测 ----
            # SLAM 就绪即可开始等待锚定：cuvslam 首帧自动锚定不依赖 NavSide，
            # 而 NavSide 启动慢（模型加载 + load_task 可达 30s）——若等它就绪
            # 才离开 BOOT，锚定的 busy/ok 事件会在 BOOT 中被忽略（状态机不认），
            # 之后永远停在「等待启动锚定」。NavSide 的命令走文件通道，先写
            # 后读不会丢（NavSide 启动后会从头读）。
            if self.state == STATE_BOOT:
                if now - boot_t0 > cfg["boot_ready_timeout_s"]:
                    self.log("ERROR: SLAM 启动就绪超时")
                    self.exit_code = 1
                    break
                if slam_ready:
                    self.state = STATE_WAIT_STARTUP_ANCHOR
                    self.startup_anchor_t0 = now
                    self.log("等待 cuvslam 启动自动锚定（首帧 localize）..."
                             + ("" if nav_ready else "（NavSide 仍在启动，命令会在其文件通道排队）"))
            # NavSide 就绪超时兜底（NavSide 窗口没开/崩溃时能退出而不是干等）。
            if not nav_ready and now - boot_t0 > cfg["boot_ready_timeout_s"]:
                self.log("ERROR: NavSide 启动就绪超时")
                self.exit_code = 1
                break

            # ---- 超时检查 ----
            if self.state == STATE_WAIT_STARTUP_ANCHOR:
                if now - self.startup_anchor_t0 > cfg["startup_anchor_timeout_s"]:
                    self.log("ERROR: 启动锚定超时")
                    self.exit_code = 1
                    break
            if self.state == STATE_ANCHORING:
                if now - self.anchor_t0 > cfg["anchor_timeout_s"]:
                    self.on_anchor_fail("锚定超时")
            if self.state == STATE_SEGMENT:
                if now - self.segment_t0 > cfg["segment_arrive_timeout_s"]:
                    self.log(
                        f"警告：第 {self.idx + 1} 段已走 {int(now - self.segment_t0)}s 仍未到达"
                        f"（超时 {cfg['segment_arrive_timeout_s']}s），继续等待"
                    )
                    self.segment_t0 = now  # 重计，避免每 tick 刷屏
            if self.state == STATE_ARRIVED and not self.arrival_prompted:
                self.arrival_prompted = True
                self.log(
                    f"已到任务点 {self.idx + 1}/{len(points)}：保持静止，输入 localize 重定位；"
                    "以该次实际位置规划到下一目标，规划成功后继续。"
                )

            # ---- cuvslam 崩溃自动重启（延迟：等相机 USB 重新枚举）----
            if self.slam_restart_pending and time.monotonic() >= self.slam_restart_at:
                self.slam_restart_pending = False
                self._restart_slam()

            # ---- 用户命令 ----
            while True:
                try:
                    cmd = self.user_queue.get_nowait()
                except queue.Empty:
                    break
                self.handle_user(cmd)

            # Consume user stop/quit and crash events before accepting an A*
            # result, so an expired worker cannot start a segment afterwards.
            self.poll_planning()
            self.poll_path_timeout()

            time.sleep(0.05)

        self.shutdown(self.exit_code)
        return self.exit_code


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError('调度配置必须是字典')
    sru_settings = cfg.get('sru', {})
    if not isinstance(sru_settings, dict) or type(sru_settings.get('enabled', True)) is not bool:
        raise ValueError('sru.enabled 必须是 true/false')
    path_settings = cfg.get('path_aware', {})
    if not isinstance(path_settings, dict) or type(path_settings.get('enabled', False)) is not bool:
        raise ValueError('path_aware.enabled 必须是 true/false')
    for key in ('ready_timeout_s', 'running_timeout_s'):
        value = float(path_settings.get(key, 60 if key == 'ready_timeout_s' else 10))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(key+' 必须是有限正数')
    # Old YAML task_points are intentionally not used: goals now come only from
    # the reviewed map picker or a map-bound file saved by that picker.
    cfg['task_points'] = []
    return cfg


def main() -> int:
    default_config = str(Path(__file__).resolve().parents[1] / "config" / "task_nav.yaml")
    parser = argparse.ArgumentParser(description="任务导航调度器：分段导航编排（cuVSLAM + NavSide）")
    parser.add_argument("--config", default=default_config, help=f"调度配置 YAML（默认 {default_config}）")
    parser.add_argument('--pick-only', action='store_true', help='只在地图选取并保存目标，不启动相机/导航')
    parser.add_argument('--task-points-file', help='复用地图选点保存的 JSON，跳过选点窗口')
    parser.add_argument('--check-only', action='store_true', help='只检查已保存选点和地图身份，不启动窗口/相机/导航')
    sru_switch = parser.add_mutually_exclusive_group()
    sru_switch.add_argument('--no-sru', dest='sru_enabled', action='store_false',
                            help='本次关闭 SRU：仅 SLAM + 调度器 + A*，移动/停车由独立遥控负责')
    sru_switch.add_argument('--sru', dest='sru_enabled', action='store_true',
                            help='本次开启 SRU 自动导航，覆盖 YAML 的 sru.enabled')
    parser.set_defaults(sru_enabled=None)
    parser.add_argument(
        "--arrive-tolerance", type=float, default=None,
        help="单独调整到达判定容差 arrive_tolerance_m（米，VIO 位姿距任务点 XZ 距离），"
             "不改 YAML 只改本次运行",
    )
    args = parser.parse_args()
    if args.pick_only and (args.task_points_file or args.check_only):
        parser.error('--pick-only 不与 --task-points-file/--check-only 同时使用')

    cfg = load_config(args.config)
    if args.sru_enabled is not None:
        cfg.setdefault('sru', {})['enabled'] = args.sru_enabled
    if args.arrive_tolerance is not None:
        if not math.isfinite(args.arrive_tolerance) or args.arrive_tolerance <= 0:
            parser.error('--arrive-tolerance 必须为有限正数')
        cfg["arrive_tolerance_m"] = args.arrive_tolerance

    return TaskNavScheduler(cfg, args).run()


if __name__ == "__main__":
    sys.exit(main())
