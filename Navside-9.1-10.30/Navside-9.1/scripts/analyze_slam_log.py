#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_slam_log.py —— 分析 cuvslam 位姿流卡顿与掉帧（无第三方依赖）

用法：
    python3 scripts/analyze_slam_log.py                      # 最新的 slam_sched 日志
    python3 scripts/analyze_slam_log.py <slam_sched.log>     # 指定文件
    python3 scripts/analyze_slam_log.py --all                # 所有 slam_sched 日志

判读方法：
    - 位姿流每 0.1s 一行 t=<墙钟秒>；相邻间隔 >0.5s 记为「空洞」。
    - 空洞按事件行分阶段：anchor=busy→ok/fail 之间 = 锚定搜索（正常）；
      paused=1 之后 = 到达停车（正常）；paused=0 且无锚定 = 行进段 ——
      这里的空洞才是异常卡顿（主线程阻塞）。
    - 与同时间窗 slam_*.log 的 Camera/IMU drop 警告交叉验证：
      空洞秒数 ≈ 掉帧秒数 → 主线程持 GIL 饿死采集线程的实锤。
"""
import glob
import os
import re
import sys
from datetime import datetime

GAP_S = 0.5  # 位姿流相邻时间戳间隔超过该值记为空洞（正常 ~0.1s）


def analyze(sched_path: str) -> None:
    # 事件行没有 t=，按文件顺序交错推进阶段（sched 文件即时间线）。
    phase = "boot"           # boot / anchor / walk / paused
    poses = []               # (t, phase)
    events = []
    for line in open(sched_path, encoding="utf-8"):
        line = line.strip()
        m = re.search(r"t=([0-9.]+)", line)
        if "anchor=busy" in line:
            phase = "anchor"
            events.append("anchor=busy")
        elif "anchor=ok" in line or "anchor=fail" in line:
            phase = "walk"   # 锚定结束，等 resume（resume 后即行进）
            events.append("anchor=ok/fail")
        elif "paused=1" in line:
            phase = "paused"
            events.append("paused=1")
        elif "paused=0" in line:
            phase = "walk"
            events.append("paused=0")
        if m and "pose=" in line:
            poses.append((float(m.group(1)), phase))

    print("=" * 70)
    print("文件:", sched_path)
    print("位姿行数:", len(poses), " 事件:", events)

    print("\n-- 位姿流空洞 (>%.1fs) --" % GAP_S)
    gaps = [(i, poses[i][0] - poses[i - 1][0], poses[i - 1][1])
            for i in range(1, len(poses))
            if poses[i][0] - poses[i - 1][0] > GAP_S]
    if not gaps:
        print("   无空洞，位姿流稳定 ~10Hz")
    walk_bad = 0
    for i, dt, ph in gaps:
        ts = datetime.fromtimestamp(poses[i - 1][0]).strftime("%H:%M:%S")
        flag = ""
        if ph == "walk":
            walk_bad += 1
            flag = "  <== 行进段异常卡顿"
        print(f"   {ts}  {dt:6.1f}s  阶段={ph}{flag}")
    if walk_bad:
        print(f"   行进段异常卡顿共 {walk_bad} 次 —— 主线程被 SLAM 内联工作阻塞")


def cross_check(sched_path: str) -> None:
    """与同时间窗 slam_*.log 的掉帧警告交叉验证。"""
    d = os.path.dirname(sched_path) or "."
    name = os.path.basename(sched_path).replace("slam_sched", "slam")
    slam_log = os.path.join(d, name)
    if not os.path.isfile(slam_log):
        print("\n（未找到对应 slam 日志:", slam_log, "）")
        return
    print("\n-- 对应 slam 日志的掉帧警告（交叉验证） --")
    found = False
    for line in open(slam_log, encoding="utf-8", errors="replace"):
        if "drop" in line.lower() or "timestamp gap" in line.lower():
            print("  ", line.strip())
            found = True
    if not found:
        print("   无掉帧警告")


def main() -> int:
    args = sys.argv[1:]
    all_paths = sorted(glob.glob("logs/task_nav/slam_sched_*.log"),
                       key=os.path.getmtime)
    if "--all" in args:
        paths = all_paths
    elif args and not args[0].startswith("-"):
        paths = [args[0]]
    else:
        paths = all_paths[-1:]  # 最新的运行
    if not paths:
        print("未找到 slam_sched 日志")
        return 1
    for p in paths:
        analyze(p)
        cross_check(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
