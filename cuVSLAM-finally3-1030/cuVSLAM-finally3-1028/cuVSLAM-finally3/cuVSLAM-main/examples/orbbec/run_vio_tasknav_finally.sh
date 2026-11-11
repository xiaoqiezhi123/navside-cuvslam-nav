#!/usr/bin/env bash
# finally3 版任务导航启动脚本：用隔离 venv 跑 run_vio_tasknav_finally.py，不碰系统环境。
# 由 task_nav_scheduler.py 以 `--ref-map <map> --scheduler --no-viz` 调用；也可手动运行：
#   ./run_vio_tasknav_finally.sh --ref-map 10261 --no-viz
# 终端指令（机器人到达任务点静止后输入）：
#   localize / l   用参考地图锚定全局位姿（作为下一段子任务起点）
#   pause / resume / quit
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
# pyorbbecsdk 的 .so 带无效 RUNPATH，靠 LD_LIBRARY_PATH 定位 venv 里的 libOrbbecSDK.so.2
VENV="${CUVSLAM_VENV:-/home/amov/cuvslam_migrate/venv}"
SP_DIR="$(echo "$VENV"/lib/python*/site-packages)"
export LD_LIBRARY_PATH="$SP_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Rerun Viewer 可执行文件（rerun）在 venv/bin 里；rr.spawn() 从 PATH 里找，必须把 venv/bin 加进 PATH。
export PATH="$VENV/bin:$PATH"
# Rerun Viewer 需要显示器。SSH 无 X 转发时 DISPLAY 为空，这里指到本机物理桌面 :0（GDM 会话）。
if [ -z "${DISPLAY:-}" ]; then
  export DISPLAY=:0
  if [ -f "/run/user/$(id -u)/gdm/Xauthority" ]; then
    export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
  fi
fi
exec "$VENV/bin/python" run_vio_tasknav_finally.py "$@"
