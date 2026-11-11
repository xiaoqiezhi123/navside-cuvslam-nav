#!/usr/bin/env bash
# 一键任务导航：调度器同时拉起 cuVSLAM（位姿）与 NavSide（SRU 推理），
# 把一条大路径导航切成多个小段：到点停车 → 关 VIO → 手动 localize → 继续下一段。
# 用法：
#   ./run_task_nav.sh                          # 先在地图选目标，确认后启动分段导航
#   ./run_task_nav.sh --pick-only              # 只选点保存，不启动相机/导航
#   ./run_task_nav.sh --task-points-file config/task_points_selected.json  # 复用选点
#   ./run_task_nav.sh --check-only             # 只检查地图及已保存目标
#   ./run_task_nav.sh --no-sru --task-points-file config/task_points_selected.json # 仅定位/规划
#   ./run_task_nav.sh --sru --task-points-file config/task_points_selected.json    # 完整导航
#   ./run_task_nav.sh --config <path>          # 自定义调度配置
#   ./run_task_nav.sh --config config/task_nav_legacy.yaml  # 回退原单目标 SRU
#   ./run_task_nav.sh --arrive-tolerance 0.3   # 单独调整到达判定容差（米，本次运行）
# 显示布局（三个终端）：
#   1) 本终端：调度器 [TASK] 信息 + 运行中指令输入；
#   2) cuvSlam 观察窗口：SLAM 原始打印（tail logs/task_nav/slam_*.log）；
#   3) NavSide 交互终端窗口：原始模式面板 + 键盘，按键 A/S/D/F/G 最高优先级、
#      随时可停车；调度器与它经文件通道通信（指令/状态）。无显示器时降级为日志。
# 运行中终端指令（在本终端输入后回车）：
#   localize / l   到点后开始下一段；中途停车/失败后继续未完成的当前目标
#   pause         暂停当前段；SRU OFF 时须通过独立遥控停车
#   force          仅跳过成功锚定的偏差校验；不能跳过 A* 失败或缺失定位
#   status / s     打印当前状态
#   quit / q       优雅退出（两端全部收尾，NavSide 最后发零速）
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv_navside/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi
exec "$PY" "$HERE/scripts/task_nav_scheduler.py" "$@"
