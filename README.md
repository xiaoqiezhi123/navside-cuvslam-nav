# navside-cuvslam-nav：cuVSLAM + A* + NavSide 分段导航

小推车机器人分段导航系统。相机 B（双目 IR）提供 cuVSLAM 位姿，相机 A（深度）提供障碍感知；A* 在 2D 规划地图上生成每段 15 个参考点，NavSide 依参考点做局部避障并输出速度命令。

## 本次更新内容

1. **新版本 cuVSLAM** —— 实机测试失败率降低
2. **新 A* 版本**
3. **2D 地图标注更新**
4. **NavSide 速度输出平滑加速** —— 梯形速度输出
5. **最好效果** —— 实现从工厂门口导航到机加工中间

## 目录

```text
navside-cuvslam-nav/
├── README.md
├── .gitignore
├── 指令.txt                              # 启动 / 工作流程说明
├── Navside-9.1-10.30/
│   └── Navside-9.1/                      # NavSide 导航侧
│       ├── navside/                      # 深度 / 位姿 / 桥接 / 状态核心模块
│       ├── scripts/                      # 分段调度器 + A* 桥接(task_nav_astar.py)
│       ├── config/                       # task_nav_finally3.yaml 等（已脱敏）
│       ├── perception/                   # 深度感知（Orbbec / RealSense / ZED）
│       ├── tests/  tools/  docs/         # 测试 / 工具 / 文档
│       ├── run_task_nav.sh               # 启动入口
│       └── *.md                          # 架构 / 排查 / 实测文档
└── cuVSLAM-finally3-1030/cuVSLAM-finally3-1028/cuVSLAM-finally3/cuVSLAM-main/
    ├── examples/orbbec/                  # cuVSLAM 调用脚本 + map2d A* 规划
    │   ├── run_vio_*.py  run_*.py        # VIO / 重定位 / 建图 / 位姿输出
    │   ├── map2d_*.py  plan_map2d.py     # 2D 地图构建 + 8 邻域加权 A* + 15 参考点
    │   └── *.sh                          # 启动脚本
    └── LICENSE                           # NVIDIA Community License
```

| 模块 | 用途 | 主要入口 |
| --- | --- | --- |
| cuVSLAM 调用 | 双目惯性定位、建图、重定位、位姿输出 | [run_vio_tasknav_finally.py](cuVSLAM-finally3-1030/cuVSLAM-finally3-1028/cuVSLAM-finally3/cuVSLAM-main/examples/orbbec/run_vio_tasknav_finally.py) |
| 2D 地图 + A* | 地图构建、全局规划、15 参考点 | [map2d_builder.py](cuVSLAM-finally3-1030/cuVSLAM-finally3-1028/cuVSLAM-finally3/cuVSLAM-main/examples/orbbec/map2d_builder.py) / [map2d_astar.py](cuVSLAM-finally3-1030/cuVSLAM-finally3-1028/cuVSLAM-finally3/cuVSLAM-main/examples/orbbec/map2d_astar.py) / [task_nav_astar.py](Navside-9.1-10.30/Navside-9.1/scripts/task_nav_astar.py) |
| NavSide | 深度 → VAE/SRU → 速度命令 | [task_nav_scheduler.py](Navside-9.1-10.30/Navside-9.1/scripts/task_nav_scheduler.py) / [run_nav.py](Navside-9.1-10.30/Navside-9.1/scripts/run_nav.py) |

## 数据流

```text
相机 B（双目 IR）→ cuVSLAM 定位 ──→ A* 全局规划（每段 15 参考点）──→ NavSide（VAE/SRU 避障）──→ 速度 UDP → 机器人控制端
相机 A（深度）→ 深度 ───────────────────────────────────────────────┘
```

## 本次保留与排除

保留 cuVSLAM 调用脚本（`examples/orbbec/`，含完整 A* 部分）、NavSide 源码 / 配置 / 测试 / 文档，以及 NVIDIA Community License。

排除 NVIDIA cuVSLAM 源码（`libs/python/cmake/tools/…`）、SRU/VAE/policy 模型权重、机器人网格、地图数据库（`data.mdb`/`grid.npz`）、现场任务点坐标、日志与录制、压缩包。详见 [.gitignore](.gitignore)。

脱敏：相机序列号 → `<SLAM_CAM_SERIAL>` / `<NAVSIDE_CAM_SERIAL>`，内网 IP → `<INTERNAL_IP>`，部署前需填回真实值。

## 在新机器上使用前

1. 安装 CUDA、Orbbec SDK 与 Python 依赖，重建 cuVSLAM 与 NavSide 环境（原环境是 `/home/amov/…` 下的软链接，不能随源码迁移）。
2. 在配置中填回实际相机序列号（当前为占位符）。
3. 需要重定位时自行恢复地图；仓库不包含历史地图与 `*_last_pose.txt`。
4. 按目标平台部署机器人控制端。

## 上游来源

cuVSLAM 上游为 [NVIDIA cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM)，原始许可保留在 [LICENSE](cuVSLAM-finally3-1030/cuVSLAM-finally3-1028/cuVSLAM-finally3/cuVSLAM-main/LICENSE)。本仓库未替换各模块已有的版权与许可声明。
