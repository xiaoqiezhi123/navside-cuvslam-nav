# navside-cuvslam-nav

小推车机器人分段导航系统：**cuVSLAM 视觉定位 + A\* 全局规划 + SRU/传统局部避障**。

## 本次更新内容

1. 新版本 cuVSLAM —— 实机测试失败率降低
2. 新 A* 版本
3. 2D 地图标注更新
4. NavSide 速度输出添加平滑加速（梯形输出）
5. 最好效果：实现从工厂门口导航到机加工中间

## 架构

```
cuVSLAM（双目 IR VIO）──定位 + 参考地图 data.mdb
        │
        ▼
map2d 管道：data.mdb → map_extractor → source_map.json → 三值栅格（未知/自由/障碍）
        │
        ▼
A* 全局规划（8 邻域加权，碰撞校验）→ 每段 15 个参考点
        │
        ▼
NavSide 分段调度器（task_nav_scheduler）→ 下发参考点给局部避障（SRU / 传统算法）
```

相机分工：一台 Orbbec Gemini 336L 双目 IR 相机跑 cuVSLAM（被动双目，激光关）；另一台 Gemini 336L 深度相机做 NavSide 深度感知。

## 目录结构

```
.
├── 指令.txt                            # 启动/工作流程说明
├── Navside-9.1-10.30/Navside-9.1/      # NavSide 导航侧
│   ├── navside/                        # 深度/位姿/桥接/状态核心模块
│   ├── scripts/                        # 分段调度器 + A* 桥接(task_nav_astar.py)
│   ├── config/                         # task_nav_finally3.yaml 等配置（已脱敏）
│   ├── perception/                     # 深度感知
│   ├── tests/  tools/  docs/           # 测试 / 工具 / 文档
│   └── *.md                            # 架构/排查/实测文档
└── cuVSLAM-finally3-1030/.../examples/orbbec/
    ├── run_vio_*.py / run_*.py         # cuVSLAM 调用脚本（VIO/重定位/建图）
    ├── map2d_*.py / plan_map2d.py      # 2D 地图构建 + A* 规划（完整 A* 部分）
    ├── pose_transform.py               # 坐标系变换（OpenCV → Z-up）
    └── *.sh                            # 启动脚本
```

## 未上传（有意排除）

| 内容 | 原因 |
|------|------|
| `asset/models/*.onnx`（SRU/VAE/policy 权重） | 模型参数隐藏 |
| `asset/robot/`（机器人网格） | 二进制大文件，非源码 |
| NVIDIA cuVSLAM 源码（libs/python/cmake/tools/…） | 仅上传「调用部分」，源码不上传 |
| `data.mdb` / `grid.npz` / 地图包（10261 等） | 地图数据，含现场环境信息 |
| `*.rrd` 追踪日志 / `logs/` / `Log/` | 运行时日志 |
| `*.zip` / `*.tar.gz` 归档 | 大文件 |

## 授权

`examples/orbbec/` 下的 cuVSLAM 调用脚本含 NVIDIA Community License 头，完整许可见
[cuVSLAM-finally3-1030/cuVSLAM-finally3-1028/cuVSLAM-finally3/cuVSLAM-main/LICENSE](cuVSLAM-finally3-1030/cuVSLAM-finally3-1028/cuVSLAM-finally3/cuVSLAM-main/LICENSE)。

## 注意

配置文件中的相机序列号与内网 IP 已脱敏为 `<SLAM_CAM_SERIAL>` / `<NAVSIDE_CAM_SERIAL>` / `<INTERNAL_IP>` 占位符，部署前需填回真实值。
