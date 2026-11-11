# 分段导航 A* + PathAware SRU：快速使用

明天现场按[测试步骤](明日实测步骤与排查_20260930.md)执行；[全部快速指令](今日产出_全部快速指令_20260929.md)汇总地图制作到导航回退的命令；[架构与上下文交接](现场测试与上下文交接_20260929.md)保存已确认需求及恢复对话指令。

当前默认配置已将 A* 的 **15 个参考点接入新版 PathAware SRU**：包含起点和终点，模型路径高度 Z=0.5，使用原 Orbbec mix 编码器，策略频率 5 Hz。每段一次交付目标与整条路径；15 点不是 15 个停车任务点。代码及离线验证完成，实地效果待测试。

2026-09-30 更新：短时位姿断流先零速等待，稳定恢复后继续原段；长断流/突变/非法数据及人工停车仍须 `localize`。见[位姿等待恢复：参数、日志与测试](位姿等待恢复_测试说明_20260930.md)。

2026-09-30 后续：已修复 VIO/IMU 定位交接，正式调度器默认启用中心偏好的 A*，简化时增加净空约束。重启后生效；先按 [修复复测步骤](VIO衔接与居中规划_修复复测_20260930.md) 做关闭 SRU 的对照。已有 5 个目标无需重新选点。

2026-10-08 更新：地图选点窗口已加入相邻任务目标的 A* 路线和 15 点预览；已有目标会自动载入并规划。打开方式仍为 `./run_task_nav.sh --pick-only`，见下方“选点时查看 A*”。

## 启动

**重新在地图选任务点用 `--pick-only`；`--check-only` 只检查已保存的任务点，不弹出地图。** 当前检查模式会明确打印这个区别。若提示“尚无地图选点文件”，先执行 `--pick-only` 保存；若提示地图指纹不匹配，要在当前地图重新选点，不能绕过身份校验。

调度器会自动启动 SLAM 并重定位，**不需要先单独运行 `run_orbbec.sh --mode localize`**。如果先单独检查过定位，须退出该程序并等相机关闭，再启动调度器。相机找不到、任务点匹配失败和位姿跳变需分别排查，见[两轮实测排查与启动顺序](两轮实测排查与启动顺序_20260930.md)。

```bash
cd /home/amov/navside_real/NavSide_log/Navside-9.1

# 只选任务点并保存，不启动相机或机器人导航
./run_task_nav.sh --pick-only

# 检查保存的任务点和地图，不启动窗口/相机/导航
./run_task_nav.sh --check-only

# 先单独测试调度器、定位和 A*（仍会启动 SLAM 相机）
./run_task_nav.sh --no-sru --task-points-file config/task_points_selected.json

# 开启 SRU，使用同一组选点运行完整导航
./run_task_nav.sh --sru --task-points-file config/task_points_selected.json
```

也可直接运行 `./run_task_nav.sh`：先弹出选点窗口，确认保存后启动定位和导航；取消/关闭选点窗口则退出，不启动子进程。

## SRU 总开关与排查

在 `config/task_nav.yaml` 设置：

```yaml
sru:
  enabled: true  # false 关闭下游；默认 true
```

`--no-sru` / `--sru` 优先于 YAML，只影响本次运行。切换前在调度器输入 `quit` 退出本轮，再重新启动；这不是运行中热切换，也不会接管另外手动启动的 NavSide 进程。`path_aware.enabled` 用于选择模型接口，**不是总开关**，关闭 SRU 时保持它为 true。

| 模式 | 运行内容 | 移动、到达与下一段 |
|---|---|---|
| `--no-sru` | 调度器 + SLAM + A*，显示并保存 15 点；不启动 NavSide、模型、下游深度相机或运动控制 | 可原地检查路径；完整分段测试由独立遥控移动/停车，真实 VIO 判定到达后停稳并输入 `localize` |
| `--sru` | 上述流程 + SRU 加载路径、确认启动、推理与运动控制 | SRU 行进，到达停车后输入 `localize` |

SRU 关闭时不等待下游 ready/running，不伪造到达，也不自动跳到下一目标。调度器终端 `status` 和 SLAM 参考点面板会显示 `SRU OFF`；`pause` 暂停当前段，**独立遥控停车**后输入 `localize`，继续当前未完成目标。规划失败、SLAM 崩溃恢复仍沿用原目标。

先用关闭模式检查定位、15 点和多段切换：这里出错，应检查地图、定位或调度器。关闭模式正常、开启后才异常，再检查路径交付确认、位姿时效、深度输入和 SRU 推理；也可能是双相机/计算资源竞争，不能仅凭开关就断定模型有问题。

## 地图选点

- 默认地图：`/home/amov/cuvslam_migrate-9.1/maps2d/viz_test_02/v6_reviewed/`。
- 在深绿色内部左键**按顺序选择目标 1、2、3……**。不要预选机器人起点，起点来自实际重定位。
- 滚轮缩放，中键拖动；Z 撤销末点、C 清空、H 全图、Enter 确认保存、Esc 取消。
- 已有有效选点会自动载入，可以修改后确认。不可通行点和连续重复点不接受。
- 保存到 `config/task_points_selected.json`，绑定地图名、源数据库和栅格指纹。换图后需重新选点。
- 不再从 YAML `task_points` 或旧 `--points` 参数读取目标。2026-09-30 已有正式选点文件；新场景或换图后重新选点。

## 选点时查看 A*

```bash
cd /home/amov/navside_real/NavSide_log/Navside-9.1
./run_task_nav.sh --pick-only
```

已有目标自动载入；也可以继续左键加目标。至少两个目标后，后台自动规划 **T1→T2、T2→T3……**，沿用当前 YAML 的中心偏好、15 点协议和碰撞检查。橙色 `T1/T2…` 是任务目标；蓝线和编号蓝点是当前预览的 15 点，灰蓝线是其他相邻路线，虚线为当前密集 A*。右侧显示参考点 X/Z、长度、最小地图净空；参考点展示 Y=0，目标文件仍为 Y=0.695。

| 按钮 / 快捷键 | 用途 |
|---|---|
| 上一段 / 下一段，← / → | 切换高亮路线及右侧 15 点坐标 |
| 目标/参考点，Tab | 切换任务目标表与参考点表 |
| 本段放大，F | 放大当前预览及周围地图 |
| 保存预览，P | 保存当前画面 PNG 和所有相邻路线 JSON 到 `logs/task_point_previews/preview_时间/`；完整路径打印在终端 |
| 撤销末点 Z / 清空 C | 撤销目标并取消对应旧预览；过期结果不会重新出现 |
| 全图 H | 回到地图全貌 |
| 确认目标 Enter | 保存任务目标并关闭窗口；`--pick-only` 到此结束，不启动导航 |
| 取消 Esc / 关闭窗口 | 不保存本次目标修改，不启动导航 |

**启动位置→T1 尚无真实定位，选点窗口不会虚构这条首段路线。** 此处只是目标到目标的几何预览；正式导航每段仍从本次实际重定位位置重新规划。预览结果不下发给 SRU，也不会写入任务目标文件。“保存预览”和“确认目标”是两个独立操作。

少于两个目标时仅显示目标；无路、点数不足等情况会标明“预览失败”，不画假直线。预览失败不自动删目标，也不阻止保存目标：正式起点来自重定位，实际能否规划仍由运行时检查决定。

预览沿用已有选点窗口，不增加相机或导航进程；开启/关闭诊断对它没有影响。使用 `--task-points-file ...` 直接启动会跳过选点窗口，要先看预览就先执行上面的 `--pick-only`。

本次当前地图 5 个目标的 4 条相邻路线均通过 15 点和端点检查：[全图效果](logs/astar_picker_preview_20261008/picker_overview.png)、[本段放大](logs/astar_picker_preview_20261008/picker_focused.png)。21 项相关测试通过，含后台取消/旧结果隔离、无路提示、导出与目标文件隔离，以及原有规划/调度回归；UI 最终排版另通过 5 项复测。使用 Agg 无界面绘制验证，未启动实际相机或机器人。

## 每段何时规划

```text
启动重定位成功
  → 起点 = 该次定位的地图 X、Z；终点 = 地图选取的目标 1
  → 偏中心 A* + 净空约束简化 + 15 点连线检查 → SLAM 终端显示/保存
  → 下发整段目标和路径 → 下游 ready
  → 恢复 VIO、申请启动 → 下游收到新位姿后 running → SRU 行进

到达目标 i → 停推理、暂停 VIO → 静止后输入 localize
  → 定位和原有任务点偏差校验通过
  → 起点 = 新定位 X、Z；终点 = 目标 i+1
  → A* + 打印/保存 → 重新交付并等待 ready/running → SRU 行进

中途人工按 A/F/G 停车 → PAUSED，不标记任务完成
  → 在调度器输入 localize → 新定位、新路径 → 继续当前未完成目标

短时位姿断流 → WAIT_POSE：零速，保留目标和路径，VIO 继续
  → 稳定恢复且通过连续性检查 → 继续同一段，不重新规划
  → 等待超时/定位突变/非法数据 → 停车，要求 localize

SLAM 崩溃 → 停车、取消旧规划 → 延迟重启并重新定位
  → 起点 = 重启后的新定位；终点 = 尚未完成的当前目标
  → 重新规划，不推进任务点
```

最终任务点重定位确认后结束，不生成不存在的下一段。

**失败处理**：起点/终点不可通行、无路、点数不足、搜索/保存失败、下游拒绝路径或确认超时，会进入 `PLAN_FAILED`，保持原目标并停车。短时位姿超时先零速等待；长断流、定位突变、非法位姿、深度失败或推理错误则保持停车。需要重定位时输入 `localize`，以新定位重试当前目标；如需更换目标，退出后重新选点。`force` 只能跳过实际到点成功定位后的偏差校验，不能绕过中途暂停、缺失定位或规划失败。

人工停车后，S/D 和旧启动命令不能恢复旧段；用调度器的 `localize` 恢复。正常行进时 S/D 仍可切换原速度模式。原模式限幅保留：S 为 vx≤0.68 m/s、|wz|≤0.8 rad/s；D 为 vx≤0.8 m/s、|wz|≤0.45 rad/s。

## 打印与产出

每次成功规划都会在 **SLAM 终端（cuvslam 位姿观察窗口）** 显示完整的 15 行，位于实时位姿上方，随面板刷新持续显示：

```text
=== A* 第 1 段 | 定位 #1 | 15 个参考点 (m) ===
[A*] 段 1 参考点 01/15: X=... Y=0.000000 Z=...
...
[A*] 段 1 参考点 15/15: X=... Y=0.000000 Z=...
```

到达、取消或重新规划时清除旧点；新规划成功后显示新的段号和定位序号。调度器终端只显示规划摘要及结果目录。显示内容同时写入 `logs/task_nav/slam_*.log`；SLAM 重启后显示在新开的观察窗口。窗口建议至少 80 列、28 行，以便同时看全参考点和位姿。已经运行的调度器需退出并重新启动，才会使用新的显示方式。

X、Z 是原 cuVSLAM 地图坐标；**15 个参考点打印及导出的 Y 固定为 0，仅作平面占位**。原始重定位的 `X,Y,Z,qx,qy,qz,qw` 另存。实际交给模型的是 `path_w=(地图Z,-地图X,0.5)`，最终目标为 `(地图Z,-地图X,0.695)`，两者高度不同。已有任务选点文件仍沿用 Y=0.695 的格式，可直接复用。

默认共 15 点：**起点 + 13 个中间点 + 终点**。保留必要转角，检查全部连线；点数不足以保留转角则暂停，不强行抽点穿墙。NavSide 每帧用实时位姿编码这组固定全局点，不删除经过的点或逐点切换目标。

结果位于 `logs/task_nav/astar/run_时间/`：

| 文件 | 内容 |
|---|---|
| `session.json` | 本次地图身份及选取的目标列表 |
| `segment_段号_anchor_定位序号_时间/route.json` | 原始锚定位姿、任务点、15 点 XY/XYZ、A* 和参考路径、检查结果 |
| 同目录 `reference_poses.csv` | 15 点 XYZ，Y=0，与终端显示一致 |
| 同目录 `reference_points.csv` | 15 点二维 XZ，兼容原独立规划工具 |
| 同目录 `reference_points.txt` | 完整 15 行打印文本 |
| 同目录 `segment_delivery.json` | 真正下发的整段数据：唯一标识、目标、15 点 NavSide 坐标 |
| 同目录 `delivery_events.jsonl` | 下游 ready/running 确认，包含对应段标识 |
| `failures.jsonl` | 失败段、当时定位和目标、失败原因 |

重启或失败重试使用同一段号、新定位序号和新交付标识，不覆盖之前结果。`route.json` 中 `sent_to_sru` 初始为 false，在收到匹配的 ready 后更新为 true；实际启动看 `delivery_events.jsonl` 中的 running。仅有路径文件不代表机器人已运行。被取消的后台结果不会触发行进。

SRU 关闭时，`session.json` 根字段和 `route.json` 的 scheduler 字段记录 `sru_enabled=false`，route 中的 `sent_to_sru` 保持 false，不生成 `segment_delivery.json` 和 `delivery_events.jsonl`。

## 配置与备份

**中心偏好开关和参数：** 在 `config/task_nav.yaml` 修改已有的 `astar.center_preference`，不要重复添加 `astar` 节点：

```yaml
center_preference:                  # 位于 astar 下
  enabled: true                     # false 恢复原地图代价和 0.30 m 简化
  decay_length_m: 0.8                # 离墙代价衰减距离，单位 m
  cost_weight: 4.0                   # 越大越偏向净空大的路线，可能更绕
  simplification_tolerance_m: 0.10   # 简化时允许偏离密集路线的距离，单位 m
  max_clearance_loss_m: 0.05         # 每条简化捷径允许损失的最小净空，单位 m
```

这会让路线偏向通道内部，不保证处处落在几何中线。地图硬障碍、膨胀半径及选点坐标不变；起终点不会吸附到中心。狭窄通道或靠墙的固定端点仍会限制净空。参数修改后重启调度器，启动摘要应显示 `A* 居中偏好 ON`；每段 `route.json → center_preference` 保存实际参数。独立地图编辑器/A* 示例未自动启用这组调度器参数。

`config/task_nav.yaml` 默认开启 PathAware，并使用 `config/nav_pathaware.yaml`。`ref_map` 是定位地图；`astar.map_package` 相对 `cuvslam_repo` 指定审核规划包；二者源数据库指纹必须一致。更换地图后重新选点，并同时更新定位地图和规划包路径。

`nav.yaml` 与 `nav_pathaware.yaml` 按 `--config` 单独加载，不合并、不继承，文件之间不会覆盖。新版通用参数参考原版；默认 vx_max=0.9、wz_max=0.5 已对齐，新增/差异项有中文注释。真机 S/D 模式仍覆盖 YAML 默认速度。保持独立配置不表示可以同时运行两个导航进程。

**参数位置（相对本工程目录；修改后重新启动）：**

| 修改内容 | 文件与字段 | 当前值/说明 |
|---|---|---|
| SRU 总开关 | `config/task_nav.yaml` → `sru.enabled` | true；命令行 `--no-sru` / `--sru` 可覆盖 |
| 定位地图 / A* 地图包 | 同上 → `ref_map`、`astar.map_package` | viz_test_02 / maps2d/viz_test_02/v6_reviewed |
| 包含起点终点 | 同上 → `astar.include_start`、`astar.include_goal` | 均为 true，总计 15 点；当前 PathAware 必须保留 |
| 任务目标 / 输出目录 | 同上 → `task_points_file`、`astar.output_directory` | 目标通过 `--pick-only` 修改；输出 logs/task_nav/astar |
| 到达判定 | 同上 → `arrive_tolerance_m`、`arrive_confirm_s` | 0.7 m、1.0 s；本次可用 `--arrive-tolerance 0.3` 改容差 |
| 重定位偏差校验 | 同上 → `anchor_verify_tolerance_m` | 5.0 m |
| 等待超时 | 同上 → `boot_ready_timeout_s` / `startup_anchor_timeout_s` / `anchor_timeout_s` / `segment_arrive_timeout_s` | 60 / 180 / 240 / 600 s；单段超时仅告警 |
| SRU 确认超时 | 同上 → `path_aware.ready_timeout_s` / `running_timeout_s` | 60 / 10 s；关闭 SRU 时不等待 |
| 相机序列号 / 启动参数 | 同上 → `env.cuvslam`、`env.navside`、`cuvslam_cmd`、`navside_cmd` | 分别指定 SLAM 与深度相机；`--no-viz` 关闭实时 Rerun |
| 策略模型 / VAE | `config/nav_pathaware.yaml` → `models.policy_path`、`models.encoder_path` | nav_policy_pathaware.onnx + vae_orbbec_mix_ep30_deploy.onnx |
| 推理频率 / 位姿过期时间 | 同上 → `control.dry_run_hz`、`policy.state_max_age_s` | 5 Hz、1.0 s；当前调度器校验固定为 5 Hz |
| 位姿自动恢复 | 同上 → `policy.pose_recovery` | 最多等待约 5 s；连续稳定 0.6 s 且至少 3 个新采样；人工停车不自动恢复 |
| 通用默认限速 | 同上 → `control.vx_max`、`control.wz_max` | 0.9 / 0.5，与 nav.yaml 一致；真机最终使用下面的 S/D 模式值 |
| SRU 近目标停车 / 行走阈值 | 同上 → `control.goal_pos_tolerance`、`control.walk_threshold` | 见实际 YAML；与调度器的 0.7 m 到达阈值分开设置 |
| 深度裁剪与过滤 | 同上 → `depth.*` | 1280×720、min 0.1 m、max 10 m、超范围值 6 m |
| 下游日志 | 同上 → `logging.*` | CSV 开启、摘要 1 Hz、verbose_sru=false |
| **真机 S/D 速度上限** | **`navside/mode.py` → `_DECISIONS`** | S：vx 0.68、wz 0.8；D：vx 0.8、wz 0.45。真机使用模式限幅，会覆盖 YAML 的 vx_max/wz_max |
| 机器人通信 / SLAM 位姿端口 | `config/nav_deploy.yaml` | cmd_remote_ip/port、pose_transport、pose_udp_host/port 等 |
| 参考点打印高度 | `scripts/task_nav_astar.py` → `REFERENCE_Y_M` | Y=0，仅打印/导出 |
| 模型路径高度 / 目标高度 | `scripts/task_nav_scheduler.py` → `_load_path_segment`；协议校验 `navside/segments.py` | Z=0.5 / 0.695；这是当前部署协议，不能只改一处。实际机器人高度在 `navside/bridge.py` |
| A* 中心偏好与简化 | `config/task_nav.yaml` → `astar.center_preference` | 默认开启；权重 4.0、衰减 0.8 m、简化 0.10 m、净空损失上限 0.05 m |
| A* 搜索上限 | 上游 `orbbec/map2d_astar.py` → `astar_search` | 最多 300000 次扩展、20 s；关闭中心偏好时沿用原地图代价，权重 2.0、简化 0.30 m |
| 地图分辨率 / 膨胀距离 | 上游规划地图包 `map.json` 查看；在地图工具中重新导出 | 当前 0.05 m、机器人半径 0.5 m、安全余量 0、soft_band 0.3 m；直接修改元数据不会重建栅格 |

15 点数量固定在调度器 A* 适配和 SRU 接口中，不是可独立调节的 YAML 参数。`config/nav.yaml`、`config/task_nav_legacy.yaml` 用于旧版回退，默认新版运行不读取 `nav.yaml`。

模型：`asset/models/nav_policy_pathaware.onnx` + 原 `vae_orbbec_mix_ep30_deploy.onnx`。深度沿用 mix 的原实际处理：1280×720 整帧缩到 64×40、最小 0.1m、超 10m 置 6m；预览框与实际输入一致。实际接收位姿超过 1 秒未更新会零速等待，读取旧缓存不会刷新这个时限，也不能充当恢复的新采样。

回退原单目标模型（保留 A* 规划/打印）：

```bash
./run_task_nav.sh --config config/task_nav_legacy.yaml --task-points-file config/task_points_selected.json
```

本次 PathAware 接入前完整备份（含已有 A* 和 SLAM 参考点面板）：

```text
/home/amov/architecture_backups/pathaware_before_20260929_094354/
  architecture.tar
  manifest.json
  恢复说明.md
```

包含上游和下游完整工程、地图、模型、配置；外部虚拟环境保留原符号链接。恢复时按备份内说明先解压到另一个空目录核对，不直接覆盖新工作。

增加 SRU 总开关前的相关文件快照：`/home/amov/architecture_backups/sru_switch_before_20260929_104439/`，含原调度器、配置、启动脚本、快速文档及 SHA256 清单。

09-29 PathAware 接入时修改调度器、A* 交付接口、mode/real/runtime/adapter，新增段协议 `navside/segments.py`、独立策略和配置；当时未修改上游定位与共享 A* 核心。09-30 已另行修复上游 tracker/IMU 交接、扩展共享 A* 并接入中心偏好，备份：`/home/amov/architecture_backups/vio_center_fix_before_20260930_121603/`。

## 验证范围

09-30 本轮完整离线测试：**上游 38 项、下游 85 项通过**，包括实际相机 worker 的 mock tracker/IMU 交接、失败/取消后基线、居中与净空约束、真实 CPU ONNX 及原有段协议/停车规则。另对当前 5 个目标和 3 条历史路线完成正式规划器回放；见 `logs/vio_center_fix_20260930/`。没有启动真实相机或机器人，实机效果待测。

以下保留 09-29 的历史验证记录：当时 **33 项离线测试通过**，其中 SRU 总开关新增 5 项：关闭时无下游启动/命令通道/确认等待；真实 A* 15 点、VIO 到达与下一段；暂停重定位保持当前目标；规划失败和 SLAM 崩溃不推进；配置及命令行覆盖。相机和控制进程均被替换，不作为实地测试结果。

2026-09-29 后续配置整理：核对新旧配置独立选择与通用参数一致性，3 项相关模型/真机循环替身测试通过，见 [本轮记录](logs/handoff_config_20260929/verification.json)。

28 项离线测试覆盖原有功能、新旧真实 ONNX 推理、错误模型拒绝、mix 深度处理、5 Hz 调用门控、观测顺序、h/c 连续与重置、15 点坐标转换、加载/启动确认及超时、人工停车竞态、同目标恢复和 SLAM 重启。real 主循环测试替换了相机和通信，确认路径传入实际模型，并验证推理期间人工停车后只输出零速。未运行真实相机或机器人；5 Hz 为配置频率，真机吞吐与导航效果仍需现场验证。

```bash
.venv_navside/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

离线示例回放：[日志](logs/task_nav/astar_offline_demo_20260929/scheduler_replay.log)、[检查报告](logs/task_nav/astar_offline_demo_20260929/verification.json)。示例目标和定位事件均为测试数据，所有发送动作已替换为日志，没有运行机器人进程，也没有写入正式选点文件。
