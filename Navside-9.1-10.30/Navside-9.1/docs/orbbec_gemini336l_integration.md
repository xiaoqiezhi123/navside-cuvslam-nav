# Orbbec Gemini 336L 接入 NavSide

本文档记录 Orbbec Gemini 336L 深度相机接入 NavSide 感知栈的环境配置、代码改动与启动方式。

日期:2026-08-17

---

## 1. 环境与依赖

### 1.1 项目环境 `.venv_navside`(正式运行环境)

路径:`/home/amov/nav_arm_mujoco/.venv_navside/`

- numpy **1.26.4**(注意:系统 python 是 numpy 2.2.6,但项目 venv 是 1.26.4)
- pyorbbecsdk2 **2.1.2**(以 `--no-deps` 只装核心,避免拉进 dash/plotly/open3d/opencv 等重依赖)
- onnxruntime 1.23.2 / pyzed 5.4 / pyrealsense2 2.58.3(原有,未动)

### 1.2 隔离环境 `venv_orbbec`(独立测试用)

路径:`/home/amov/venv_orbbec/`

- numpy 1.26.4 + pyorbbecsdk2 2.1.2(带全套依赖,含 open3d/opencv/dash 等)

### 1.3 wheel 安装包(已下载持久化)

```
/home/amov/orbbec/pyorbbecsdk2-2.1.2-cp310-cp310-manylinux_2_27_aarch64.whl
```

来源:GitHub releases `orbbec/pyorbbecsdk`,PyPI 上**没有**此包。

---

## 2. 权限(udev 规则)

没有这条规则时,`ob.Pipeline()` 会报 `usbEnumerator openUsbDevice failed!`。

文件:`/etc/udev/rules.d/99-orbbec.rules`

```
SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", MODE="0666"
```

安装方式(需 sudo):

```bash
sudo cp /home/amov/orbbec/99-orbbec.rules /etc/udev/rules.d/99-orbbec.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 3. 代码改动

仓库根:`/home/amov/navside_real/NavSide_log/NavSide/`

| 文件 | 改动 |
|---|---|
| `perception/orbbec_depth_perception.py` | 新增,类 `OrbbecDepthPerception`,对齐 realsense/zed 接口 |
| `perception/__init__.py` | 加 `orbbec/gemini/gemini336l/336l` 分支 |

### 后端接口契约(与 ZED/RealSense 完全一致)

```python
cam = create_depth_perception("orbbec")   # 工厂
cam.start()
out = cam.read()                           # -> DepthPerceptionOutput
#   out.success: bool
#   out.depth_input: np.ndarray, float32, 单位米, shape (720, 1280)
#   out.depth_feature: None(VAE 编码由 adapter 完成)
#   out.front_distance_m: float(中心 ROI 第10百分位)
#   out.error: str
cam.close()
```

---

## 4. 启动命令

### 4.1 单独验证后端(严格校验,17 项断言)

```bash
/home/amov/nav_arm_mujoco/.venv_navside/bin/python /home/amov/orbbec/verify_orbbec.py
```

### 4.2 完整真机导航循环

切换后端只需改 `config/nav.yaml` 一行:

```yaml
depth:
  backend: orbbec      # 原为 realsense
```

启动:

```bash
cd /home/amov/navside_real/NavSide_log/NavSide
.venv_navside/bin/python scripts/run_nav.py --real --config config/nav.yaml
```

> `run_nav.py` 会自动 re-exec 进 `.venv_navside`。

---

## 5. 关键接口事实

- **import 名** = `pyorbbecsdk`(包名是 `pyorbbecsdk2`,别搞混)。
- **深度流**:Y16 格式,采用 `1280×720@30`(对齐现有 RAW 720×1280)。
- **换算**:`depth_frame.get_depth_scale()` = 1.0 → 原始 uint16 单位是**毫米**,米 = `raw / 1000.0`;0 与 65535 是无效哨兵值。
- **抓帧**:`Pipeline` + `Config` + `get_stream_profile_list(OBSensorType.DEPTH_SENSOR)` + `get_video_stream_profile(w,h,OBFormat.Y16,fps)` + `enable_stream` + `start` + `wait_for_frames(ms)`(超时返回 None)+ `get_depth_frame()` + `np.frombuffer(get_data(), np.uint16)`。

---

## 6. 踩坑记录

1. **numpy 版本**:pyorbbecsdk2 对 Python 3.10 要求 `numpy<2.0`;项目 `.venv_navside` 正好是 numpy 1.26.4,无冲突(2.2.6 是系统 python,不是项目环境)。
2. **本地 wheel 安装**:`pip install xxx.whl` 时文件名必须带完整版本号(存成通用名会报 `Invalid wheel filename`)。
3. **从国内下 GitHub 大文件**:直连极慢(36KB/s),用 `https://ghfast.top/` 前缀(实测 7.6MB/s),配合 `curl -C -` 断点续传。
4. **Orin WiFi 抖动**:会掉线/重启,`/tmp` 在重启后被清空;重要文件放 `/home/amov/` 下,下载/安装脚本建议放后台跑(`nohup`)抗断连。
