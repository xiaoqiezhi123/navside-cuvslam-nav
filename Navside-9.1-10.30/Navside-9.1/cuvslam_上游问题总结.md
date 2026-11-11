# cuVSLAM 绑定层 / 环境问题总结(提交上游用)

生成日期:2026-09-14
运行环境:Jetson(Orin/tegra,Linux 5.15.148-tegra)、Python 3.10、pyorbbecsdk、
cuvslam 绑定为预编译 .so(pycuvslam.cpython-310-aarch64-linux-gnu.so,nanobind)。

---

## P0:绑定层 GIL 保护缺失 → 进程硬中止(最高优先级,建议上游优先修复)

### 现象

调用 `Tracker.localize_in_map(...)`(async 模式,sync_mode=False)期间进程随机死亡,
无 Python traceback,只有:

```
Fatal Python error: PyThreadState_Get: the function must be called with the GIL held,
but the GIL is released (the current Python thread state is NULL)
```

### 实测证据

- 崩点不固定:一次崩在等待 finish_cb 期间(主线程停在 `Event.wait`),一次崩在
  localize_in_map **尚未被调用**时(主线程停在 result_queue.get)——说明是后台线程
  在随机时刻触发的回调问题,不是宿主代码的调用顺序问题;
- 与「进程内第几次调用」无关:新进程的第一次 localize 也崩过;
- 相邻两次 localize 间隔越短,触发概率越高(背靠背触发过 2 次);
- 每次进程退出都伴随 `nanobind: leaked N instances / types / functions`
  (官方文档明确「likely caused by a reference counting issue in the binding code」)。

### 根因(已定位:与调用方式相关,非 .so 必然缺陷)

`SlamConfig.sync_mode` 决定 `localize_in_map` 的执行方式:

- **sync_mode=True**(run_vio.py 的 `--mode localize` 一直用):搜索在**调用线程内
  同步执行**,start_cb/finish_cb **同步触发**(主线程,持 GIL)——用户长期反复
  localize **从未出现 GIL 问题**;
- **sync_mode=False**(tasknav 脚本原配置):localize_in_map 排队到**后台 SLAM
  线程**,回调从后台线程触发。该路径下绑定层存在**某个回调位点缺少
  `nb::gil_scoped_acquire`**(怀疑 start_cb,「localization 开始时调用」)——后台
  线程在无 GIL、无 Python 线程状态时调用 Python 对象 → CPython 硬中止
  （Fatal Python error: PyThreadState_Get）。崩点随机（一次在等待 finish_cb、
  一次在 localize_in_map 尚未被调用时）、与进程内第几次调用无关——与
  「后台线程在随机时机触发回调」完全吻合。

另注意:宿主侧尝试传 `None` 给 start_cb 规避时,绑定直接抛 TypeError
(签名要求 `Callable`),说明该参数在 C++ 侧被无条件使用。

**宿主侧已根治**:tasknav 脚本改为 `SLAM_SYNC_MODE = True`(与 localize 模式
一致),同步搜索期间采集线程挂起（pause 协议），无并发风险。

### 建议的上游修改方案(源码级)

1. **审计并修复所有从非 Python 线程触发的回调位点**,统一包 GIL:
   ```cpp
   // 例:定位线程触发 start_cb 时
   void invoke_start_cb() {
       nb::gil_scoped_acquire gil;   // 必须先获取 GIL + 挂线程状态
       if (!start_cb.is_none()) {
           start_cb();
       }
   }
   ```
   定位搜索线程、SLAM 跟踪线程、任何 `std::thread`/线程池里调 Python 对象的位置
   都要逐一排查。

2. **修复引用计数泄漏**:进程退出时的 nanobind leaked 报告(Config、Pose、
   MultisensorSettings、Rig、Landmark、PoseGraph、State 等类型 + 346 个函数)
   指向绑定类定义/返回策略的问题——检查 `nb::class_` 的所有权策略、回调参数
   的 `keep_alive`、跨线程传递 Python 对象时的引用管理。GIL 问题与泄漏问题
   很可能是同一套线程/引用管理缺陷的两个表现。

3. **建议增加定位线程生命周期接口**(如 localize 空闲通知),宿主可据此安全地
   串行化 localize 请求,进一步降低竞态面。

4. 若短期无法修复,建议让 `start_cb` 支持传 `None`(跳过该回调)——目前宿主想
   规避都规避不了(TypeError)。

### 宿主侧临时缓解(本环境已实施)

- 两次锚定之间强制 10s 冷却(挡住背靠背触发);
- 进程崩溃后由调度器延迟 10s 自动重启,重启后走「启动自动锚定」路径续接任务点;
- [SCHED] 事件双写文件通道,崩溃后留痕可查。

---

## P1:Orbbec USB 栈在快速开/关相机后失稳(中优先级)

### 现象

- 同一相机在多次快速「打开→关闭→再打开」后:
  1. `Failed to get frames for calibration`(打开后取不到标定帧);
  2. `pyorbbecsdk.OBError: Device not found by serial number: <SLAM_CAM_SERIAL>`,
     `lsusb` 里相机彻底消失——**必须物理拔插 USB 才能恢复**;
- 每次打开相机都有 `[UsbEnumeratorLibusb.cpp:162] Failed to get string descriptor:
  error=Operation timed out`(串号描述符读取超时,偶发)。

### 推断根因

Orbbec SDK 的 USB 释放/重新枚举在快速循环下存在竞态:设备句柄未干净释放、
枚举缓存陈旧、或固件/USB3 协商异常后降级。串号描述符读取超时是同一问题的
早期信号。

### 建议的上游修改方案

- 设备关闭时确保 `stop → close` 完成后延迟释放(或加释放确认);
- `query_devices()` 加设备列表失效刷新机制;
- 提供「软复位」接口(重新枚举而非物理拔插);
- 开流前对「标定帧不可用」做重开重试(目前只在打开阶段重试)。

### 宿主侧临时缓解

- 崩溃重启前固定等待 10s 再重开相机;
- 打开相机阶段重试延长到 15s(按串号 + 按索引双路径)。

---

## P2:相机 B(<SLAM_CAM_SERIAL>)打不开 1280x720 深度流(硬件/USB 带宽问题)

### 现象

- 相机 B 请求 1280x720@30 深度流失败,只有 1280x720@5(USB2 降速特征);
- IR 流只能拿到 848x480@10 回落档;
- 同环境相机 A(<NAVSIDE_CAM_SERIAL>)正常。

### 推断

相机 B 与该 USB 口的 USB3 协商失败(线材/口/固件)。与 P1 的枚举问题可能
同源(该相机 USB 状态整体不稳)。

### 建议

- 检查/更换 USB3 线材与端口,确认 SuperSpeed 协商;
- 更新相机固件;
- SDK 侧:枚举后打印/暴露 USB 协商速率,便于诊断。

### 宿主侧处理

深度感知已切换到相机 A;相机 B 仅用于 cuVSLAM 被动双目(IR 848x480@10,
激光关闭),不影响任务导航功能。

---

## P3:地图使用注意事项(非代码问题)

- `localize_in_map` 的 4D 搜索(x/y/z/航向)不覆盖滚转/俯仰:若建图结束时相机
  有明显倾斜(如手持),重建图后的水平姿态会 localize 失败。建议建图时固定
  相机姿态或文档中明确约束;
- 首帧自动锚定的 coarse 半径(10m)与冷启动全图搜索(25m)的差异属于宿主导入
  策略,供参考。

---

## 附录:本环境验证过的正常路径(供回归参考)

- 每个进程的**第一次** `localize_in_map`(启动自动锚定)成功率较高(约 7/8);
- 进程内**第二次**调用曾 2/2 崩溃,新进程首次调用也有崩溃记录——GIL 竞态
  与调用次数无严格对应,与时机/并发状态相关;
- 崩溃后延迟重启 + 启动锚定续接,流程可自愈(前提:相机 USB 未彻底掉线)。
