# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.

"""cuVSLAM → NavSide 深度帧共享内存通道（双缓冲 + 序号，latest-wins）。

把 cuVSLAM 独占相机后拿到、且已做填洞（HoleFillingFilter）的原始深度 uint16(mm)
零拷贝地转交给同机 NavSide（VAE/SRU 消费，控制环 10Hz）。

字节布局（写方创建时即确定 W/H，读方 attach 后从 header 读回）：

    [0 .. 24)              header: seq(u64) ts_ns(u64) width(u32) height(u32)
    [24 .. 24+N)           slot0: uint16[H*W]           N = W*H*2
    [24+N .. 24+2N)        slot1: uint16[H*W]

并发协议（单生产者 / 单消费者，无锁）：
  - 写方：把整帧拷进 ``slot[seq % 2]``，再 ``seq += 1``。
  - 读方：读 ``seq``，变了就读 ``slot[(seq-1) % 2]`` 并拷贝返回。
  - 双缓冲保证读者永远读到「完整写完的那一槽」；写方正在写的是另一槽，不会相互覆盖。

注意：本文件在 cuVSLAM 与 NavSide 两侧各有一份，**字节布局与并发协议必须保持一致**，
改这里要两边同步。
"""

import time
from multiprocessing import shared_memory

import numpy as np

_HEADER_DTYPE = np.dtype(
    [
        ("seq", "<u8"),    # 已完成的帧序号（写方写完一帧后 +1）
        ("ts_ns", "<u8"),  # 最近一帧的时间戳（ns，仅诊断用，非精确对齐）
        ("width", "<u4"),
        ("height", "<u4"),
    ]
)
_HEADER_BYTES = _HEADER_DTYPE.itemsize  # 24
_N_SLOTS = 2
_U16 = np.dtype("<u2")


def _total_bytes(width: int, height: int) -> int:
    return _HEADER_BYTES + _N_SLOTS * width * height * _U16.itemsize


class DepthShmWriter:
    """写方（cuVSLAM 侧）：创建共享内存块并写入深度帧。"""

    def __init__(self, name: str, width: int, height: int):
        self.name = name
        self.width = width
        self.height = height
        size = _total_bytes(width, height)

        # 处理上次异常退出遗留的块：存在则先 unlink 再重建。
        # 注意：unlink 会移除名字，若此时读者仍附着在旧块上，读者会读旧块而失去新数据。
        # 正常启动顺序是 cuVSLAM 先起、NavSide 后 attach，此场景无影响；若 cuVSLAM 中途
        # 重启，需让 NavSide 一并重启（见 DepthShmReader 重试 attach 说明）。
        try:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.unlink()
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=size)

        self._header = np.ndarray((1,), dtype=_HEADER_DTYPE, buffer=self._shm.buf, offset=0)
        self._slots = np.ndarray(
            (_N_SLOTS, height, width), dtype=_U16, buffer=self._shm.buf, offset=_HEADER_BYTES
        )
        # seq=0 表示「尚未写入任何帧」；W/H 立即写入，供读方 attach 后读回。
        self._header[0] = (0, 0, width, height)

    def write(self, depth_uint16: np.ndarray, ts_ns: int) -> None:
        """写入一帧深度。``depth_uint16`` 必须是 (H, W) uint16，值为毫米。"""
        if depth_uint16.shape != (self.height, self.width):
            raise ValueError(f"depth shape {depth_uint16.shape} != {(self.height, self.width)}")
        seq = int(self._header["seq"][0])
        self._slots[seq % _N_SLOTS] = depth_uint16  # 整帧拷入共享内存
        self._header["ts_ns"][0] = ts_ns
        self._header["seq"][0] = seq + 1  # 写完后才推进序号（读方据此判断槽已完成）

    def close(self) -> None:
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:  # noqa: BLE001
            pass


class DepthShmReader:
    """读方（NavSide 侧）：附着共享内存块并读取最新深度帧。"""

    def __init__(self, name: str, timeout_s: float = 5.0):
        self.name = name
        # 轮询等待写方创建块（启动顺序不固定：NavSide 可能先于 cuVSLAM 起）。
        deadline = time.monotonic() + timeout_s
        shm = None
        while shm is None:
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
            except FileNotFoundError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"等待共享内存 '{name}' 超时（写方未启动？）")
                time.sleep(0.1)
        self._shm = shm
        self._header = np.ndarray((1,), dtype=_HEADER_DTYPE, buffer=shm.buf, offset=0)
        # 写方「创建块」到「写入 W/H」之间有一极小窗口（块已存在但 header 仍为 0），
        # 重试读直到 W/H 非零，避免读到未初始化的 0x0。
        deadline = time.monotonic() + 1.0
        while True:
            self.width = int(self._header["width"][0])
            self.height = int(self._header["height"][0])
            if self.width > 0 and self.height > 0:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"共享内存 '{name}' 已附着但 header W/H 未初始化")
            time.sleep(0.001)
        self._slots = np.ndarray(
            (_N_SLOTS, self.height, self.width), dtype=_U16, buffer=shm.buf, offset=_HEADER_BYTES
        )
        self._last_seq = int(self._header["seq"][0])

    def read_latest(self):
        """返回最新完成的 ``(depth_uint16, ts_ns)``；无新帧则返回 ``None``。

        返回的 depth 是私有拷贝（已脱离共享内存），调用方可安全改写。
        """
        seq = int(self._header["seq"][0])
        if seq == self._last_seq:
            return None
        # 最近完整帧在 slot[(seq-1) % 2]；写方正写 slot[seq % 2]，二者不同槽，读它安全。
        frame = self._slots[(seq - 1) % _N_SLOTS].copy()
        ts_ns = int(self._header["ts_ns"][0])
        self._last_seq = seq
        return frame, ts_ns

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:  # noqa: BLE001
            pass
