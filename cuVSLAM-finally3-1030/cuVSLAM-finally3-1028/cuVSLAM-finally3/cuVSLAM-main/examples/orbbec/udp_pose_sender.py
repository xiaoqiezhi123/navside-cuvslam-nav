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
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""UDP 位姿发送器：把 cuVSLAM 全局位姿以位姿包发给同机下游 RobotComm。

协议（下游 bridge.py RobotComm 按此解析）：
  struct "<7d" = [px, py, pz, qw, qx, qy, qz]（7×float64 = 56 字节，小端）
    - 位置为 Z-up 世界系（+X 前、+Y 左、+Z 上）
    - 姿态为 Z-up 世界系单位四元数（wxyz 顺序）
只传位姿，不含速度/重力 —— 下游 RobotComm 复用其差分/重力/z=0.695 逻辑自行计算。

相比 A* 版 udp_pose_sender.py：去掉 pose_trace 诊断钩子（本基座端口不引入该依赖），
坐标系变换从 pose_transform.opencv_pose_to_zup 导入。
"""

import socket
import struct

from pose_transform import opencv_pose_to_zup


class UdpPoseSender:
    """把 cuVSLAM 相机位姿变换为 Z-up 机体位姿后经 UDP 发送。

    UDP 的 sendto 非阻塞，`send_pose()` 可直接在主线程逐帧调用，无需额外线程。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8082):
        self.addr = (host, port)
        self._packer = struct.Struct("<7d")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_pose(self, position_xyz, quaternion_xyzw) -> None:
        """变换 OpenCV→Z-up 并发送一帧位姿。

        Args:
            position_xyz: (3,) 相机在 OpenCV 世界下的位置 (x右, y下, z前)。
            quaternion_xyzw: (4,) 相机朝向 (x, y, z, w)。
        """
        p_zup, q_zup = opencv_pose_to_zup(position_xyz, quaternion_xyzw)
        qx, qy, qz, qw = q_zup  # opencv_pose_to_zup 返回 xyzw，报文用 wxyz
        packet = self._packer.pack(
            float(p_zup[0]), float(p_zup[1]), float(p_zup[2]),
            float(qw), float(qx), float(qy), float(qz),
        )
        self._sock.sendto(packet, self.addr)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
