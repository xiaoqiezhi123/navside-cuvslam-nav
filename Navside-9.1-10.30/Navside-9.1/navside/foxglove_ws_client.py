"""Foxglove WebSocket client with ROS1 binary deserialization.

Connects to a Foxglove WebSocket bridge server, subscribes to ROS topics,
and parses the binary message payloads into Python/numpy values.

Supports:
  - nav_msgs/Odometry   -> position, quaternion (wxyz), rpy
  - geometry_msgs/Twist  -> linear velocity, angular velocity

Protocol reference:
  - Foxglove WS: https://github.com/foxglove/ws-protocol
  - ROS1 binary: little-endian, no alignment padding
"""

import asyncio
import json
import math
import os
import struct
import time

import numpy as np
import websockets


# ---------------------------------------------------------------------------
# ROS1 binary deserialization
# ---------------------------------------------------------------------------

class Ros1Reader:
    """Streaming reader for ROS1 binary-serialized messages.

    ROS1 format: little-endian, no alignment padding between fields.
    Primitives are written directly; strings are uint32 length-prefixed;
    arrays are uint32 count-prefixed.
    """

    def __init__(self, data: bytes):
        self._buf = data
        self._pos = 0

    def uint32(self) -> int:
        value = struct.unpack_from("<I", self._buf, self._pos)[0]
        self._pos += 4
        return value

    def float64(self) -> float:
        value = struct.unpack_from("<d", self._buf, self._pos)[0]
        self._pos += 8
        return value

    def string(self) -> str:
        length = self.uint32()
        if length == 0:
            return ""
        raw = self._buf[self._pos:self._pos + length]
        self._pos += length
        return raw.decode("utf-8", errors="replace")

    def skip(self, n: int) -> None:
        if self._pos + n > len(self._buf):
            raise ValueError(
                f"ROS1 read overflow: need {n} bytes at offset {self._pos}"
            )
        self._pos += n


def deserialize_odometry(data: bytes):
    """Deserialize nav_msgs/Odometry from ROS1 binary format.

    Message layout:
      Header header
        uint32 seq
        uint32 stamp.secs
        uint32 stamp.nsecs
        string frame_id
      string child_frame_id
      geometry_msgs/Pose pose
        Point position
          float64 x, y, z
        Quaternion orientation   (ROS1 order: x, y, z, w)
          float64 x, y, z, w
        float64[36] covariance
      geometry_msgs/Twist twist
        Vector3 linear
          float64 x, y, z
        Vector3 angular
          float64 x, y, z
        float64[36] covariance

    Returns:
        (position_xyz, quaternion_wxyz, rpy) as numpy arrays, or None on error.
    """
    try:
        r = Ros1Reader(data)

        # Header
        _seq = r.uint32()
        _secs = r.uint32()
        _nsecs = r.uint32()
        _frame_id = r.string()

        # child_frame_id
        _child_frame_id = r.string()

        # Pose (position)
        px = r.float64()
        py = r.float64()
        pz = r.float64()
        position = np.array([px, py, pz])

        # Pose (orientation) — ROS1 order is x, y, z, w
        ox = r.float64()
        oy = r.float64()
        oz = r.float64()
        ow = r.float64()
        quat_wxyz = np.array([ow, ox, oy, oz])  # convert to wxyz

        # Pose covariance: float64[36] = 288 bytes
        r.skip(288)

        # Twist (linear velocity)
        lx = r.float64()
        ly = r.float64()
        lz = r.float64()

        # Twist (angular velocity)
        ax = r.float64()
        ay = r.float64()
        az = r.float64()

        # Twist covariance: float64[36] = 288 bytes
        r.skip(288)

        # Compute RPY from quaternion
        rpy = _quaternion_wxyz_to_rpy(quat_wxyz)

        return position, quat_wxyz, rpy

    except (ValueError, struct.error) as e:
        print(f"Odometry deserialize error: {e}")
        return None


def deserialize_twist(data: bytes):
    """Deserialize geometry_msgs/Twist from ROS1 binary format.

    Message layout:
      Vector3 linear
        float64 x, y, z
      Vector3 angular
        float64 x, y, z

    Returns:
        (linear_xyz, angular_xyz) as numpy arrays, or None on error.
    """
    try:
        r = Ros1Reader(data)

        lx = r.float64()
        ly = r.float64()
        lz = r.float64()
        linear = np.array([lx, ly, lz])

        ax = r.float64()
        ay = r.float64()
        az = r.float64()
        angular = np.array([ax, ay, az])

        return linear, angular

    except (ValueError, struct.error) as e:
        print(f"Twist deserialize error: {e}")
        return None


def _quaternion_wxyz_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to (roll, pitch, yaw) in radians."""
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm > 1e-12:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


# Map ROS message types to deserializers
_DESERIALIZERS = {
    "nav_msgs/Odometry": deserialize_odometry,
    "nav_msgs/msg/Odometry": deserialize_odometry,
    "geometry_msgs/Twist": deserialize_twist,
    "geometry_msgs/msg/Twist": deserialize_twist,
}


# ---------------------------------------------------------------------------
# Foxglove WebSocket client
# ---------------------------------------------------------------------------

class FoxgloveWsClient:
    """Async Foxglove WebSocket client for ROS topic subscription.

    Connects to a Foxglove bridge server, subscribes to topics,
    and dispatches parsed messages via callbacks.

    Usage:
        client = FoxgloveWsClient("ws://<INTERNAL_IP>:8765")
        client.on_odom = lambda pos, quat, rpy: print(f"pos={pos}")
        client.on_cmd_vel = lambda lin, ang: print(f"vel={lin}")

        async with client:
            await client.subscribe("/odom", "nav_msgs/Odometry")
            await client.run()
    """

    def __init__(self, uri: str):
        self.uri = uri
        self._ws = None
        self._connected = False
        self._running = False

        # Topic -> channel ID mapping (populated from serverInfo)
        self._channels: dict = {}

        # Subscription ID counter
        self._sub_id_counter = 0
        # subscription_id -> (topic, msg_type, deserializer)
        self._subscriptions: dict = {}

        # Callbacks (set by owner before run())
        self.on_odom = None      # async callable(pos, quat_wxyz, rpy)
        self.on_cmd_vel = None   # async callable(linear, angular)

    @property
    def connected(self) -> bool:
        return self._connected

    async def __aenter__(self):
        ok = await self.connect()
        if not ok:
            raise ConnectionError(f"Failed to connect to {self.uri}")
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    async def connect(self) -> bool:
        """Connect to the Foxglove WebSocket server and parse serverInfo."""
        try:
            # Temporarily clear proxy env vars — websockets reads them
            # and routes through the system proxy which can't reach the robot.
            proxy_vars = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                          "ALL_PROXY", "all_proxy")
            saved = {k: os.environ.pop(k, None) for k in proxy_vars}
            try:
                self._ws = await websockets.connect(
                    self.uri,
                    ping_interval=20,
                    ping_timeout=10,
                    subprotocols=["foxglove.websocket.v1"],
                )
            finally:
                for k, v in saved.items():
                    if v is not None:
                        os.environ[k] = v
        except Exception as e:
            print(f"FoxgloveWsClient: connect to {self.uri} failed: {e}")
            self._connected = False
            return False

        # Wait for serverInfo
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
        except asyncio.TimeoutError:
            print("FoxgloveWsClient: timeout waiting for serverInfo")
            self._connected = False
            return False

        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            print(f"FoxgloveWsClient: invalid serverInfo: {raw[:200]}")
            self._connected = False
            return False

        if info.get("op") != "serverInfo":
            print(
                f"FoxgloveWsClient: expected serverInfo, got {info.get('op')}"
            )
            self._connected = False
            return False

        print(f"FoxgloveWsClient: got serverInfo, name={info.get('name', '?')}")

        # Wait for initial advertise message(s) — channels arrive AFTER serverInfo.
        # The server pushes advertise/advertiseServices/parameterValues after handshake.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if isinstance(raw, bytes):
                continue  # skip binary during handshake

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            op = msg.get("op", "")
            if op == "advertise":
                self._update_channels(msg.get("channels", []))
                if self._channels:
                    break  # got our channels
            elif op == "advertiseServices":
                pass  # services not used yet
            elif op == "parameterValues":
                pass  # parameters not used yet

        self._connected = True
        print(
            f"FoxgloveWsClient: connected to {self.uri} "
            f"({len(self._channels)} channels available)"
        )
        return True

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        self._running = False
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def subscribe(self, topic: str, msg_type: str) -> bool:
        """Subscribe to a ROS topic.

        Args:
            topic: ROS topic name (e.g. "/high_frequency_odometry")
            msg_type: ROS message type (e.g. "nav_msgs/Odometry")

        Returns:
            True if subscription was sent.
        """
        if not self._connected or self._ws is None:
            print(
                f"FoxgloveWsClient: not connected, cannot subscribe to {topic}"
            )
            return False

        # Look up channel ID
        channel_id = self._channels.get(topic)
        if channel_id is None:
            print(
                f"FoxgloveWsClient: channel not found for '{topic}', "
                f"trying by topic name"
            )
            channel_id = topic

        sub_id = self._sub_id_counter
        self._sub_id_counter += 1

        subscribe_msg = json.dumps({
            "op": "subscribe",
            "subscriptions": [
                {"id": sub_id, "channelId": channel_id}
            ],
        })

        await self._ws.send(subscribe_msg)

        deserializer = _DESERIALIZERS.get(msg_type)
        self._subscriptions[sub_id] = (topic, msg_type, deserializer)

        print(
            f"FoxgloveWsClient: subscribed to {topic} "
            f"(id={sub_id}, channelId={channel_id}, type={msg_type})"
        )
        return True

    async def run(self) -> None:
        """Run the receive loop (blocking until disconnected or stopped)."""
        if not self._connected:
            print("FoxgloveWsClient: not connected")
            return

        self._running = True

        while self._running and self._connected:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                print("FoxgloveWsClient: connection closed")
                self._connected = False
                break

            if isinstance(message, bytes):
                await self._handle_binary(message)
            elif isinstance(message, str):
                await self._handle_json(message)

    async def _handle_binary(self, data: bytes) -> None:
        """Handle a binary message from the WebSocket.

        Foxglove binary format:
          1 byte  — opcode (0x01 = MESSAGE_DATA)
          4 bytes LE — subscription_id
          8 bytes LE — timestamp (nanoseconds)
          remaining  — ROS1 binary payload
        """
        if len(data) < 13:
            return

        opcode = data[0]
        if opcode != 0x01:  # MESSAGE_DATA
            return

        sub_id = struct.unpack_from("<I", data, 1)[0]
        # timestamp = struct.unpack_from("<Q", data, 5)[0]  # nanoseconds, unused
        payload = data[13:]

        sub_info = self._subscriptions.get(sub_id)
        if sub_info is None:
            return

        topic, msg_type, deserializer = sub_info
        if deserializer is None:
            return

        # Deserialize and dispatch
        if msg_type in ("nav_msgs/Odometry", "nav_msgs/msg/Odometry"):
            result = deserializer(payload)
            if result is not None and self.on_odom is not None:
                pos, quat, rpy = result
                await self.on_odom(pos, quat, rpy)

        elif msg_type in ("geometry_msgs/Twist", "geometry_msgs/msg/Twist"):
            result = deserializer(payload)
            if result is not None and self.on_cmd_vel is not None:
                linear, angular = result
                await self.on_cmd_vel(linear, angular)

    async def _handle_json(self, message: str) -> None:
        """Handle a JSON message (advertise, subscription status, etc.)."""
        try:
            msg = json.loads(message)
            op = msg.get("op", "")
            if op == "advertise":
                self._update_channels(msg.get("channels", []))
            elif op == "subscriptionStatus":
                for sub in msg.get("subscriptions", []):
                    status = sub.get("status", "UNKNOWN")
                    if status != "OK":
                        print(
                            f"FoxgloveWsClient: subscription "
                            f"id={sub.get('id')} status={status}"
                        )
        except json.JSONDecodeError:
            pass

    def _update_channels(self, channels: list) -> None:
        """Add channels from an advertise message to the channel map."""
        for ch in channels:
            topic = ch.get("topic", "")
            ch_id = ch.get("id", -1)
            if topic and ch_id >= 0:
                self._channels[topic] = ch_id
