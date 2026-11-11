"""Opt-in pose tracing. Standard library only; never opens network sockets.

Legacy wire: <7d (56 bytes). Diagnostic wire: same 56 bytes + PTD1 trailer.
Trailer: magic, sender UUID, sequence, camera ns, SDK receipt monotonic ns,
tracking-complete monotonic ns, send monotonic ns, anchor epoch.
The trailer is observational; it is not used to authorize robot movement.
"""
import atexit
import json
import os
from pathlib import Path
import queue
import struct
import threading
import time
import uuid

POSE = struct.Struct('<7d')
TRAILER = struct.Struct('<4s16s6Q')
MAGIC = b'PTD1'


def decode_pose(data):
    if len(data) == POSE.size:
        return POSE.unpack(data), {'wire_version': 1}
    if len(data) != POSE.size + TRAILER.size:
        raise ValueError('unexpected_pose_packet_size')
    magic, stream, seq, source, received, processed, sent, epoch = TRAILER.unpack(data[POSE.size:])
    if magic != MAGIC:
        raise ValueError('unknown_pose_trailer')
    return POSE.unpack(data[:POSE.size]), dict(wire_version=2, stream_id=stream.hex(), source_sequence=seq,
        source_timestamp_ns=source, frame_received_ns=received, processed_ns=processed, send_ns=sent,
        anchor_epoch=epoch)


class NullTrace:
    enabled = False
    stream_id = ''
    def emit(self, stage, **fields):
        pass
    def close(self):
        pass
    def encode(self, data, **context):
        return data, {}


NULL_TRACE = NullTrace()


class PoseTrace:
    """Bounded, asynchronous writer. Slow disk drops trace rows, never blocks control.

Rows carry a cumulative drop count; normal close also writes final statistics.
Forced process termination may lose buffered diagnostic rows.
"""
    enabled = True
    def __init__(self, directory, role, max_queue=4096):
        self.stream_id = uuid.uuid4().hex
        self.role = role
        self._sequence = 0
        self._queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self.dropped = 0
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory/f'{role}_{os.getpid()}_{self.stream_id[:8]}.jsonl'
        self._file = self.path.open('x', encoding='utf-8')
        try:
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except OSError:
            boot = ''
        self.emit('trace_start', boot_id=boot, pid=os.getpid(), schema=1)
        self._thread = threading.Thread(target=self._write, daemon=True, name='pose-trace-writer')
        self._thread.start()
        atexit.register(self.close)

    def emit(self, stage, **fields):
        if not self.enabled or self._stop.is_set():
            return
        row = dict(stage=stage, role=self.role, logger_id=self.stream_id,
                   mono_ns=time.monotonic_ns(), wall_ns=time.time_ns(), trace_dropped=self.dropped, **fields)
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def encode(self, data, *, source_timestamp_ns=0, frame_received_ns=0, processed_ns=0, anchor_epoch=0):
        self._sequence += 1
        sent = time.monotonic_ns()
        tail = TRAILER.pack(MAGIC, bytes.fromhex(self.stream_id), self._sequence,
                            source_timestamp_ns, frame_received_ns, processed_ns, sent, anchor_epoch)
        packet = data + tail
        return packet, decode_pose(packet)[1]

    def _write(self):
        try:
            last_flush = time.monotonic()
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    row = self._queue.get(timeout=.2)
                except queue.Empty:
                    row = None
                if row is not None:
                    self._file.write(json.dumps(row, separators=(',', ':'))+'\n')
                    self._queue.task_done()
                if time.monotonic()-last_flush >= .2:
                    self._file.flush()
                    last_flush = time.monotonic()
            self._file.write(json.dumps(dict(stage='trace_end', role=self.role, logger_id=self.stream_id,
                mono_ns=time.monotonic_ns(), wall_ns=time.time_ns(), trace_dropped=self.dropped))+'\n')
        except Exception as exc:
            self.enabled = False
            print(f'[POSE-DIAG] trace write failed: {exc}', flush=True)
        finally:
            self._file.close()

    def close(self):
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2.)


_traces = {}
def get_trace(role='nav'):
    directory = os.environ.get('NAV_POSE_DIAG_DIR', '')
    if not directory:
        return NULL_TRACE
    if role not in _traces:
        try:
            _traces[role] = PoseTrace(directory, role)
        except OSError as exc:
            print(f'[POSE-DIAG] cannot create trace: {exc}', flush=True)
            _traces[role] = NULL_TRACE
    return _traces[role]


def trace_packet(trace, stage, packet, **fields):
    if not trace.enabled:
        return
    data = dict(fields)
    if packet is not None:
        data.update(getattr(packet, 'pose_trace', None) or {})
        data.update(pose_sequence=packet.pose_sequence, received_ns=int(packet.received_monotonic*1e9),
                    state_read_timestamp_sec=packet.timestamp_sec, state_position=packet.robot_pos_w.tolist(),
                    state_quaternion=packet.robot_quat_wxyz.tolist(), linear_velocity=packet.linear_vel_b.tolist(),
                    angular_velocity=packet.angular_vel_b.tolist())
    trace.emit(stage, **data)
