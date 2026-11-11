"""Pure segment protocol. The mode controller serializes access with its lock."""
from dataclasses import dataclass
import json
import re

import numpy as np

FRAME = 'navside_zup'


@dataclass(frozen=True)
class Segment:
    session_id: str
    revision: int
    segment_id: str
    goal_w: np.ndarray
    path_w: np.ndarray
    resume_token: int | None
    fingerprint: str

    @classmethod
    def parse(cls, payload):
        obj = json.loads(payload) if isinstance(payload, str) else payload
        if (not isinstance(obj, dict) or type(obj.get('schema_version')) is not int
                or obj.get('schema_version') != 1 or obj.get('frame') != FRAME):
            raise ValueError('invalid_schema_or_frame')
        session, revision = obj.get('session_id'), obj.get('revision')
        if (not isinstance(session, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', session)
                or type(revision) is not int or revision < 1):
            raise ValueError('invalid_segment_identity')
        segment_id = f'{session}:{revision}'
        if obj.get('segment_id') != segment_id:
            raise ValueError('invalid_segment_id')
        goal = np.asarray(obj.get('goal_w'), dtype=np.float32)
        path = np.asarray(obj.get('path_w'), dtype=np.float32)
        if goal.shape != (3,) or path.shape != (15,3) or not all(np.isfinite(v).all() for v in (goal,path)):
            raise ValueError('invalid_goal_or_path')
        if not np.isclose(goal[2], .695, atol=1e-6) or not np.allclose(path[:,2], .5, atol=1e-6):
            raise ValueError('invalid_height')
        if not np.allclose(path[-1,:2], goal[:2], rtol=0, atol=1e-5):
            raise ValueError('path_goal_mismatch')
        token = obj.get('resume_token')
        if token is not None and (type(token) is not int or token < 1):
            raise ValueError('invalid_resume_token')
        fingerprint = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)
        goal.setflags(write=False)
        path.setflags(write=False)
        return cls(session, revision, segment_id, goal, path, token, fingerprint)


class SegmentProtocol:
    def __init__(self, session_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', session_id):
            raise ValueError('PathAware 需要调度器提供 NAVSIDE_SESSION_ID')
        self.session_id = session_id
        self.segment = None
        self.phase = 'empty'
        self.epoch = 0
        self.manual_token = 0
        self.manual_hold = False
        self.pending_start = False
        self.loaded_pose_sequence = 0

    def event(self, phase, *, segment_id=None, reason=None):
        event = dict(path=phase, session_id=self.session_id,
                     segment_id=segment_id or (self.segment.segment_id if self.segment else 'none'),
                     manual_token=str(self.manual_token))
        if reason:
            event['reason'] = reason
        return event

    def stop(self, manual=False, reason=None):
        self.epoch += 1
        self.pending_start = False
        if manual:
            self.manual_token += 1
            self.manual_hold = True
        self.phase = 'paused' if self.manual_hold else 'stopped'
        return self.event('paused' if manual else 'stopped', reason=reason)

    def load(self, payload, *, stopped, pose_sequence, reset):
        segment = Segment.parse(payload)
        sid = segment.segment_id
        if segment.session_id != self.session_id:
            return self.event('rejected', segment_id=sid, reason='wrong_session')
        if self.segment and segment.revision <= self.segment.revision:
            if (sid == self.segment.segment_id and segment.fingerprint == self.segment.fingerprint
                    and self.phase in ('ready', 'running', 'waiting_pose') and not self.manual_hold):
                return self.event(self.phase)
            return self.event('rejected', segment_id=sid, reason='stale_or_conflicting_segment')
        if not stopped:
            return self.event('rejected', segment_id=sid, reason='must_stop_before_load')
        if self.manual_hold and segment.resume_token != self.manual_token:
            return self.event('rejected', segment_id=sid, reason='manual_stop_requires_localize')
        reset()
        self.segment = segment
        self.loaded_pose_sequence = pose_sequence
        self.epoch += 1
        self.manual_hold = False
        self.pending_start = False
        self.phase = 'ready'
        return self.event('ready')

    def start(self, segment_id):
        if not self.segment or segment_id != self.segment.segment_id:
            return self.event('rejected', segment_id=segment_id, reason='wrong_segment')
        if self.manual_hold or self.phase not in ('ready', 'running'):
            return self.event('rejected', reason='segment_not_ready')
        if self.phase == 'running':
            return self.event('running')
        self.pending_start = True
        return None

    def tick(self, pose_sequence, pose_fresh):
        if (self.pending_start and not self.manual_hold and self.phase == 'ready'
                and pose_fresh and pose_sequence > self.loaded_pose_sequence):
            self.pending_start = False
            self.phase = 'running'
            return self.event('running')
        return None

    def fault(self, reason):
        self.stop()
        self.phase = 'error'
        return self.event('error', reason=reason)

    def wait_for_pose(self, reason):
        self.epoch += 1  # invalidate commands already in inference
        self.pending_start = False
        self.phase = 'waiting_pose'
        return self.event('waiting_pose', reason=reason)

    def resume_pose(self):
        self.epoch += 1
        self.phase = 'running'
        return self.event('pose_resumed', reason='pose_stable')

    @property
    def running(self):
        return self.phase == 'running' and not self.manual_hold
