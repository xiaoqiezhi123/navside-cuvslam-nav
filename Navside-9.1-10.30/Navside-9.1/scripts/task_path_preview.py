"""Cancellable goal-to-goal previews; one worker, no GUI or navigation IO."""
from collections import OrderedDict
import queue
import threading


def pair_key(start, goal):
    return tuple(start), tuple(goal)


class PreviewWorker:
    def __init__(self, planner):
        self.planner = planner
        self.results = queue.Queue()
        self._condition = threading.Condition()
        self._pending = None
        self._cancel = threading.Event()
        self._closed = False
        self._revision = 0
        self._cache = OrderedDict()
        self._thread = None

    def request(self, points):
        snapshot = tuple(tuple(p) for p in points)
        with self._condition:
            if self._closed:
                raise RuntimeError('预览窗口已关闭')
            self._cancel.set()
            self._cancel = threading.Event()
            self._revision += 1
            self._pending = self._revision, snapshot, self._cancel
            # Superseded results must neither occupy the UI nor be displayed.
            while True:
                try:
                    self.results.get_nowait()
                except queue.Empty:
                    break
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name='task-path-preview', daemon=True)
                self._thread.start()
            self._condition.notify()
            return self._revision

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._cancel.set()
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                revision, points, cancel = self._pending
                self._pending = None
            for start, goal in zip(points, points[1:]):
                if cancel.is_set():
                    break
                key = pair_key(start, goal)
                try:
                    if key in self._cache:
                        outcome = self._cache[key]
                        self._cache.move_to_end(key)
                    else:
                        outcome = self.planner.preview(start, goal, cancel=cancel)
                        if not cancel.is_set():
                            self._cache[key] = outcome
                            if len(self._cache) > 64:
                                self._cache.popitem(last=False)
                except Exception as exc:
                    outcome = exc
                # Serializing publication with request() also prevents stale queue growth.
                with self._condition:
                    if self._closed or cancel.is_set():
                        break
                    self.results.put((revision, key, outcome))
