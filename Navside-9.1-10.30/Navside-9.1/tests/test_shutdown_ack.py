"""Shutdown acknowledgement tests: fake processes, real local event-file reader."""
import copy
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import task_nav_scheduler as scheduler


class ShutdownAckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = scheduler.load_config(str(ROOT/'config/task_nav.yaml'))
        self.cfg['task_points'] = [(-6., .695, 7.)]
        self.cfg['astar']['output_directory'] = str(Path(self.tmp.name)/'plans')
        self.s = scheduler.TaskNavScheduler(copy.deepcopy(self.cfg), SimpleNamespace())
        self.s.planner = SimpleNamespace(pm=SimpleNamespace(meta={key:'test' for key in
            ('map_id', 'map_version', 'grid_sha256', 'source_db_sha256')}))
        self.s.log = Mock()
        self.addCleanup(self.s._cancel_plan)

    def test_run_reads_ack_after_input_thread_stop_and_joins_event_reader(self):
        actual_thread = threading.Thread
        readers = []
        fake_slam = SimpleNamespace(tag='SLAM', reference_panel_text='', alive=lambda:False,
            start=lambda:self.s.events.put(('slam', {'__ready__':'1'})))
        def make_thread(*args, **kwargs):
            if kwargs.get('target') is scheduler.sched_file_loop and kwargs['args'][3] == 'nav':
                self.assertIsNot(kwargs['args'][2], self.s.stop_event)
                worker = actual_thread(*args, **kwargs)
                readers.append(worker)
                return worker
            return SimpleNamespace(start=lambda:None)
        def send(command):
            self.assertEqual(command, 'quit')
            self.assertTrue(self.s.stop_event.is_set())
            self.assertFalse(self.s._nav_file_stop.is_set())
            with open(self.s.nav_sched_file, 'a') as stream:
                stream.write('[SCHED] process=exit\n')
            return True
        with patch.object(self.s, 'prepare_planning', return_value=True), \
             patch.object(self.s, '_spawn_nav_terminal'), \
             patch.object(self.s, 'poll_planning', side_effect=lambda:setattr(self.s, 'done', True)), \
             patch.object(self.s, 'nav_send', side_effect=send), \
             patch.object(scheduler, '__file__', str(Path(self.tmp.name)/'scripts/scheduler.py')), \
             patch.object(scheduler, 'ChildProc', return_value=fake_slam), \
             patch.object(scheduler.threading, 'Thread', side_effect=make_thread), \
             patch.object(scheduler.signal, 'signal'), \
             patch.object(scheduler, 'open_viewer_window', return_value=False), \
             patch.object(scheduler.subprocess, 'Popen', side_effect=AssertionError('no real processes')):
            self.assertEqual(self.s.run(), 0)
        self.assertTrue(self.s.nav_exited)
        self.assertTrue(self.s._nav_file_stop.is_set())
        self.assertTrue(self.s._slam_file_stop.is_set())
        for worker in readers:
            worker.join(timeout=1.)
            self.assertFalse(worker.is_alive())
        self.assertFalse(any('未收到 NavSide' in c.args[0] for c in self.s.log.call_args_list))

    def test_missing_ack_is_not_reported_as_successful_full_exit(self):
        self.s.nav = SimpleNamespace()
        self.s.nav_send = Mock(return_value=True)
        clock = iter(range(100, 140))
        with patch.object(scheduler.time, 'monotonic', side_effect=lambda:next(clock)), \
             patch.object(scheduler.time, 'sleep'):
            self.s.shutdown(0)
        self.assertFalse(self.s.nav_exited)
        self.assertEqual(self.s.exit_code, 1)
        messages = [c.args[0] for c in self.s.log.call_args_list]
        self.assertTrue(any('子进程尚未全部确认退出' in msg for msg in messages))
        self.assertFalse(any(msg.startswith('全部退出') for msg in messages))

    def test_forced_kill_is_followed_by_wait_for_process_exit(self):
        self.s.sru_enabled = False
        proc = Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = [scheduler.subprocess.TimeoutExpired('fake', 3), -9]
        proc.kill.side_effect = lambda:setattr(proc.poll, 'return_value', -9)
        self.s.slam = SimpleNamespace(proc=proc, send=Mock(), reference_panel_text='',
                                      alive=lambda:proc.poll() is None)
        clock = iter(range(100, 140))
        with patch.object(scheduler.time, 'monotonic', side_effect=lambda:next(clock)), \
             patch.object(scheduler.time, 'sleep'):
            self.s.shutdown(0)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        self.assertEqual(proc.wait.call_count, 2)
        self.assertFalse(self.s.slam.alive())


if __name__ == '__main__':
    unittest.main()
