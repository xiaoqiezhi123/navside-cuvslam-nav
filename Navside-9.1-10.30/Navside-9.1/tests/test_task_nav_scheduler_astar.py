"""Exercise real A* through scheduler events; all process/command IO is replaced."""
import copy
import io
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import task_nav_scheduler as scheduler
from task_nav_astar import SegmentAStarPlanner


def pose_text(pose):
    return '('+','.join(str(v) for v in pose)+')'


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = scheduler.load_config(str(ROOT/'config/task_nav_legacy.yaml'))
        cls.planner = SegmentAStarPlanner(cls.config)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cfg = copy.deepcopy(self.config)
        cfg['task_points'] = [(-6.6, .695, 5.2), (-30.2, .695, 5.25)]
        self.s = scheduler.TaskNavScheduler(cfg, SimpleNamespace())
        self.s.planner = self.planner
        self.s.plan_output_dir = Path(self.tmp.name)/'plans'
        self.s.log_dir = Path(self.tmp.name)
        self.s.slam = SimpleNamespace(tag='SLAM')
        self.s.nav = SimpleNamespace(tag='NAV')
        self.history = []
        self.panels_at_resume = []
        self.s.log = lambda message: self.history.append(('log', message))

        def send(child, message):
            self.history.append(('send', child.tag, message))
            if child.tag == 'SLAM' and message == 'resume':
                self.panels_at_resume.append(getattr(child, 'reference_panel_text', ''))
            return True
        self.s.send = send
        self.addCleanup(self.s._cancel_plan)
        # Tests must fail immediately if they accidentally start real processes.
        self.process_guard = patch.object(scheduler.subprocess, 'Popen', side_effect=AssertionError('robot process forbidden in test'))
        self.process_guard.start()
        self.addCleanup(self.process_guard.stop)

    def commands(self):
        return [(item[1], item[2]) for item in self.history if item[0] == 'send']

    def anchor(self, pose, generation=None):
        marker = {} if generation is None else {'__generation__': str(generation)}
        self.s.handle_sched('slam', dict(anchor='busy', **marker))
        self.s.handle_sched('slam', dict(anchor='ok', pose=pose_text(pose), **marker))

    def await_plan(self):
        until = time.monotonic()+10
        while self.s.state == scheduler.STATE_PLANNING and time.monotonic() < until:
            with patch.object(scheduler.time, 'sleep', return_value=None):
                self.s.poll_planning()
            time.sleep(.01)
        self.assertNotEqual(self.s.state, scheduler.STATE_PLANNING, 'planner failed to finish')

    def arrive(self):
        with patch.object(scheduler.time, 'sleep', return_value=None):
            self.s.arrive()

    def test_startup_next_segment_and_final_completion(self):
        self.s.handle_sched('slam', {'__ready__': '1'})
        start = (-.5, -.21, 2., 0, 0, 0, 1)
        self.anchor(start)
        self.await_plan()
        self.assertEqual(self.s.idx, 0)
        self.assertEqual(self.s.current_route['start_xy'], [-.5, 2.])
        self.assertEqual(self.s.current_route['goal_xy'], [-6.6, 5.2])
        # The SLAM viewer must retain all 15 points across repeated ANSI clears.
        panel = self.panels_at_resume[-1]
        references = scheduler.reference_lines(self.s.current_route)
        self.assertEqual(panel.count('/15:'), 15)
        self.assertTrue(all(line in panel for line in references))
        self.assertIn('定位 #1', panel)
        fake_stdout = io.StringIO(2 * (scheduler.SLAM_PANEL_PREFIX + '=== cuVSLAM ===\n'
                                      '位姿: test\n[SCHED] pose=(0,0,0,0,0,0,1)\n'))
        viewer_child = SimpleNamespace(proc=SimpleNamespace(stdout=fake_stdout),
                                       reference_panel_text=panel)
        viewer_log = Path(self.tmp.name)/'slam_viewer.log'
        scheduler.reader_loop(viewer_child, scheduler.queue.Queue(), 'slam', viewer_log)
        frames = viewer_log.read_text(encoding='utf-8').split(scheduler.SLAM_PANEL_PREFIX)[1:]
        self.assertEqual(len(frames), 2)
        for frame in frames:
            self.assertEqual(frame.count('/15:'), 15)
            self.assertIn('位姿: test', frame)
            self.assertNotIn('[SCHED]', frame)
        self.assertIn(('NAV', 'goal 5.200 6.600 0.695'), self.commands())
        self.assertEqual([cmd for target, cmd in self.commands() if target == 'NAV'], ['A', 'goal 5.200 6.600 0.695', 'S'])
        # A rolling VIO pose must never replace the next segment's actual anchor.
        self.s.slam_state['pose'] = '(99,99,99,0,0,0,1)'
        self.arrive()
        self.assertIsNone(self.s.anchor_pose)
        self.assertEqual(self.s.slam.reference_panel_text, '')
        self.s.handle_user('force')
        self.assertEqual(self.s.idx, 0)
        next_start = (-6.7, .42, 5.25, 0, 0, 0, 1)
        self.s.handle_user('localize')
        self.anchor(next_start)
        self.await_plan()
        self.assertEqual(self.s.idx, 1)
        self.assertEqual(self.s.current_route['start_xy'], [-6.7, 5.25])
        self.assertEqual(self.s.current_route['goal_xy'], [-30.2, 5.25])
        self.assertEqual(self.s.current_route['scheduler']['anchor_pose_xyz_qxyzw'], list(next_start))
        self.assertIn('A* 第 2 段', self.panels_at_resume[-1])
        self.assertIn('定位 #2', self.panels_at_resume[-1])
        self.assertNotIn('段 1 参考点', self.panels_at_resume[-1])
        self.arrive()
        self.anchor((-30.2, .3, 5.25, 0, 0, 0, 1))
        self.assertTrue(self.s.done)
        self.assertEqual(len(list(self.s.plan_output_dir.glob('segment_*'))), 2)

    def test_early_startup_busy_before_ready_is_not_lost(self):
        self.anchor((-.5, 0, 2, 0, 0, 0, 1))
        self.await_plan()
        self.assertEqual(self.s.state, scheduler.STATE_SEGMENT)
        self.assertEqual(self.s.idx, 0)

    def test_invalid_startup_pose_does_not_launch_or_force(self):
        self.s.handle_sched('slam', {'anchor': 'busy'})
        self.s.handle_sched('slam', {'anchor': 'ok', 'pose': '(1,nan,2,0,0,0,1)'})
        self.assertEqual(self.s.state, scheduler.STATE_WAIT_STARTUP_ANCHOR)
        self.assertIsNone(self.s.anchor_pose)
        self.s.handle_user('force')
        self.assertEqual(self.s.idx, 0)
        self.assertNotIn(('NAV', 'S'), self.commands())
        self.assertNotIn(('SLAM', 'resume'), self.commands())

    def test_failed_plan_stops_and_reanchor_retries_same_target(self):
        self.anchor((1000, 0, 1000, 0, 0, 0, 1))
        self.await_plan()
        self.assertEqual(self.s.state, scheduler.STATE_PLAN_FAILED)
        self.assertEqual(self.s.slam.reference_panel_text, '')
        self.assertNotIn(('NAV', 'S'), self.commands())
        self.assertFalse(any(cmd.startswith('goal ') for _, cmd in self.commands()))
        self.assertNotIn(('SLAM', 'resume'), self.commands())
        self.s.handle_user('force')
        self.assertEqual(self.s.idx, 0)
        self.s.handle_user('localize')
        self.anchor((-.5, -.3, 2, 0, 0, 0, 1))
        self.await_plan()
        self.assertEqual(self.s.idx, 0)
        self.assertEqual(self.s.state, scheduler.STATE_SEGMENT)
        self.assertEqual(self.s.current_route['scheduler']['anchor_sequence'], 2)
        self.assertTrue((self.s.plan_output_dir/'failures.jsonl').is_file())

    def test_restart_keeps_current_goal_and_rejects_old_instance(self):
        self.s.idx = 1
        self.s.state = scheduler.STATE_SEGMENT
        self.s.handle_sched('slam', {'__exit__': '1', '__generation__': '0'})
        self.assertEqual(self.s.state, scheduler.STATE_RESTART_WAIT)
        self.assertIn(('NAV', 'A'), self.commands())
        self.s.handle_sched('slam', {'__exit__': '1', '__generation__': '0'})
        self.assertEqual(self.s.slam_crashes, 1)
        self.s.slam_restart_pending = False
        fake_child = SimpleNamespace(start=lambda: None, tag='SLAM')
        with patch.object(scheduler, 'ChildProc', return_value=fake_child), \
             patch.object(scheduler.threading.Thread, 'start', return_value=None), \
             patch.object(scheduler, 'open_viewer_window', return_value=False):
            self.s._restart_slam()
        self.assertEqual(self.s.idx, 1)
        self.anchor((-1, 0, 2, 0, 0, 0, 1), generation=0)
        self.assertEqual(self.s.state, scheduler.STATE_WAIT_STARTUP_ANCHOR)
        new_pose = (-6.6, -.4, 5.2, 0, 0, 0, 1)
        self.anchor(new_pose, generation=1)
        self.await_plan()
        self.assertEqual(self.s.idx, 1)
        self.assertFalse(self.s.done)
        self.assertEqual(self.s.current_route['start_xy'], [-6.6, 5.2])
        self.assertEqual(self.s.current_route['goal_xy'], [-30.2, 5.25])
        self.assertEqual(self.panels_at_resume[-1].count('/15:'), 15)
        self.assertIn('A* 第 2 段', self.panels_at_resume[-1])

    def test_crash_invalidates_inflight_result(self):
        release = threading.Event()
        started = threading.Event()
        result = self.planner.plan((-.5, 0, 2, 0, 0, 0, 1), self.s.cfg['task_points'][0], 0, anchor_sequence=1)
        def slow_plan(*args, **kwargs):
            started.set()
            release.wait(3)
            return result
        with patch.object(self.planner, 'plan', side_effect=slow_plan):
            self.anchor((-.5, 0, 2, 0, 0, 0, 1))
            self.assertTrue(started.wait(1))
            generation, sequence = self.s.plan_generation, self.s.anchor_sequence
            self.s.handle_sched('slam', {'__exit__': '1'})
            release.set()
            self.s.plan_results.put((generation, 0, sequence, (result, Path(self.tmp.name))))
            self.s.poll_planning()
            self.assertNotIn(('NAV', 'S'), self.commands())
            self.assertNotIn(('SLAM', 'resume'), self.commands())
            self.assertIsNone(self.s.current_route)

    def test_quit_during_resume_delay_never_sends_go(self):
        self.s.state = scheduler.STATE_PLANNING
        with patch.object(scheduler.time, 'sleep', side_effect=lambda _: self.s.handle_user('quit')):
            self.s._resume_segment(0)
        self.assertTrue(self.s.done)
        self.assertNotIn(('NAV', 'S'), self.commands())
        self.assertFalse(any(cmd.startswith('goal ') for _, cmd in self.commands()))

    def test_old_nav_arrival_does_not_finish_new_segment(self):
        self.s.state = scheduler.STATE_SEGMENT
        self.s.segment_ready_at = time.time()
        self.s.handle_sched('nav', dict(mode='LOW_SPEED', zero_reason='goal_reached',
                                      __received_at__=str(self.s.segment_ready_at-1)))
        self.assertEqual(self.s.state, scheduler.STATE_SEGMENT)
        self.assertEqual(self.commands(), [])

    def test_picker_cancel_and_check_failure_never_start_processes(self):
        with patch.object(self.s, 'prepare_planning', return_value=False):
            self.assertEqual(self.s.run(), 0)
        self.assertEqual(self.commands(), [])
        with patch.object(self.s, 'prepare_planning', side_effect=ValueError('wrong map')):
            self.assertEqual(self.s.run(), 2)
        self.assertEqual(self.commands(), [])


if __name__ == '__main__':
    unittest.main()
