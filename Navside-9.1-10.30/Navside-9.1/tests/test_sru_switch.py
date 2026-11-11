"""Scheduler-only operation: real A*, fake SLAM IO, no robot processes."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import task_nav_scheduler as scheduler
from task_nav_astar import SegmentAStarPlanner


class SruSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = scheduler.load_config(str(ROOT/'config/task_nav.yaml'))
        cls.planner = SegmentAStarPlanner(cls.config)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = copy.deepcopy(self.config)
        self.cfg['sru']['enabled'] = False
        self.cfg['task_points'] = [(-6.6, .695, 5.2), (-30.2, .695, 5.25)]
        self.s = scheduler.TaskNavScheduler(self.cfg, SimpleNamespace())
        self.s.planner = self.planner
        self.s.plan_output_dir = Path(self.tmp.name)/'plans'
        self.s.log = Mock()
        self.s.slam = SimpleNamespace(tag='SLAM', reference_panel_text='', send=Mock(return_value=True))
        self.s.nav_send = Mock(side_effect=AssertionError('SRU command forbidden'))
        self.addCleanup(self.s._cancel_plan)
        guard = patch.object(scheduler.subprocess, 'Popen', side_effect=AssertionError('hardware forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def anchor(self, pose=(-.5, 0, 2, 0, 0, 0, 1)):
        self.s.handle_sched('slam', {'anchor': 'busy'})
        self.s.handle_sched('slam', dict(anchor='ok', pose='('+','.join(map(str, pose))+')'))
        until = time.monotonic()+10
        while self.s.state == scheduler.STATE_PLANNING and time.monotonic() < until:
            self.s.poll_planning()
            time.sleep(.01)
        self.assertNotEqual(self.s.state, scheduler.STATE_PLANNING)

    def vio_arrive(self):
        x, y, z = self.cfg['task_points'][self.s.idx]
        self.cfg['arrive_confirm_s'] = 0
        for _ in range(2):
            self.s.handle_sched('slam', dict(pose=f'({x},{y},{z},0,0,0,1)', t=str(time.time())))
        self.assertEqual(self.s.state, scheduler.STATE_ARRIVED)

    def test_full_manual_route_preserves_endpoints_exports_and_real_arrival(self):
        self.anchor()
        self.assertEqual(self.s.state, scheduler.STATE_SEGMENT)
        route = self.s.current_route
        self.assertEqual(len(route['references_xy']), 15)
        self.assertEqual(route['references_xy'][0], route['start_xy'])
        self.assertEqual(route['references_xy'][-1], route['goal_xy'])
        saved = json.loads((self.s.path_directory/'route.json').read_text())
        self.assertFalse(saved['scheduler']['sent_to_sru'])
        self.assertFalse(saved['scheduler']['sru_enabled'])
        self.assertFalse((self.s.path_directory/'segment_delivery.json').exists())
        self.assertIn('SRU OFF', self.s.slam.reference_panel_text)
        self.assertEqual(self.s.slam.reference_panel_text.count('/15:'), 15)
        # Neither elapsed delivery timeouts nor stray NavSide events finish a segment.
        self.s.path_deadline = 0
        self.s.poll_path_timeout()
        self.s.handle_sched('nav', dict(zero_reason='goal_reached', __exit__='1'))
        self.s.handle_user('send nav S')
        self.assertEqual(self.s.state, scheduler.STATE_SEGMENT)
        self.assertFalse(self.s.done)
        self.vio_arrive()
        self.assertEqual(self.s.idx, 0)
        self.s.handle_user('localize')
        self.anchor((-6.7, .42, 5.25, 0, 0, 0, 1))
        self.assertEqual(self.s.idx, 1)
        self.assertEqual(self.s.current_route['start_xy'], [-6.7, 5.25])
        self.vio_arrive()
        self.s.handle_user('localize')
        self.anchor((-30.2, .2, 5.25, 0, 0, 0, 1))
        self.assertTrue(self.s.done)
        self.s.nav_send.assert_not_called()

    def test_pause_failure_and_slam_crash_keep_unfinished_target(self):
        self.anchor()
        self.s.handle_user('pause')
        self.assertEqual(self.s.state, scheduler.STATE_PAUSED)
        self.s.handle_user('localize')
        self.anchor((-1, 0, 2.1, 0, 0, 0, 1))
        self.assertEqual(self.s.state, scheduler.STATE_SEGMENT)
        self.assertEqual(self.s.idx, 0)
        self.s.handle_user('pause')
        self.s.handle_user('localize')
        self.anchor((10000, 0, 10000, 0, 0, 0, 1))
        self.assertEqual(self.s.state, scheduler.STATE_PLAN_FAILED)
        self.s.handle_user('force')
        self.assertEqual(self.s.idx, 0)
        self.s.handle_user('localize')
        self.anchor()
        self.s.handle_sched('slam', {'__exit__': '1'})
        self.assertEqual(self.s.state, scheduler.STATE_RESTART_WAIT)
        self.assertTrue(self.s.slam_restart_pending)
        self.assertEqual(self.s.idx, 0)
        self.assertIsNone(self.s.current_route)
        self.s.nav_send.assert_not_called()

    def test_off_preflight_does_not_require_downstream_config(self):
        del self.cfg['navside_cmd']
        self.s.args = SimpleNamespace(task_points_file=str(
            ROOT/'logs/task_nav/astar_offline_demo_20260929/demo_goals.json'))
        with patch.object(scheduler, 'SegmentAStarPlanner', return_value=self.planner):
            self.assertTrue(self.s.prepare_planning())
        self.assertEqual(len(self.cfg['task_points']), 3)

    def test_run_skips_nav_launch_channels_ready_wait_and_shutdown_wait(self):
        del self.cfg['navside_cmd']
        self.cfg['boot_ready_timeout_s'] = 0
        self.cfg['astar']['output_directory'] = str(Path(self.tmp.name)/'run_plans')
        fake_slam = SimpleNamespace(tag='SLAM', reference_panel_text='', alive=lambda: False,
            start=lambda: self.s.events.put(('slam', {'__ready__': '1'})))
        def finish_iteration():
            self.assertEqual(self.s.state, scheduler.STATE_WAIT_STARTUP_ANCHOR)
            self.s.done = True
        with patch.object(self.s, 'prepare_planning', return_value=True), \
                patch.object(self.s, '_spawn_nav_terminal', side_effect=AssertionError('NavSide launch forbidden')), \
                patch.object(self.s, 'poll_planning', side_effect=finish_iteration) as poll, \
                patch.object(scheduler, '__file__', str(Path(self.tmp.name)/'scripts/scheduler.py')), \
                patch.object(scheduler, 'ChildProc', return_value=fake_slam), \
                patch.object(scheduler, 'NavSideChild', side_effect=AssertionError('NavSide channel forbidden')), \
                patch.object(scheduler.threading, 'Thread'), \
                patch.object(scheduler.signal, 'signal'), \
                patch.object(scheduler, 'open_viewer_window', return_value=False), \
                patch.object(scheduler.time, 'sleep') as sleep:
            self.assertEqual(self.s.run(), 0)
            poll.assert_called_once()
            self.assertEqual([c.args for c in sleep.call_args_list], [(.05,)])
        self.assertIsNone(self.s.nav)
        self.assertEqual(self.s.nav_cmd_file, '')
        self.assertFalse(list(self.s.log_dir.glob('nav_*.log')))
        metadata = json.loads((self.s.plan_output_dir/'session.json').read_text())
        self.assertFalse(metadata['sru_enabled'])
        self.assertFalse(metadata['sru_reference_input_connected'])

    def test_cli_overrides_yaml_and_invalid_switch_is_rejected(self):
        path = Path(self.tmp.name)/'config.yaml'
        for stored, flag, expected in [(True, '--no-sru', False), (False, '--sru', True),
                                       (False, None, False), (True, None, True)]:
            path.write_text(scheduler.yaml.safe_dump({'sru': {'enabled': stored}}))
            args = ['scheduler', '--config', str(path), '--check-only'] + ([flag] if flag else [])
            with patch.object(sys, 'argv', args), patch.object(scheduler, 'TaskNavScheduler') as cls:
                cls.return_value.run.return_value = 0
                self.assertEqual(scheduler.main(), 0)
                self.assertIs(cls.call_args.args[0]['sru']['enabled'], expected)
        for invalid in ['false', 0, None]:
            path.write_text(scheduler.yaml.safe_dump({'sru': {'enabled': invalid}}))
            with self.assertRaisesRegex(ValueError, 'sru.enabled'):
                scheduler.load_config(str(path))


if __name__ == '__main__':
    unittest.main()
