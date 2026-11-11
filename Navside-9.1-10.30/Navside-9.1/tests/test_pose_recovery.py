"""Deterministic dropout/recovery tests, with no robot or camera access."""
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navside.bridge import NavStatePacketV2
from navside.mode import NavMode, NavModeController
from navside.pose_health import PoseHealthMonitor, PoseRecoveryConfig
import test_pathaware as base
from test_pathaware import payload, scheduler, ROOT
import navside.real as real
import navside.mode as mode
from navside.runtime import load_nav_config


def packet(seq=1, at=100., x=0., yaw=0.):
    return NavStatePacketV2(pose_sequence=seq, received_monotonic=at,
        robot_pos_w=np.array([x, 0., .695]),
        robot_quat_wxyz=np.array([math.cos(yaw/2), 0., 0., math.sin(yaw/2)]))


class PoseMonitorTests(unittest.TestCase):
    def setUp(self):
        self.monitor = PoseHealthMonitor()
        self.first = packet()
        self.assertEqual(self.monitor.update(self.first, 100.)[0], 'ok')

    def begin_wait(self):
        self.assertEqual(self.monitor.update(self.first, 101.1)[0], 'wait')

    def test_short_dropout_requires_distinct_stable_samples(self):
        self.begin_wait()
        for seq, at in enumerate((101.4, 101.6, 101.8), 2):
            p = packet(seq, at, x=.3)
            self.assertEqual(self.monitor.update(p, at)[0], 'wait')
            for _ in range(10):
                self.assertEqual(self.monitor.update(p, at+.001)[0], 'wait')
        action, _, details = self.monitor.update(packet(5, 102., x=.3), 102.)
        self.assertEqual(action, 'resume')
        self.assertEqual(details['stable_samples'], 4)
        self.assertEqual(self.monitor.update(packet(6, 102.2, x=.4), 102.2)[0], 'ok')

    def test_gap_in_recovery_restarts_stability(self):
        self.begin_wait()
        for seq, at in enumerate((101.4, 101.6, 102.0, 102.2, 102.4), 2):
            self.assertEqual(self.monitor.update(packet(seq, at), at)[0], 'wait')
        self.assertEqual(self.monitor.update(packet(7, 102.6), 102.6)[0], 'resume')

    def test_cache_cannot_recover_even_after_enough_wall_time(self):
        self.begin_wait()
        p = packet(2, 101.4)
        self.monitor.update(p, 101.4)
        for at in (101.5, 101.7, 102., 103.):
            self.assertEqual(self.monitor.update(p, at)[0], 'wait')
        self.assertEqual(self.monitor.stable_samples, 0)

    def test_missing_data_timeout_and_late_fresh_packet(self):
        for latest in (None, packet(2, 106.1)):
            with self.subTest(latest=latest is None):
                monitor = PoseHealthMonitor()
                monitor.update(self.first, 100.)
                monitor.update(None, 101.1)
                self.assertEqual(monitor.update(latest, 106.1)[:2],
                                 ('error', 'pose_timeout_requires_localize'))

    def test_outage_detected_even_if_loop_only_observes_fresh_packet(self):
        self.assertEqual(self.monitor.update(packet(2, 101.5), 101.5)[0], 'wait')
        monitor = PoseHealthMonitor()
        monitor.update(self.first, 100.)
        self.assertEqual(monitor.update(packet(2, 107.), 107.)[:2],
                         ('error', 'pose_timeout_requires_localize'))

    def test_position_rotation_and_sequence_changes_latch_error(self):
        for p in (packet(2, 101.4, x=1.01), packet(2, 101.4, yaw=math.radians(61)),
                  packet(1, 101.4), packet(2, 99.)):
            with self.subTest(packet=p):
                monitor = PoseHealthMonitor()
                monitor.update(self.first, 100.)
                monitor.update(self.first, 101.1)
                self.assertEqual(monitor.update(p, 101.4)[0], 'error')
        # A jump during normal running is also rejected.
        self.assertEqual(self.monitor.update(packet(2, 100.1, x=2.), 100.1)[0], 'error')

    def test_recovery_compares_with_pre_outage_pose_not_just_neighbours(self):
        self.begin_wait()
        self.monitor.update(packet(2, 101.4, x=.6), 101.4)
        self.assertEqual(self.monitor.update(packet(3, 101.6, x=1.1), 101.6)[0], 'error')

    def test_quaternion_sign_change_is_same_orientation(self):
        p = packet(2, 100.1)
        p.robot_quat_wxyz *= -1
        self.assertEqual(self.monitor.update(p, 100.1)[0], 'ok')

    def test_recorded_catchup_offsets_wait_for_settling_then_resume(self):
        # 06:46 and 06:17 runs: cumulative travel while pose output was delayed.
        for stale_age, total in ((1.0359, 1.0033), (1.0069, 1.5102)):
            with self.subTest(total=total):
                monitor = PoseHealthMonitor()
                monitor.update(self.first, 100.)
                action, _, details = monitor.update(self.first, 100.+stale_age, max_linear_speed_mps=.68)
                self.assertEqual(action, 'wait')
                self.assertAlmostEqual(details['wait_position_limit_m'], 1.+.68*stale_age, places=4)
                for seq, (at, x) in enumerate(((101.59, .1), (101.65, .6),
                                               (101.76, .9), (101.8583, total)), 2):
                    self.assertEqual(monitor.update(packet(seq, at, x=x), at)[0], 'wait')
                for seq, at in enumerate((102.0583, 102.2583), 6):
                    self.assertEqual(monitor.update(packet(seq, at, x=total), at)[0], 'wait')
                self.assertEqual(monitor.update(packet(8, 102.4583, x=total), 102.4583)[0], 'resume')

    def test_allowance_frozen_at_zero_and_cumulative_limit_still_enforced(self):
        self.monitor.update(self.first, 101.1, max_linear_speed_mps=.68)
        limit = self.monitor.wait_position_limit
        self.monitor.update(packet(2, 101.4, x=.9), 101.4, max_linear_speed_mps=5.)
        self.assertEqual(self.monitor.wait_position_limit, limit)
        self.monitor.update(packet(3, 102., x=1.6), 102.)
        action, reason, details = self.monitor.update(packet(4, 103., x=1.8), 103.)
        self.assertEqual((action, reason), ('error', 'pose_jump_requires_localize'))
        self.assertEqual(details['jump_reference'], 'pre_wait_pose')
        self.assertAlmostEqual(details['position_limit_m'], 1.748)

    def test_true_adjacent_jump_not_hidden_by_outage_allowance(self):
        self.monitor.update(self.first, 101.1, max_linear_speed_mps=.68)
        self.monitor.update(packet(2, 101.4, x=.2), 101.4)
        action, _, details = self.monitor.update(packet(3, 101.6, x=1.3), 101.6)
        self.assertEqual(action, 'error')
        self.assertEqual(details['jump_reference'], 'previous_sample')
        self.assertEqual(details['position_limit_m'], 1.)

    def test_loop_that_missed_stale_interval_still_waits_for_stable_catchup(self):
        action, _, details = self.monitor.update(packet(2, 101.5, x=1.2), 101.5,
                                                max_linear_speed_mps=.68)
        self.assertEqual(action, 'wait')
        self.assertEqual(details['wait_position_limit_m'], 2.02)

    def test_continuing_position_or_heading_changes_cannot_auto_resume(self):
        for change in ('position', 'heading'):
            with self.subTest(change=change):
                monitor = PoseHealthMonitor()
                monitor.update(self.first, 100.)
                monitor.update(self.first, 101.1, max_linear_speed_mps=.68)
                for seq in range(2, 24):
                    at = 101.4+(seq-2)*.2
                    p = packet(seq, at, x=.3*(seq % 2) if change == 'position' else 0.,
                               yaw=math.radians(12)*(seq % 2) if change == 'heading' else 0.)
                    self.assertEqual(monitor.update(p, at)[0], 'wait')
                self.assertEqual(monitor.update(packet(24, 106.1), 106.1)[:2],
                                 ('error', 'pose_timeout_requires_localize'))

    def test_invalid_fields_fail_without_waiting(self):
        cases = [('robot_pos_w', np.array([np.nan, 0., 0.])),
                 ('robot_pos_w', np.zeros(2)), ('robot_quat_wxyz', np.zeros(4)),
                 ('robot_quat_wxyz', np.array([2., 0, 0, 0])),
                 ('linear_vel_b', np.full(3, np.inf)), ('angular_vel_b', np.full(3, np.nan)),
                 ('received_monotonic', float('nan')), ('received_monotonic', 100.2),
                 ('pose_sequence', -1)]
        for field, value in cases:
            p = packet(2, 100.1)
            setattr(p, field, value)
            with self.subTest(field=field, value=value):
                self.assertEqual(self.monitor.update(p, 100.1)[:2],
                                 ('error', 'pose_invalid_requires_localize'))

    def test_expired_inference_enters_wait_even_if_newest_pose_is_fresh(self):
        self.assertEqual(self.monitor.update(packet(2, 100.1), 100.1,
                         force_wait_reason='inference_pose_expired')[:2], ('wait', 'inference_pose_expired'))

    def test_invalid_config_rejected(self):
        for settings in ({'timeout_s': -1}, {'stable_s': 5}, {'min_samples': 1},
                         {'min_samples': 2.5}, {'max_sample_gap_s': float('nan')},
                         {'max_position_jump_m': 0}, {'max_rotation_jump_deg': 181},
                         {'stable_position_span_m': 0}, {'stable_rotation_span_deg': 181}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PoseRecoveryConfig(**settings)


class ModePoseRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.nav = NavModeController(path_aware=True, session_id='test')
        self.reset = Mock()
        self.nav.apply_sched_line('segment_load '+json.dumps(payload()))
        self.nav.process_segments(reset=self.reset, pose_sequence=0)
        self.nav.apply_sched_line('segment_start test:1')
        self.nav.process_segments(reset=self.reset, pose_sequence=1, pose_fresh=True)
        self.check(packet(), 100.)
        self.old_segment, self.old_epoch, _ = self.nav.path_snapshot()

    def check(self, p, at):
        return self.nav.check_path_pose(p, reset=self.reset, now=at)

    def stable(self):
        for seq, at in enumerate((101.4, 101.6, 101.8, 102.), 2):
            self.check(packet(seq, at, x=.3), at)

    def test_wait_sends_zero_then_resumes_same_path_speed_and_clears_recurrence(self):
        self.nav.apply_line('D')
        self.assertFalse(self.check(packet(), 101.1))
        self.assertEqual(self.nav.path_phase(), 'waiting_pose')
        sender = Mock()
        self.nav.guarded_path_send(self.old_epoch, sender, [1, 0, 1])
        np.testing.assert_allclose(sender.call_args.args, 0.)
        self.stable()
        segment, epoch, running = self.nav.path_snapshot()
        self.assertIs(segment, self.old_segment)
        self.assertTrue(running)
        self.assertEqual(self.nav.get_mode(), NavMode.MEDIUM_SPEED)
        self.assertEqual(self.reset.call_count, 2)
        self.assertGreater(epoch, self.old_epoch)
        self.nav.guarded_path_send(self.old_epoch, sender, [1, 0, 1])
        np.testing.assert_allclose(sender.call_args.args, 0.)
        self.nav.guarded_path_send(epoch, sender, [1, 0, 1])
        np.testing.assert_allclose(sender.call_args.args, [.8, 0, .45])
        self.assertEqual([e['path'] for e in self.nav.poll_path_events()], ['waiting_pose', 'pose_resumed'])

    def test_manual_stop_or_scheduler_pause_while_waiting_cannot_auto_resume(self):
        for key in ('A', 'F', 'G', 'scheduler_A'):
            with self.subTest(key=key):
                self.setUp()
                self.check(packet(), 101.1)
                if key == 'scheduler_A':
                    self.nav.apply_sched_line('A')
                else:
                    self.nav.apply_line(key)
                self.stable()
                self.assertFalse(self.nav.path_snapshot()[2])
                self.assertNotIn('pose_resumed', [e['path'] for e in self.nav.poll_path_events()])
                self.assertFalse(self.nav.apply_line('S').accepted)
                # A late old inference check cannot undo the manual stop.
                self.assertFalse(self.nav.check_path_pose(packet(6, 102.2), reset=self.reset,
                    now=102.2, expected_epoch=self.old_epoch, force_wait_reason='inference_pose_expired'))

    def test_hard_fault_does_not_recover_when_stream_returns(self):
        self.check(packet(), 101.1)
        self.check(packet(2, 106.1), 106.1)
        for seq, at in enumerate((106.2, 106.4, 106.6, 106.8), 3):
            self.assertFalse(self.check(packet(seq, at), at))
        self.assertEqual(self.nav.path_phase(), 'error')
        self.assertFalse(self.nav.apply_line('S').accepted)

    def test_actual_mode_speed_allows_catchup_but_holds_zero_until_stable(self):
        self.assertFalse(self.check(packet(), 101.0359))
        events = self.nav.poll_path_events()
        self.assertEqual(events[0]['wait_position_limit_m'], 1.7044)
        sender = Mock()
        for seq, at in enumerate((101.8583, 102.0583, 102.2583), 2):
            self.assertFalse(self.check(packet(seq, at, x=1.0033), at))
            _, epoch, _ = self.nav.path_snapshot()
            self.nav.guarded_path_send(epoch, sender, [1., 0., 0.])
            np.testing.assert_allclose(sender.call_args.args, 0.)
        self.assertTrue(self.check(packet(5, 102.4583, x=1.0033), 102.4583))
        self.assertIs(self.nav.path_snapshot()[0], self.old_segment)
        self.assertEqual(self.reset.call_count, 2)

    def test_new_localized_segment_resets_pose_continuity_baseline(self):
        self.check(packet(2, 100.1, x=5.), 100.1)
        self.assertEqual(self.nav.path_phase(), 'error')
        self.nav.apply_sched_line('segment_load '+json.dumps(payload(2)))
        self.nav.process_segments(reset=self.reset, pose_sequence=2)
        self.nav.apply_sched_line('segment_start test:2')
        self.nav.process_segments(reset=self.reset, pose_sequence=3, pose_fresh=True)
        self.assertTrue(self.check(packet(3, 100.2, x=5.), 100.2))


class SchedulerPoseRecoveryTests(unittest.TestCase):
    setUpClass = classmethod(base.SchedulerPathTests.setUpClass.__func__)
    setUp = base.SchedulerPathTests.setUp
    anchor = base.SchedulerPathTests.anchor
    pump = base.SchedulerPathTests.pump
    running = base.SchedulerPathTests.running

    def begin_wait(self):
        self.anchor()
        self.running()
        self.nav.check_path_pose(packet(), reset=self.reset, now=100.)
        self.nav.check_path_pose(packet(), reset=self.reset, now=101.1)
        self.pump()
        self.assertEqual(self.s.state, scheduler.STATE_WAIT_POSE)

    def test_wait_preserves_path_vio_and_goal_then_resumes_without_redelivery(self):
        self.begin_wait()
        segment_id, route = self.s.path_segment_id, copy.deepcopy(self.s.current_route)
        commands = list(self.sent)
        # Old goal_reached/VIO-at-goal during waiting cannot advance the task.
        self.s.handle_sched('nav', dict(session_id=self.s.session_id, segment_id=segment_id,
                                       mode='STANDBY', zero_reason='goal_reached'))
        self.s.handle_sched('slam', dict(pose='(-6.6,0,5.2,0,0,0,1)', t=str(scheduler.time.time())))
        self.assertEqual(self.s.state, scheduler.STATE_WAIT_POSE)
        for seq, at in enumerate((101.4, 101.6, 101.8, 102.), 2):
            self.nav.check_path_pose(packet(seq, at), reset=self.reset, now=at)
            self.pump()
        self.assertEqual(self.s.state, scheduler.STATE_SEGMENT)
        self.assertEqual(self.s.path_segment_id, segment_id)
        self.assertEqual(self.s.current_route, route)
        self.assertEqual(self.sent, commands)
        self.assertEqual(self.s.idx, 0)
        events = (self.s.path_directory/'delivery_events.jsonl').read_text()
        self.assertIn('waiting_pose', events)
        self.assertIn('pose_resumed', events)

    def test_manual_pause_invalidates_late_recovery_and_localize_keeps_current_target(self):
        self.begin_wait()
        old_id = self.s.path_segment_id
        self.s.handle_user('pause')
        self.assertEqual(self.s.state, scheduler.STATE_PAUSED)
        self.s.handle_sched('nav', dict(path='pose_resumed', session_id=self.s.session_id, segment_id=old_id))
        self.assertEqual(self.s.state, scheduler.STATE_PAUSED)
        self.s.handle_user('localize')
        self.anchor()
        self.running()
        self.assertEqual(self.s.idx, 0)
        self.assertNotEqual(self.s.path_segment_id, old_id)

    def test_long_timeout_requires_localize_not_automatic_restart(self):
        self.begin_wait()
        old_id = self.s.path_segment_id
        self.nav.check_path_pose(None, reset=self.reset, now=106.1)
        self.pump()
        self.assertEqual(self.s.state, scheduler.STATE_PLAN_FAILED)
        self.s.handle_sched('nav', dict(path='pose_resumed', session_id=self.s.session_id, segment_id=old_id))
        self.assertEqual(self.s.state, scheduler.STATE_PLAN_FAILED)
        self.assertEqual(self.s.idx, 0)
        self.assertIn(('SLAM', 'pause'), self.sent)

    def test_unexpected_relocalization_and_slam_restart_invalidate_wait(self):
        for event, expected in (({'anchor': 'busy'}, scheduler.STATE_PLAN_FAILED),
                                ({'anchor': 'ok'}, scheduler.STATE_PLAN_FAILED),
                                ({'__exit__': '1'}, scheduler.STATE_RESTART_WAIT)):
            with self.subTest(event=event):
                self.setUp()
                self.begin_wait()
                old_id = self.s.path_segment_id
                self.s.handle_sched('slam', event)
                self.pump()
                self.s.handle_sched('nav', dict(path='pose_resumed', session_id=self.s.session_id, segment_id=old_id))
                self.assertEqual(self.s.state, expected)
                self.assertFalse(self.nav.path_snapshot()[2])
                self.assertEqual(self.s.idx, 0)


class RealLoopRecoveryTests(unittest.TestCase):
    def run_scenario(self, *, slow_inference=False, long_outage=False):
        class Clock:
            value = 100.
            def time(self): return self.value
            def monotonic(self): return self.value
            def monotonic_ns(self): return int(self.value*1e9)
            def perf_counter(self): return self.value
            def sleep(self, seconds): self.value += max(seconds, .01)
        clock = Clock()
        nav = NavModeController(path_aware=True, session_id='test')
        nav.apply_sched_line('segment_load '+json.dumps(payload()))
        sent, calls, cleanup = [], [], []
        adapter = SimpleNamespace(reset_recurrent_state=Mock(), last_action=np.zeros(3), next_tick=0.)
        adapter.should_tick = lambda now: now >= adapter.next_tick
        def step(**kwargs):
            calls.append((clock.value, kwargs['path_w']))
            if slow_inference and len(calls) == 1:
                clock.sleep(1.2)
            adapter.next_tick = clock.value + .2
            return dict(control=dict(raw_cmd=np.array([.3,0.,0.]), final_cmd=np.array([.3,0.,0.]), zero_reason='none'),
                goal_dist=5., diag=dict(raw_action=np.zeros(3), linear_vel_b=np.zeros(3), angular_vel_b=np.zeros(3),
                    projected_gravity_b=np.array([0,0,-1.]), target_position=np.zeros(4)))
        app = SimpleNamespace(config=load_nav_config(str(ROOT/'config/nav_pathaware.yaml')),
                              adapter=adapter, default_goal=lambda:np.array([5,0,.695]), step=step)
        class Comm:
            started = False
            def start(self): pass
            def stop(self): cleanup.append('transport')
            def load_task(self): return False
            def send_zero(self): self.send_command(0., 0., 0.)
            def send_command(self, *cmd): sent.append((clock.value, nav._segments.phase, tuple(cmd)))
            def get_latest_state(self):
                if nav.path_phase() == 'ready' and not self.started:
                    nav.apply_sched_line('segment_start test:1')
                    self.started = True
                if clock.value > (108.5 if long_outage else 104.5):
                    nav.apply_sched_line('quit')
                slot = int((clock.value-100.)*10 + 1e-6)
                if not slow_inference and 101. <= clock.value < (108. if long_outage else 102.5):
                    slot = 9
                return packet(slot+1, 100.+slot/10.)
        def close_camera():
            self.assertNotIn('process=exit', events.read_text())
            self.assertEqual(cleanup, ['transport'])
            cleanup.append('camera')
        camera = SimpleNamespace(start=lambda:None, close=close_camera,
            read=lambda:SimpleNamespace(success=True, depth_input=np.ones((40,64))))
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp)/'events.log'
            args = SimpleNamespace(config=str(ROOT/'config/nav_pathaware.yaml'), goal=None,
                                   deploy_config=None, csv_dir=None, show_depth=False)
            with patch.dict('os.environ', {'NAVSIDE_SESSION_ID':'test','NAVSIDE_SCHED_FILE':str(events)}), \
                 patch.object(real, 'time', clock), patch.object(mode, 'time', clock), \
                 patch.object(real.NavSideApp, 'from_config', return_value=app), \
                 patch.object(real, 'NavModeController', return_value=nav), \
                 patch.object(nav, 'start_input_thread'), patch.object(nav, 'stop_input_thread'), \
                 patch.object(nav, 'start_sched_input'), patch.object(nav, 'stop_sched_input'), \
                 patch.object(real, 'create_depth_perception', return_value=camera), \
                 patch.object(real, 'RobotComm', return_value=Comm()), patch.object(real, 'CsvTickLogger'), \
                 patch('socket.socket', side_effect=AssertionError('network forbidden')), \
                 patch('subprocess.Popen', side_effect=AssertionError('process forbidden')), \
                 patch('sys.stdout', new_callable=io.StringIO):
                real.run(args)
            self.assertEqual(cleanup, ['transport', 'camera'])
            self.assertIn('process=exit', events.read_text())
            return sent, calls, events.read_text()

    def test_actual_loop_zero_during_wait_and_same_path_after_recovery(self):
        sent, calls, events = self.run_scenario()
        self.assertIn('path=waiting_pose', events)
        self.assertIn('path=pose_resumed', events)
        waiting = [cmd for _, phase, cmd in sent if phase == 'waiting_pose']
        self.assertGreater(len(waiting), 1)
        self.assertTrue(all(np.allclose(cmd, 0) for cmd in waiting))
        self.assertTrue(any(at > 103.1 and cmd[0] > 0 for at, _, cmd in sent))
        self.assertTrue(all(path is calls[0][1] for _, path in calls))
        np.testing.assert_allclose(calls[0][1], payload()['path_w'])
        self.assertTrue(all(at < 101.91 or at >= 103.09 for at, _ in calls))

    def test_slow_inference_output_discarded_before_recovery(self):
        sent, calls, events = self.run_scenario(slow_inference=True)
        self.assertIn('reason=inference_pose_expired', events)
        self.assertIn('path=pose_resumed', events)
        first_send = next(cmd for at, _, cmd in sent if at >= calls[0][0]+1.19)
        np.testing.assert_allclose(first_send, 0.)

    def test_long_outage_stays_zero_after_fresh_packets_return(self):
        sent, calls, events = self.run_scenario(long_outage=True)
        self.assertIn('reason=pose_timeout_requires_localize', events)
        self.assertNotIn('path=pose_resumed', events)
        self.assertTrue(all(np.allclose(cmd, 0) for at, _, cmd in sent if at >= 101.91))
        self.assertTrue(all(at < 101.91 for at, _ in calls))


if __name__ == '__main__':
    unittest.main()
