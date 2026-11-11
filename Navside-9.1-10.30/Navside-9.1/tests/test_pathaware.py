"""Offline PathAware integration; cameras, sockets and robot processes forbidden."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from navside.adapter import SruNavAdapter
from navside.bridge import NavStatePacketV2, RobotComm
from navside.mode import NavMode, NavModeController
from navside.runtime import NavSideApp, load_nav_config
from navside.segments import Segment
from navside.real import state_packet_is_fresh
import navside.real as real
import task_nav_scheduler as scheduler
from task_nav_astar import SegmentAStarPlanner


def payload(revision=1, token=None, session='test'):
    return dict(schema_version=1, session_id=session, revision=revision,
                segment_id=f'{session}:{revision}', frame='navside_zup',
                goal_w=[5.,0.,.695], path_w=[[float(x),0.,.5] for x in np.linspace(0,5,15)],
                resume_token=token)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.nav = NavModeController(path_aware=True, session_id='test')
        self.reset = Mock()

    def pump(self, seq=10, fresh=True):
        return self.nav.process_segments(reset=self.reset, pose_sequence=seq, pose_fresh=fresh)

    def load(self, obj):
        return self.nav.apply_sched_line('segment_load '+json.dumps(obj))

    def start(self, obj=None):
        self.load(obj or payload())
        self.assertEqual(self.pump()[-1]['path'], 'ready')
        self.nav.apply_sched_line('segment_start '+(obj or payload())['segment_id'])
        self.assertEqual(self.pump(seq=11)[-1]['path'], 'running')

    def test_loading_never_starts_and_start_waits_for_new_pose(self):
        self.load(payload())
        self.assertEqual(self.nav.get_mode(), NavMode.STANDBY)
        self.pump(seq=10)
        self.nav.apply_sched_line('segment_start test:1')
        self.assertEqual(self.pump(seq=10), [])
        self.assertFalse(self.nav.path_snapshot()[2])
        self.assertEqual(self.pump(seq=11, fresh=False), [])
        self.assertEqual(self.pump(seq=11)[-1]['path'], 'running')
        self.assertEqual(self.nav.get_mode(), NavMode.LOW_SPEED)
        self.reset.assert_called_once()
        self.load(payload())
        self.pump(seq=12)
        self.reset.assert_called_once()  # duplicate delivery does not reset h/c

    def test_manual_stop_blocks_queued_and_delayed_start_until_new_authorized_load(self):
        self.start()
        _, old_epoch, _ = self.nav.path_snapshot()
        self.nav.apply_line('F')
        token = self.pump()[-1]['manual_token']
        self.nav.apply_sched_line('A')  # scheduler stop cannot clear manual latch
        self.nav.apply_sched_line('segment_start test:1')
        self.pump(seq=12)
        sent = []
        self.nav.guarded_path_send(old_epoch, lambda *cmd: sent.append(cmd), [.4,0,.2])
        np.testing.assert_array_equal(sent[-1], [0,0,0])
        self.load(payload(2))
        self.assertEqual(self.pump()[-1]['reason'], 'manual_stop_requires_localize')
        self.start(payload(2, int(token)))
        self.nav.apply_sched_line('segment_start test:1')
        self.assertEqual(self.pump(seq=20)[-1]['path'], 'rejected')
        self.assertTrue(self.nav.path_snapshot()[2])
        self.assertEqual(self.reset.call_count, 2)

    def test_manual_stop_after_load_queued_does_not_install_or_release(self):
        self.load(payload())
        self.nav.apply_line('A')
        self.pump()
        self.assertIsNone(self.nav.path_snapshot()[0])
        self.assertEqual(self.reset.call_count, 0)
        self.assertFalse(self.nav.apply_sched_line('S').accepted)
        self.assertFalse(self.nav.apply_line('S').accepted)
        self.assertFalse(self.nav.apply_sched_line('goal 1 2 .695').accepted)

    def test_invalid_and_cross_session_payloads_never_run(self):
        for changes in [dict(path_w=[[0,0,.5]]), dict(frame='cuvslam'),
                        dict(goal_w=[1,2,.695]), dict(path_w=[[0,0,float('nan')]]*15),
                        dict(path_w=[[0,0,.695]]*15), dict(revision=True)]:
            with self.subTest(changes=changes):
                value=payload()
                value.update(changes)
                with self.assertRaises(ValueError):
                    Segment.parse(value)
        self.load(payload(session='foreign'))
        self.assertEqual(self.pump()[-1]['reason'], 'wrong_session')
        self.assertIsNone(self.nav.path_snapshot()[0])
        self.nav.apply_sched_line('segment_load {bad json')
        self.assertEqual(self.pump()[-1]['path'], 'error')

    def test_legacy_parser_does_not_treat_segment_commands_as_s(self):
        nav=NavModeController()
        self.assertFalse(nav.apply_sched_line('segment_load '+json.dumps(payload())).accepted)
        self.assertEqual(nav.get_mode(), NavMode.STANDBY)
        self.assertTrue(nav.apply_sched_line('goal 5 0 .695').accepted)
        self.assertTrue(nav.apply_sched_line('S').accepted)

    def test_receipt_metadata_stays_old_when_read_again(self):
        comm=RobotComm.__new__(RobotComm)  # never construct sockets / HTTP clients
        comm._odom_lock=threading.Lock()
        comm._vel_lock=threading.Lock()
        comm._lin_vel=np.zeros(3)
        comm._ang_vel=np.zeros(3)
        comm._pose_sequence=0
        comm._pose_received_monotonic=0.
        comm._seq=0
        comm._compute_velocities=Mock()
        comm._handle_pose(np.zeros(3), np.array([1.,0,0,0]))
        first, second=comm.get_latest_state(), comm.get_latest_state()
        self.assertEqual(first.pose_sequence, second.pose_sequence)
        self.assertEqual(first.received_monotonic, second.received_monotonic)
        self.assertTrue(state_packet_is_fresh(second, 1, now=first.received_monotonic+.1))
        self.assertFalse(state_packet_is_fresh(second, 1, now=first.received_monotonic+2))


class SchedulerPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=scheduler.load_config(str(ROOT/'config/task_nav.yaml'))
        cls.planner=SegmentAStarPlanner(cls.cfg)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cfg=copy.deepcopy(self.cfg)
        cfg['task_points']=[(-6.6,.695,5.2),(-30.2,.695,5.25)]
        self.s=scheduler.TaskNavScheduler(cfg, SimpleNamespace())
        self.s.planner=self.planner
        self.s.plan_output_dir=Path(self.tmp.name)/'plans'
        self.s.log_dir=Path(self.tmp.name)
        self.s.slam=SimpleNamespace(tag='SLAM',reference_panel_text='')
        self.s.nav=SimpleNamespace(tag='NAV')
        self.nav=NavModeController(path_aware=True, session_id=self.s.session_id)
        self.pose_seq=10
        self.sent=[]
        self.reset=Mock()
        self.s.log=Mock()
        def send(child, command):
            self.sent.append((child.tag,command))
            if child.tag=='NAV':
                self.nav.apply_sched_line(command)
            elif command=='resume':
                self.pose_seq+=1
            return True
        self.s.send=send
        self.addCleanup(self.s._cancel_plan)
        guard=patch.object(scheduler.subprocess,'Popen',side_effect=AssertionError('hardware process forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def anchor(self, pose=(-.5,0,2,0,0,0,1)):
        self.s.handle_sched('slam',dict(anchor='busy'))
        self.s.handle_sched('slam',dict(anchor='ok',pose='('+','.join(map(str,pose))+')'))
        until=time.monotonic()+10
        while self.s.state==scheduler.STATE_PLANNING and time.monotonic()<until:
            self.s.poll_planning()
            time.sleep(.01)
        self.assertEqual(self.s.state,scheduler.STATE_WAIT_PATH_READY)

    def pump(self):
        for event in self.nav.process_segments(reset=self.reset,pose_sequence=self.pose_seq,pose_fresh=True):
            self.s.handle_sched('nav',event)

    def running(self):
        self.pump()
        self.assertEqual(self.s.state,scheduler.STATE_WAIT_PATH_RUNNING)
        self.pump()
        self.assertEqual(self.s.state,scheduler.STATE_SEGMENT)

    def test_astar_endpoints_conversion_acknowledgements_and_next_segment(self):
        self.anchor()
        route=self.s.current_route
        refs=route['references_xy']
        self.assertEqual(len(refs),15)
        self.assertEqual(refs[0],route['start_xy'])
        self.assertEqual(refs[-1],route['goal_xy'])
        self.assertEqual(self.s.slam.reference_panel_text.count('/15:'),15)
        self.assertNotIn(('SLAM','resume'),self.sent)
        self.running()
        segment=self.nav.path_snapshot()[0]
        np.testing.assert_allclose(segment.path_w[:,2],.5)
        np.testing.assert_allclose(segment.path_w[:,:2],[[z,-x] for x,z in refs])
        self.assertTrue(all('goal ' not in cmd and cmd != 'S' for tag,cmd in self.sent if tag=='NAV'))
        self.assertTrue((self.s.path_directory/'segment_delivery.json').is_file())
        self.assertEqual(len((self.s.path_directory/'delivery_events.jsonl').read_text().splitlines()),2)
        with patch.object(scheduler.time,'sleep',return_value=None):
            self.s.arrive()
        self.s.handle_user('localize')
        self.anchor((-6.6,.3,5.2,0,0,0,1))
        self.running()
        self.assertEqual(self.s.idx,1)

    def test_manual_localize_replans_same_target_and_old_start_is_rejected(self):
        self.anchor()
        middle=self.s.current_route['references_xy'][3]
        self.running()
        old_id=self.s.path_segment_id
        self.nav.apply_line('F')
        self.pump()
        self.assertEqual(self.s.state,scheduler.STATE_PAUSED)
        self.s.handle_user('force')
        self.assertEqual(self.s.idx,0)
        self.nav.apply_sched_line('segment_start '+old_id)
        self.pump()
        self.assertFalse(self.nav.path_snapshot()[2])
        self.s.handle_user('localize')
        self.anchor((middle[0],.3,middle[1],0,0,0,1))
        self.running()
        self.assertEqual(self.s.idx,0)
        self.assertNotEqual(self.s.path_segment_id,old_id)
        self.assertEqual(self.nav.path_snapshot()[0].resume_token,1)

    def test_timeouts_and_old_session_ack_never_start(self):
        self.anchor()
        self.s.handle_sched('nav',dict(path='ready',session_id='old',segment_id=self.s.path_segment_id))
        self.assertEqual(self.s.state,scheduler.STATE_WAIT_PATH_READY)
        self.s.path_deadline=time.monotonic()-1
        self.s.poll_path_timeout()
        self.pump()
        self.assertEqual(self.s.state,scheduler.STATE_PLAN_FAILED)
        self.assertNotIn(('SLAM','resume'),self.sent)
        self.assertFalse(self.nav.path_snapshot()[2])
        self.s.handle_user('localize')
        self.anchor()
        self.pump()  # ready received; start is still queued at NavSide
        self.assertEqual(self.s.state,scheduler.STATE_WAIT_PATH_RUNNING)
        self.s.path_deadline=time.monotonic()-1
        self.s.poll_path_timeout()
        self.pump()  # delayed start cannot win against the timeout stop
        self.assertEqual(self.s.state,scheduler.STATE_PLAN_FAILED)
        self.assertFalse(self.nav.path_snapshot()[2])

    def test_crash_invalidates_delivery_and_restart_replans_current_target(self):
        self.anchor()
        self.running()
        old_id=self.s.path_segment_id
        self.s.handle_sched('slam',{'__exit__':'1'})
        self.pump()
        self.assertFalse(self.nav.path_snapshot()[2])
        self.s.handle_sched('nav',dict(path='running',session_id=self.s.session_id,segment_id=old_id))
        self.assertEqual(self.s.state,scheduler.STATE_RESTART_WAIT)
        fake=SimpleNamespace(tag='SLAM',start=lambda:None)
        self.s.slam_restart_pending=False
        with patch.object(scheduler,'ChildProc',return_value=fake), \
             patch.object(scheduler.threading.Thread,'start',return_value=None), \
             patch.object(scheduler,'open_viewer_window',return_value=False):
            self.s._restart_slam()
        self.anchor()
        self.running()
        self.assertEqual(self.s.idx,0)
        self.assertNotEqual(self.s.path_segment_id,old_id)


class ModelAndRealLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import onnxruntime as ort
        with patch.object(ort,'get_available_providers',return_value=['CPUExecutionProvider']):
            cls.app=NavSideApp.from_config(str(ROOT/'config/nav_pathaware.yaml'))
        cls.depth=np.full((720,1280),3.,dtype=np.float32)

    def setUp(self):
        self.app.adapter.reset_recurrent_state()

    def test_mix_preprocessing_and_wrong_policy_rejection(self):
        adapter=self.app.adapter
        seen=[]
        def encode(names, feed):
            seen.append(next(iter(feed.values())).copy())
            return [np.zeros((1,64,5,8),dtype=np.float32)]
        with patch.object(adapter.encoder_session,'run',side_effect=encode):
            adapter.depth_preprocess(np.full((720,1280),11.,dtype=np.float32))
            adapter.depth_preprocess(np.full((720,1280),np.nan,dtype=np.float32))
        self.assertEqual(seen[0].shape,(1,1,40,64))
        np.testing.assert_allclose(seen[0],6.)
        np.testing.assert_allclose(seen[1],0.)
        with self.assertRaises(ValueError):
            adapter.depth_preprocess(np.zeros((480,640),dtype=np.float32))
        wrong=copy.copy(self.app.config)
        wrong.policy_path=str(ROOT/'asset/models/nav_policy.onnx')
        import onnxruntime as ort
        with patch.object(ort,'get_available_providers',return_value=['CPUExecutionProvider']):
            with self.assertRaisesRegex(ValueError,'模型接口不匹配'):
                NavSideApp(wrong)

    def test_actual_models_obs_order_rate_recurrence_and_legacy(self):
        app=self.app
        self.assertEqual(app.config.dry_run_hz,5)
        self.assertIn('mix',app.config.encoder_path)
        segment=Segment.parse(payload())
        state=app.default_state()
        obs=[]
        original=app.adapter.policy_session.run
        def run(names, feed):
            obs.append({key:value.copy() for key,value in feed.items()})
            return original(names,feed)
        with patch.object(app.adapter.policy_session,'run',side_effect=run):
            first=app.step(self.depth,state,segment.goal_w,timestamp=100.,path_w=segment.path_w,print_control=False)
            self.assertIsNone(app.step(self.depth,state,segment.goal_w,timestamp=100.05,path_w=segment.path_w))
            hidden=app.adapter.h_state.copy()
            state.robot_quat_wxyz=np.array([np.sqrt(.5),0,0,np.sqrt(.5)],dtype=np.float32)
            second=app.step(self.depth,state,segment.goal_w,timestamp=100.21,path_w=segment.path_w,print_control=False)
        self.assertEqual(len(obs),2)
        self.assertEqual(obs[0]['obs'].shape,(1,2636))
        np.testing.assert_allclose(obs[1]['h_in'],hidden)
        np.testing.assert_allclose(obs[1]['obs'][0,9:12],first['diag']['raw_action'])
        np.testing.assert_allclose(obs[1]['obs'][0,16:76],second['diag']['path_obs'])
        self.assertFalse(np.allclose(first['diag']['path_obs'],second['diag']['path_obs']))
        self.assertTrue(np.isfinite(second['diag']['raw_action']).all())
        app.adapter.reset_recurrent_state()
        self.assertFalse(np.any(app.adapter.h_state))
        with self.assertRaises(ValueError):
            app.step(self.depth,state,segment.goal_w,timestamp=101.,path_w=None)
        import onnxruntime as ort
        with patch.object(ort,'get_available_providers',return_value=['CPUExecutionProvider']):
            legacy=NavSideApp.from_config(str(ROOT/'config/nav.yaml'))
        old=legacy.step(self.depth,legacy.default_state(),segment.goal_w,print_control=False)
        self.assertEqual(old['diag']['obs_shape'].tolist(),[1,2576])

    def test_real_loop_forwards_full_path_and_stop_during_inference_sends_zero(self):
        app=self.app
        with tempfile.TemporaryDirectory() as tmp:
            sched_file=Path(tmp)/'events.log'
            controller=NavModeController(path_aware=True,session_id='test')
            controller.apply_sched_line('segment_load '+json.dumps(payload()))
            calls=[]
            sent=[]
            class FakeComm:
                def start(self): pass
                def stop(self): pass
                def load_task(self): return False
                def send_zero(self): sent.append((0.,0.,0.))
                def send_command(self,*cmd): sent.append(cmd)
                def get_latest_state(self):
                    self.seq=getattr(self,'seq',0)+1
                    if len(calls)>=1:
                        controller.apply_sched_line('quit')
                    elif sched_file.exists() and 'path=ready' in sched_file.read_text() and not getattr(self,'started',False):
                        controller.apply_sched_line('segment_start test:1')
                        self.started=True
                    return NavStatePacketV2(robot_pos_w=np.array([0.,0.,.695]),pose_sequence=self.seq,
                                            received_monotonic=time.monotonic())
            camera=SimpleNamespace(start=lambda:None,close=lambda:None,
                read=lambda:SimpleNamespace(success=True,depth_input=self.depth))
            original=app.step
            def step(*args,**kwargs):
                calls.append(kwargs['path_w'].copy())
                result=original(*args,**kwargs)
                controller.apply_line('F')  # stop arrives while inference is finishing
                return result
            args=SimpleNamespace(config=str(ROOT/'config/nav_pathaware.yaml'),goal=None,
                                 deploy_config=None,csv_dir=None,show_depth=False)
            with patch.dict('os.environ',{'NAVSIDE_SESSION_ID':'test','NAVSIDE_SCHED_FILE':str(sched_file)}), \
                 patch.object(real.NavSideApp,'from_config',return_value=app), \
                 patch.object(real,'NavModeController',return_value=controller), \
                 patch.object(controller,'start_input_thread'),patch.object(controller,'stop_input_thread'), \
                 patch.object(controller,'start_sched_input'),patch.object(controller,'stop_sched_input'), \
                 patch.object(real,'create_depth_perception',return_value=camera), \
                 patch.object(real,'RobotComm',return_value=FakeComm()), \
                 patch.object(real,'CsvTickLogger'),patch.object(app,'step',side_effect=step), \
                 patch('socket.socket',side_effect=AssertionError('network forbidden')), \
                 patch('subprocess.Popen',side_effect=AssertionError('process forbidden')), \
                 patch('sys.stdout',new_callable=io.StringIO):
                real.run(args)
            self.assertEqual(len(calls),1)
            np.testing.assert_allclose(calls[0],payload()['path_w'])
            self.assertTrue(all(np.allclose(cmd,0) for cmd in sent))
            self.assertIn('path=ready',sched_file.read_text())
            self.assertIn('path=running',sched_file.read_text())


if __name__=='__main__':
    unittest.main()
