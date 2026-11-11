"""Pose wire/trace/report regression tests; no cameras or real sockets."""
import importlib.util
import ast
from collections import deque
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace
from typing import Deque, Optional
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navside.pose_trace import PoseTrace, NULL_TRACE, POSE, decode_pose, trace_packet
from navside.bridge import RobotComm
import navside.real as real
import test_pose_recovery as recovery

spec = importlib.util.spec_from_file_location('pose_diagnostics', ROOT/'scripts/pose_diagnostics.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class DiagnosticTests(unittest.TestCase):
    def test_production_sender_preserves_legacy_and_extended_wire_pose(self):
        upstream=Path('/home/amov/cuvslam_migrate-9.1/orbbec')
        # The real pure coordinate transform has no server startup side effects.
        spec=importlib.util.spec_from_file_location('foxglove_odom_server',upstream/'foxglove_odom_server.py')
        transform=importlib.util.module_from_spec(spec);spec.loader.exec_module(transform)
        with patch.dict(sys.modules,{'foxglove_odom_server':transform,
                                     'pose_trace_hook':SimpleNamespace(get_trace=lambda:NULL_TRACE)}):
            spec=importlib.util.spec_from_file_location('tested_sender',upstream/'udp_pose_sender.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            trace=PoseTrace(tmp,'slam')
            try:
                for tracing in (NULL_TRACE,trace):
                    sender=module.UdpPoseSender.__new__(module.UdpPoseSender)
                    sender.trace=tracing;sender._sock=Mock();sender._packer=POSE;sender.addr=('127.0.0.1',8082)
                    sender.send_pose([1.,2.,3.],[0.,0.,0.,1.],source_timestamp_ns=123,
                                     frame_received_ns=100,processed_ns=110,anchor_epoch=2)
                    data=sender._sock.sendto.call_args.args[0]
                    pose,meta=decode_pose(data)
                    np.testing.assert_allclose(pose,[3.,-1.,-2.,1.,0.,0.,0.],atol=1e-12)
                    self.assertEqual(meta['wire_version'],2 if tracing.enabled else 1)
                    if tracing.enabled:self.assertEqual(meta['source_timestamp_ns'],123)
            finally:trace.close()

    def test_imu_diagnostics_preserve_registration_order_and_report_late_data(self):
        path=Path('/home/amov/cuvslam_migrate-9.1/orbbec/run_vio_tasknav.py')
        tree=ast.parse(path.read_text())
        selected=ast.Module([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='register_imu_until'],[])
        import queue
        trace=Mock()
        scope=dict(queue=queue,vslam=SimpleNamespace(Tracker=object,ImuMeasurement=SimpleNamespace),
                   Deque=Deque,Optional=Optional,ImuSample=object,get_trace=lambda:trace)
        exec(compile(selected,str(path),'exec'),scope)
        pending=deque();samples=queue.Queue();tracker=Mock()
        for t in (40,60,90,110):
            samples.put(SimpleNamespace(timestamp_ns=t,linear_accelerations=(0,0,0),angular_velocities=(0,0,0)))
        last=scope['register_imu_until'](tracker,samples,pending,100,50)
        self.assertEqual(last,90)
        self.assertEqual([c.args[1].timestamp_ns for c in tracker.register_imu_measurement.call_args_list],[60,90])
        self.assertEqual(pending[0].timestamp_ns,110)
        stats=trace.emit.call_args.kwargs
        self.assertEqual(stats['registered'],2);self.assertEqual(stats['skipped_old'],1)
        self.assertAlmostEqual(stats['image_to_last_imu_s'],1e-8)

    def test_legacy_packet_unchanged_and_malformed_packets_rejected(self):
        packet = POSE.pack(1,2,3,1,0,0,0)
        self.assertEqual(NULL_TRACE.encode(packet)[0], packet)
        self.assertEqual(decode_pose(packet), ((1,2,3,1,0,0,0), {'wire_version':1}))
        for malformed in (packet[:-1], packet+b'bad', packet+b'X'*68):
            with self.subTest(length=len(malformed)), self.assertRaises(ValueError):
                decode_pose(malformed)

    def test_metadata_round_trip_and_restart_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = PoseTrace(tmp,'slam'), PoseTrace(tmp,'slam')
            try:
                body = POSE.pack(1,2,3,1,0,0,0)
                packet, metadata = first.encode(body, source_timestamp_ns=123, frame_received_ns=10, processed_ns=11, anchor_epoch=2)
                self.assertEqual(packet[:56],body)
                self.assertEqual(decode_pose(packet)[1],metadata)
                self.assertEqual(metadata['source_sequence'],1)
                self.assertEqual(metadata['anchor_epoch'],2)
                self.assertNotEqual(first.stream_id,second.stream_id)
                self.assertEqual(first.encode(body)[1]['source_sequence'],2)
            finally:
                first.close(); second.close()

    def test_bridge_receives_extended_packet_and_cache_keeps_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tx,rx = PoseTrace(tmp,'slam'),PoseTrace(tmp,'nav')
            comm = RobotComm.__new__(RobotComm)
            comm.pose_trace=rx
            comm._odom_lock=threading.Lock();comm._vel_lock=threading.Lock()
            comm._lin_vel=np.zeros(3);comm._ang_vel=np.zeros(3)
            comm._pose_sequence=0;comm._seq=0;comm._running=True
            comm._compute_velocities=Mock()
            body=POSE.pack(5,6,-1,1,0,0,0)
            packet,metadata=tx.encode(body,source_timestamp_ns=1000000000,anchor_epoch=3)
            tx.emit('tx', **metadata,wire_pose=list(POSE.unpack(body)))
            comm._pose_sock=Mock()
            comm._pose_sock.recvfrom.side_effect=[(packet,('127.0.0.1',1234)),OSError('done')]
            with patch('socket.socket',side_effect=AssertionError('network forbidden')):
                comm._run_udp_loop()
                a,b=comm.get_latest_state(),comm.get_latest_state()
                trace_packet(rx,'inference_start',b,inference_id=1)
            self.assertEqual(a.pose_trace['source_timestamp_ns'],1000000000)
            self.assertEqual(a.pose_trace,b.pose_trace)
            self.assertEqual(a.received_monotonic,b.received_monotonic)
            self.assertEqual(a.pose_sequence,b.pose_sequence)
            np.testing.assert_allclose(a.robot_pos_w,[5,6,.695])
            tx.close();rx.close()
            rows,errors=diagnostics.read_rows(Path(tmp))
            summary,samples,_=diagnostics.analyze(rows)
            self.assertEqual(errors,0)
            self.assertEqual(summary['matched_tx_rx'],1)
            self.assertEqual(summary['mismatches'],[])
            self.assertEqual(samples[0]['inference_use_count'],1)
            self.assertGreaterEqual(samples[0]['send_to_receive_s'],0)

    def test_writer_full_queue_reports_loss_without_blocking_caller(self):
        # Prevent the writer from draining to deterministically fill the queue.
        with tempfile.TemporaryDirectory() as tmp, patch.object(threading.Thread,'start'), patch.object(threading.Thread,'join'):
            trace=PoseTrace(tmp,'nav',max_queue=1)
            for _ in range(20): trace.emit('rx')
            self.assertEqual(trace.dropped,20)
            trace.close()
            trace._write()
            trace._thread=threading.current_thread()
            rows,_=diagnostics.read_rows(Path(tmp))
            self.assertEqual(rows[-1]['trace_dropped'],20)

    def test_actual_real_loop_records_input_and_expired_inference_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace=PoseTrace(tmp,'nav')
            with patch.object(real,'get_trace',return_value=trace):
                sent,calls,events=recovery.RealLoopRecoveryTests().run_scenario(slow_inference=True)
            rows,_=diagnostics.read_rows(Path(tmp))
            starts=[r for r in rows if r['stage']=='inference_start']
            ends=[r for r in rows if r['stage']=='inference_end']
            self.assertEqual(len(starts),len(calls))
            self.assertEqual(starts[0]['pose_sequence'],ends[0]['pose_sequence'])
            self.assertAlmostEqual(ends[0]['duration_s'],1.2)
            self.assertTrue(any(r.get('reason')=='inference_pose_expired' for r in rows))
            self.assertTrue(all(np.allclose(cmd,0) for _,phase,cmd in sent if phase=='waiting_pose'))

    def synthetic_rows(self):
        rows=[dict(stage='trace_start',logger_id='s',boot_id='same',mono_ns=1),
              dict(stage='trace_start',logger_id='r',boot_id='same',mono_ns=1)]
        for seq,x in ((1,0),(2,42.68)):
            t=1000000000+seq*100000000
            shared=dict(stream_id='s',source_sequence=seq,source_timestamp_ns=seq*100000000,
                        frame_received_ns=t,processed_ns=t+1000000,send_ns=t+2000000,anchor_epoch=1,wire_version=2)
            rows += [dict(stage='vio',logger_id='s',mono_ns=t+1000000,source_timestamp_ns=seq*100000000,
                         frame_received_ns=t,track_started_ns=t,track_done_ns=t+1000000,local_position=[x,0,0]),
                     dict(stage='tx',logger_id='s',mono_ns=t+2000000,wire_pose=[x,0,0,1,0,0,0],**shared),
                     dict(stage='rx',logger_id='r',mono_ns=t+3000000,received_ns=t+3000000,
                          socket_received_ns=t+3000000,wire_pose=[x,0,0,1,0,0,0],**shared)]
        return rows

    def test_report_identifies_jump_already_in_raw_vio(self):
        summary,samples,raw=diagnostics.analyze(self.synthetic_rows())
        self.assertAlmostEqual(summary['largest_steps'][0]['horizontal_step_m'],42.68)
        self.assertAlmostEqual(summary['largest_steps'][0]['raw_vio_step_3d_m'],42.68)
        self.assertEqual(summary['mismatches'],[])
        self.assertAlmostEqual(samples[-1]['send_to_receive_s'],.001)
        self.assertAlmostEqual(raw[-1]['source_dt_s'],.1)

    def test_report_distinguishes_transform_jump_from_raw_vio(self):
        rows=self.synthetic_rows()
        [r for r in rows if r['stage']=='vio'][-1]['local_position']=[.1,0,0]
        summary,_,_=diagnostics.analyze(rows)
        self.assertAlmostEqual(summary['largest_steps'][0]['raw_vio_step_3d_m'],.1)
        self.assertAlmostEqual(summary['largest_steps'][0]['global_step_3d_m'],42.68)

    def test_report_does_not_compare_epochs_or_incompatible_host_clocks(self):
        rows=self.synthetic_rows()
        rows[1]['boot_id']='different-host'
        for r in rows:
            if r.get('source_sequence')==2: r['anchor_epoch']=2
        summary,samples,_=diagnostics.analyze(rows)
        self.assertEqual(summary['largest_steps'],[])
        self.assertTrue(all(s['send_to_receive_s'] is None for s in samples))

    def test_local_reset_is_not_drift_even_when_anchor_epoch_is_unchanged(self):
        for metadata in [({'tracker_generation':1,'tracker_id':7}, {'tracker_generation':2,'tracker_id':7}),
                         ({'tracker_id':7}, {'tracker_id':8})]:
            with self.subTest(metadata=metadata):
                rows=self.synthetic_rows()
                raw_rows=[r for r in rows if r['stage']=='vio']
                for row,values in zip(raw_rows,metadata):row.update(values)
                summary,samples,raw=diagnostics.analyze(rows)
                # A failed/cancelled anchor resets the local tracker without an epoch change.
                # Still retain any global jump so this fix cannot hide a transform error.
                self.assertAlmostEqual(samples[-1]['global_step_3d_m'],42.68)
                self.assertIsNone(samples[-1]['raw_vio_step_3d_m'])
                self.assertFalse(samples[-1]['raw_vio_frame_continuous'])
                self.assertIsNone(raw[-1]['local_step_m'])
                self.assertFalse(raw[-1]['local_frame_continuous'])
                self.assertEqual(summary['vio_metrics']['local_step_m']['count'],0)

    def test_report_marks_mismatch_and_missing_packets_without_claiming_loss(self):
        rows=self.synthetic_rows()
        [r for r in rows if r['stage']=='rx'][-1]['wire_pose'][0]=999
        rows=[r for r in rows if not (r['stage']=='rx' and r['source_sequence']==1)]
        summary,_,_=diagnostics.analyze(rows)
        self.assertEqual(summary['tx_without_rx'],1)
        self.assertEqual(summary['mismatches'][0]['kind'],'tx_rx_pose_mismatch')

    def test_prepare_and_report_are_offline_and_generate_reviewable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp, patch('socket.socket',side_effect=AssertionError('network forbidden')), \
             patch('subprocess.Popen',side_effect=AssertionError('process forbidden')), \
             patch.object(diagnostics,'BASE',Path(tmp)/'base'), patch('sys.stdout',new_callable=io.StringIO):
            folder=diagnostics.prepare(str(Path(tmp)/'run'))
            self.assertIn('NAV_POSE_DIAG_MODULE', (folder/'enable.sh').read_text())
            self.assertEqual((folder/'nav_pathaware.yaml').read_bytes(),(ROOT/'config/nav_pathaware.yaml').read_bytes())
            (folder/'synthetic.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.synthetic_rows()))
            summary=diagnostics.report(folder)
            self.assertEqual(summary['matched_tx_rx'],2)
            self.assertTrue((folder/'poses.csv').is_file())
            self.assertIn('不等同纯网络延迟',(folder/'report.md').read_text())

    def explain(self, rows):
        rows = sorted(rows, key=lambda r:r['mono_ns'])
        summary, samples, _ = diagnostics.analyze(rows)
        summary['malformed_or_partial_lines'] = 0
        findings = diagnostics.interpret(rows, summary, samples)
        report = diagnostics.render_report(Path('/test/run'), rows, summary, samples, findings, [])
        return findings, report

    def waiting_rows(self):
        rows = self.synthetic_rows()
        for r in rows:
            if r.get('source_sequence') == 2 or (r['stage'] == 'vio' and r['source_timestamp_ns'] == 200000000):
                for field in ('mono_ns', 'frame_received_ns', 'processed_ns', 'send_ns',
                              'socket_received_ns', 'received_ns', 'track_started_ns', 'track_done_ns'):
                    if field in r:r[field] += 1500000000
                if r['stage'] == 'vio':r['track_started_ns'] = 1200000000
            if r['stage'] == 'rx':r['pose_sequence'] = r['source_sequence']
        rows += [dict(stage='config',logger_id='r',mono_ns=2,state_max_age_s=1.),
                 dict(stage='control',logger_id='s',mono_ns=900000000,command='resume'),
                 dict(stage='path_event',logger_id='r',mono_ns=2200000000,path='waiting_pose',
                      reason='pose_stale',pose_sequence=1,pose_age_s=1.097,segment_id='one'),
                 dict(stage='path_event',logger_id='r',mono_ns=3400000000,path='pose_resumed',
                      reason='pose_stable',pose_sequence=3,wait_s=1.2,stable_samples=3,segment_id='one')]
        return rows

    def test_wait_explanation_correlates_actual_gap_and_slow_upstream_track(self):
        findings, report = self.explain(self.waiting_rows())
        wait = findings['waits'][0]
        self.assertTrue(wait['upstream_gap'])
        self.assertAlmostEqual(wait['receive_gap_s'],1.6)
        self.assertAlmostEqual(wait['next_delivery_s'],.001)
        self.assertAlmostEqual(wait['slow_track']['duration_s'],1.501)
        self.assertEqual(wait['outcome']['path'],'pose_resumed')
        self.assertIn('自动恢复',report)

    def test_wait_does_not_attribute_slow_track_across_incomparable_clocks(self):
        rows = self.waiting_rows()
        rows[1]['boot_id'] = 'other-host'
        findings, _ = self.explain(rows)
        self.assertFalse(findings['waits'][0]['upstream_gap'])
        self.assertIsNone(findings['waits'][0]['next_delivery_s'])

    def test_pause_gap_is_not_a_delivery_delay_even_when_epoch_unchanged(self):
        rows = self.synthetic_rows()
        rows += [dict(stage='control',logger_id='s',mono_ns=900000000,command='resume'),
                 dict(stage='control',logger_id='s',mono_ns=1150000000,command='pause'),
                 dict(stage='control',logger_id='s',mono_ns=1190000000,command='resume')]
        findings, report = self.explain(rows)
        self.assertEqual(findings['intervals'][0]['kind'],'跨暂停/重定位')
        self.assertIn('不能当成一个 UDP 包传输了这么久',report)

    def test_vio_only_report_does_not_call_missing_downstream_healthy(self):
        rows = [r for r in self.synthetic_rows() if r['stage'] in ('vio','trace_start')]
        findings, report = self.explain(rows)
        self.assertEqual(findings['inference_count'],0)
        self.assertIn('数据不足：',report)
        self.assertIn('不能判断下游是否正常',report)
        self.assertNotIn('已匹配的发送/接收坐标一致',report)

    def test_new_worker_owned_tracker_generations_match_actual_frame_ids(self):
        rows=self.synthetic_rows()
        rows += [dict(stage='tracker_switched',logger_id='s',mono_ns=1000000000,tracker_generation=0,tracker_id=7),
                 dict(stage='tracker_switched',logger_id='s',mono_ns=1500000000,tracker_generation=1,tracker_id=8)]
        for r in rows:
            if r['stage']=='vio':r.update(tracker_generation=0,tracker_id=7)
        findings, report=self.explain(rows)
        self.assertEqual(findings['tracker_generations'][0]['frame_count'],2)
        self.assertEqual(findings['tracker_generations'][0]['mismatch_count'],0)
        self.assertEqual(findings['tracker_generations'][1]['frame_count'],0)
        self.assertIn('0 帧表示尚无恢复跟踪的证据',report)
        [r for r in rows if r['stage']=='vio'][0]['tracker_id']=99
        findings,_=self.explain(rows)
        self.assertEqual(findings['tracker_generations'][0]['mismatch_count'],1)

    def test_manual_pause_ends_wait_without_claiming_later_automatic_resume(self):
        rows = self.waiting_rows()
        rows += [dict(stage='path_event',logger_id='r',mono_ns=2300000000,path='paused',
                      reason='operator',segment_id='one')]
        findings, report = self.explain(rows)
        self.assertEqual(findings['waits'][0]['outcome']['path'],'paused')
        self.assertFalse(findings['waits'][0]['upstream_gap'])
        self.assertIn('人工暂停 `operator`',report)
        self.assertIn('0 次自动恢复',report)

    def test_task_evidence_joins_exact_session_not_latest_folder(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(diagnostics,'ROOT',Path(tmp)):
            base=Path(tmp)/'logs/task_nav'
            folder=base/'astar/run_20260930_094638_088844'
            folder.mkdir(parents=True)
            (folder/'session.json').write_text(json.dumps(dict(session_id='wanted')))
            (folder/'failures.jsonl').write_text(json.dumps(dict(error='起点不可通行'))+'\n')
            (base/'nav_sched_20260930_094638.log').write_text('session_id=unrelated zero_reason=goal_reached\n')
            (base/'slam_sched_20260930_094638.log').write_text('[localize] FAILED\n')
            self.assertEqual(diagnostics.related_task_evidence([dict(session_id='elsewhere')]),[])
            matched=diagnostics.related_task_evidence([dict(session_id='wanted')])[0]
            self.assertEqual(len(matched['failures']),1)
            self.assertFalse(matched['localize_failed'])
            self.assertEqual(matched['arrivals'],[])


if __name__=='__main__':
    unittest.main()
