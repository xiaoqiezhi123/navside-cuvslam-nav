"""Preview cancellation, rendering and isolation from saved navigation targets."""
from pathlib import Path
import json
import queue
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from task_path_preview import PreviewWorker
from task_nav_astar import SegmentAStarPlanner, _load_modules, save_task_points
from task_point_picker import TaskPointPicker


class WorkerTests(unittest.TestCase):
    def test_superseded_work_cannot_publish_and_uses_one_worker(self):
        entered,release=threading.Event(),threading.Event()
        calls=[]
        def preview(a,b,cancel):
            calls.append(threading.get_ident())
            if len(calls)==1:
                entered.set()
                if not release.wait(2):raise RuntimeError('test release timeout')
            return dict(start=a,goal=b)  # Deliberately ignore cancellation.
        worker=PreviewWorker(SimpleNamespace(preview=preview))
        self.addCleanup(worker.close)
        self.addCleanup(release.set)
        worker.request([(0,0,0),(1,0,1)])
        self.assertTrue(entered.wait(1))
        revision=worker.request([(2,0,2),(3,0,3)])
        release.set()
        generation,key,result=worker.results.get(timeout=2)
        self.assertEqual(generation,revision)
        self.assertEqual(result['start'],(2,0,2))
        self.assertEqual(len(calls),2)
        self.assertEqual(len(set(calls)),1)
        self.assertTrue(worker.results.empty())
        worker.close();worker._thread.join(1)
        self.assertFalse(worker._thread.is_alive())

    def test_cache_and_failure_are_independent_of_other_pairs(self):
        calls=[]
        def preview(a,b,cancel):
            calls.append((a,b))
            if b[0]==2:raise ValueError('无路')
            return dict(start=a,goal=b)
        worker=PreviewWorker(SimpleNamespace(preview=preview))
        self.addCleanup(worker.close)
        a,b,c=(0,0,0),(1,0,1),(2,0,2)
        worker.request([a,b]);worker.results.get(timeout=1)
        revision=worker.request([a,b,c])
        r1=worker.results.get(timeout=1);r2=worker.results.get(timeout=1)
        self.assertEqual(r1[0],revision)
        self.assertIsInstance(r1[2],dict)
        self.assertIsInstance(r2[2],ValueError)
        self.assertEqual(calls,[(a,b),(b,c)])


class PickerTests(unittest.TestCase):
    def make_picker(self,initial=None):
        data,algorithm=_load_modules('/home/amov/cuvslam_migrate-9.1')
        mask=np.zeros((16,20),bool);mask[1:-1,1:-1]=True;mask[3:13,9]=False
        pm=data.PlanningMap(Path('/tmp/preview-map'),
            dict(origin_xz=[0.,0.],resolution_m=1.,robot_radius_m=0.,safety_margin_m=0.,
                 map_id='test',grid_sha256='test-grid',source_db_sha256='test-db',map_version='test'),
            np.where(mask,0,100).astype(np.int8),mask,np.ones(mask.shape),np.where(mask,0.,np.inf),True)
        planner=SegmentAStarPlanner.__new__(SegmentAStarPlanner)
        planner.pm=pm;planner.algorithm=algorithm;planner.include_start=True;planner.include_goal=True
        planner.center_settings=dict(enabled=False);planner.plan_options={}
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        target=Path(tmp.name)/'goals.json'
        if initial:save_task_points(planner,initial,target)
        picker=TaskPointPicker(planner,target,initial,preview_root=Path(tmp.name)/'previews')
        self.addCleanup(plt.close,picker.fig);self.addCleanup(picker.on_close)
        return picker

    def wait_for_previews(self,picker):
        until=time.monotonic()+5
        while time.monotonic()<until:
            picker.poll_previews()
            if all(k in picker.previews for k in picker.pairs()):return
            time.sleep(.01)
        self.fail('preview did not finish')

    def test_loaded_goals_render_exact_15_points_and_export_without_confirming(self):
        points=[[2.5,.695,5.5],[13.5,.695,5.5],[13.5,.695,11.5]]
        picker=self.make_picker(points)
        original=picker.output.read_bytes()
        self.wait_for_previews(picker)
        self.assertEqual(len(picker.previews),2)
        route=picker.current_preview()
        self.assertEqual(len(route['references_xy']),15)
        self.assertEqual(route['references_xy'][0],[2.5,5.5])
        self.assertEqual(route['references_xy'][-1],[13.5,5.5])
        self.assertTrue(route['all_reference_segments_collision_free'])
        labels={x.get_text() for x in picker.ax.texts}
        self.assertTrue({'T1','T2','T3','1','15'}<=labels)
        picker.next_pair();self.assertEqual(picker.selected_pair,1)
        self.assertIn('T2 → T3',picker.table.texts[0].get_text())
        picker.focus();self.assertGreater(picker.ax.get_xlim()[0],0.)
        picker.save_preview()
        directory=picker.last_preview_directory
        payload=json.loads((directory/'preview.json').read_text())
        self.assertTrue(payload['preview_only'])
        self.assertEqual(len(payload['routes']),2)
        self.assertGreater((directory/'preview.png').stat().st_size,1000)
        self.assertFalse(picker.accepted)
        self.assertEqual(picker.output.read_bytes(),original)
        picker.cancel()
        self.assertEqual(picker.output.read_bytes(),original)

    def test_undo_clear_and_stale_result_never_restore_deleted_route(self):
        picker=self.make_picker()
        picker.add(2.5,5.5);picker.add(13.5,5.5)
        self.wait_for_previews(picker)
        key=picker.pairs()[0];old=picker.previews[key];revision=picker.revision
        picker.undo()
        picker.preview_worker.results.put((revision,key,old))
        picker.poll_previews()
        self.assertEqual(len(picker.points),1)
        self.assertEqual(picker.previews,{})
        self.assertFalse(picker.route_artists)
        picker.add(9.5,5.5)  # Wall; must not change targets/revision.
        self.assertEqual(len(picker.points),1)
        picker.clear()
        self.assertEqual(picker.points,[])
        picker.save_preview()
        self.assertIsNone(picker.last_preview_directory)

    def test_no_route_shows_failure_without_invented_straight_line(self):
        picker=self.make_picker()
        picker.planner.pm.traversable[:,9]=False
        picker.add(2.5,5.5);picker.add(13.5,5.5)
        self.wait_for_previews(picker)
        self.assertIsInstance(picker.current_preview(),Exception)
        self.assertFalse(picker.route_artists)
        self.assertIn('失败 1',picker.status.get_text())
        self.assertFalse(picker.output.exists())


if __name__=='__main__':unittest.main()
