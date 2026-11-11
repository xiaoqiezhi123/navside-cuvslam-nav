"""Offline bridge tests against the current reviewed map; no robot processes."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from task_nav_astar import (SegmentAStarPlanner, SegmentPlanningError, validate_anchor_pose,
                           save_task_points, load_task_points, reference_lines)


class SegmentPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = dict(cuvslam_repo='/home/amov/cuvslam_migrate-9.1', ref_map='viz_test_02',
                       astar=dict(map_package='maps2d/viz_test_02/v6_reviewed', include_goal=False))
        cls.planner = SegmentAStarPlanner(cls.cfg)

    def test_enabled_center_plan_keeps_protocol_identity_and_exact_endpoints(self):
        from task_nav_scheduler import load_config
        cfg=load_config(str(ROOT/'config/task_nav.yaml'))
        planner=SegmentAStarPlanner(cfg)
        self.assertTrue(planner.center_enabled)
        anchor=(-.2069,-.0342,1.3384,0,0,0,1)
        task=(-6.502375719518561,.695,6.631336473050762)
        result=planner.plan(anchor,task,0,anchor_sequence=1)
        self.assertEqual(len(result['references_xyz']),15)
        self.assertEqual(result['references_xyz'][0],[anchor[0],0.,anchor[2]])
        self.assertEqual(result['references_xyz'][-1],[task[0],0.,task[2]])
        self.assertTrue(result['includes_start']);self.assertTrue(result['includes_goal'])
        self.assertGreater(result['min_clearance_m'],.8)
        self.assertEqual(result['grid_sha256'],self.planner.pm.meta['grid_sha256'])
        self.assertEqual(result['center_preference']['max_clearance_loss_m'],.05)
        self.assertTrue(result['all_reference_segments_collision_free'])
        preview=planner.preview((anchor[0],.695,anchor[2]),task)
        self.assertEqual(preview['references_xyz'],result['references_xyz'])
        self.assertEqual(preview['center_preference'],result['center_preference'])
        self.assertTrue(preview['preview_only'])
        self.assertNotIn('scheduler',preview)

    def test_anchor_event_is_start_and_goal_uses_map_xz(self):
        anchor = (-.5, -.234, 2., 0., .3, 0., .9539)
        task = (-30.2, .456, 13.5)
        result = self.planner.plan(anchor, task, 2, anchor_sequence=4)
        self.assertEqual(result['start_xy'], [-.5, 2.])
        self.assertEqual(result['goal_xy'], [-30.2, 13.5])
        self.assertEqual(result['scheduler']['anchor_pose_xyz_qxyzw'], list(anchor))
        self.assertEqual(result['scheduler']['task_point_xyz'], list(task))
        self.assertEqual(result['scheduler']['segment_number'], 3)
        self.assertEqual(result['scheduler']['anchor_sequence'], 4)
        self.assertFalse(result['scheduler']['sent_to_sru'])
        self.assertEqual(len(result['references_xy']), 15)
        self.assertFalse(result['includes_start'])
        self.assertFalse(result['includes_goal'])
        self.assertEqual(result['references_xyz'], [[x, 0., z] for x, z in result['references_xy']])
        self.assertEqual(len(reference_lines(result)), 15)
        self.assertTrue(all('Y=0.000000' in line for line in reference_lines(result)))
        self.assertTrue(result['all_reference_segments_collision_free'])
        with tempfile.TemporaryDirectory() as tmp:
            path = self.planner.export(result, Path(tmp)/'segment')
            self.assertTrue((path/'route.json').is_file())
            self.assertTrue((path/'reference_points.csv').is_file())
            self.assertTrue((path/'reference_poses.csv').is_file())
            self.assertEqual(len((path/'reference_points.txt').read_text().splitlines()), 15)

    def test_invalid_anchors_never_fall_back_to_previous_or_zero(self):
        invalid = [None, (1, 2, 3), (1, 2, 3, 0, 0, 0, 0),
                   (float('nan'), 2, 3, 0, 0, 0, 1), (1, 2, 3, 0, 0, float('inf'), 1)]
        for anchor in invalid:
            with self.subTest(anchor=anchor), self.assertRaises(SegmentPlanningError):
                validate_anchor_pose(anchor)

    def test_actual_config_blocked_goals_are_not_moved(self):
        for task in [(-6., 0., 13.), (-30., 0., 15.)]:
            status = self.planner.diagnose_goal(task)
            self.assertFalse(status['traversable'])
            self.assertEqual(status['task_point_xyz'], list(task))
            with self.assertRaises(ValueError):
                self.planner.plan((-.5, 0, 2, 0, 0, 0, 1), task, 0, anchor_sequence=1)

    def test_reference_identity_mismatch_is_rejected(self):
        with patch.object(self.planner.data, 'load_planning_map', return_value=self.planner.pm), \
             patch.object(self.planner.data, 'sha256_file', return_value='different-source-db'):
            with self.assertRaises(SegmentPlanningError) as error:
                SegmentAStarPlanner(self.cfg)
            self.assertEqual(error.exception.code, 'map_identity_mismatch')

    def test_map_picked_goals_roundtrip_and_identity_guard(self):
        points = [[-6.6, .695, 5.2], [-30.2, .695, 5.25]]
        with tempfile.TemporaryDirectory() as tmp:
            path = save_task_points(self.planner, points, Path(tmp)/'goals.json')
            self.assertEqual(load_task_points(self.planner, path), points)
            contents = path.read_text()
            for invalid in [[], [[-6, .695, 13]], [[-.5, 0, 2]], [points[0], points[0]]]:
                with self.assertRaises(ValueError):
                    save_task_points(self.planner, invalid, path)
                self.assertEqual(path.read_text(), contents)
            payload = json.loads(contents)
            payload['grid_sha256'] = 'different-grid'
            path.write_text(json.dumps(payload))
            with self.assertRaises(SegmentPlanningError) as error:
                load_task_points(self.planner, path)
            self.assertEqual(error.exception.code, 'task_map_mismatch')

    def test_picker_select_undo_save_cancel_without_camera(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from task_point_picker import TaskPointPicker
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'goals.json'
            picker = TaskPointPicker(self.planner, path)
            try:
                picker.confirm()
                self.assertFalse(picker.accepted)
                self.assertFalse(path.exists())
                picker.add(-6, 13)  # blocked
                self.assertEqual(picker.points, [])
                picker.add(-6.6, 5.2)
                picker.add(-30.2, 5.25)
                picker.undo()
                self.assertEqual(picker.points, [[-6.6, .695, 5.2]])
                picker.confirm()
                self.assertTrue(picker.accepted)
                self.assertEqual(load_task_points(self.planner, path), picker.points)
                previous = path.read_text()
                second = TaskPointPicker(self.planner, path, picker.points)
                second.clear()
                second.cancel()
                self.assertFalse(second.accepted)
                self.assertEqual(path.read_text(), previous)
            finally:
                plt.close('all')


if __name__ == '__main__':
    unittest.main()
