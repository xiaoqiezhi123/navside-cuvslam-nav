"""Read-only bridge from scheduler anchor poses to the existing 2D A*.

This module never starts cameras, sends NavSide commands, or changes task points.
Inputs retain cuVSLAM's global XYZ/quaternion convention; planning uses (X,Z).
"""
from __future__ import annotations

import importlib
import csv
import json
import math
from pathlib import Path
import sys

REFERENCE_Y_M = 0.0  # 参考点打印/导出的平面占位值，不参与 A* 或模型高度
TASK_POINT_Y_M = 0.695  # 保持已有选点文件格式


class SegmentPlanningError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def finite_vector(values, size, name):
    try:
        vector = tuple(float(v) for v in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SegmentPlanningError('invalid_input', f'{name}必须包含 {size} 个有限数值') from exc
    if len(vector) != size or not all(math.isfinite(v) for v in vector):
        raise SegmentPlanningError('invalid_input', f'{name}必须包含 {size} 个有限数值')
    return vector


def validate_anchor_pose(pose):
    pose = finite_vector(pose, 7, '重定位位姿 (X,Y,Z,qx,qy,qz,qw)')
    if math.hypot(*pose[3:]) < 1e-8:
        raise SegmentPlanningError('invalid_anchor_rotation', '重定位位姿四元数不能为零')
    return pose


def _load_modules(repo):
    directory = (Path(repo)/'orbbec').resolve()
    names = ('map2d_data', 'map2d_astar')
    for name in names:
        path = directory/(name+'.py')
        if not path.is_file():
            raise SegmentPlanningError('missing_planner', f'缺少 A* 模块：{path}')
        cached = sys.modules.get(name)
        if cached is not None and Path(cached.__file__).resolve() != path:
            raise SegmentPlanningError('planner_module_conflict', f'{name} 已从另一个工程加载')
    sys.path.insert(0, str(directory))
    try:
        return tuple(importlib.import_module(name) for name in names)
    finally:
        sys.path.remove(str(directory))


class SegmentAStarPlanner:
    """Load/check the map once, then plan each segment from a fresh anchor event."""

    def __init__(self, cfg):
        settings = cfg.get('astar', {})
        if not isinstance(settings, dict):
            raise SegmentPlanningError('invalid_config', 'astar 必须是配置字典')
        package = settings.get('map_package')
        if not isinstance(package, str) or not package.strip():
            raise SegmentPlanningError('invalid_config', '请设置 astar.map_package')
        include_goal = settings.get('include_goal', False)
        if not isinstance(include_goal, bool):
            raise SegmentPlanningError('invalid_config', 'astar.include_goal 必须是 true/false')
        self.include_goal = include_goal
        self.include_start = settings.get('include_start', False)
        if not isinstance(self.include_start, bool) or (self.include_start and not include_goal):
            raise SegmentPlanningError('invalid_config', 'include_start=true 时必须同时 include_goal=true')
        self.repo = Path(cfg['cuvslam_repo']).resolve()
        self.data, self.algorithm = _load_modules(self.repo)
        package = Path(package).expanduser()
        if not package.is_absolute():
            package = self.repo/package
        self.pm = self.data.load_planning_map(package.resolve(), expected_map_id=str(cfg['ref_map']))
        # Map names alone do not establish coordinate identity. Verify the actual
        # reference database used by the SLAM launcher against the planning package.
        reference_db = self.repo/'orbbec'/str(cfg['ref_map'])/'data.mdb'
        if not reference_db.is_file():
            raise SegmentPlanningError('missing_reference_map', f'找不到定位参考地图：{reference_db}')
        if self.data.sha256_file(reference_db) != self.pm.meta['source_db_sha256']:
            raise SegmentPlanningError('map_identity_mismatch', '定位参考数据库与 A* 地图源指纹不一致')
        self.reference_db = reference_db
        center = settings.get('center_preference', {})
        if not isinstance(center, dict) or not isinstance(center.get('enabled', False), bool):
            raise SegmentPlanningError('invalid_config', 'astar.center_preference.enabled 必须为 true/false')
        self.center_enabled = center.get('enabled', False)
        self.center_settings = dict(enabled=self.center_enabled)
        self.plan_options = {}
        if self.center_enabled:
            defaults = dict(decay_length_m=.8, cost_weight=4., simplification_tolerance_m=.10,
                            max_clearance_loss_m=.05)
            for name, default in defaults.items():
                value = center.get(name, default)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise SegmentPlanningError('invalid_config', f'astar.center_preference.{name} 必须为有限非负数')
                if name == 'decay_length_m' and value == 0:
                    raise SegmentPlanningError('invalid_config', '中心偏好衰减距离必须大于 0')
                self.center_settings[name] = float(value)
            self.pm = self.algorithm.center_preference_map(self.pm, self.center_settings['decay_length_m'])
            self.plan_options = {name:self.center_settings[name] for name in defaults if name != 'decay_length_m'}

    def diagnose_goal(self, task_point):
        x, y, z = finite_vector(task_point, 3, '任务点 (X,Y,Z)')
        cell = self.pm.world_to_grid(x, z)
        status = dict(task_point_xyz=[x, y, z], map_xz=[x, z], cell=cell,
                      traversable=False, clearance_m=None)
        if cell is not None:
            status['clearance_m'] = float(self.pm.clearance_m[cell])
            status['traversable'] = self.algorithm.world_segment_is_free(self.pm, (x, z), (x, z))
        return status

    def _plan_geometry(self, start, goal, cancel=None):
        """Shared geometry/options for real anchors and explicitly labelled previews."""
        result = self.algorithm.plan_reference_path(
            self.pm, (start[0], start[2]), (goal[0], goal[2]),
            count=13 if self.include_start else 15,
            include_goal=False if self.include_start else self.include_goal, cancel=cancel, **self.plan_options)
        result['center_preference'] = dict(self.center_settings)
        if self.include_start:
            # Core already checks this complete 15-position polyline. Reuse it
            # including both endpoints; never resample a second time downstream.
            result['references_xy'] = result['reference_polyline_xy']
            result.update(reference_count=15, includes_start=True, includes_goal=True)
        result['references_xyz'] = [[x, REFERENCE_Y_M, z] for x, z in result['references_xy']]
        result['reference_xyz_convention'] = f'map X, fixed Y={REFERENCE_Y_M:g} m, map Z; display/export only, not measured vertical Y'
        return result

    def preview(self, start_task_point, end_task_point, cancel=None):
        """Preview selected goal -> goal; never invent a localization/segment event."""
        start, goal = validate_task_points(self, [start_task_point, end_task_point])
        result = self._plan_geometry(start, goal, cancel)
        result.update(preview_only=True, start_source='selected_task_point',
                      note='选点预览；正式导航必须用本次重定位起点重新规划，不下发本预览。')
        return result

    def plan(self, anchor_pose, task_point, segment_index, *, anchor_sequence, cancel=None):
        anchor = validate_anchor_pose(anchor_pose)
        goal = finite_vector(task_point, 3, '任务点 (X,Y,Z)')
        if isinstance(segment_index, bool) or not isinstance(segment_index, int) or segment_index < 0:
            raise SegmentPlanningError('invalid_input', '段下标必须为非负整数')
        if isinstance(anchor_sequence, bool) or not isinstance(anchor_sequence, int) or anchor_sequence < 1:
            raise SegmentPlanningError('invalid_input', '重定位序号必须为正整数')
        # No fallback to VIO, previous task coordinates, or an old anchor. No snapping.
        result = self._plan_geometry(anchor, goal, cancel)
        result['scheduler'] = dict(segment_index=segment_index, segment_number=segment_index+1,
                                   anchor_sequence=anchor_sequence,
                                   start_source='SLAM anchor=ok event',
                                   anchor_pose_xyz_qxyzw=list(anchor), task_point_xyz=list(goal),
                                   sent_to_sru=False)
        return result

    def export(self, result, directory):
        directory = self.algorithm.export_coordinates(result, directory)
        with (directory/'reference_poses.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'X_map_m', 'Y_fixed_m', 'Z_map_m'])
            writer.writerows((i, *p) for i, p in enumerate(result['references_xyz'], 1))
        (directory/'reference_points.txt').write_text('\n'.join(reference_lines(result))+'\n', encoding='utf-8')
        return directory


def reference_lines(result):
    number = result['scheduler']['segment_number']
    return [f"[A*] 段 {number} 参考点 {i:02d}/15: X={x:.6f} Y={y:.6f} Z={z:.6f}"
            for i, (x, y, z) in enumerate(result['references_xyz'], 1)]


def save_task_points(planner, points, path):
    """Persist map-picked goals with map identity; never write the YAML task list."""
    import os
    import tempfile
    validated = validate_task_points(planner, points)
    meta = planner.pm.meta
    payload = dict(schema_version=1, source='planning_map_picker',
                   map_id=meta['map_id'], map_version=meta['map_version'],
                   source_db_sha256=meta['source_db_sha256'], grid_sha256=meta['grid_sha256'],
                   axes='X=map X, Y=fixed 0.695 m, Z=map Z', task_points_xyz=validated)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def validate_task_points(planner, points):
    if not isinstance(points, (list, tuple)) or not points:
        raise SegmentPlanningError('empty_task_points', '请在地图上至少选择一个目标任务点')
    result = []
    for i, point in enumerate(points, 1):
        x, y, z = finite_vector(point, 3, f'任务点 {i}')
        if not math.isclose(y, TASK_POINT_Y_M, rel_tol=0, abs_tol=1e-9):
            raise SegmentPlanningError('invalid_height', f'任务点 {i} 的展示 Y 必须为 {TASK_POINT_Y_M}')
        if not planner.diagnose_goal((x, y, z))['traversable']:
            raise SegmentPlanningError('blocked_task_point', f'任务点 {i} 不在地图可通行区域内部')
        if result and math.hypot(x-result[-1][0], z-result[-1][2]) <= 1e-7:
            raise SegmentPlanningError('duplicate_task_point', f'任务点 {i} 与前一点重合')
        result.append([x, TASK_POINT_Y_M, z])
    return result


def load_task_points(planner, path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or payload.get('schema_version') != 1 or payload.get('source') != 'planning_map_picker':
        raise SegmentPlanningError('invalid_task_file', '任务点文件不是当前地图选点格式，请重新选点')
    for key in ('map_id', 'source_db_sha256', 'grid_sha256'):
        if payload.get(key) != planner.pm.meta[key]:
            raise SegmentPlanningError('task_map_mismatch', f'任务点文件与当前规划地图不匹配：{key}')
    return validate_task_points(planner, payload.get('task_points_xyz'))
