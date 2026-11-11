"""Independent 2D A* and collision-checked reference points. No robot IO.

All xy values mean (cuVSLAM map X, map Z), in metres. By default the
15 references are strictly BETWEEN the separately retained start and goal.
"""
from __future__ import annotations

import csv
import copy
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np

from map2d_data import _segment_cells_grid


class PlanningError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def center_preference_map(pm, decay_length_m=.8):
    """Runtime-only soft cost; preserve source identity and hard obstacle mask."""
    if not math.isfinite(decay_length_m) or decay_length_m <= 0:
        raise ValueError('中心偏好衰减距离必须为有限正数')
    preferred = copy.copy(pm)
    required = float(pm.meta['robot_radius_m'])+float(pm.meta['safety_margin_m'])
    preferred.cost = np.full(pm.cost.shape, np.inf, dtype=np.float64)
    preferred.cost[pm.traversable] = np.exp(-np.maximum(0., pm.clearance_m[pm.traversable]-required)/decay_length_m)
    return preferred


def astar_search(pm, start, goal, cost_weight=2.0, max_expansions=300_000,
                 cancel=None, timeout_s=20.0):
    """8-neighbour weighted-cost A*; octile lower bound, no corner cutting."""
    if not math.isfinite(cost_weight) or cost_weight < 0 or max_expansions <= 0:
        raise ValueError("Invalid search limits or cost weight")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Invalid timeout")
    failure = lambda reason, count=0: dict(status=reason, expanded=count, path_cells=[], total_cost=None)
    if start is None or goal is None or not pm.is_traversable(*start) or not pm.is_traversable(*goal):
        return failure("blocked_endpoint")
    res = pm.meta['resolution_m']
    began = time.monotonic()

    def heuristic(cell):
        dr, dc = abs(cell[0]-goal[0]), abs(cell[1]-goal[1])
        return res*(max(dr, dc)+(math.sqrt(2)-1)*min(dr, dc))

    moves = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    queue = [(heuristic(start), 0.0, start)]
    best, parent = {start: 0.0}, {}
    expanded = 0
    while queue:
        if expanded % 128 == 0:
            if cancel is not None and cancel.is_set():
                return failure('cancelled', expanded)
            if time.monotonic()-began > timeout_s:
                return failure('timeout', expanded)
        _, distance, cell = heapq.heappop(queue)
        if distance > best[cell]:
            continue
        if cell == goal:
            path = [cell]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return dict(status='ok', expanded=expanded, path_cells=path[::-1], total_cost=distance)
        if expanded >= max_expansions:
            return failure('expansion_limit', expanded)
        expanded += 1
        for dr, dc in moves:
            nxt = (cell[0]+dr, cell[1]+dc)
            if not pm.edge_is_free(cell, nxt):
                continue
            step = res*math.hypot(dr, dc)*(1+cost_weight*(pm.cost[cell]+pm.cost[nxt])/2)
            distance_next = distance+step
            if distance_next < best.get(nxt, math.inf):
                best[nxt], parent[nxt] = distance_next, cell
                heapq.heappush(queue, (distance_next+heuristic(nxt), distance_next, nxt))
    return failure('no_path', expanded)


def world_segment_cells(pm, a, b):
    """Exact supercover for arbitrary world endpoints, including grid-edge touches."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
        return []
    if pm.world_to_grid(*a) is None or pm.world_to_grid(*b) is None:
        return []
    origin = np.asarray(pm.meta['origin_xz'])
    res = pm.meta['resolution_m']
    return _segment_cells_grid((a-origin)/res, (b-origin)/res)


def world_segment_is_free(pm, a, b):
    cells = world_segment_cells(pm, a, b)
    return bool(cells) and all(pm.is_traversable(r, c) for r, c in cells)


def check_endpoint(pm, point, name='点'):
    point = np.asarray(point, float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise PlanningError('invalid_coordinate', f'{name}必须是两个有限的米制坐标')
    cell = pm.world_to_grid(*point)
    if cell is None:
        raise PlanningError('out_of_bounds', f'{name}超出地图范围，请在深绿色区域选点')
    if not world_segment_is_free(pm, point, point):
        raise PlanningError('blocked_endpoint', f'{name}不在可通行区域内（或贴着边界），请在深绿色内部选点')
    return tuple(point)


def simplify_path(pm, path, tolerance_m=.30, cancel=None, max_clearance_loss_m=None):
    """Iterative RDP, retaining bends whenever their replacement collides.

    The tolerance bounds geometric deviation from the original weighted A*.
    Every accepted shortcut is independently checked in continuous coordinates.
    """
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('简化容差必须为有限非负数')
    if max_clearance_loss_m is not None and (not math.isfinite(max_clearance_loss_m) or max_clearance_loss_m < 0):
        raise ValueError('简化净空损失上限必须为有限非负数')
    points = np.asarray(path, float)
    edge_clearances = None
    if max_clearance_loss_m is not None:
        edge_clearances = []
        for a, b in zip(points, points[1:]):
            if cancel is not None and cancel.is_set():
                raise PlanningError('cancelled', '规划已取消')
            cells = world_segment_cells(pm, a, b)
            edge_clearances.append(min((float(pm.clearance_m[c]) for c in cells), default=0.))
    keep = {0, len(points)-1}
    pending = [(0, len(points)-1)]
    while pending:
        if cancel is not None and cancel.is_set():
            raise PlanningError('cancelled', '规划已取消')
        first, last = pending.pop()
        if last-first <= 1:
            continue
        a, b = points[first], points[last]
        delta = b-a
        u = np.clip((points[first+1:last]-a) @ delta / max(float(delta@delta), 1e-30), 0, 1)
        distances = np.linalg.norm(points[first+1:last]-(a+u[:, None]*delta), axis=1)
        offset = int(np.argmax(distances))
        if distances[offset] <= tolerance_m:
            cells = world_segment_cells(pm, a, b)
            collision_free = bool(cells) and all(pm.is_traversable(*c) for c in cells)
            clearance_ok = edge_clearances is None or (bool(cells) and
                min(float(pm.clearance_m[c]) for c in cells)+max_clearance_loss_m+1e-9 >=
                min(edge_clearances[first:last]))
            if collision_free and clearance_ok:
                continue
        middle = first+1+offset if distances[offset] > 1e-10 else (first+last)//2
        keep.add(middle)
        pending.extend([(first, middle), (middle, last)])
    return points[sorted(keep)]


def sample_references(anchors, count=15, include_goal=False):
    """Keep every corner; distribute remaining subdivisions by segment length.

    Equal spacing is local to each straight segment, never across a corner.
    Return references and the complete start -> refs -> goal polyline.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('参考点数量必须是正整数')
    anchors = np.asarray(anchors, float)
    lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    if not len(lengths) or np.any(lengths <= 1e-10):
        raise PlanningError('same_endpoint', '起终点距离过近，无法生成互不重复的参考点')
    budget = count if include_goal else count+1
    if len(lengths) > budget:
        raise PlanningError('reference_budget', f'当前路线转折较多，{count} 个点不足以保留必要转角')
    subdivisions = np.ones(len(lengths), dtype=int)
    for _ in range(budget-len(lengths)):
        subdivisions[int(np.argmax(lengths/subdivisions))] += 1
    full = [anchors[0]]
    for a, b, n in zip(anchors, anchors[1:], subdivisions):
        full.extend(b.copy() if k == n else a+(b-a)*(k/int(n)) for k in range(1, int(n)+1))
    full = np.asarray(full)
    return (full[1:] if include_goal else full[1:-1]), full


def plan_reference_path(pm, start_xy, goal_xy, count=15, include_goal=False,
                        cost_weight=2.0, cancel=None, simplification_tolerance_m=.30,
                        max_clearance_loss_m=None):
    began = time.monotonic()
    if not pm.reviewed:
        raise PlanningError('unreviewed_map', '当前地图尚未审核')
    start = check_endpoint(pm, start_xy, '起点')
    goal = check_endpoint(pm, goal_xy, '终点')
    if math.dist(start, goal) <= 1e-7:
        raise PlanningError('same_endpoint', '起终点重合或距离过近，请重新选择')
    search = astar_search(pm, pm.world_to_grid(*start), pm.world_to_grid(*goal),
                          cost_weight=cost_weight, cancel=cancel)
    if search['status'] != 'ok':
        messages = {'no_path': '起终点之间没有可通行路径', 'cancelled': '规划已取消',
                    'timeout': '搜索超时（20 秒）', 'expansion_limit': '搜索达到扩展上限'}
        raise PlanningError(search['status'], messages.get(search['status'], '起终点无法规划'))
    # Retain the actual click coordinates; the grid centres are only search nodes.
    dense = [start, *[pm.grid_to_world(*c) for c in search['path_cells']], goal]
    dense = np.asarray([p for i, p in enumerate(dense) if i == 0 or math.dist(dense[i-1], p) > 1e-10])
    dense[0], dense[-1] = start, goal
    if not all(world_segment_is_free(pm, a, b) for a, b in zip(dense, dense[1:])):
        raise PlanningError('endpoint_connection', '点击点到搜索格心的连接不安全，请重新选点')
    # Same-cell selections can connect directly without a detour through the centre.
    if len(search['path_cells']) == 1:
        dense = np.asarray([start, goal])
    anchors = simplify_path(pm, dense, tolerance_m=simplification_tolerance_m, cancel=cancel,
                            max_clearance_loss_m=max_clearance_loss_m)
    refs, full = sample_references(anchors, count=count, include_goal=include_goal)
    touched = set()
    for a, b in zip(full, full[1:]):
        cells = world_segment_cells(pm, a, b)
        if not cells or not all(pm.is_traversable(*c) for c in cells):
            raise PlanningError('reference_collision', '参考点连线碰撞检查失败，未输出路线')
        touched.update(cells)
    if cancel is not None and cancel.is_set():
        raise PlanningError('cancelled', '规划已取消')
    meta = pm.meta
    return dict(schema_version=1, status='ok', map_directory=str(pm.directory),
                map_id=meta.get('map_id'), map_version=meta.get('map_version'),
                source_db_sha256=meta.get('source_db_sha256'), grid_sha256=meta.get('grid_sha256'),
                annotations_sha256=meta.get('annotations_sha256'),
                coordinate_frame='cuvslam_map_frame', units='metres',
                axes={'x': 'map X', 'y': 'map Z (horizontal, NOT original vertical Y)'},
                resolution_m=meta['resolution_m'], origin_xz=meta['origin_xz'],
                robot_radius_m=meta.get('robot_radius_m'), safety_margin_m=meta.get('safety_margin_m'),
                start_xy=list(start), goal_xy=list(goal), reference_count=count,
                includes_start=False, includes_goal=bool(include_goal),
                simplification_tolerance_m=simplification_tolerance_m,
                max_simplification_clearance_loss_m=max_clearance_loss_m,
                sampling='preserve bends; length-based subdivisions within straight segments',
                references_xy=refs.tolist(), anchors_xy=anchors.tolist(),
                reference_polyline_xy=full.tolist(), astar_path_xy=dense.tolist(),
                astar_cost_weight=cost_weight, astar_total_cost=float(search['total_cost']),
                astar_expanded=search['expanded'],
                astar_length_m=float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()),
                reference_length_m=float(np.linalg.norm(np.diff(full, axis=0), axis=1).sum()),
                min_clearance_m=min(float(pm.clearance_m[c]) for c in touched),
                all_reference_segments_collision_free=True,
                planning_seconds=time.monotonic()-began)


def export_coordinates(result, directory):
    """Create a new output directory; do not overwrite a previous plan."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'route.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (directory/'reference_points.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'x_m_map_X', 'y_m_map_Z'])
        writer.writerows((i, format(x, '.17g'), format(y, '.17g'))
                         for i, (x, y) in enumerate(result['references_xy'], 1))
    with (directory/'endpoints.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['role', 'x_m_map_X', 'y_m_map_Z'])
        writer.writerow(['start', *result['start_xy']])
        writer.writerow(['goal', *result['goal_xy']])
    return directory
