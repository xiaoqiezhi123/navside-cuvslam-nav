# -*- coding: utf-8 -*-
"""map2d_data.py —— 二维规划地图核心数据与算法（V1）

职责（见《二维规划地图 V1 改动说明》§3-§8）：
  - 调用匹配版本的 map_extractor 从 cuVSLAM 保存地图目录（data.mdb）离线导出
    全局 landmarks / poses / edges（source_map.json）；
  - 三值占用栅格（int8：-1 未知 / 0 自由 / 100 障碍，初始全未知）；
  - 人工标注多边形合成（优先级：人工障碍 > 人工未知 > 候选障碍 > 人工自由 > 默认未知）；
  - 机器人圆形包络膨胀（欧氏距离变换）与软代价；
  - 地图包读写（grid.npz + map.json + annotations.json + preview.png）与校验；
  - 供 A* 使用的查询接口：load_planning_map / world_to_grid / grid_to_world /
    is_traversable / edge_is_free。

本模块不加载 cuVSLAM、不启动相机；栅格计算与测试独立于导出器二进制。
坐标约定：与 cuVSLAM 参考地图一致（X 右、Y 下、Z 前），二维平面为 XZ。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from scipy.ndimage import distance_transform_edt

UNKNOWN = -1
FREE = 0
OBSTACLE = 100

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# 坐标与栅格纯函数
# ---------------------------------------------------------------------------

def world_to_grid(x: float, z: float, origin_xz: Tuple[float, float],
                  resolution_m: float, shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """世界坐标 (x, z) → (row, col)；越界返回 None。

    row = floor((z - z_min) / r)，col = floor((x - x_min) / r)；
    有效边界左闭右开 [x_min, x_min + width*r) × [z_min, z_min + height*r)。
    """
    x_min, z_min = origin_xz
    height, width = shape
    if resolution_m <= 0 or not np.isfinite(resolution_m):
        raise ValueError("resolution_m 必须为有限正数")
    if not np.isfinite(x) or not np.isfinite(z):
        return None
    col = int(np.floor((x - x_min) / resolution_m))
    row = int(np.floor((z - z_min) / resolution_m))
    if not (0 <= col < width and 0 <= row < height):
        return None
    return (row, col)


def grid_to_world(row: int, col: int, origin_xz: Tuple[float, float],
                  resolution_m: float, shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    """(row, col) → 格子中心世界坐标 (x, z)；越界返回 None。"""
    height, width = shape
    if not (0 <= col < width and 0 <= row < height):
        return None
    x_min, z_min = origin_xz
    return (x_min + (col + 0.5) * resolution_m, z_min + (row + 0.5) * resolution_m)


def align_bounds(x_min: float, z_min: float, x_max: float, z_max: float,
                 resolution_m: float) -> Tuple[Tuple[float, float], Tuple[int, int]]:
    """把世界范围向外对齐到栅格边界，返回 (origin_xz, (height, width))。"""
    x0 = np.floor(x_min / resolution_m) * resolution_m
    z0 = np.floor(z_min / resolution_m) * resolution_m
    width = int(np.ceil((x_max - x0) / resolution_m))
    height = int(np.ceil((z_max - z0) / resolution_m))
    return (x0, z0), (height, width)


# ---------------------------------------------------------------------------
# 多边形栅格化（标注用）
# ---------------------------------------------------------------------------

def _points_in_polygon(pts: np.ndarray, x, z):
    """Vectorized ray casting; polygon boundaries are handled separately."""
    x, z = np.broadcast_arrays(x, z)
    inside = np.zeros(x.shape, dtype=bool)
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        if a[1] == b[1]:
            continue
        inside ^= ((a[1] > z) != (b[1] > z)) & (
            x < (b[0] - a[0]) * (z - a[1]) / (b[1] - a[1]) + a[0])
    return inside


def _point_in_polygon(px, pz, x, z) -> bool:
    return bool(_points_in_polygon(np.column_stack((px, pz)), x, z))


def _segments_intersect(a, b, c, d) -> bool:
    """Closed segment intersection, including collinear overlap/touching."""
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    def on_segment(p, q, r):
        return np.all(r >= np.minimum(p, q) - 1e-10) and np.all(r <= np.maximum(p, q) + 1e-10)
    ab, cd = b - a, d - c
    v = (cross(ab, c-a), cross(ab, d-a), cross(cd, a-c), cross(cd, b-c))
    eps = 1e-10 * max(1.0, np.linalg.norm(ab), np.linalg.norm(cd))
    if ((v[0] > eps and v[1] < -eps) or (v[0] < -eps and v[1] > eps)) and \
       ((v[2] > eps and v[3] < -eps) or (v[2] < -eps and v[3] > eps)):
        return True
    return any(abs(t) <= eps and on_segment(p, q, r) for t, p, q, r in
               ((v[0], a, b, c), (v[1], a, b, d), (v[2], c, d, a), (v[3], c, d, b)))


def validate_polygon(vertices) -> np.ndarray:
    """Reject zero area, repeated vertices, self touching and overlapping edges."""
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("多边形至少需要 3 个 (x, z) 顶点")
    if not np.all(np.isfinite(pts)):
        raise ValueError("多边形顶点含 NaN/Inf")
    if len(np.unique(pts, axis=0)) != len(pts):
        raise ValueError("多边形含重复顶点（无需重复首顶点）")
    local = pts - pts[0]
    area2 = np.sum(local[:, 0] * np.roll(local[:, 1], -1) - np.roll(local[:, 0], -1) * local[:, 1])
    if abs(area2) < 1e-12:
        raise ValueError("多边形面积为零")
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        prev = pts[(i - 1) % n] - a
        nxt = b - a
        cross = prev[0] * nxt[1] - prev[1] * nxt[0]
        if abs(cross) <= 1e-12 and np.dot(prev, nxt) > 0:
            raise ValueError("多边形相邻边重叠")
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _segments_intersect(a, b, pts[j], pts[(j + 1) % n]):
                raise ValueError("多边形自交或边界自接触")
    return pts


def _segment_cells_grid(start, end):
    """Closed continuous segment in (col,row) units; include every touched cell.

    Split at grid crossings and sample crossings and interval interiors. A corner
    includes all four neighbours; a segment on a grid line includes both sides.
    Unlike Bresenham this also works for wall endpoints away from cell centres.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    delta = end - start
    times = [np.array([0.0, 1.0])]
    for axis in (0, 1):
        if delta[axis] != 0:
            lo, hi = sorted((start[axis], end[axis]))
            crossings = np.arange(np.ceil(lo), np.floor(hi) + 1)
            t = (crossings - start[axis]) / delta[axis]
            times.append(t[(t >= 0) & (t <= 1)])
    t = np.unique(np.concatenate(times))
    t = np.concatenate((t, (t[:-1] + t[1:]) / 2))
    points = start + t[:, None] * delta
    near = np.isclose(points, np.rint(points), rtol=0, atol=1e-9)
    points = np.where(near, np.rint(points), points)
    base = np.floor(points).astype(np.int64)
    cells = set()
    for dx, dz in ((0, 0), (1, 0), (0, 1), (1, 1)):
        keep = (near[:, 0] if dx else np.ones(len(points), bool)) & \
               (near[:, 1] if dz else np.ones(len(points), bool))
        cells.update((int(r-dz), int(c-dx)) for c, r in base[keep])
    return sorted(cells)


def rasterize_polygon(pts, origin_xz, resolution_m, shape, mode):
    """Conservative rasterization without per-cell Python geometry loops.

    Obstacle/unknown includes centres inside AND every cell touched by an edge,
    even sub-cell walls. Free releases interior cells only: boundary cells are
    withheld conservatively, including concave notches between the four corners.
    """
    if mode not in ("conservative", "full_cover"):
        raise ValueError(f"未知栅格化模式: {mode}")
    h, w = shape
    grid_pts = (np.asarray(pts) - np.asarray(origin_xz)) / resolution_m
    lo = np.maximum(0, np.floor(grid_pts.min(axis=0)).astype(int) - 1)
    hi = np.minimum([w-1, h-1], np.floor(grid_pts.max(axis=0)).astype(int))
    if np.any(lo > hi):
        return np.empty((0, 2), dtype=np.int64)
    boundary = set()
    for a, b in zip(grid_pts, np.roll(grid_pts, -1, axis=0)):
        boundary.update((r, c) for r, c in _segment_cells_grid(a, b)
                        if 0 <= r < h and 0 <= c < w)
    mask = np.zeros((hi[1]-lo[1]+1, hi[0]-lo[0]+1), dtype=bool)
    # Vectorized scan conversion, bounded temporaries; no millions of Python tuples.
    xs = np.arange(lo[0], hi[0] + 1)[None, :] + 0.5
    for first in range(int(lo[1]), int(hi[1]) + 1, 128):
        zs = np.arange(first, min(first + 128, hi[1] + 1))[:, None] + 0.5
        mask[first-lo[1]:first-lo[1]+len(zs)] = _points_in_polygon(grid_pts, xs, zs)
    for r, c in boundary:
        if lo[1] <= r <= hi[1] and lo[0] <= c <= hi[0]:
            mask[r-lo[1], c-lo[0]] = mode == "conservative"
    cells = np.argwhere(mask)
    cells += [lo[1], lo[0]]
    return cells


def wall_enclosed_mask(segments, origin_xz, resolution_m, shape, snap_m: float = 0.5,
                       shrink_m: float = 0.0):
    """由墙线段围成的封闭区域 → 内部格掩码（供「由墙围成的可通行区」使用）。

    流程：
      1) 栅格化墙线段；
      2) 端点距离 ≤ snap_m 的对接缝，栅格化补线弥合（手绘小缝隙）；
      3) 剩余「开放端」（deg=1 节点）按最近邻两两配对、连成线段（自动补上未画的下墙/门洞）；
      4) 从地图边界 4 邻域洪泛，无法从边界到达的格子判为「内部」；
      5) 可选 shrink_m > 0 时，把内部掩码向内腐蚀 shrink_m 米（离墙更保守、自由区更小）。

    最近邻配对能正确闭合「建筑外轮廓」的下沿开口；内部隔墙仍保持障碍、不参与内外判定。
    返回 bool 掩码（内部=可通行候选）。
    """
    from collections import defaultdict
    from scipy.ndimage import binary_erosion, label

    segments = [np.asarray(s, dtype=np.float64) for s in (segments or [])]
    h, w = shape
    wall = np.zeros(shape, dtype=bool)

    def rasterize_seg(a, b):
        coords = (np.array([a, b]) - np.asarray(origin_xz)) / resolution_m
        for row, col in _segment_cells_grid(coords[0], coords[1]):
            if 0 <= row < h and 0 <= col < w:
                wall[row, col] = True

    for s in segments:
        if s.shape != (2, 2) or not np.all(np.isfinite(s)):
            raise ValueError(f"墙线段必须为 2 个 (x,z) 端点: {s}")
        rasterize_seg(s[0], s[1])

    pair_connections = []  # 最近邻自动补线的「开放端对」，稍后判断内外、只保留外轮廓对
    if snap_m > 0 and len(segments) >= 2:
        n = len(segments)
        pts = []
        for i, s in enumerate(segments):
            pts.append((i, 0, s[0]))
            pts.append((i, 1, s[1]))
        m = 2 * n
        parent = list(range(m))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(m):
            pi = pts[i][2]
            for j in range(i + 1, m):
                if pts[i][0] == pts[j][0]:
                    continue
                if np.hypot(pi[0] - pts[j][2][0], pi[1] - pts[j][2][1]) <= snap_m:
                    union(i, j)
                    rasterize_seg(pi, pts[j][2])  # 物理弥合对接缝
        groups = defaultdict(list)
        for idx in range(m):
            groups[find(idx)].append(idx)
        node_coord = {g: np.mean([pts[k][2] for k in mem], axis=0) for g, mem in groups.items()}
        deg = defaultdict(int)
        for i in range(n):
            a, b = find(2 * i), find(2 * i + 1)
            if a != b:
                deg[a] += 1
                deg[b] += 1
        open_ends = [g for g in deg if deg[g] == 1]
        remaining = list(open_ends)
        while len(remaining) >= 2:
            a = remaining.pop(0)
            best, best_d = None, np.inf
            for b in remaining:
                d = np.hypot(*(node_coord[a] - node_coord[b]))
                if d < best_d:
                    best_d, best = d, b
            remaining.remove(best)
            pair_connections.append((node_coord[a], node_coord[best]))

    # 把每对自动补线单独栅格化，便于逐一判断该缺口属于「外轮廓开口（补）」还是
    # 「内部门洞（不补）」。判断依据：临时放开这对补线后，若内部区会被边界洪泛到
    # （即内部漏到地图外），则它必须是外轮廓开口，予以保留；否则是内部隔墙门洞，放开。
    pair_masks = []
    for ca, cb in pair_connections:
        pm = np.zeros(shape, dtype=bool)
        coords = (np.array([ca, cb]) - np.asarray(origin_xz)) / resolution_m
        for row, col in _segment_cells_grid(coords[0], coords[1]):
            if 0 <= row < h and 0 <= col < w:
                pm[row, col] = True
        pair_masks.append(pm)

    def outside_of(wall_mask):
        labels, n_lab = label(~wall_mask)
        border = set(labels[0, :].tolist()) | set(labels[-1, :].tolist()) \
            | set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
        border.discard(0)
        outside = np.zeros(shape, dtype=bool)
        for lb in border:
            outside[labels == lb] = True
        return outside

    # 全部补线闭合时的内部区（作为「真正内部」的参考）
    wall_all = wall.copy()
    for pm in pair_masks:
        wall_all |= pm
    enclosed_all = ~outside_of(wall_all) & ~wall_all

    keep_closed = []
    for i, pm in enumerate(pair_masks):
        w_reopen = wall.copy()
        for j, pmj in enumerate(pair_masks):
            if j != i:
                w_reopen |= pmj
        if (enclosed_all & outside_of(w_reopen)).any():
            keep_closed.append(i)  # 外轮廓开口：放开会漏到图外，必须保留

    wall_final = wall.copy()
    for i in keep_closed:
        wall_final |= pair_masks[i]
    enclosed = ~outside_of(wall_final) & ~wall_final
    if shrink_m > 0:
        iters = max(1, int(np.ceil(shrink_m / resolution_m)))
        enclosed = binary_erosion(enclosed, structure=np.ones((3, 3), dtype=bool), iterations=iters)
    return enclosed


# ---------------------------------------------------------------------------
# 栅格合成与膨胀
# ---------------------------------------------------------------------------

def build_occupancy(origin_xz: Tuple[float, float], resolution_m: float,
                    shape: Tuple[int, int],
                    candidate_cells: List[Tuple[int, int]],
                    annotations: Optional[dict] = None,
                    free_from_walls: bool = False,
                    wall_snap_m: float = 0.5,
                    free_shrink_m: float = 0.0) -> np.ndarray:
    """人工障碍 > 人工未知 > 未剔除候选障碍 > 人工自由 > 默认未知。

    candidate_remove removes candidate cells touched by the selected polygon;
    it does not erase manual obstacles or declare any unknown space free.

    free_from_walls=True 时，忽略 annotations 里的 free 多边形，改为用墙线段
    （obstacle_segments）围成的封闭区域内部作为自由区（见 wall_enclosed_mask）；
    free_shrink_m 把该自由区再向内收缩（离墙更保守）。
    """
    height, width = shape
    grid = np.full((height, width), UNKNOWN, dtype=np.int8)
    annotations = annotations or {}
    candidates = np.zeros(shape, dtype=bool)
    for r, c in candidate_cells:
        if 0 <= r < height and 0 <= c < width:
            candidates[r, c] = True
    for verts in annotations.get("candidate_remove", []):
        cells = rasterize_polygon(validate_polygon(verts), origin_xz, resolution_m, shape, "conservative")
        candidates[cells[:, 0], cells[:, 1]] = False
    if free_from_walls:
        enclosed = wall_enclosed_mask(annotations.get("obstacle_segments", []),
                                      origin_xz, resolution_m, shape,
                                      snap_m=wall_snap_m, shrink_m=free_shrink_m)
        grid[enclosed] = FREE
    else:
        for verts in annotations.get("free", []):
            cells = rasterize_polygon(validate_polygon(verts), origin_xz, resolution_m, shape, "full_cover")
            grid[cells[:, 0], cells[:, 1]] = FREE
    grid[candidates] = OBSTACLE
    for name, value in (("unknown", UNKNOWN), ("obstacle", OBSTACLE)):
        for verts in annotations.get(name, []):
            cells = rasterize_polygon(validate_polygon(verts), origin_xz, resolution_m, shape, "conservative")
            grid[cells[:, 0], cells[:, 1]] = value

    # 墙线段（WallSeg 标注）：两两端点，保守覆盖所有相交格，按障碍优先级。
    for seg in annotations.get("obstacle_segments", []):
        seg = np.asarray(seg, dtype=np.float64)
        if seg.shape != (2, 2) or not np.all(np.isfinite(seg)):
            raise ValueError(f"墙线段必须为 2 个 (x,z) 端点: {seg}")
        c0 = world_to_grid(seg[0, 0], seg[0, 1], origin_xz, resolution_m, shape)
        c1 = world_to_grid(seg[1, 0], seg[1, 1], origin_xz, resolution_m, shape)
        if c0 is None or c1 is None:
            raise ValueError(f"墙线段端点越界: {seg}")
        coords = (seg - np.asarray(origin_xz)) / resolution_m
        for row, col in _segment_cells_grid(coords[0], coords[1]):
            if 0 <= row < height and 0 <= col < width:
                grid[row, col] = OBSTACLE
    return grid


def compute_inflation(occupancy: np.ndarray, resolution_m: float,
                      robot_radius_m: float, safety_margin_m: float,
                      soft_band_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由原始占用栅格计算 clearance / traversable / cost（不改动 occupancy）。

    保守圆形包络：R_required = robot_radius_m + safety_margin_m。
    只对自由掩码补零外圈后做欧氏距离变换（sampling=r），
    clearance = max(0, 中心距 - sqrt(2)*r)（保守下界，语义写入元数据）。
    traversable = 自由 且 clearance > R_required + eps；其余（障碍/未知/地图外）不可通行。
    cost = clip(1 - (clearance - R_required)/soft_band, 0, 1)（仅可通行格），否则 +inf。
    """
    if resolution_m <= 0 or not np.isfinite(resolution_m):
        raise ValueError("resolution_m 必须为有限正数")
    if robot_radius_m <= 0 or not np.isfinite(robot_radius_m):
        raise ValueError("robot_radius_m 必须为有限正数")
    if safety_margin_m < 0 or not np.isfinite(safety_margin_m):
        raise ValueError("safety_margin_m 必须为有限非负数")
    if soft_band_m <= 0 or not np.isfinite(soft_band_m):
        raise ValueError("soft_band_m 必须为有限正数")
    free_mask = occupancy == FREE
    padded = np.pad(free_mask, 1, mode="constant", constant_values=False)
    edt = distance_transform_edt(padded, sampling=resolution_m)
    center_dist = edt[1:-1, 1:-1]
    clearance = np.maximum(0.0, center_dist - np.sqrt(2.0) * resolution_m)
    r_required = robot_radius_m + safety_margin_m
    eps = 1e-9
    traversable = free_mask & (clearance > r_required + eps)
    band = np.clip(1.0 - (clearance - r_required) / soft_band_m, 0.0, 1.0)
    cost = np.full_like(clearance, np.inf, dtype=np.float64)
    cost[traversable] = band[traversable]
    return clearance, traversable, cost


# ---------------------------------------------------------------------------
# 查询接口（A* 用）
# ---------------------------------------------------------------------------

def edge_is_free(occ: np.ndarray, traversable: np.ndarray,
                 cell_a: Tuple[int, int], cell_b: Tuple[int, int]) -> bool:
    """两格是否可直接通行：
      - 8 邻域；斜向必须同时检查两个相邻正交格（禁止穿角）；
      - 线段查询保守覆盖所有相交格子（不只检查端点）。
    """
    (ra, ca), (rb, cb) = cell_a, cell_b
    h, w = occ.shape
    if not (0 <= ra < h and 0 <= ca < w and 0 <= rb < h and 0 <= cb < w):
        return False
    if not (traversable[ra, ca] and traversable[rb, cb]):
        return False
    dr, dc = rb - ra, cb - ca
    if max(abs(dr), abs(dc)) > 1:
        # 长线段：超覆盖光栅化，所有相交格子都必须可通行
        for (r, c) in _supercover_line(ra, ca, rb, cb):
            if not traversable[r, c]:
                return False
        return True
    if abs(dr) == 1 and abs(dc) == 1:
        # 斜向：穿角检查
        if not traversable[ra, cb] or not traversable[rb, ca]:
            return False
    return True


def _supercover_line(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """Supercover between cell centres, including corner-touching neighbours."""
    return _segment_cells_grid((c0 + 0.5, r0 + 0.5), (c1 + 0.5, r1 + 0.5))


# ---------------------------------------------------------------------------
# 导出器调用与 source_map.json
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_extractor(extractor: Path, map_dir: Path, timeout_s: float = 600.0) -> dict:
    """把静态源地图复制到临时目录，用只读方式调用 map_extractor 导出 JSON。

    返回解析后的 dict；源目录不创建锁文件、不写回。
    """
    if not extractor.is_file() or not os.access(extractor, os.X_OK):
        raise RuntimeError(f"导出器不可执行: {extractor}")
    db = map_dir / "data.mdb"
    if not db.is_file() or db.stat().st_size == 0:
        raise RuntimeError(f"地图目录缺少有效 data.mdb: {map_dir}")
    with tempfile.TemporaryDirectory(prefix="map2d_src_") as src_tmp, \
            tempfile.TemporaryDirectory(prefix="map2d_out_") as out_tmp:
        shutil.copy2(db, Path(src_tmp) / "data.mdb")
        t0 = time.monotonic()
        proc = subprocess.run(
            [str(extractor), f"--map_path={src_tmp}"],
            cwd=out_tmp,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"map_extractor 退出码 {proc.returncode}:\n"
                f"{proc.stdout.decode(errors='replace')[-2000:]}\n"
                f"{proc.stderr.decode(errors='replace')[-2000:]}")
        out_json = Path(out_tmp) / "map.json"
        if not out_json.is_file():
            raise RuntimeError("map_extractor 未生成 map.json")
        data = json.loads(out_json.read_text(encoding="utf-8"))
        _validate_source_json(data)
        return data


def _validate_source_json(data: dict) -> None:
    if "landmarks" not in data or not isinstance(data["landmarks"], list):
        raise RuntimeError("导出 JSON 缺少 landmarks 列表")
    if not data["landmarks"]:
        raise RuntimeError("导出 JSON 无任何地图点")
    ids = set()
    for lm in data["landmarks"]:
        if not isinstance(lm.get("id"), int) or "pose" not in lm:
            raise RuntimeError("地图点结构非法: 需要 int id 与 pose")
        p = lm["pose"]
        for k in ("x", "y", "z"):
            v = p.get(k)
            if not isinstance(v, (int, float)) or not np.isfinite(float(v)):
                raise RuntimeError(f"地图点 {lm.get('id')} 的 {k} 坐标非法: {v}")
        if lm["id"] in ids:
            raise RuntimeError(f"地图点 id 重复: {lm['id']}")
        ids.add(lm["id"])
    if "poses" not in data or not isinstance(data["poses"], list):
        raise RuntimeError("导出 JSON 缺少 poses 列表")
    if "edges" not in data or not isinstance(data["edges"], list):
        raise RuntimeError("导出 JSON 缺少 edges 列表")


def landmark_xyz(data: dict) -> np.ndarray:
    """landmarks → (N,3) 全局 XYZ 数组（cuVSLAM 系：X 右、Y 下、Z 前）。"""
    pts = np.array([[lm["pose"]["x"], lm["pose"]["y"], lm["pose"]["z"]]
                    for lm in data["landmarks"]], dtype=np.float64)
    return pts


def exclude_landmarks(pts: np.ndarray, ids: np.ndarray, erased_ids) -> tuple:
    """按擦除点 id 剔除地图点（标注橡皮擦语义：被擦除的点不再参与障碍判定）。

    返回 (过滤后 pts, 过滤后 ids)。erased_ids 为可迭代的 int id 集合。
    """
    pts = np.asarray(pts, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    erased = set(int(i) for i in (erased_ids or []))
    if not erased:
        return pts, ids
    keep = ~np.isin(ids, np.array(sorted(erased), dtype=np.int64))
    return pts[keep], ids[keep]


def map_stats(data: dict) -> dict:
    pts = landmark_xyz(data)
    return {
        "num_landmarks": len(data["landmarks"]),
        "num_keyframes": len(data["poses"]),
        "num_edges": len(data["edges"]),
        "xyz_min": [float(v) for v in pts.min(axis=0)],
        "xyz_max": [float(v) for v in pts.max(axis=0)],
    }


# ---------------------------------------------------------------------------
# 地图包读写
# ---------------------------------------------------------------------------

@dataclass
class PlanningMap:
    """可重复加载的地图包对象。"""
    directory: Path
    meta: dict
    occupancy: np.ndarray
    traversable: np.ndarray
    clearance_m: np.ndarray
    cost: np.ndarray
    reviewed: bool

    def world_to_grid(self, x: float, z: float) -> Optional[Tuple[int, int]]:
        return world_to_grid(x, z, self.meta["origin_xz"],
                             self.meta["resolution_m"], self.occupancy.shape)

    def grid_to_world(self, row: int, col: int) -> Optional[Tuple[float, float]]:
        return grid_to_world(row, col, self.meta["origin_xz"],
                             self.meta["resolution_m"], self.occupancy.shape)

    def is_traversable(self, row: int, col: int) -> bool:
        h, w = self.occupancy.shape
        if not (0 <= row < h and 0 <= col < w):
            return False
        return bool(self.traversable[row, col])

    def edge_is_free(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return edge_is_free(self.occupancy, self.traversable, a, b)


def save_map_package(out_dir: Path, meta: dict, occupancy: np.ndarray,
                     traversable: np.ndarray, clearance_m: np.ndarray,
                     cost: np.ndarray, annotations: dict,
                     source_map: Optional[dict] = None) -> Path:
    """写地图包到 out_dir（版本化目录）。返回发布目录。

    先写临时目录，校验通过后再发布；默认不覆盖已存在版本。
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"输出目录已存在且非空: {out_dir}")
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        shapes = [occupancy.shape, traversable.shape, clearance_m.shape, cost.shape]
        if any(s != occupancy.shape for s in shapes):
            raise ValueError("四个栅格数组形状不一致")
        if occupancy.dtype != np.int8:
            occupancy = occupancy.astype(np.int8)
        np.savez_compressed(tmp / "grid.npz", occupancy=occupancy,
                            traversable=traversable, clearance_m=clearance_m, cost=cost)
        (tmp / "annotations.json").write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
        if source_map is not None:
            (tmp / "source_map.json").write_text(
                json.dumps(source_map, ensure_ascii=False), encoding="utf-8")
        meta = dict(meta)
        meta.setdefault("schema_version", SCHEMA_VERSION)
        meta.setdefault("reviewed", False)
        if source_map is not None:
            meta["source_geometry_sha256"] = sha256_file(tmp / "source_map.json")
        meta["grid_sha256"] = sha256_file(tmp / "grid.npz")
        meta["annotations_sha256"] = sha256_file(tmp / "annotations.json")
        (tmp / "map.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        # 全部写入后再发布
        tmp.rename(out_dir)
        return out_dir
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def load_map_package(directory: Path, expected_map_id: Optional[str] = None,
                     require_reviewed: bool = True) -> PlanningMap:
    """加载并校验地图包。规划用途必须 require_reviewed=True（拒绝未审核地图）。"""
    directory = Path(directory)
    meta_path = directory / "map.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"缺少 map.json: {directory}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"schema_version 不匹配: {meta.get('schema_version')}")
    if expected_map_id is not None and meta.get("map_id") != expected_map_id:
        raise RuntimeError(f"map_id 不匹配: 期望 {expected_map_id}, 实际 {meta.get('map_id')}")
    if require_reviewed and not meta.get("reviewed", False):
        raise RuntimeError(f"地图未通过审核（reviewed=false），不能用于规划: {directory}")
    grid_path = directory / "grid.npz"
    if not grid_path.is_file():
        raise FileNotFoundError(f"缺少 grid.npz: {directory}")
    if sha256_file(grid_path) != meta.get("grid_sha256"):
        raise RuntimeError("grid.npz 校验和不匹配（文件损坏或被篡改）")
    for name, key in (("annotations.json", "annotations_sha256"),
                      ("source_map.json", "source_geometry_sha256")):
        if meta.get(key) and (not (directory / name).is_file() or
                              sha256_file(directory / name) != meta[key]):
            raise RuntimeError(f"{name} 校验和不匹配")
    with np.load(grid_path, allow_pickle=False) as npz:
        occupancy = npz["occupancy"]
        traversable = npz["traversable"]
        clearance_m = npz["clearance_m"]
        cost = npz["cost"]
    if occupancy.shape != traversable.shape or occupancy.shape != clearance_m.shape \
            or occupancy.shape != cost.shape:
        raise RuntimeError("栅格数组形状不一致")
    if not set(np.unique(occupancy)).issubset({UNKNOWN, FREE, OBSTACLE}):
        raise RuntimeError("occupancy 含非法数值")
    if occupancy.ndim != 2 or occupancy.dtype != np.int8 or traversable.dtype != bool:
        raise RuntimeError("栅格维数或数据类型非法")
    if (meta.get("height", occupancy.shape[0]), meta.get("width", occupancy.shape[1])) != occupancy.shape:
        raise RuntimeError("元数据尺寸与栅格不一致")
    if not np.all(np.isfinite(clearance_m)) or np.any(clearance_m < 0):
        raise RuntimeError("clearance_m 非法")
    if np.any(traversable & (occupancy != FREE)):
        raise RuntimeError("非自由格不能通行")
    if not np.all(np.isfinite(cost[traversable])) or np.any(cost[traversable] < 0) or np.any(cost[traversable] > 1) or not np.all(np.isposinf(cost[~traversable])):
        raise RuntimeError("cost 语义非法")
    return PlanningMap(directory=directory, meta=meta, occupancy=occupancy,
                       traversable=traversable, clearance_m=clearance_m, cost=cost,
                       reviewed=bool(meta.get("reviewed", False)))


def load_planning_map(directory: Path, expected_map_id: Optional[str] = None) -> PlanningMap:
    """A* 规划入口：拒绝未审核地图（编辑器草稿请用 load_map_package(..., require_reviewed=False)）。"""
    return load_map_package(directory, expected_map_id=expected_map_id, require_reviewed=True)
