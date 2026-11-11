# -*- coding: utf-8 -*-
"""map2d_builder.py —— 二维规划地图制作命令行（V1）

按《二维规划地图 V1 改动说明》§4.1.2 的契约：
  必须参数：--map-dir --extractor --output --resolution --robot-radius-m --safety-margin-m
  可选参数：--map-id --ground-y --height-min --height-max --soft-band-m
            --annotations --review --force
流程：
  data.mdb → 校验/指纹 → map_extractor 导出 source_map.json
           → 全局点统计 → （可选）高度过滤生成候选障碍
           → 人工标注合成三值栅格 → 包络膨胀/软代价
           → 地图包（grid.npz + map.json + annotations.json + preview.png）

不接相机、不启动 VIO、不实现 A*。地面高度未配置时不做自动障碍识别，
栅格保持未知，仅输出点云底图供人工核验。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import map2d_data as m2d


def build_preview(out_dir: Path, pts: np.ndarray, occ: np.ndarray,
                  traversable: np.ndarray, origin_xz, resolution_m: float,
                  candidates: np.ndarray | None) -> Path:
    """XZ 俯视底图：特征点（散点）+ 栅格叠加。Z 向上、等比例。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字体（无 CJK 字体时回退英文标题，避免预览出现方框）
    from matplotlib import font_manager
    _cjk = [f.name for f in font_manager.fontManager.ttflist
            if "CJK" in f.name or "WenQuanYi" in f.name]
    if _cjk:
        plt.rcParams["font.family"] = _cjk[0]

    x_min, z_min = origin_xz
    h, w = occ.shape
    extent = [x_min, x_min + w * resolution_m, z_min, z_min + h * resolution_m]
    fig, ax = plt.subplots(figsize=(max(8, w / 200), max(8, h / 200)), dpi=120)
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 2], s=0.3, c="0.35", label="地图特征点", rasterized=True)
    if candidates is not None and len(candidates):
        ax.scatter(candidates[:, 0], candidates[:, 2], s=1.5, c="tab:red",
                   label="候选障碍点", rasterized=True)
    img = np.full((h, w, 4), (0, 0, 0, 0), dtype=np.float32)
    img[occ == m2d.OBSTACLE] = (0.0, 0.0, 0.0, 0.55)
    img[occ == m2d.FREE] = (0.0, 0.9, 0.0, 0.18)
    img[traversable] = (0.0, 0.5, 0.0, 0.5)
    ax.imshow(img, extent=extent, origin="lower", interpolation="nearest")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("二维规划地图预览（XZ 俯视，Z 向上）")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(markerscale=8, loc="upper right")
    path = out_dir / "preview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="二维规划地图制作（V1，XZ 平面）")
    p.add_argument("--map-dir", required=True, help="cuVSLAM 保存地图目录（含 data.mdb）")
    p.add_argument("--extractor", required=True, help="map_extractor 可执行文件路径")
    p.add_argument("--output", required=True, help="地图包输出目录（版本化，不允许覆盖非空）")
    p.add_argument("--resolution", type=float, required=True, help="栅格分辨率（米）")
    p.add_argument("--robot-radius-m", type=float, required=True, help="机器人圆形包络半径（米）")
    p.add_argument("--safety-margin-m", type=float, required=True, help="安全余量（米，≥0）")
    p.add_argument("--map-id", default=None, help="地图标识（默认取目录名）")
    p.add_argument("--map-version", default=None, help="版本名（默认取输出目录名）")
    p.add_argument("--ground-y", type=float, default=None,
                   help="地面高度 Y（已知才填；不填则关闭高度过滤，不做自动障碍识别）")
    p.add_argument("--height-min", type=float, default=0.0, help="碰撞高度下限（相对地面，米）")
    p.add_argument("--height-max", type=float, default=1.5, help="碰撞高度上限（相对地面，米）")
    p.add_argument("--soft-band-m", type=float, default=0.3, help="软代价带宽（米，>0）")
    p.add_argument("--annotations", default=None,
                   help="人工标注 JSON（free/obstacle/unknown 多边形，米制 (x,z) 顶点）")
    p.add_argument("--review", action="store_true",
                   help="标记为已审核（需人工核验后使用；默认 reviewed=false）")
    p.add_argument("--no-preview", action="store_true", help="不生成 preview.png")
    p.add_argument("--force", action="store_true",
                   help="覆盖已存在的输出目录（仅草稿迭代用；已发布版本仍默认拒绝覆盖）")
    p.add_argument("--free-from-walls", action="store_true",
                   help="忽略 free 多边形，改为用墙线段(obstacle_segments)围成的封闭区域作为可通行区")
    p.add_argument("--wall-snap-m", type=float, default=0.5,
                   help="墙线段端点弥合距离（米，弥合手绘小缝隙；默认 0.5）")
    p.add_argument("--free-shrink-m", type=float, default=0.0,
                   help="自由区向内收缩量（米，>0 时离墙更保守、浅绿更少；默认 0）")
    args = p.parse_args(argv)

    map_dir = Path(args.map_dir).resolve()
    extractor = Path(args.extractor).resolve()
    out_dir = Path(args.output).resolve()
    db = map_dir / "data.mdb"
    if not db.is_file() or db.stat().st_size == 0:
        print(f"错误：{map_dir} 缺少有效 data.mdb", file=sys.stderr)
        return 1
    for v, name in ((args.robot_radius_m, "robot-radius-m"),
                    (args.safety_margin_m, "safety-margin-m")):
        if not np.isfinite(v) or (name == "robot-radius-m" and v <= 0) or v < 0:
            print(f"错误：{name} 非法（{v}）", file=sys.stderr)
            return 1
    if args.resolution <= 0 or not np.isfinite(args.resolution):
        print(f"错误：--resolution 非法（{args.resolution}）", file=sys.stderr)
        return 1
    if args.soft_band_m <= 0 or not np.isfinite(args.soft_band_m):
        print(f"错误：--soft-band-m 非法（{args.soft_band_m}）", file=sys.stderr)
        return 1
    if args.free_shrink_m < 0 or not np.isfinite(args.free_shrink_m):
        print(f"错误：--free-shrink-m 非法（{args.free_shrink_m}）", file=sys.stderr)
        return 1

    if not np.isfinite(args.height_min) or not np.isfinite(args.height_max) or args.height_min > args.height_max:
        p.error("高度区间必须有限且 min <= max")
    if args.ground_y is not None and not np.isfinite(args.ground_y):
        p.error("ground-y 必须为有限数")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            p.error("输出目录非空，请选择新版本或用 --force 重建草稿")
        old_meta = out_dir / "map.json"
        if old_meta.is_file() and json.loads(old_meta.read_text()).get("reviewed"):
            p.error("不能覆盖已审核版本，请使用新目录")
    annotations = {"free": [], "obstacle": [], "unknown": [], "candidate_remove": [],
                   "obstacle_segments": [], "erased_point_ids": []}
    if args.annotations:
        user_ann = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
        for k in annotations:
            annotations[k] = list(user_ann.get(k, []))
        if "erased_point_ids" in user_ann:
            annotations["erased_point_ids"] = list(user_ann["erased_point_ids"])

    print(f"[1/5] 源地图校验: {map_dir}")
    db_sha = m2d.sha256_file(db)
    db_size = db.stat().st_size
    print(f"      data.mdb sha256={db_sha[:16]}... size={db_size}")
    print(f"[2/5] 调用导出器: {extractor}")
    source = m2d.run_extractor(extractor, map_dir)
    if m2d.sha256_file(db) != db_sha:
        raise RuntimeError("导出期间源数据库发生变化，请停止写入后重试")
    stats = m2d.map_stats(source)
    print(f"      地图点={stats['num_landmarks']} 关键帧={stats['num_keyframes']} 边={stats['num_edges']}")
    print(f"      XYZ 范围: min={stats['xyz_min']} max={stats['xyz_max']}")

    print("[3/5] 合成三值栅格")
    pts = m2d.landmark_xyz(source)
    ids = np.array([lm["id"] for lm in source["landmarks"]], dtype=np.int64)
    # 橡皮擦语义：被擦除的点不参与候选障碍判定，也不显示在预览里
    pts, ids = m2d.exclude_landmarks(pts, ids, annotations.get("erased_point_ids", []))
    print(f"      擦除点 {len(annotations.get('erased_point_ids', []))} 个，"
          f"剩余 {len(pts)} 点参与合成")
    if not len(pts):
        raise ValueError("所有特征点已擦除，无法确定地图范围")
    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())
    if x_max - x_min < args.resolution or z_max - z_min < args.resolution:
        print("错误：地图点范围小于一个栅格，无法建图", file=sys.stderr)
        return 1
    # Include the maximum feature and annotation vertices inside the half-open map.
    vertices = [v for key in ("free", "obstacle", "unknown", "candidate_remove", "obstacle_segments")
                for poly in annotations.get(key, []) for v in poly]
    if vertices:
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or not np.all(np.isfinite(verts)):
            raise ValueError("标注坐标非法")
        x_min, z_min = np.minimum([x_min, z_min], verts.min(axis=0))
        x_max, z_max = np.maximum([x_max, z_max], verts.max(axis=0))
    origin_xz, shape = m2d.align_bounds(x_min, z_min, x_max + args.resolution,
                                       z_max + args.resolution, args.resolution)
    if shape[0] * shape[1] > 50_000_000:
        raise ValueError(f"地图 {shape} 超过 5000 万格，请核验离群点/标注范围")
    print(f"      原点(origin_xz)={origin_xz} 形状(height,width)={shape}")

    candidates = None
    candidate_cells = []
    if args.ground_y is not None:
        h_rel = args.ground_y - pts[:, 1]
        mask = (h_rel >= args.height_min) & (h_rel <= args.height_max)
        candidates = pts[mask]
        for (x, y, z) in candidates:
            cell = m2d.world_to_grid(x, z, origin_xz, args.resolution, shape)
            if cell is not None:
                candidate_cells.append(cell)
        print(f"      高度过滤: ground_y={args.ground_y} [{args.height_min},{args.height_max}] "
              f"→ 候选障碍点 {len(candidates)}")
    else:
        print("      未配置 --ground-y：关闭高度过滤，栅格保持未知（不自动识别障碍）")
    occ = m2d.build_occupancy(origin_xz, args.resolution, shape, candidate_cells, annotations,
                              free_from_walls=args.free_from_walls, wall_snap_m=args.wall_snap_m,
                              free_shrink_m=args.free_shrink_m)
    if args.free_from_walls:
        print(f"      自由区来源：墙线段围成的封闭区域"
              f"（wall_snap_m={args.wall_snap_m}m, free_shrink_m={args.free_shrink_m}m，忽略 free 多边形）")
    n_free = int((occ == m2d.FREE).sum())
    n_obs = int((occ == m2d.OBSTACLE).sum())
    n_unk = int((occ == m2d.UNKNOWN).sum())
    print(f"      自由={n_free} 障碍={n_obs} 未知={n_unk}")

    print("[4/5] 包络膨胀与软代价")
    clearance, traversable, cost = m2d.compute_inflation(
        occ, args.resolution, args.robot_radius_m, args.safety_margin_m, args.soft_band_m)
    print(f"      可通行格子={int(traversable.sum())} "
          f"(R_required={args.robot_radius_m + args.safety_margin_m}m)")

    print("[5/5] 写入地图包")
    map_id = args.map_id or map_dir.name
    meta = {
        "map_id": map_id,
        "map_version": args.map_version or out_dir.name,
        "source_type": "cuvslam_data_mdb",
        "source_map_dir": str(map_dir),
        "source_db_sha256": db_sha,
        "source_db_size_bytes": db_size,
        "extractor_version_or_sha256": m2d.sha256_file(extractor),
        "source_geometry_sha256": "",  # 保存时计算实际 source_map.json 的指纹
        "coordinate_frame": "cuvslam_map_frame (X右 Y下 Z前; 平面=XZ)",
        "resolution_m": args.resolution,
        "origin_xz": list(origin_xz),
        "width": shape[1],
        "height": shape[0],
        "row_axis": "Z",
        "col_axis": "X",
        "unknown_policy": "不可通行（与障碍同）",
        "robot_radius_m": args.robot_radius_m,
        "safety_margin_m": args.safety_margin_m,
        "soft_band_m": args.soft_band_m,
        "clearance_semantics": "conservative lower bound: max(0, center_dist - sqrt(2)*r)",
        "height_filter": None if args.ground_y is None else {
            "ground_y": args.ground_y, "min": args.height_min, "max": args.height_max},
        "reviewed": bool(args.review),
        "review_note": "人工核验后置 reviewed=true" if not args.review else "已审核",
    }
    if args.force and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
        print("      （--force：已删除旧目录）")
    out_dir = m2d.save_map_package(out_dir, meta, occ, traversable, clearance, cost,
                                   annotations, source_map=source)
    if not args.no_preview:
        pv = build_preview(out_dir, pts, occ, traversable, origin_xz,
                           args.resolution, candidates)
        print(f"      预览: {pv}")
    print(f"完成：{out_dir}")
    if not args.review:
        print("注意：未核验（reviewed=false）的地图仅供草稿使用，"
              "load_planning_map 会拒绝加载。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
