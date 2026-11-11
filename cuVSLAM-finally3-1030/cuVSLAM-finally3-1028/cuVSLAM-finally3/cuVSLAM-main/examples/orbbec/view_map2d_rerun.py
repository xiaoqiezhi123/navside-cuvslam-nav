"""Export existing map geometry/annotations for Rerun inspection (no cameras).

The viewer shows X right and Z up in the 2D panel. Rerun's 2D screen Y is down,
so that panel logs (X,-Z); original XYZ is retained in the 3D panel. This display
transform never changes annotations.json or the planning map coordinate frame.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import map2d_data as m2d


def export_recording(source_path, annotations_path, output, ground_y=None,
                     height_min=0.5, height_max=5.0):
    import rerun as rr
    import rerun.blueprint as rrb

    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    m2d._validate_source_json(source)
    ann = json.loads(Path(annotations_path).read_text(encoding="utf-8")) if annotations_path else {}
    pts = m2d.landmark_xyz(source)
    ids = np.array([lm["id"] for lm in source["landmarks"]], dtype=np.int64)
    pts, _ = m2d.exclude_landmarks(pts, ids, ann.get("erased_point_ids", []))
    if not len(pts):
        raise ValueError("没有剩余特征点")
    if ground_y is not None and (not np.all(np.isfinite([ground_y, height_min, height_max])) or height_min > height_max):
        raise ValueError("地面/高度区间非法")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"输出已存在，请使用新文件名: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # A fresh recording for each export; no modification of the source map.
    rec = rr.RecordingStream("map2d_inspection")
    blueprint = rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial2DView(origin="map_xz", name="XZ map (X right / Z up; displayed y=-Z)"),
        rrb.Spatial3DView(origin="map_xyz", name="Original cuVSLAM XYZ")))
    rec.save(output, default_blueprint=blueprint)
    rec.log("map_xyz", rr.ViewCoordinates.RDF, static=True)
    rec.log("map_xyz/features", rr.Points3D(pts, colors=[130, 130, 130], radii=0.025), static=True)
    rec.log("map_xz/features", rr.Points2D(pts[:, [0, 2]] * [1, -1],
                                          colors=[130, 130, 130], radii=0.025), static=True)
    if ground_y is not None:
        heights = ground_y - pts[:, 1]
        cand = pts[(heights >= height_min) & (heights <= height_max)]
        rec.log("map_xyz/candidates", rr.Points3D(cand, colors=[230, 60, 40], radii=0.04), static=True)
        rec.log("map_xz/candidates", rr.Points2D(cand[:, [0, 2]] * [1, -1],
                                                colors=[230, 60, 40], radii=0.04), static=True)
    colors = {"free": [40, 190, 65], "obstacle": [30, 30, 30],
              "unknown": [45, 110, 230], "candidate_remove": [240, 145, 30],
              "obstacle_segments": [30, 30, 30]}
    for key, color in colors.items():
        strips = []
        for vertices in ann.get(key, []):
            arr = np.asarray(vertices, dtype=float)
            if key != "obstacle_segments":
                arr = m2d.validate_polygon(vertices)
                arr = np.vstack((arr, arr[0]))
            elif arr.shape != (2, 2) or not np.all(np.isfinite(arr)):
                raise ValueError("墙线段坐标非法")
            strips.append(arr)
        if strips:
            rec.log(f"map_xz/{key}", rr.LineStrips2D([s * [1, -1] for s in strips],
                                                    colors=color, radii=0.05), static=True)
            rec.log(f"map_xyz/{key}", rr.LineStrips3D(
                [np.column_stack((s[:, 0], np.full(len(s), ground_y or 0.0), s[:, 1])) for s in strips],
                colors=color, radii=0.05), static=True)
    rec.log("notes", rr.TextDocument(
        "Inspection only. Edit/save annotations with run_map2d_annotator.sh. "
        "2D display: (X,-Z); 3D: original XYZ. Grey/red points are sparse features/candidates, "
        "not confirmed free space. Candidate removal polygons are shown as outlines; "
        "the builder applies their grid semantics. This recording is not a planning input."), static=True)
    rec.flush()
    rec.disconnect()
    return output


def main():
    p = argparse.ArgumentParser(description="导出 Rerun 2D/3D 核验视图，不启动相机")
    p.add_argument("--source-json", required=True)
    p.add_argument("--annotations")
    p.add_argument("--output", required=True, help="新 .rrd 文件路径")
    p.add_argument("--ground-y", type=float)
    p.add_argument("--height-min", type=float, default=0.5)
    p.add_argument("--height-max", type=float, default=5.0)
    args = p.parse_args()
    output = export_recording(args.source_json, args.annotations, args.output,
                              args.ground_y, args.height_min, args.height_max)
    print(f"已导出: {output}\n用 Rerun 打开该文件；标注编辑仍使用 run_map2d_annotator.sh。")


if __name__ == "__main__":
    main()
