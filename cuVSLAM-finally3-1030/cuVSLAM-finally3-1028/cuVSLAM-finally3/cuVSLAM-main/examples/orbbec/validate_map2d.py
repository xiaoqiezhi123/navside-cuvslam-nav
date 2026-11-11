"""Offline map audit and A* probes. Does not control a robot or approve a map.

Draft packages are deliberately allowed for comparison. Runtime planning must
use map2d_data.load_planning_map, which enforces the human-review gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import label

import map2d_data as m2d


# Shared search implementation; this audit alone permits draft packages.
from map2d_astar import astar_search as astar_probe


def point_status(pm, xz, components):
    cell = pm.world_to_grid(*xz)
    result = dict(xz=xz, cell=cell, traversable=False, component=0)
    if cell is not None:
        result.update(occupancy=int(pm.occupancy[cell]), traversable=pm.is_traversable(*cell),
                      clearance_m=float(pm.clearance_m[cell]), component=int(components[cell]),
                      cell_center_xz=pm.grid_to_world(*cell))
    return result


def audit(pm, probes, task_config=None):
    # 4-connected components have exactly the same connectivity as an 8-connected
    # graph that prohibits corner cutting (each valid diagonal has a 2-edge detour).
    components, count = label(pm.traversable)
    sizes = np.bincount(components.ravel())[1:]
    meta = pm.meta
    clearance, traversable, cost = m2d.compute_inflation(
        pm.occupancy,meta['resolution_m'],meta['robot_radius_m'],meta['safety_margin_m'],meta['soft_band_m'])
    checks = {
        'inflation_recomputed': bool(np.allclose(clearance,pm.clearance_m,rtol=0,atol=1e-10)),
        'traversability_recomputed': bool(np.array_equal(traversable,pm.traversable)),
        'cost_recomputed': bool(np.allclose(cost,pm.cost,rtol=0,atol=1e-10)),
        'nonfree_never_traversable': not bool(np.any(pm.traversable & (pm.occupancy!=m2d.FREE))),
    }
    del clearance, traversable, cost
    rows, cols = np.nonzero(pm.traversable)
    samples = np.linspace(0,len(rows)-1,min(1000,len(rows)),dtype=int) if len(rows) else []
    checks['coordinate_roundtrip'] = all(
        pm.world_to_grid(*pm.grid_to_world(int(rows[i]),int(cols[i]))) == (rows[i],cols[i]) for i in samples)
    source_db = Path(meta['source_map_dir'])/'data.mdb'
    checks['source_db_matches'] = source_db.is_file() and m2d.sha256_file(source_db)==meta['source_db_sha256']
    points = {p['id']: point_status(pm,p['xz'],components) for p in probes['points']}
    paths=[]
    for a,b in probes['pairs']:
        start, goal = points[a], points[b]
        if start['traversable'] and goal['traversable'] and start['component']!=goal['component']:
            result=dict(status='different_components',expanded=0,path_cells=[],total_cost=None)
        else:
            result=astar_probe(pm,start['cell'],goal['cell'])
        cells=result.pop('path_cells')
        result.update(start=a,goal=b,cell_count=len(cells),
                      path_xz=[pm.grid_to_world(*c) for c in cells],
                      every_edge_collision_checked=bool(cells) and all(pm.edge_is_free(c,d) for c,d in zip(cells,cells[1:])),
                      length_m=sum(math.dist(c,d)*meta['resolution_m'] for c,d in zip(cells,cells[1:])),
                      min_clearance_m=min((float(pm.clearance_m[c]) for c in cells),default=None))
        paths.append(result)
    report=dict(map_directory=str(pm.directory),map_id=meta['map_id'],map_version=meta['map_version'],
                reviewed=pm.reviewed, source_db_sha256=meta['source_db_sha256'],
                source_geometry_sha256=meta['source_geometry_sha256'],
                annotations_sha256=meta['annotations_sha256'],grid_sha256=meta['grid_sha256'],
                resolution_m=meta['resolution_m'],origin_xz=meta['origin_xz'],shape=list(pm.occupancy.shape),
                robot_radius_m=meta['robot_radius_m'],safety_margin_m=meta['safety_margin_m'],
                height_filter=meta.get('height_filter'),
                occupancy_counts={str(v):int(np.count_nonzero(pm.occupancy==v)) for v in (-1,0,100)},
                traversable_cells=len(rows),traversable_area_m2=len(rows)*meta['resolution_m']**2,
                connected_components=count,component_sizes=sorted(sizes.tolist(),reverse=True),
                checks=checks,probe_points=points,probe_paths=paths,
                all_probe_paths_pass=bool(paths) and all(p['status']=='ok' and p['every_edge_collision_checked'] for p in paths),
                probe_note='Synthetic validation positions at grid cell centres, not deployed task points.',
                runtime_review_gate_open=pm.reviewed)
    if task_config:
        same_map=task_config.get('ref_map')==meta['map_id']
        reference_db=Path(meta['source_map_dir']).parent/str(task_config.get('ref_map'))/'data.mdb'
        reference_sha=m2d.sha256_file(reference_db) if reference_db.is_file() else None
        report['existing_task_config']=dict(ref_map=task_config.get('ref_map'),same_map_name=same_map,
            reference_db_sha256=reference_sha,same_map_identity=reference_sha==meta['source_db_sha256'],
            numeric_checks_only=reference_sha!=meta['source_db_sha256'],
            points=[point_status(pm,[p[0],p[2]],components) for p in task_config.get('task_points',[])])
    return report, components


def render(pm, report, components, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    occupied=np.argwhere(pm.occupancy!=m2d.UNKNOWN)
    if not len(occupied):
        return
    pad=int(math.ceil(1/pm.meta['resolution_m']))
    lo=np.maximum(occupied.min(0)-pad,0)
    hi=np.minimum(occupied.max(0)+pad+1,pm.occupancy.shape)
    region=np.s_[lo[0]:hi[0],lo[1]:hi[1]]
    res=pm.meta['resolution_m']; x,z=pm.meta['origin_xz']
    extent=[x+lo[1]*res,x+hi[1]*res,z+lo[0]*res,z+hi[0]*res]
    occ=pm.occupancy[region]
    semantic=np.zeros(occ.shape,dtype=np.uint8)
    semantic[occ==0]=1; semantic[occ==100]=2; semantic[pm.traversable[region]]=3
    fig,axs=plt.subplots(2,1,figsize=(14,12),constrained_layout=True)
    axs[0].imshow(semantic,origin='lower',extent=extent,interpolation='nearest',vmin=0,vmax=3,
                  cmap=ListedColormap(['#e5e5e5','#d9efc1','#161616','#30975a']))
    axs[0].set_title('Grey: unknown | light green: free | black: wall | dark green: traversable')
    image=np.ma.masked_equal(components[region],0)
    axs[1].set_facecolor('#eeeeee')
    axs[1].imshow(image,origin='lower',extent=extent,interpolation='nearest',cmap='tab10',vmin=0,vmax=10)
    for path in report['probe_paths']:
        if path['path_xz']:
            xy=np.asarray(path['path_xz'])
            axs[1].plot(xy[:,0],xy[:,1],linewidth=1.5,label=f"{path['start']} -> {path['goal']}")
    for name,p in report['probe_points'].items():
        for ax in axs:
            ax.plot(*p['xz'],'o',color='#002a73',markersize=4)
            ax.annotate(name,p['xz'],xytext=(4,5),textcoords='offset points',fontsize=8)
    axs[1].set_title(f"{report['connected_components']} traversable component(s); offline A* probes")
    if any(p['path_xz'] for p in report['probe_paths']):
        axs[1].legend(loc='upper center',fontsize=8,ncol=3)
    for ax in axs:
        ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_aspect('equal')
        ax.set_xlim(*extent[:2]); ax.set_ylim(*extent[2:])
    fig.suptitle(f"{report['map_version']} | resolution={res} m | radius={pm.meta['robot_radius_m']} m | reviewed={pm.reviewed}")
    fig.savefig(output,dpi=130)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map',required=True)
    parser.add_argument('--probes',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--task-config')
    args=parser.parse_args()
    pm=m2d.load_map_package(Path(args.map),require_reviewed=False)
    probes=json.loads(Path(args.probes).read_text())
    cfg=None
    if args.task_config:
        import yaml
        cfg=yaml.safe_load(Path(args.task_config).read_text())
    report,components=audit(pm,probes,cfg)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    (out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    render(pm,report,components,out/'validation.png')
    print(json.dumps({k:report[k] for k in ('map_version','reviewed','traversable_cells','connected_components','checks','all_probe_paths_pass')},ensure_ascii=False,indent=2))
    return 0 if all(report['checks'].values()) and report['all_probe_paths_pass'] else 2


if __name__=='__main__':
    raise SystemExit(main())
