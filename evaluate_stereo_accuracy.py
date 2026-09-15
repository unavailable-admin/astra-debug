"""Offline-only accuracy comparison. Never imported by the robot controller.

The supplied reference is archived authored AABBs, NOT live PhysX feedback.
Use only on a freshly reset, unchanged authored layout. AABB midpoint is the
cube center; the USD prim translation in this asset is at the cube bottom.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def evaluate(run_dir, reference):
    run_dir=Path(run_dir)
    report=json.loads((run_dir/'report.json').read_text())
    observation=next(e for e in report['events'] if e['state']=='observe')
    truth=json.loads(Path(reference).read_text())
    rows={}
    for letter,estimate in observation['vision']['letters'].items():
        bounds=truth['letter_cube_'+letter]
        center=(np.array(bounds['bbox_min'])+bounds['bbox_max'])/2
        delta=np.array(estimate['estimated_xyz'])-center
        rows[letter]={'estimated_center_m':estimate['estimated_xyz'],'reference_center_m':center.tolist(),
            'error_xyz_mm':(delta*1000).tolist(),'error_3d_mm':float(np.linalg.norm(delta)*1000),
            'error_xy_mm':float(np.linalg.norm(delta[:2])*1000)}
    values=[r['error_3d_mm'] for r in rows.values()]
    result={'reference':str(reference),'reference_kind':'archived_authored_AABB_center',
        'live_physx_accuracy_verified':False,'requires_unchanged_authored_layout':True,
        'frame':observation['frame'],'letters':rows,
        'rmse_3d_mm':float(np.sqrt(np.mean(np.square(values)))) if values else None,
        'max_3d_mm':max(values) if values else None,
        'all_requested_within_10mm':bool(rows) and set(rows)==set(report['word']) and max(values)<10}
    (run_dir/'initial_accuracy.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir')
    p.add_argument('--reference',default='runs/scene_authored_geometry.json')
    args=p.parse_args()
    print(json.dumps(evaluate(args.run_dir,args.reference),indent=2))
