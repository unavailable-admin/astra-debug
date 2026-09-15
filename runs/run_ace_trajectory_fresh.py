"""ACE continuous-profile experiment; require fresh grid before the first pick."""
import asyncio
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from astra_vision import AstraVision, identified_pixels
from track_letters import Tracker
from closed_loop_spell import spell
original=AstraVision.scene_candidates
async def checked_scene(self,image,word='ACE'):
    decision=await original(self,image,word)
    if not getattr(self,'fresh_grid_checked',False):
        tracker=Tracker(ROOT/'runs/initial.jpg',ROOT/'runs/scene_authored_geometry.json',targets=word,camera_forward_shift=.1)
        vision=tracker.locate(image,identified=identified_pixels(decision,cv2.imread(str(image)).shape,word))
        geometry=json.loads((ROOT/'runs/scene_authored_geometry.json').read_text())
        errors={l:float(np.linalg.norm(np.array(v['estimated_xyz'][:2])-geometry['letter_cube_'+l]['position'][:2])) for l,v in vision.get('letters',{}).items()}
        valid=vision.get('ok') and all(errors.get(l,float('inf'))<.02 for l in word)
        (self.directory.parent/'initial_layout_check.json').write_text(json.dumps({'fresh_grid_verified':bool(valid),'errors_m':errors,'vision':vision},indent=2))
        print('FRESH_GRID',valid,errors,flush=True)
        if not valid:raise RuntimeError('Fresh ACE grid not visually verified; no pick permitted')
        self.fresh_grid_checked=True
    return decision
AstraVision.scene_candidates=checked_scene
result=asyncio.run(spell('ACE',motion_profile='trajectory',api_timeout=120,api_max_attempts=10))
raise SystemExit(0 if result['success_verified'] else 1)
