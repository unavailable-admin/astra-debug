"""Scene11 stereo -> VLM identities -> measured 3D -> pick/place -> reobserve.

Task row coordinates and 40mm object size are configured priors. Object world
poses and the old reference-letter homography are never read by this runner.
"""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import websockets
from scipy.spatial.transform import Rotation

from astra_sim_agent import Sim, ROOT
from astra_vision import AstraVision, identified_pixels
from closed_loop_spell import observe_scene, select_action, placement_errors
from grasp_letter_demo import Grasp, execute_pick_place
from stereo_letters import StereoTracker
from test_stereo_live import validate_status
from stereo_geometry import StereoGeometry


def targets_for(word,start_x=-.26):
    word=word.strip().upper()
    if not word or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in word) or len(set(word))!=len(word):
        raise ValueError('Expected unique ASCII letters')
    if len(word)>4:raise ValueError('Up to four cubes fit the configured left-arm row')
    if not np.isfinite(start_x) or start_x<-.35 or start_x+.09*(len(word)-1)>.05:
        raise ValueError('Target row exceeds left-arm workspace')
    return {l:{'letter':l,'word':word,'target_x':start_x+.09*i,'target_y':1.66,
               'front_corridor_y':1.71} for i,l in enumerate(word)}


def completion_checks(estimates,targets,hand_points,tolerance=.018):
    """Current VLM-identified upright cubes + measured positions + hands clear.

    Do not let a coarse VLM 'original grid' label override measured placement.
    The original model decision remains in the report for disagreement audits.
    """
    errors=placement_errors(estimates,targets)
    hands=np.asarray(hand_points,float)
    result={'ok':False,'position_errors_m':errors,'minimum_hand_distances_m':{}}
    if hands.ndim!=2 or hands.shape[1]!=3 or len(hands)==0 or not np.isfinite(hands).all():return result
    for letter in targets:
        if letter not in estimates or not np.isfinite(errors.get(letter,np.nan)) or errors[letter]>tolerance:return result
        xyz=np.asarray(estimates[letter]['estimated_xyz'],float)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():return result
        distance=float(np.min(np.linalg.norm(hands-xyz,axis=1)))
        result['minimum_hand_distances_m'][letter]=distance
        if distance<.10:return result
    result['ok']=True
    return result


def export_video(out, sim):
    if sim.frame_count<1:return None
    dt=sim.frame.get('dt',1/30)
    destination=out/f'stereo_{sim.speed:g}x.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-threads','1',
        '-framerate',str(1/dt),'-i',str(out/'frame_%04d.jpg'),
        '-threads','1','-framerate',str(1/dt),'-i',str(out/'frame_%04d_right.jpg'),
        '-filter_complex_threads','1','-filter_complex','[0:v][1:v]hstack=inputs=2[v]',
        '-map','[v]','-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',
        str(destination)],check=True)
    return str(destination)


async def run(args):
    targets=targets_for(args.word,args.start_x);word=''.join(targets)
    speed=getattr(args,'speed',1.)
    if not np.isfinite(speed) or not .25<=speed<=3.:raise ValueError('Speed must be finite and within [0.25, 3]')
    out=Path(args.output) if getattr(args,'output',None) else ROOT/'runs'/time.strftime('stereo_spell_%Y%m%d_%H%M%S')
    out.mkdir()
    print('RUN_DIR',out,flush=True)
    report={'word':word,'speed':speed,'motion_profile':'trajectory','success_verified':False,
            'skills_completed':0,'events':[],'perception':'stereo','object_truth_used_for_control':False,
            'targets':targets,'inspect_only':args.inspect_only}
    sim=None
    def save():
        if sim:
            report.update(actions_sent=sim.actions_sent,frames=sim.frame_count,
                simulation_seconds=sim.frame['sim_time']-sim.initial_sim_time if sim.frame else 0)
        (out/'report.json').write_text(json.dumps(report,indent=2))
    try:
        api=AstraVision(out/'api',timeout=args.api_timeout)
        async with websockets.connect(args.uri,proxy=None,ping_interval=None,open_timeout=10,max_size=128*1024*1024) as ws:
            sim=Sim(ws,out,root_pose={'pos':[.003253,1.495587,.76]})
            sim.expected_scene='showroom_scene_11_stereo'
            sim.motion_profile='trajectory';sim.speed=speed
            sim.adaptive_ik=True
            await sim.send({'type':'status'})
            status=await sim.receive('status_response')
            validate_status(status,StereoGeometry.load('sim'))
            await sim.start()
            if sim.status['action_layout']!='full36':raise ValueError('Requires full36')
            tracker=StereoTracker(sim);controller=Grasp(sim,tracker=tracker)
            for side,delta in [('right',[.10,-.02,.06]),('left',[-.12,0,.08])]:
                wrist=sim.kin.fk(sim.full_q(),side+'_wrist_yaw_link')
                result=await sim.move_wrist(side,(wrist[:3,3]+delta).tolist(),
                    Rotation.from_matrix(wrist[:3,:3]).as_quat()[[3,0,1,2]].tolist())
                if result['position_error_m']>.01:raise RuntimeError('Observation pose not reached')
            controller.rest={n:v for n,v in sim.full_q().items() if n in sim.names and n.startswith('left_')}
            refresh_count=0
            for turn in range(args.max_skills+4):
                frame=out/f'frame_{sim.frame_count-1:04d}.jpg'
                decision=await observe_scene(api,sim,frame,word,args.api_max_attempts)
                await sim.check_live_status()
                pixels=identified_pixels(decision,cv2.imread(str(frame)).shape,word)
                vision=tracker.locate(frame,identified=pixels)
                estimates=vision['letters'] if vision['ok'] else {}
                action,letter,reason=select_action(decision,estimates,targets)
                q=sim.full_q()
                hands=[sim.kin.fk(q,name)[:3,3] for name in sim.kin.by_child
                       if name.startswith(('lh_','rh_'))]
                completion=completion_checks(estimates,targets,hands)
                if completion['ok']:
                    action,letter,reason='finish',None,'current_identity_stereo_position_and_hand_clearance_passed'
                elif action=='finish':
                    action,letter,reason='stop',None,'finish_not_supported_by_hand_clearance'
                event={'state':'observe','frame':str(frame),'api_decision':decision,'vision':vision,
                    'validated_action':action,'reason':reason,'completion_checks':completion,
                    'position_errors':placement_errors(estimates,targets)}
                report['events'].append(event);save()
                print('STEREO_OBSERVE',json.dumps(event),flush=True)
                if action=='finish':
                    report.update(success_verified=True,reason=reason);break
                if args.inspect_only:
                    report['reason']='inspection_complete';break
                if action!='pick_place':
                    if len(estimates)<len(targets) and refresh_count<2:
                        refresh_count+=1
                        await sim.move({})
                        report['events'].append({'state':'refresh_observation','attempt':refresh_count})
                        save();continue
                    report['reason']=reason;break
                if report['skills_completed']>=args.max_skills:
                    report['reason']='skill_budget_exhausted';break
                # Contact and travel heights come from the current stereo surfaces.
                table=vision['table_z']
                for target in targets.values():
                    target['contact_z']=table+.04-.005
                    target['carry_height']=table+.095
                skill={'letter':letter,'start_frame':sim.frame_count-1}
                report['events'].append({'state':'pick_place','skill':skill});save()
                await execute_pick_place(sim,controller,letter,targets[letter],skill,None,estimates[letter])
                report['skills_completed']+=1;save()
            await sim.send({'type':'unsubscribe_step_result'})
    except Exception as exc:
        report.update(reason='stopped_on_error',error=f'{type(exc).__name__}: {exc}')
        print('STOP',report['error'],flush=True)
    finally:
        save()
        if sim and sim.frame_count:
            try:report['video']=export_video(out,sim)
            except Exception as exc:report['video_error']=str(exc)
        save()
    print('REPORT',json.dumps({k:v for k,v in report.items() if k!='events'}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uri',default='ws://10.19.4.253:8081')
    p.add_argument('--word',default='ACE')
    p.add_argument('--speed',type=float,default=1.,help='Trajectory speed multiplier, 0.25 to 3; simulation dt unchanged')
    p.add_argument('--output',type=Path,help='New run directory (must not already exist)')
    p.add_argument('--start-x',type=float,default=-.26)
    p.add_argument('--inspect-only',action='store_true')
    p.add_argument('--max-skills',type=int,default=3)
    p.add_argument('--api-timeout',type=float,default=120)
    p.add_argument('--api-max-attempts',type=int,default=10)
    args=p.parse_args()
    if not np.isfinite(args.speed) or not .25<=args.speed<=3.:p.error('Speed must be within [0.25, 3]')
    if args.max_skills<1 or args.api_max_attempts<1 or not np.isfinite(args.api_timeout) or args.api_timeout<=0:
        p.error('Budgets and timeout must be positive')
    result=asyncio.run(run(args))
    raise SystemExit(0 if result['success_verified'] or result.get('reason')=='inspection_complete' else 1)
