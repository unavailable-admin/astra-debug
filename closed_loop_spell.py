"""Astra scene decisions -> bounded WS skills -> Astra verification, for configurable Scene11 words.

No continue.json and no saved model plan are consumed. This controller stops on
ambiguous observations, invalid decisions, failed skills, or exhausted budgets.
It is a fixed-scene prototype, not a general grasp policy or a real-robot driver.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import urllib.error
import cv2
import numpy as np
import websockets
from scipy.spatial.transform import Rotation
from astra_sim_agent import Sim,ROOT
from astra_vision import AstraVision,identified_pixels
from grasp_letter_demo import Grasp,execute_pick_place
from track_letters import Tracker, ANCHORS

def build_targets(word, geometry, start_x=-.26, spacing=.09, row_y=1.66):
    """Public task interface: one upright Scene11 cube per unique letter."""
    word = word.strip().upper()
    if not word or any(l not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for l in word):
        raise ValueError('word must contain only ASCII letters A-Z')
    if len(set(word)) != len(word):
        raise ValueError('Scene11 has one cube per letter; repeated letters are unsupported')
    missing = [l for l in word if 'letter_cube_'+l not in geometry]
    if missing:
        raise ValueError(f'Letters absent from scene: {missing}')
    if len(set(ANCHORS)-set(word)) < 6:
        raise ValueError('Keep at least six reference letters outside the target word')
    if not all(np.isfinite(v) for v in (start_x, spacing, row_y)) or spacing < .06:
        raise ValueError('Row coordinates must be finite and spacing must be >= 0.06 m')
    if not 1.64 <= row_y <= 1.67:
        raise ValueError('Front row y must be within [1.64, 1.67] m')
    targets = {l: {'letter': l, 'target_x': start_x+i*spacing, 'target_y': row_y,
                   'word': word, 'carry_height': .852, 'front_corridor_y': 1.71}
               for i, l in enumerate(word)}
    if any(not -.35 <= t['target_x'] <= .05 for t in targets.values()):
        raise ValueError('Target row must fit the left-arm workspace x=[-0.35, 0.05] m')
    return targets

async def observe_scene(api,sim,frame,word,max_attempts=10):
    if type(max_attempts) is not int or max_attempts<1:
        raise ValueError('max_attempts must be a positive integer')
    for attempt in range(max_attempts):
        await sim.check_live_status()
        try:
            return await api.scene_candidates(frame,word)
        except (urllib.error.HTTPError,TimeoutError) as exc:
            if isinstance(exc,urllib.error.HTTPError) and exc.code not in (502,503,504):
                raise
            if attempt+1==max_attempts:raise
            delay=min(30,2**(min(attempt,4)+1))
            print('API_RETRY',f'next={attempt+2}/{max_attempts}',f'wait={delay}s',type(exc).__name__,str(exc),flush=True)
            await asyncio.sleep(delay)

def placement_errors(estimates,targets):
    return {letter:float(np.linalg.norm(np.array(value['estimated_xyz'][:2])-
                [targets[letter]['target_x'],targets[letter]['target_y']]))
            for letter,value in estimates.items() if letter in targets}

def select_action(decision,estimates,targets,tolerance=.018):
    """A model decision needs current visual coordinates and code-side checks."""
    errors=placement_errors(estimates,targets)
    action=decision.get('next_action',{})
    if not isinstance(action,dict):return 'stop',None,'invalid_action'
    if action.get('action')=='finish':
        if decision.get('row_complete') is True and all(errors.get(l,float('inf'))<=tolerance for l in targets):
            return 'finish',None,'visual_and_position_checks_passed'
        return 'stop',None,'finish_not_supported_by_measurements'
    letter=action.get('letter')
    if action.get('action')=='pick_place' and letter in targets and letter in estimates:
        if errors[letter]<=tolerance:return 'stop',None,'selected_letter_already_at_target'
        x,y,_=estimates[letter]['estimated_xyz']
        in_workspace=(-.35<=x<=.25 and 1.63<=y<=1.96)
        if not in_workspace:return 'stop',None,'outside_skill_workspace'
        return 'pick_place',letter,'validated'
    return 'stop',None,'uncertain_or_invalid_scene_decision'

async def run(args):
    out=ROOT/'runs'/time.strftime('closed_loop_%Y%m%d_%H%M%S');out.mkdir()
    print('RUN_DIR',out,flush=True)
    report={'success_verified':False,'mode':'inspect' if args.inspect_only else 'execute',
            'uri':args.uri,'events':[],'skills_completed':0,
            'motion_profile':args.motion_profile, 'speed':args.speed, 'word':args.word,'lift_verification':'skipped_trust_skill','scope':'Fixed Scene11; model visual decisions plus deterministic skills; no human checkpoint.'}
    geometry=json.loads((ROOT/'runs/scene_authored_geometry.json').read_text())
    targets=build_targets(args.word, geometry, args.start_x, args.spacing, args.row_y)
    args.word=''.join(targets)
    report['word']=args.word
    if args.max_skills is None: args.max_skills=len(targets)
    report['targets']=targets
    report['motion_parameters']={
        'speed':args.speed,
        'timing_rule':'ceil(base interpolation frames / speed), ceil(base hold frames / speed); minimum 2 interpolation and 4 nonzero hold frames; listed rates are speed=1 baselines',
        'minimum_settle_frames':4,
        'minimum_interpolation_frames':2,
        'cartesian_step_m':.036 if args.motion_profile=='trajectory' else .018,
        'transit_joint_step_rad':.0225 if args.motion_profile=='trajectory' else .015,
        'terminal_joint_step_rad':.015,
        'release_clearance_m':.035 if args.motion_profile=='trajectory' else .012,
        'terminal_hold_frames':10}

    report['api_max_attempts']=args.api_max_attempts
    api=AstraVision(out/'api',timeout=args.api_timeout)
    report.update(api.client.metadata())
    sim=None
    def save():
        report['actions_sent']=sim.actions_sent if sim else 0
        report['motion_stats']=sim.motion_stats if sim else {}
        report['simulation_seconds']=(sim.frame['sim_time']-sim.initial_sim_time
            if sim and sim.frame and sim.initial_sim_time is not None else 0)
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    async with websockets.connect(args.uri,proxy=None,ping_interval=None,max_size=128*1024*1024) as ws:
        sim=Sim(ws,out,root_pose={'pos':[.003253,1.495587,.76]})
        sim.motion_profile=args.motion_profile
        sim.speed=args.speed
        try:
            await sim.start()
            if sim.status['action_layout']!='full36':raise RuntimeError('Requires an idle full36 Scene11 worker')
            controller=Grasp(sim)
            controller.tracker=Tracker(ROOT/'runs/initial.jpg',ROOT/'runs/scene_authored_geometry.json',targets=args.word,camera_forward_shift=.1)
            if not args.inspect_only:
                wrist=sim.kin.fk(sim.full_q(),'right_wrist_yaw_link')
                result=await sim.move_wrist('right',(wrist[:3,3]+[.10,-.02,.06]).tolist(),
                    Rotation.from_matrix(wrist[:3,:3]).as_quat()[[3,0,1,2]].tolist())
                report['events'].append({'state':'clear_other_arm','result':result});save()
                if result['position_error_m']>.01:raise RuntimeError('Other arm did not clear the workspace')
                # Park outside the front-left placement row, which the original
                # rest pose obscures. Reuse this measured pose after every skill.
                wrist=sim.kin.fk(sim.full_q(),'left_wrist_yaw_link')
                result=await sim.move_wrist('left',(wrist[:3,3]+[-.12,0,.08]).tolist(),
                    Rotation.from_matrix(wrist[:3,:3]).as_quat()[[3,0,1,2]].tolist())
                report['events'].append({'state':'set_observation_pose','result':result});save()
                if result['position_error_m']>.01:raise RuntimeError('Left arm did not reach observation pose')
                controller.rest={n:v for n,v in sim.full_q().items() if n in sim.names and n.startswith('left_')}
            cleared=False
            for turn in range(args.max_skills+3):
                frame=out/f'frame_{sim.frame_count-1:04d}.jpg'
                print('STATE OBSERVE',turn,frame.name,flush=True)
                decision=await observe_scene(api,sim,frame,args.word,args.api_max_attempts)
                await sim.check_live_status()
                pixels=identified_pixels(decision,cv2.imread(str(frame)).shape,word=args.word)
                vision=controller.tracker.locate(frame,identified=pixels)
                estimates=vision.get('letters',{}) if vision.get('ok') else {}
                action,letter,reason=select_action(decision,estimates,targets)
                event={'state':'observe','frame':str(frame),'api_decision':decision,'vision':vision,
                       'validated_action':action,'reason':reason,'position_errors':placement_errors(estimates,targets)}
                report['events'].append(event);save()
                print('DECISION',action,letter,reason,flush=True)
                if action=='finish':
                    report['success_verified']=True;report['reason']=reason
                    break
                if args.inspect_only:
                    report['reason']='inspection_complete_no_skill_executed';break
                if action=='stop':
                    # One bounded view-clearing action, never an unlocalized grasp.
                    if not cleared and len(estimates)<len(targets):
                        wrist=sim.kin.fk(sim.full_q(),'left_wrist_yaw_link')
                        await sim.move_wrist('left',(wrist[:3,3]+[-.08,0,.04]).tolist(),
                            Rotation.from_matrix(wrist[:3,:3]).as_quat()[[3,0,1,2]].tolist())
                        cleared=True;report['events'].append({'state':'clear_view'});save();continue
                    report['reason']=reason;break
                if report['skills_completed']>=args.max_skills:
                    report['reason']='skill_budget_exhausted';break
                skill_report={'letter':letter,'grasp_success_verified':False,'start_frame':sim.frame_count-1}
                report['events'].append({'state':'pick_place','skill':skill_report});save()
                # Trust the scripted pinch/lift; inspect again after release and retreat.
                await execute_pick_place(sim,controller,letter,targets[letter],skill_report,None,estimates[letter])
                report['skills_completed']+=1;save()
                # Next iteration must inspect the released block before finish.
            else:report['reason']='observation_budget_exhausted'
        except Exception as exc:
            report['reason']='stopped_on_error';report['error']=f'{type(exc).__name__}: {exc}'
            print('STOP',report['error'],flush=True)
        finally:
            save()
            try:
                if sim.frame is not None:await sim.send({'type':'unsubscribe_step_result'})
            except Exception:pass
    print('REPORT',json.dumps({k:v for k,v in report.items() if k!='events'},ensure_ascii=False),flush=True)
    return report

async def spell(word='ACE', *, uri='ws://10.19.4.253:8081', start_x=-.26,
                spacing=.09, row_y=1.66, motion_profile='legacy', speed=1.,
                max_skills=None, inspect_only=False, api_timeout=120, api_max_attempts=10):
    """Run the standalone loop and return its report; no CLI parsing required."""
    if type(speed) not in (int,float) or not np.isfinite(speed) or not .25<=speed<=3.:
        raise ValueError('speed must be within [0.25, 3.0]')
    if motion_profile not in ('continuous', 'legacy', 'trajectory'):
        raise ValueError('Unknown motion profile')
    if not np.isfinite(api_timeout) or api_timeout <= 0:
        raise ValueError('api_timeout must be positive and finite')
    if type(api_max_attempts) is not int or api_max_attempts < 1:
        raise ValueError('api_max_attempts must be a positive integer')
    if max_skills is not None and (type(max_skills) is not int or max_skills < 1):
        raise ValueError('max_skills must be a positive integer')
    geometry=json.loads((ROOT/'runs/scene_authored_geometry.json').read_text())
    build_targets(word, geometry, start_x, spacing, row_y)
    return await run(argparse.Namespace(word=word, uri=uri, start_x=start_x,
        spacing=spacing, row_y=row_y, motion_profile=motion_profile, speed=speed,
        max_skills=max_skills, inspect_only=inspect_only,
        api_timeout=api_timeout, api_max_attempts=api_max_attempts))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uri',default='ws://10.19.4.253:8081')
    p.add_argument('--word',default='ACE',help='Unique scene letters in desired left-to-right order')
    p.add_argument('--start-x',type=float,default=-.26,help='First cube center in world meters')
    p.add_argument('--spacing',type=float,default=.09,help='Center spacing in meters (minimum .06)')
    p.add_argument('--row-y',type=float,default=1.66)
    p.add_argument('--motion-profile',choices=['continuous','legacy','trajectory'],default='legacy')
    p.add_argument('--speed',type=float,default=1.,help='Trajectory time scale, .25 to 3; 1 preserves original timing')
    p.add_argument('--inspect-only',action='store_true',help='Observe/API/validate only; subscription sets the robot stance but sends no joint actions')
    p.add_argument('--max-skills',type=int,default=None,help='Default: one move per requested letter')
    p.add_argument('--api-timeout',type=float,default=120)
    p.add_argument('--api-max-attempts',type=int,default=10,help='Total attempts per scene observation, including the first request (default: 10)')
    args=p.parse_args()
    if not np.isfinite(args.speed) or not .25<=args.speed<=3:p.error('--speed must be within [.25, 3]')
    if args.api_timeout<=0:p.error('--api-timeout must be positive')
    if args.api_max_attempts<1:p.error('--api-max-attempts must be positive')
    if args.max_skills is not None and args.max_skills<1:p.error('--max-skills must be positive')
    try:
        build_targets(args.word,json.loads((ROOT/'runs/scene_authored_geometry.json').read_text()),args.start_x,args.spacing,args.row_y)
    except ValueError as exc:p.error(str(exc))
    result=asyncio.run(spell(**vars(args)))
    raise SystemExit(0 if result['success_verified'] or (args.inspect_only and result.get('reason')=='inspection_complete_no_skill_executed') else 1)
