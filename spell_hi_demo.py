"""Execute a company-Astra placement plan with visual feedback and a pointing finger.

Resets the selected Scene11 worker with robot Y+10cm. The model chooses placement
targets; deterministic vision/IK executes bounded strokes. No object teleporting.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import websockets
from astra_sim_agent import Sim, ROOT
from track_letters import Tracker


def continuous_ik_step(kin,q,goal,goal_r,fraction):
    """Refine a Cartesian step rather than jumping to a remote IK branch."""
    current=kin.fk(q,'left_wrist_yaw_link')
    start_r=Rotation.from_matrix(current[:3,:3]);rv=(start_r.inv()*goal_r).as_rotvec()
    failure=None
    for _ in range(8):
        subpos=current[:3,3]+fraction*(goal-current[:3,3])
        subr=start_r*Rotation.from_rotvec(fraction*rv)
        try:
            target,error=kin.solve(q,'left',subpos,subr.as_quat()[[3,0,1,2]])
            actual=kin.fk(dict(q,**target),'left_wrist_yaw_link')
            angle=(subr.inv()*Rotation.from_matrix(actual[:3,:3])).magnitude()
            delta=max(abs(v-q[n]) for n,v in target.items())
            if error<=.003 and angle<=.03 and delta<=.35:
                return target,error,fraction
            failure=f'IK continuity/residual: joint={delta}, pos={error}, angle={angle}'
        except ValueError as exc:failure=str(exc)
        fraction*=.5
    raise ValueError('No continuous Cartesian step: '+str(failure))


class Pointer:
    def __init__(self, sim, tracker=None):
        self.sim = sim
        self.side = 'left'
        self.rest = {n:v for n,v in sim.full_q().items() if n in sim.names and n.startswith('left_')}
        self.rotation = Rotation.from_euler('z',150,degrees=True) * Rotation.from_euler('y',90,degrees=True)
        self.quat = self.rotation.as_quat()[[3,0,1,2]]
        self.tip_local = np.array([.006,0,.036,1.])  # from USD distal-link bounds
        self.tracker = tracker if tracker is not None else Tracker(ROOT/'runs/initial.jpg', ROOT/'runs/scene_authored_geometry.json',
                               targets='HI', camera_forward_shift=.1)
        exemplar = ROOT/'runs/hi_demo_20260910_082128/frame_1556.jpg'
        if tracker is None and exemplar.exists():
            self.tracker.add_visual_exemplar('H',exemplar,[481.5,553.5])

    def tip_offset(self):
        q = self.sim.full_q()
        return (np.linalg.inv(self.sim.kin.fk(q,'left_wrist_yaw_link'))
                @ self.sim.kin.fk(q,'lh_index_distal') @ self.tip_local)[:3]

    def vision(self):
        result = self.tracker.locate(self.sim.directory/f'frame_{self.sim.frame_count-1:04d}.jpg')
        (self.sim.directory/f'vision_{self.sim.frame_count-1:04d}.json').write_text(json.dumps(result,indent=2))
        return result.get('letters',{})

    async def point(self):
        targets = {'lh_index_mcp_pitch':.03, 'lh_middle_mcp_pitch':1.25,
                   'lh_ring_mcp_pitch':1.25, 'lh_pinky_mcp_pitch':1.25,
                   'lh_thumb_cmc_yaw':.15, 'lh_thumb_cmc_pitch':.1}
        for _ in range(12):
            q = self.sim.full_q()
            error = max(abs(v-q[n]) for n,v in targets.items())
            print('HAND_PREP',_,round(error,4),flush=True)
            if error<.05:
                return
            await self.sim.move({n:float(q[n]+np.clip(v-q[n],-.32,.32)) for n,v in targets.items()})
        raise RuntimeError('Pointing hand did not settle')

    async def goto(self, pos, quat=None, *, settle=True):
        goal = np.asarray(pos,float)
        goal_r = self.rotation if quat is None else Rotation.from_quat(np.asarray(quat)[[1,2,3,0]])
        bias = np.zeros(7)
        previous_error, stalled = float('inf'),0
        for i in range(85):
            q = self.sim.full_q()
            current = self.sim.kin.fk(q,'left_wrist_yaw_link')
            r = Rotation.from_matrix(current[:3,:3])
            rotvec = (r.inv()*goal_r).as_rotvec()
            pe,oe = float(np.linalg.norm(goal-current[:3,3])),float(np.linalg.norm(rotvec))
            if pe<.003 and oe<.03:
                print('GOTO_REACHED',np.round(goal,4).tolist(),pe,oe,flush=True)
                return
            step = .036 if self.sim.motion_profile == "trajectory" else .018
            fraction = min(1., step/max(pe,1e-9), .16/max(oe,1e-9))
            subpos = current[:3,3]+fraction*(goal-current[:3,3])
            subr = r*Rotation.from_rotvec(fraction*rotvec)
            subquat = subr.as_quat()[[3,0,1,2]]
            if getattr(self.sim,'adaptive_ik',False):
                target,err,fraction=continuous_ik_step(self.sim.kin,q,goal,goal_r,fraction)
            else:
                target,err = self.sim.kin.solve(q,'left',subpos,subquat)
            names = list(target)
            nominal = np.array([target[n] for n in names])
            measured = np.array([q[n] for n in names])
            command = nominal+bias
            lo = np.array([self.sim.kin.by_name[n]['lower'] for n in names])
            hi = np.array([self.sim.kin.by_name[n]['upper'] for n in names])
            command = np.clip(command,lo+1e-5,hi-1e-5)
            delta = command-measured
            delta *= min(1.,.28/max(float(np.max(np.abs(delta))),1e-9))
            await self.sim.move(dict(zip(names,(measured+delta).tolist())), continuous=fraction < .95 or not settle)
            afterq = self.sim.full_q()
            # Small bounded integral correction for the measured gravity sag.
            if fraction>.95:
                bias = np.clip(bias+.4*(nominal-np.array([afterq[n] for n in names])),-.08,.08)
            combined = pe+.12*oe
            stalled = stalled+1 if combined>=previous_error-.0001 else 0
            previous_error = combined
            if i%5==0:
                print('GOTO_PROGRESS',i,round(pe,4),round(oe,4),flush=True)
            if stalled>=14:
                raise RuntimeError(f'Cartesian motion stalled: pos error={pe}, orientation={oe}')
        raise RuntimeError('Cartesian motion step limit')

    async def place(self, letter, target_x, target_y):
        for stroke in range(5):
            measured = self.vision()
            if letter not in measured:
                await self.park()
                measured = self.vision()
                if letter not in measured:
                    raise RuntimeError(f'{letter} is occluded or not upright after parking; refusing blind stroke')
            x,y,_ = measured[letter]['estimated_xyz']
            print('LETTER',letter,stroke,x,y,'TARGET',target_x,target_y,flush=True)
            if abs(x-target_x)<.015 and abs(y-target_y)<.012:
                return
            if abs(x-target_x)>.02 or y<target_y-.012:
                raise RuntimeError('This minimal controller only supports forward-row strokes without lateral drift')
            offset = self.rotation.apply(self.tip_offset())
            start_tip = np.array([x,y+.031,.769])
            start_wrist = start_tip-offset
            hover = start_wrist.copy();hover[2]=1.11
            await self.goto(hover)
            await self.goto(start_wrist)
            desired_y = max(target_y,y-.12)
            end_tip = np.array([x,desired_y+.027,.769])
            await self.goto(end_tip-offset)
            lifted = end_tip-offset;lifted[2]=1.11
            await self.goto(lifted)
        raise RuntimeError(f'{letter} did not reach its row within five strokes')

    async def park(self):
        print('PARK_FOR_VISION',flush=True)
        for _ in range(35):
            q=self.sim.full_q()
            if max(abs(v-q[n]) for n,v in self.rest.items())<.04:
                return
            await self.sim.move({n:float(q[n]+np.clip(v-q[n],-.25,.25)) for n,v in self.rest.items()},
                                continuous=self.sim.motion_profile == "trajectory")
        raise RuntimeError('Arm parking did not converge')


async def run(args):
    response = json.loads(args.plan.read_text())
    text = response['choices'][0]['message']['content'].strip()
    if text.startswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0]
    plan = json.loads(text)
    if [p['letter'] for p in plan]!=['H','I']:
        raise ValueError('Expected a model plan spelling HI')
    geometry=json.loads((ROOT/'runs/scene_authored_geometry.json').read_text())
    for p in plan:
        if abs(p['target_y']-1.66)>.001 or abs(p['target_x']-geometry['letter_cube_'+p['letter']]['position'][0])>.003:
            raise ValueError('Model plan outside the validated demonstration workspace')
    out=ROOT/'runs'/time.strftime('hi_demo_%Y%m%d_%H%M%S');out.mkdir(parents=True)
    print('RUN_DIR',out,flush=True)
    report={'word':'HI','model':'gpt-6-astra','model_plan':plan,'plan_source':str(args.plan),
            'controller':'classical_vision_and_ik_pointer','success_verified':False}
    async with websockets.connect(args.uri,proxy=None,ping_interval=None,max_size=128*1024*1024) as ws:
        sim=Sim(ws,out,root_pose={'pos':[.003253,1.495587,.76]})
        try:
            await sim.start()
            controller=Pointer(sim)
            print('INITIAL_VISION',json.dumps(controller.vision()),flush=True)
            if args.inspect_only:
                return
            await controller.point()
            for p in plan:
                await controller.place(**p)
            final=controller.vision()
            report['final_visual_estimates']=final
            report['placement_estimate_pass']=all(p['letter'] in final and np.linalg.norm(
                np.array(final[p['letter']]['estimated_xyz'][:2])-[p['target_x'],p['target_y']])<.018 for p in plan)
        except Exception as exc:
            report['error']=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            report['actions_sent']=sim.actions_sent
            report['frames_saved']=sim.frame_count
            (out/'report.json').write_text(json.dumps(report,indent=2))
            await sim.send({'type':'unsubscribe_step_result'})
            print('REPORT',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uri',default='ws://10.19.4.253:8080')
    p.add_argument('--plan',type=Path,default=ROOT/'runs/hi_plan_response.json')
    p.add_argument('--inspect-only',action='store_true')
    asyncio.run(run(p.parse_args()))
