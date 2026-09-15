"""Full36 thumb/index grasp experiment with a visual checkpoint after lifting.

Deterministic controller experiment, not an autonomous model rollout. Uses the
existing company model's H/I front-row placement targets. Never teleports objects.
Carry 35 mm above the contact height without rotating a held block; only rotate
the empty hand. Final success is recorded separately after visual inspection.
"""
import asyncio
import json
from pathlib import Path
import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import websockets
from astra_sim_agent import Sim, ROOT
from spell_hi_demo import Pointer

NAMES=['lh_index_mcp_pitch','lh_index_dip','lh_thumb_cmc_yaw','lh_thumb_cmc_pitch','lh_thumb_ip']
# The running asset enforces DIP=.89*MCP and thumb IP=2.29*CMC pitch.
OPEN=np.array([.5,.5*.89,1.,.2,.2*2.29])
CLOSED=np.array([.72,.72*.89,1.,.32,.32*2.29])

class Grasp(Pointer):
    def __init__(self,sim,tracker=None):
        super().__init__(sim,tracker=tracker)
        self.held_hand={}
        original_move=sim.move
        async def move_with_grip(targets, **kwargs):
            merged={**self.held_hand,**targets}
            q=sim.full_q()
            # Contact can stop a finger short of its desired position. Retain
            # the desired grip while respecting the same per-call angle bound.
            bounded={n:float(q[n]+np.clip(v-q[n],-.30,.30)) for n,v in merged.items()}
            return await original_move(bounded, **kwargs)
        sim.move=move_with_grip

    def tips(self,q):
        w=np.linalg.inv(self.sim.kin.fk(q,'left_wrist_yaw_link'))
        a=(w@self.sim.kin.fk(q,'lh_index_distal')@[.006,0,.03,1])[:3]
        b=(w@self.sim.kin.fk(q,'lh_thumb_distal')@[-.005,0,.04,1])[:3]
        return a,b

    async def hand(self,targets):
        self.held_hand={}
        for _ in range(20):
            q=self.sim.full_q()
            if max(abs(v-q[n]) for n,v in targets.items())<.04:
                self.held_hand=dict(targets)
                return
            await self.sim.move({n:float(q[n]+np.clip(v-q[n],-.3,.3)) for n,v in targets.items()})
        raise RuntimeError('Hand did not reach free-space preshape')

    async def center(self,xyz,nominal=None, *, settle=True):
        q=self.sim.full_q()
        if nominal is not None:q.update(dict(zip(NAMES,nominal)))
        a,b=self.tips(q)
        await self.goto(np.array(xyz)-self.rotation.apply((a+b)/2), settle=settle)

    async def release(self,xyz):
        # Open before retracting. Near the table, full free-space convergence
        # can be blocked by fingertip/table contact; finish opening after lifting.
        opening=dict(zip(NAMES,OPEN.tolist()))
        self.held_hand.update(opening)
        for _ in range(3):
            await self.sim.move(opening)
        release_frame=self.sim.frame_count-1
        # Release before retreat; finish opening above contact at the same
        # 35 mm clearance already checked by execute_pick_place's preflight.
        rise=.035 if self.sim.motion_profile == 'trajectory' else .012
        clear=np.array(xyz)+[0.,0.,rise]
        await self.center(clear,OPEN)
        await self.hand(opening)
        return release_frame

    def set_yaw(self,yaw):
        self.yaw=float(yaw)
        self.rotation=Rotation.from_euler('z',yaw,degrees=True)*self.base_rotation
        self.quat=self.rotation.as_quat()[[3,0,1,2]]

    async def reorient(self,xyz,yaw,nominal):
        start=self.yaw
        for angle in np.linspace(start,yaw,max(2,int(abs(yaw-start)/10)+1))[1:]:
            self.set_yaw(angle)
            await self.center(xyz,nominal, settle=self.sim.motion_profile != "trajectory" or abs(angle-yaw)<1e-8)

    def preflight(self,waypoints, *, dense=False):
        """Check grasp-center waypoints, including rotations, before contact."""
        q=dict(self.sim.full_q()); errors=[]
        for xyz,yaw,hand in waypoints:
            q.update(dict(zip(NAMES,hand)))
            a,b=self.tips(q)
            rot=Rotation.from_euler('z',yaw,degrees=True)*self.base_rotation
            pos=np.array(xyz)-rot.apply((a+b)/2)
            if dense:
                start=self.sim.kin.fk(q,'left_wrist_yaw_link')
                start_r=Rotation.from_matrix(start[:3,:3])
                rv=(start_r.inv()*rot).as_rotvec()
                count=max(1,int(np.ceil(np.linalg.norm(pos-start[:3,3])/.005)),
                          int(np.ceil(np.linalg.norm(rv)/.02)))
                max_delta=0.
                for fraction in np.linspace(1/count,1,count):
                    subpos=start[:3,3]+fraction*(pos-start[:3,3])
                    subrot=start_r*Rotation.from_rotvec(fraction*rv)
                    joints,pe=self.sim.kin.solve(q,'left',subpos,subrot.as_quat()[[3,0,1,2]])
                    delta=max(abs(v-q[n]) for n,v in joints.items())
                    if delta>.35:raise RuntimeError(f'Dense preflight IK branch change: {delta:.3f} rad at {list(xyz)}, yaw={yaw}')
                    max_delta=max(max_delta,delta)
                    q.update(joints)
                    actual=self.sim.kin.fk(q,'left_wrist_yaw_link')
                    oe=(subrot.inv()*Rotation.from_matrix(actual[:3,:3])).magnitude()
                    if pe>.003 or oe>.03:
                        raise RuntimeError(f'Dense preflight residual at {list(xyz)}, yaw={yaw}: {pe}, {oe}')
                errors.append({'center':list(xyz),'yaw':yaw,'position_error':pe,'orientation_error':oe,
                               'samples':count,'max_joint_step_rad':max_delta})
                continue
            try:
                joints,pe=self.sim.kin.solve(q,'left',pos,rot.as_quat()[[3,0,1,2]])
            except ValueError as exc:
                raise ValueError(f'Preflight center={list(xyz)}, yaw={yaw}: {exc}') from exc
            q.update(joints)
            actual=self.sim.kin.fk(q,'left_wrist_yaw_link')
            oe=(rot.inv()*Rotation.from_matrix(actual[:3,:3])).magnitude()
            if pe>.003 or oe>.03:
                raise RuntimeError(f'Preflight failed at {xyz}, yaw={yaw}: {pe}, {oe}')
            errors.append({'center':list(xyz),'yaw':yaw,'position_error':pe,'orientation_error':oe})
        return errors


def plan_pick_place(c, xyz, target, transit_yaw, *, measured_height=False):
    """Separate source turning clearance from destination release clearance.

    For stereo estimates, try bounded higher source clearances before any arm
    motion. Keep the measured pinch centered; adjust only free-space waypoints.
    """
    failures=[]
    candidates=[(rise,destination_rise) for destination_rise in ([.035,.030] if measured_height else [.035])
                for rise in ([.065,.055,.045,.035] if measured_height else [.035])]
    for rise,destination_rise in candidates:
        offset=np.zeros(3)
        grasp_xyz=xyz+offset
        hover=grasp_xyz.copy();hover[2]=xyz[2]+.148
        clearance=grasp_xyz.copy();clearance[2]=xyz[2]+rise
        lifted=grasp_xyz.copy();lifted[2]=target.get('carry_height',clearance[2])
        if measured_height:lifted[2]=max(lifted[2],clearance[2]+.005)
        placed=np.array([target['target_x'],target['target_y'],target.get('contact_z',xyz[2])])+offset
        above=placed.copy();above[2]=placed[2]+destination_rise if measured_height else clearance[2]
        contact_angles=list(range(transit_yaw+10,91,10))
        transit_angles=list(range(80,transit_yaw-1,-10))
        route=[above]
        if 'front_corridor_y' in target:
            front_y=target['front_corridor_y'];staging_y=1.78 if measured_height else min(grasp_xyz[1],1.78)
            route=[np.array([grasp_xyz[0],staging_y,lifted[2]]),np.array([placed[0],staging_y,lifted[2]]),
                np.array([placed[0],front_y,lifted[2]]),np.array([placed[0],front_y,above[2]]),above]
        waypoints=[(hover,transit_yaw,OPEN),(clearance,transit_yaw,OPEN),
            *[(clearance,y,OPEN) for y in contact_angles],(grasp_xyz,90,OPEN),(grasp_xyz,90,CLOSED),
            (lifted,90,CLOSED),*[(p,90,CLOSED) for p in route],(placed,90,CLOSED),
            (placed,90,OPEN),*([(placed+[0,0,.035],90,OPEN)] if measured_height else []),
            (above,90,OPEN),*[(above,y,OPEN) for y in transit_angles]]
        try:preflight=c.preflight(waypoints,dense=measured_height)
        except (ValueError,RuntimeError) as exc:
            failures.append({'source_clearance_m':rise,'destination_clearance_m':destination_rise,'error':str(exc)})
            continue
        return {'hover':hover,'clearance':clearance,'lifted':lifted,'placed':placed,
                'above':above,'route':route,'preflight':preflight,
                'grasp_xyz':grasp_xyz,'pinch_offset_m':offset.tolist(),
                'source_clearance_m':rise,'destination_clearance_m':destination_rise,'rejected_candidates':failures}
    raise RuntimeError('No reachable pick/place clearance: '+json.dumps(failures))


async def execute_pick_place(sim,c,letter,target,report,verify_lift,estimate=None):
    """Reusable bounded skill; caller supplies an optional visual gate and placement target."""
    estimate=estimate or c.vision().get(letter)
    if not estimate:
        wrist=sim.kin.fk(sim.full_q(),'left_wrist_yaw_link')
        clear=wrist[:3,3]+[-.08,0.,.04]
        quat=Rotation.from_matrix(wrist[:3,:3]).as_quat()[[3,0,1,2]]
        await sim.move_wrist('left',clear.tolist(),quat.tolist())
        estimate=c.vision().get(letter)
    if not estimate:raise RuntimeError(f'{letter} not visible')
    xyz=np.array(estimate.get('grasp_center_xyz',estimate['estimated_xyz']),dtype=float)
    if 'grasp_center_xyz' not in estimate:xyz[2]=.792
    report['initial_letter_estimate']=estimate
    def phase(name):sim.phase=f'{letter}:{name}'
    phase('hand_prepare')
    targets=dict(zip(NAMES,OPEN))
    targets.update({f'lh_{finger}_{joint}':value for finger in ['middle','ring','pinky'] for joint,value in [('mcp_pitch',1.3),('dip',1.3*.89)]})
    await c.hand(targets)
    a,b=c.tips(sim.full_q());d=a-b;d/=np.linalg.norm(d)
    z=np.array([0.,0.,1.]);z-=d*z.dot(d);z/=np.linalg.norm(z)
    c.base_rotation=Rotation.from_matrix(np.diag([1.,-1.,-1.])@np.column_stack([d,z,np.cross(d,z)]).T)
    transit_yaw=60 if letter in ('H','A') else 30
    c.set_yaw(transit_yaw)
    print('OPEN_TIPS',a.tolist(),b.tolist(),letter,xyz.tolist(),flush=True)
    plan=plan_pick_place(c,xyz,target,transit_yaw,measured_height='grasp_center_xyz' in estimate)
    xyz=plan['grasp_xyz']
    hover,clearance,lifted,placed,above,route=[plan[k] for k in ('hover','clearance','lifted','placed','above','route')]
    place_clearance=above.copy()
    report['carry_route']=[p.tolist() for p in route]
    report['preflight']=plan['preflight']
    report['source_clearance_m']=plan['source_clearance_m']
    report['destination_clearance_m']=plan['destination_clearance_m']
    report['pinch_offset_m']=plan['pinch_offset_m']
    report['preflight_rejected_candidates']=plan['rejected_candidates']
    print('PREFLIGHT_PASSED',len(report['preflight']),flush=True)
    phase('approach_hover')
    await c.center(hover)
    phase('descend_clearance')
    await c.center(clearance,OPEN)
    phase('orient_grasp')
    await c.reorient(clearance,90,OPEN)
    phase('descend_contact')
    await c.center(xyz)
    phase('close')
    for fraction in np.linspace(.125,1.,8):
        nominal=OPEN+fraction*(CLOSED-OPEN)
        c.held_hand.update(dict(zip(NAMES,nominal.tolist())))
        await sim.move(dict(zip(NAMES,nominal.tolist())))
        await c.center(xyz,nominal)
        a,b=c.tips(sim.full_q())
        print('CLOSE',float(fraction),'tip_gap',float(np.linalg.norm(a-b)),flush=True)
    report['before_lift_frame']=sim.frame_count-1
    phase('lift')
    await c.center(lifted,CLOSED)
    report['lift_frame']=sim.frame_count-1
    print('LIFT_CHECKPOINT',sim.frame_count-1,flush=True)
    a,b=c.tips(sim.full_q())
    if verify_lift is None:
        report['lift_decision']={'verification':'skipped','source':'user_requested_trust_skill',
                                 'continue_transport':True,'grasp_verified':False}
        report['grasp_success_verified']=False
        print('LIFT_TRUST_SKILL',letter,flush=True)
    else:
        decision=await verify_lift(letter,sim.directory/f"frame_{report['before_lift_frame']:04d}.jpg",sim.directory/f"frame_{report['lift_frame']:04d}.jpg",float(np.linalg.norm(a-b)))
        report['lift_decision']=decision
        if not decision.get('grasp_verified'):
            await c.center(xyz,CLOSED)
            await c.release(xyz)
            await c.center(clearance,OPEN)
            await c.reorient(clearance,transit_yaw,OPEN)
            await c.center(hover,OPEN)
            await c.park()
            raise RuntimeError('Lift not visually verified; returned and opened hand')
        report['grasp_success_verified']=True
    phase('carry')
    for point in route:await c.center(point,CLOSED)
    phase('lower_to_place')
    await c.center(placed,CLOSED)
    phase('release')
    report['release_frame']=await c.release(placed)
    phase('retract')
    await c.center(place_clearance,OPEN)
    phase('orient_retreat')
    await c.reorient(place_clearance,transit_yaw,OPEN)
    await c.center(above,OPEN)
    phase('park')
    await c.park()
    report['final_frame']=sim.frame_count-1

async def main(letter='H'):
    out=ROOT/'runs'/time.strftime('grasp_%Y%m%d_%H%M%S');out.mkdir()
    print('RUN_DIR',out,flush=True)
    report={'controller':'full36 deterministic pinch experiment','letter':letter,'grasp_success_verified':False}
    async with websockets.connect('ws://10.19.4.253:8081',proxy=None,ping_interval=None,max_size=128*1024*1024) as ws:
        sim=Sim(ws,out,root_pose={'pos':[.003253,1.495587,.76]})
        try:
            await sim.start()
            if sim.status['action_layout']!='full36':raise RuntimeError('Requires full36')
            c=Grasp(sim)
            plan=json.loads((ROOT/'runs/hi_plan_response.json').read_text())
            target=next(p for p in json.loads(plan['choices'][0]['message']['content']) if p['letter']==letter)
            async def manual_gate(letter,before,after,gap):
                (out/'checkpoint.json').write_text(json.dumps(report,indent=2))
                for _ in range(600):
                    if (out/'continue.json').exists():
                        return json.loads((out/'continue.json').read_text())
                    await asyncio.sleep(1)
                return {'grasp_verified':False,'reason':'Visual checkpoint timed out'}
            await execute_pick_place(sim,c,letter,target,report,manual_gate)
        except Exception as exc:
            report['error']=str(exc)
            raise
        finally:
            report['actions_sent']=sim.actions_sent
            (out/'report.json').write_text(json.dumps(report,indent=2))
            await sim.send({'type':'unsubscribe_step_result'})
            print('REPORT',json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--letter',choices=['H','I'],default='H')
    asyncio.run(main(parser.parse_args().letter))
