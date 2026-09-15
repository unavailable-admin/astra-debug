"""Offline task validation and bounded motion regression tests."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import numpy as np
from astra_sim_agent import Sim, ROOT
from closed_loop_spell import build_targets, spell
from track_letters import Tracker
from grasp_letter_demo import Grasp

class TaskInterface(unittest.TestCase):
    def setUp(self):
        self.geometry=json.loads((ROOT/'runs/scene_authored_geometry.json').read_text())

    def test_new_word_preserves_order_and_layout(self):
        targets=build_targets(' face ',self.geometry,start_x=-.28,spacing=.08)
        self.assertEqual(list(targets),list('FACE'))
        np.testing.assert_allclose([t['target_x'] for t in targets.values()],[-.28,-.20,-.12,-.04])
        self.assertTrue(all(t['front_corridor_y']>t['target_y'] for t in targets.values()))

    def test_invalid_task_rejected_before_connection(self):
        for word in ('', 'BOOK', 'A C', '中文', 'ABCDEF'):
            with self.assertRaises(ValueError):build_targets(word,self.geometry)
        for params in ({'spacing':.03},{'start_x':float('nan')},{'row_y':1.9}):
            with self.assertRaises(ValueError):build_targets('ACE',self.geometry,**params)
        with self.assertRaises(ValueError):build_targets('A',{})

    def test_letter_without_template_does_not_crash(self):
        tracker=Tracker(ROOT/'runs/initial.jpg',ROOT/'runs/scene_authored_geometry.json',targets='B')
        result=tracker.locate(ROOT/'runs/initial.jpg',identified={})
        self.assertNotIn('B',result.get('letters',{}))

class PythonInterface(unittest.IsolatedAsyncioTestCase):
    async def test_script_entry_forwards_word_and_motion_configuration(self):
        with patch('closed_loop_spell.run', new=AsyncMock(return_value={'success_verified': True})) as run:
            result=await spell('FACE', start_x=-.28, spacing=.08, motion_profile='continuous', speed=1.5)
        self.assertTrue(result['success_verified'])
        args=run.call_args.args[0]
        self.assertEqual(args.word,'FACE')
        self.assertEqual(args.motion_profile,'continuous')
        self.assertEqual(args.spacing,.08)
        self.assertEqual(args.speed,1.5)

    async def test_invalid_request_never_starts_controller(self):
        with patch('closed_loop_spell.run', new=AsyncMock()) as run:
            for params in ({'word':'BOOK'}, {'motion_profile':'turbo'}, {'api_timeout':float('nan')}, {'speed':0}, {'speed':True}, {'speed':float('nan')}):
                with self.assertRaises(ValueError):await spell(**params)
            run.assert_not_awaited()

class ReleaseClearance(unittest.IsolatedAsyncioTestCase):
    async def test_opening_convergence_is_checked_after_clearance(self):
        for profile,rise in [('legacy',.012),('trajectory',.035)]:
            grasp=Grasp.__new__(Grasp)
            grasp.sim=SimpleNamespace(motion_profile=profile,frame_count=1,move=AsyncMock())
            grasp.held_hand={}
            grasp.center=AsyncMock();grasp.hand=AsyncMock()
            await grasp.release(np.array([-.17,1.66,.792]))
            self.assertEqual(grasp.sim.move.await_count,3)
            np.testing.assert_allclose(grasp.center.call_args.args[0],[-.17,1.66,.792+rise])
            grasp.hand.assert_awaited_once()

class RotationSegments(unittest.IsolatedAsyncioTestCase):
    async def test_only_final_rotation_waypoint_settles_in_trajectory_mode(self):
        for profile in ('legacy','continuous','trajectory'):
            grasp=Grasp.__new__(Grasp)
            grasp.sim=SimpleNamespace(motion_profile=profile)
            grasp.yaw=30
            grasp.set_yaw=lambda angle:setattr(grasp,'yaw',angle)
            grasp.center=AsyncMock()
            await grasp.reorient([0,0,0],90,[])
            calls=grasp.center.await_args_list
            self.assertEqual(len(calls),6)
            self.assertTrue(calls[-1].kwargs['settle'])
            self.assertEqual([c.kwargs['settle'] for c in calls[:-1]], [profile!='trajectory']*5)

class Motion(unittest.IsolatedAsyncioTestCase):
    async def actions(self,profile,continuous,delta=.03,speed=1.):
        sim=Sim(None,None)
        sim.motion_profile=profile
        sim.speed=speed
        sim.names=['joint'];sim.frame={}
        sim.q=lambda:np.array([0.])
        sim.kin=SimpleNamespace(by_name={'joint':{'lower':-1,'upper':1}})
        sim.validate_frame=Mock();sim.record=Mock();sim.observation=lambda:{}
        sim.send=AsyncMock()
        async def receive(kind,**kwargs):
            if kind=='submit_actions_response':
                return {'ok':True,'accepted_count':len(sim.send.call_args.args[0]['actions'])}
            return {'frames':[{}]}
        sim.receive=receive
        await sim.move({'joint':delta},continuous=continuous)
        return np.array(sim.send.call_args.args[0]['actions'])[:,0],sim.motion_stats

    async def test_transit_shortens_but_endpoint_still_settles(self):
        slow,_=await self.actions('legacy',True)
        fast,stats=await self.actions('continuous',True)
        end,_=await self.actions('continuous',False)
        self.assertEqual(len(slow),22)
        self.assertEqual(len(fast),4)
        np.testing.assert_allclose(end,slow)
        self.assertEqual(stats['hold_frames'],0)
        self.assertEqual(fast[-1],slow[-1])

    async def test_speed_resamples_moves_and_preserves_endpoint(self):
        base,_=await self.actions('trajectory',False,.28)
        fast,stats=await self.actions('trajectory',False,.28,speed=1.5)
        slow,_=await self.actions('trajectory',False,.28,speed=.5)
        self.assertLess(len(fast),len(base))
        self.assertGreater(len(slow),len(base))
        self.assertEqual(fast[-1],base[-1])
        self.assertEqual(stats['hold_frames'],7)
        for value in (0,-1,float('nan'),float('inf'),True,'1.5',4):
            with self.assertRaises(ValueError):await self.actions('trajectory',True,speed=value)

    async def test_trajectory_transit_faster_but_contact_rate_unchanged(self):
        baseline,_=await self.actions('continuous',True,.28)
        transit,_=await self.actions('trajectory',True,.28)
        contact,stats=await self.actions('trajectory',False,.28)
        legacy,_=await self.actions('legacy',False,.28)
        self.assertLess(len(transit),len(baseline))
        self.assertLessEqual(np.max(np.abs(np.diff(np.r_[0,transit]))),.0225+1e-12)
        np.testing.assert_allclose(contact,legacy)
        self.assertEqual(stats['hold_frames'],10)

    async def test_joint_velocity_and_call_bounds_preserved(self):
        for profile in ('legacy','continuous'):
            actions,_=await self.actions(profile,True,.28)
            self.assertLessEqual(np.max(np.abs(np.diff(np.r_[0,actions]))),.015+1e-12)
            with self.assertRaises(ValueError):await self.actions(profile,True,.36)

if __name__=='__main__':unittest.main()
