import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import cv2
import json
from pathlib import Path
from scipy.spatial.transform import Rotation
from g1_kinematics import Kinematics

from stereo_letters import StereoTracker
from closed_loop_stereo import targets_for,completion_checks
from grasp_letter_demo import Grasp,plan_pick_place
from spell_hi_demo import continuous_ik_step


class StereoIntegrationTests(unittest.TestCase):
    def test_completion_requires_positions_and_actual_hand_separation(self):
        targets=targets_for('E',-.08)
        estimates={'E':{'estimated_xyz':[-.08,1.66,.777]}}
        self.assertTrue(completion_checks(estimates,targets,[[0,1.6,1.05]])['ok'])
        self.assertFalse(completion_checks(estimates,targets,[[-.08,1.66,.80]])['ok'])
        self.assertFalse(completion_checks({},targets,[[0,1.6,1.05]])['ok'])
        self.assertFalse(completion_checks(estimates,targets,[])['ok'])
        self.assertFalse(completion_checks({'E':{'estimated_xyz':[-.08,1.70,.777]}},targets,[[0,1.6,1.05]])['ok'])

    def test_task_validation_without_geometry_file(self):
        with patch('builtins.open',side_effect=AssertionError('No geometry read permitted')):
            self.assertEqual(list(targets_for('ACE')),['A','C','E'])
            for word in ('BOOK','','ABCDEF','A C'):
                with self.assertRaises(ValueError):targets_for(word)

    def test_grasp_dependency_injection_does_not_construct_old_tracker(self):
        sim=SimpleNamespace(full_q=lambda:{},names=[],move=lambda *a,**k:None)
        tracker=object()
        with patch('spell_hi_demo.Tracker',side_effect=AssertionError('Old tracker must not run')):
            grasp=Grasp(sim,tracker=tracker)
        self.assertIs(grasp.tracker,tracker)

    def test_stereo_requires_current_identity_and_both_images(self):
        tracker=StereoTracker(None)
        self.assertFalse(tracker.locate('not_used.jpg',identified={})['ok'])
        with patch('stereo_letters.cv2.imread',return_value=None):
            with self.assertRaises(ValueError):tracker.locate('missing.jpg',identified={'A':[320,240]})

    def test_textureless_pair_never_produces_grasp(self):
        tracker=StereoTracker(None)
        blank=np.full((480,640,3),180,dtype=np.uint8)
        result=tracker.locate_images(blank,blank,{'A':[320,240]},np.eye(4))
        self.assertFalse(result['ok'])
        self.assertEqual(result['letters'],{})

    def test_metric_plane_reconstruction_from_independently_rendered_pair(self):
        tracker=StereoTracker(None);g=tracker.geometry
        texture=np.random.default_rng(5).integers(80,150,(480,640),dtype=np.uint8)
        left=cv2.cvtColor(cv2.GaussianBlur(texture,(3,3),0),cv2.COLOR_GRAY2BGR)
        cv2.putText(left,'A',(310,250),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,180,0),2)
        # Independently synthesize a frontoparallel plane at known 0.7m depth.
        homography=g.K_right@(g.R+np.outer(g.T,[0,0,1])/.7)@np.linalg.inv(g.K_left)
        right=cv2.warpPerspective(left,homography,(640,480))
        camera_to_world=np.diag([1.,-1,-1,1]);camera_to_world[2,3]=1.
        result=tracker.locate_images(left,right,{'A':[318,242]},camera_to_world)
        self.assertTrue(result['ok'])
        value=result['letters']['A']
        np.testing.assert_allclose(value['top_center_xyz'],[-2*.7/g.K_left[0,0],-2*.7/g.K_left[1,1],.3],atol=.004)
        self.assertAlmostEqual(value['grasp_center_xyz'][2],value['top_center_xyz'][2]-.005)
        failed=tracker.locate_images(left,np.full_like(right,130),{'A':[318,242]},camera_to_world)
        self.assertFalse(failed['ok'])

    def test_source_clearance_search_preserves_reachable_destination_and_grasp(self):
        data=json.loads((Path(__file__).parent/'fixtures/stereo_e_preflight.json').read_text())
        q=data['q'];root=data['root_pose']
        sim=SimpleNamespace(kin=Kinematics(root['actual_pos'],root['actual_quat_wxyz']),
            full_q=lambda:dict(q),names=data['action_names'],move=lambda *a,**k:None)
        c=Grasp(sim,tracker=object());a,b=c.tips(q);d=a-b;d/=np.linalg.norm(d)
        z=np.array([0.,0.,1.]);z-=d*z.dot(d);z/=np.linalg.norm(z)
        c.base_rotation=Rotation.from_matrix(np.diag([1.,-1.,-1.])@np.column_stack([d,z,np.cross(d,z)]).T)
        xyz=np.array(data['grasp_center_xyz'])
        with self.assertRaises(RuntimeError):plan_pick_place(c,xyz,data['target'],30)
        plan=plan_pick_place(c,xyz,data['target'],30,measured_height=True)
        self.assertGreaterEqual(plan['source_clearance_m'],.035)
        self.assertLessEqual(plan['source_clearance_m'],.065)
        self.assertAlmostEqual(plan['above'][2],data['target']['contact_z']+plan['destination_clearance_m'])
        self.assertGreaterEqual(plan['destination_clearance_m'],.030)
        self.assertLessEqual(plan['destination_clearance_m'],.035)
        np.testing.assert_array_equal(xyz,data['grasp_center_xyz'])
        self.assertTrue(all(p['position_error']<=.003 and p['orientation_error']<=.03 for p in plan['preflight']))
        np.testing.assert_allclose(plan['grasp_xyz']-xyz,plan['pinch_offset_m'])
        np.testing.assert_array_equal(plan['grasp_xyz'],xyz)
        np.testing.assert_allclose(plan['placed'][:2]-[data['target']['target_x'],data['target']['target_y']],plan['pinch_offset_m'][:2])
        self.assertTrue(all(p['max_joint_step_rad']<=.35 for p in plan['preflight']))

    def test_cartesian_step_refines_captured_runtime_ik_failure(self):
        data=json.loads((Path(__file__).parent/'fixtures/stereo_ik_step.json').read_text())
        q=data['q'];root=data['root_pose'];kin=Kinematics(root['actual_pos'],root['actual_quat_wxyz'])
        goal=np.array(data['position']);rotation=Rotation.from_quat(np.array(data['quat'])[[1,2,3,0]])
        with self.assertRaises(ValueError):kin.solve(q,'left',goal,data['quat'])
        target,error,fraction=continuous_ik_step(kin,q,goal,rotation,1.)
        self.assertLess(fraction,1.)
        self.assertLessEqual(error,.003)
        self.assertLessEqual(max(abs(v-q[n]) for n,v in target.items()),.35)


if __name__=='__main__':unittest.main()
