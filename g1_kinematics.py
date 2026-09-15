"""Kinematics from the simulation's USD joint frames (no robot motion)."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

MODEL = Path(__file__).with_name('g1_joint_model.json')


def transform(pos, quat):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = pos
    return t


class Kinematics:
    def __init__(self, root_pos, root_quat):
        self.joints = json.loads(MODEL.read_text())
        self.by_child = {j['child']: j for j in self.joints}
        self.by_name = {j['name']: j for j in self.joints}
        self.root = transform(root_pos, root_quat)
        for j in self.joints:
            j['f0'] = transform(j['pos0'], j['quat0'])
            j['invf1'] = np.linalg.inv(transform(j['pos1'], j['quat1']))

    def fk(self, q, link):
        if link not in self.by_child:
            return self.root.copy()
        j = self.by_child[link]
        t = np.eye(4)
        if j['type'] == 'PhysicsRevoluteJoint':
            axis = np.eye(3)['XYZ'.index(j['axis'])]
            t[:3, :3] = Rotation.from_rotvec(axis * q.get(j['name'], 0.)).as_matrix()
        return self.fk(q, j['parent']) @ j['f0'] @ t @ j['invf1']

    def solve(self, q, side, pos, quat=None):
        names = [f'{side}_{suffix}_joint' for suffix in (
            'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
            'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
        initial = np.array([q[n] for n in names])
        lo = np.array([self.by_name[n]['lower'] for n in names])
        hi = np.array([self.by_name[n]['upper'] for n in names])
        goal_rot = transform([0, 0, 0], quat)[:3, :3] if quat is not None else None
        def residual(x):
            t = self.fk(dict(q, **dict(zip(names, x))), f'{side}_wrist_yaw_link')
            terms = [t[:3, 3] - pos]
            if goal_rot is not None:
                terms.append(.12 * Rotation.from_matrix(goal_rot.T @ t[:3, :3]).as_rotvec())
            terms.append(.001 * (x - initial))
            return np.concatenate(terms)
        result = least_squares(residual, np.clip(initial, lo + 1e-6, hi - 1e-6),
                               bounds=(lo, hi), max_nfev=150)
        target = dict(zip(names, result.x.tolist()))
        actual = self.fk(dict(q, **target), f'{side}_wrist_yaw_link')
        err = float(np.linalg.norm(actual[:3, 3] - pos))
        if err > .015:
            raise ValueError(f'IK target unreachable: position error {err:.4f} m')
        return target, err


def export_usd(path):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(path)
    joints = []
    def quat(q):
        return [float(q.GetReal()), *map(float, q.GetImaginary())]
    for p in stage.Traverse():
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        a, b = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if not a or not b:
            continue
        item = {'name': p.GetName(), 'type': p.GetTypeName(),
                'parent': a[0].name, 'child': b[0].name,
                'pos0': list(j.GetLocalPos0Attr().Get()),
                'pos1': list(j.GetLocalPos1Attr().Get()),
                'quat0': quat(j.GetLocalRot0Attr().Get()),
                'quat1': quat(j.GetLocalRot1Attr().Get())}
        if p.IsA(UsdPhysics.RevoluteJoint):
            r = UsdPhysics.RevoluteJoint(p)
            item.update(axis=r.GetAxisAttr().Get(),
                        lower=float(np.deg2rad(r.GetLowerLimitAttr().Get())),
                        upper=float(np.deg2rad(r.GetUpperLimitAttr().Get())))
        joints.append(item)
    MODEL.write_text(json.dumps(joints, indent=2))
    print('Exported', len(joints), 'joints to', MODEL)


if __name__ == '__main__':
    import sys
    export_usd(sys.argv[1])
