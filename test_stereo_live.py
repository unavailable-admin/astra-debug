"""Live Scene11 stereo smoke test at 1x, with reset and real-frame video.

Run as a script. This holds the initial joint targets; it does NOT spell words.
Only camera calibration and robot joint feedback are used for reconstruction.
"""
import argparse
import asyncio
import base64
import json
import math
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import cv2
import numpy as np
import websockets

from g1_kinematics import Kinematics, transform
from stereo_geometry import CV_TO_GL, StereoGeometry


def decode(spec):
    raw = np.frombuffer(base64.b64decode(spec['data']), dtype=np.uint8)
    if spec['encoding'] in ('jpeg_base64', 'png_base64'):
        result = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    elif spec['encoding'] == 'raw_base64':
        result = cv2.cvtColor(raw.reshape(spec['shape']), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError('Unsupported image encoding')
    if result is None or result.shape != (480, 640, 3):
        raise ValueError('Expected 640x480 stereo images')
    return result


def validate_status(status, geometry):
    if (status['scene_id'] != 'showroom_scene_11_stereo'
            or not status.get('stereo_enabled')):
        raise ValueError('Expected stereo Scene11')
    if any(status.get(k) for k in ('step_result_subscribed', 'is_executing', 'queue_length')):
        raise ValueError('Worker is occupied')
    meta = status['stereo_calibration']
    if tuple(meta['image_size']) != geometry.image_size_wh:
        raise ValueError('Live calibration resolution differs')
    mounts = []
    for side, K in (('left', geometry.K_left), ('right', geometry.K_right)):
        pose = meta[f'render_{side}_pose_head']
        mounts.append(transform(pose['translation_m'], pose['rotation_quat_wxyz_opengl']) @ CV_TO_GL)
        if abs(meta[side]['matrix'][0][0] - K[0, 0]) > 1e-6:
            raise ValueError('Live focal length differs from local calibration')
    relative = np.linalg.inv(mounts[1]) @ mounts[0]
    for a, b in ((mounts[0], geometry.T_head_left_cv),
                 (relative[:3, :3], geometry.R), (relative[:3, 3], geometry.T)):
        if not np.allclose(a, b, atol=1e-7, rtol=0):
            raise ValueError('Live rendered mount differs from local calibration')


def analyze(left, right, geometry, q, kin, directory, label):
    sift = cv2.SIFT_create()
    kl, dl = sift.detectAndCompute(left, None)
    kr, dr = sift.detectAndCompute(right, None)
    if dl is None or dr is None or min(len(dl), len(dr)) < 2:
        raise ValueError('Insufficient stereo image features')
    matcher = cv2.BFMatcher()
    def ratio_matches(a, b):
        return {pair[0].queryIdx: pair[0] for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 and pair[0].distance < .65 * pair[1].distance}
    forward, backward = ratio_matches(dl, dr), ratio_matches(dr, dl)
    matches = [m for m in forward.values() if m.trainIdx in backward
               and backward[m.trainIdx].trainIdx == m.queryIdx]
    camera_to_root = kin.fk(q, 'head_link') @ geometry.T_head_left_cv
    rows, accepted, seen = [], [], set()
    for match in matches:
        l, r = np.array([kl[match.queryIdx].pt]), np.array([kr[match.trainIdx].pt])
        key = tuple(np.round(np.concatenate((l[0], r[0])), 2))
        if key in seen:
            continue
        seen.add(key)
        try:
            xyz = geometry.triangulate(l, r, image_size_wh=(640, 480), max_reprojection_px=.8)[0]
        except ValueError:
            continue
        root_xyz = camera_to_root[:3, :3] @ xyz + camera_to_root[:3, 3]
        rows.append({'left_px': l[0].tolist(), 'right_px': r[0].tolist(),
                     'left_camera_xyz_m': xyz.tolist(), 'robot_root_xyz_m': root_xyz.tolist()})
        accepted.append(match)
    cv2.imwrite(str(directory / f'{label}_matches.jpg'),
                cv2.drawMatches(left, kl, right, kr, accepted, None, flags=2))
    (directory / f'{label}_points.json').write_text(json.dumps(rows, indent=2))
    return {'mutual_ratio_matches': len(matches), 'unique_triangulated_points': len(rows),
            'metric_accuracy_verified': False, 'letter_identity_verified': False}


async def run(args):
    directory = Path(args.output or ('runs/stereo_live_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')))
    directory.mkdir(parents=True, exist_ok=False)
    geometry = StereoGeometry.load('sim')
    report = {'test': 'stereo_capture_and_geometry', 'speed': 1.0,
              'spelling_tested': False, 'output': str(directory), 'ok': False}
    print('OUTPUT', directory, flush=True)
    times = []
    async def receive(ws, kind):
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 60))
            if msg.get('ok') is False:
                raise RuntimeError(str(msg))
            if msg.get('type') == kind:
                return msg
    try:
        async with websockets.connect(args.uri, proxy=None, ping_interval=None,
                                      open_timeout=10, close_timeout=2, max_size=128*1024*1024) as ws:
            async def send(msg):
                await ws.send(json.dumps(msg, allow_nan=False))
            await send({'type': 'status'})
            status = await receive(ws, 'status_response')
            validate_status(status, geometry)
            # Persist calibration and robot metadata only, not object poses/layout.
            (directory / 'calibration.json').write_text(json.dumps(status['stereo_calibration'], indent=2))
            await send({'type': 'subscribe_step_result', 'force_reset': True, 'empty_queue_policy': 'pause'})
            await receive(ws, 'subscribe_step_result_response')
            initial = await receive(ws, 'step_result')
            first = initial['frames'][-1]
            names = status['full_body_joint_names']
            q = dict(zip(names, np.asarray(first['state']['joint_position']).reshape(-1)))
            # Work in the robot pelvis frame, obtainable on a real robot too.
            kin = Kinematics([0, 0, 0], [1, 0, 0, 0])
            action = [float(q[n]) for n in status['action_joint_names']]
            dt = float(first['dt'])
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError('Invalid simulation dt')
            count = math.ceil(args.seconds / dt)
            def save(frame):
                left, right = [decode(frame['images'][side]) for side in ('left', 'right')]
                n = len(times)
                if times and not np.isclose(frame['sim_time'] - times[-1], dt, atol=1e-5):
                    raise ValueError('Dropped/reset simulation frames')
                times.append(float(frame['sim_time']))
                for side, im in (('left', left), ('right', right)):
                    cv2.imwrite(str(directory / f'{side}_{n:04d}.png'), im)
                state = {'sim_time': frame['sim_time'], 'joint_names': names,
                         'joint_position': frame['state']['joint_position']}
                (directory / f'state_{n:04d}.json').write_text(json.dumps(state))
                joined = np.hstack((left, right))
                cv2.putText(joined, f'STEREO HOLD TEST | 1x | sim {frame["sim_time"]:.3f}s | no grasp',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                cv2.imwrite(str(directory / f'pair_{n:04d}.png'), joined)
                return left, right
            left, right = save(first)
            report['initial'] = analyze(left, right, geometry, q, kin, directory, 'initial')
            for offset in range(0, count, 10):
                batch = min(10, count - offset)
                await send({'type': 'submit_actions', 'actions': [action] * batch})
                ack = await receive(ws, 'submit_actions_response')
                if ack.get('accepted_count') != batch:
                    raise ValueError('Incomplete action acceptance')
                result = await receive(ws, 'step_result')
                if len(result.get('frames', [])) != batch:
                    raise ValueError('Incomplete recording feedback')
                for frame in result['frames']:
                    left, right = save(frame)
                print('SIM_SECONDS', round(times[-1] - times[0], 3), flush=True)
            q = dict(zip(names, np.asarray(frame['state']['joint_position']).reshape(-1)))
            report['final'] = analyze(left, right, geometry, q, kin, directory, 'final')
            report.update(sim_seconds=times[-1]-times[0], frames=len(times), dt=dt)
            if min(report[k]['unique_triangulated_points'] for k in ('initial', 'final')) < 10:
                raise ValueError('Insufficient geometric matches')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                        '-threads', '1', '-framerate', str(1/dt), '-i', str(directory/'pair_%04d.png'),
                        '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(directory/'stereo_1x.mp4')], check=True)
        report.update(ok=True, video=str(directory/'stereo_1x.mp4'))
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (directory/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uri', default='ws://10.19.4.253:8081')
    p.add_argument('--seconds', type=float, default=3, help='Simulation seconds holding initial joints, at 1x')
    p.add_argument('--output')
    args = p.parse_args()
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= 10:
        p.error('--seconds must be in (0, 10]')
    asyncio.run(run(args))
