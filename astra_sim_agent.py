#!/usr/bin/env python3
"""Company Chat Completions agent driving the EASIM Scene11 simulation.

Only named joint targets are executable. Every chunk is interpolated and followed
by fresh feedback. Logs distinguish a model success claim from verified success.
"""
import argparse
import asyncio
import base64
import io
import json
import math
import re
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image
import websockets
from g1_kinematics import Kinematics, transform
from openai_connection import OfficialOpenAI

ROOT = Path(__file__).resolve().parent


def image_bytes(spec):
    raw = base64.b64decode(spec['data'])
    if spec['encoding'] in ('jpeg_base64', 'png_base64'):
        im = Image.open(io.BytesIO(raw)).convert('RGB')
    else:
        h, w, _ = spec['shape']
        im = Image.frombytes('RGB', (w, h), raw)
    out = io.BytesIO()
    im.save(out, format='JPEG', quality=90)
    return out.getvalue()


def api_call(history, tools):
    # Bounded context: current telemetry, recent decisions, fresh visual estimates.
    # Full audit history remains on disk; duplicated joint histories are not sent.
    word = re.search(r'spell (\w+)', history[0]['content']).group(1)
    observation_items = next(m['content'] for m in reversed(history) if isinstance(m.get('content'), list))
    observed = json.loads(next(c['text'] for c in observation_items if c['type'] == 'input_text'))
    current = {'eef_world': observed['eef'], 'finger_link_origins_world': observed['finger_link_positions'],
               'hand_angles_rad': {n:v for n,v in observed['joints'].items() if n.startswith(('lh_', 'rh_'))}}
    calls = [{'tool': m['name'], 'arguments': json.loads(m['arguments'])} for m in history if m.get('type') == 'function_call'][-2:]
    feedback = next((m['content'] for m in reversed(history) if isinstance(m.get('content'), str)
                     and m['content'].startswith('External visual')), 'No current visual estimate. Do not assume any block has moved or been grasped.')
    last_result = next((json.loads(m['output']) for m in reversed(history) if m.get('type') == 'function_call_output'), {})
    last_result = {k:v for k,v in last_result.items() if k not in ('joints','eef','finger_link_positions')}
    prompt = f'''Control a simulated G1/O6 hand to arrange blocks spelling {word} left to right, row Y=1.66, X=-0.08/0/+0.08. Start with C. Table top Z=.7566; blocks 4cm. World +Y forward, +X right, +Z up. Wrist is NOT fingertip. Initial block centers (superseded by visual feedback): A=(-.00145,1.9035,.777), C=(.1387,1.7651,.777), E=(-.1537,1.8347,.777).
Return ONLY JSON {{"tool":name,"arguments":object}} for ONE command:
move_wrist: {{"side":"right" or "left","position":[x,y,z],"quat":[w,x,y,z] optional,"note":"purpose"}}. IK moves up to .18m, executes bounded feedback-controlled chunks; omitted quat holds orientation.
move_joints: {{"targets":{{joint_name:absolute_radians}},"note":"purpose"}}. At most .35rad change per joint. O6 MCP range 0..1.6, thumb yaw0..1.36 pitch0...58. DIP/IP held near initial (arm26).
finish: {{"success":false,"note":"honest outcome"}}. No automatic success; never claim a grasp from wrist movement alone. Push/slide is allowed. Finger link origins are not tips. Check feedback before contact, avoid pressing hand into table.
Current state: {json.dumps(current,separators=(',',':'))}
Previous commands: {json.dumps(calls,separators=(',',':'))}
Last execution feedback: {json.dumps(last_result,separators=(',',':'))}
{feedback}
Choose the next useful movement; if orientation is still approaching an earlier target, finish that approach with the improved multi-chunk controller. No images means use telemetry and externally measured glyph positions, not imagined visual evidence.'''
    content = [{'type': 'text', 'text': prompt}]
    content.extend({'type': 'image_url', 'image_url': {'url': c['image_url'], 'detail': 'low'}}
                   for c in observation_items if c['type'] == 'input_image')
    chat_messages = [{'role': 'user', 'content': content if len(content)>1 else prompt}]
    payload = {'model': 'gpt-6-astra', 'messages': chat_messages,
               'reasoning_effort': 'low'}
    result = OfficialOpenAI().chat(payload, timeout=180)
    choice = result['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise RuntimeError('Model completion did not finish normally')
    text = choice['message']['content']
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    command = json.loads(text)
    if command.get('tool') not in {t['name'] for t in tools} or not isinstance(command.get('arguments'), dict):
        raise ValueError('Model returned an invalid JSON command')
    result['raw_output'] = result.get('output')
    result['status'] = 'completed'
    result['endpoint'] = 'chat/completions'
    result['command_transport'] = 'validated_json_text'
    result['output'] = [{'type': 'function_call', 'name': command['tool'],
                         'call_id': 'json_' + str(time.time_ns()), 'arguments': json.dumps(command['arguments'])}]
    return result


class SimulationResetError(RuntimeError):
    """The live simulator no longer matches the observation used for planning."""


class Sim:
    def __init__(self, ws, directory, root_pose=None):
        self.ws, self.directory = ws, directory
        self.root_pose = root_pose
        self.frame = None
        self.frame_count = 0
        self.actions_sent = 0
        self.last_step = None
        self.initial_sim_time = None
        self.motion_profile = "legacy"
        self.speed = 1.0
        self.motion_stats = {"moves": 0, "interpolation_frames": 0, "hold_frames": 0}

    async def receive(self, kind, timeout=45):
        async def read():
            while True:
                msg = json.loads(await self.ws.recv())
                if msg.get('type') == kind:
                    return msg
                if msg.get('ok') is False:
                    raise RuntimeError(str(msg))
        return await asyncio.wait_for(read(), timeout)

    async def send(self, msg):
        await self.ws.send(json.dumps(msg, allow_nan=False))

    def record(self, frame):
        self.frame = frame
        if self.initial_sim_time is None:
            self.initial_sim_time = frame.get("sim_time")
        stem = self.directory / f'frame_{self.frame_count:04d}'
        self.frame_count += 1
        stem.with_suffix('.jpg').write_bytes(image_bytes(frame['image']))
        if 'right' in frame.get('images', {}):
            stem.with_name(stem.name + '_right').with_suffix('.jpg').write_bytes(image_bytes(frame['images']['right']))
        stem.with_suffix('.json').write_text(json.dumps({k:v for k,v in frame.items() if k not in ('image', 'images')}))
        self.validate_frame(frame)

    def validate_frame(self, frame):
        from scipy.spatial.transform import Rotation
        q = dict(zip(self.status['full_body_joint_names'],
                     np.asarray(frame['state']['joint_position']).reshape(-1)))
        for side in ('left', 'right'):
            predicted = self.kin.fk(q, side + '_wrist_yaw_link')
            measured = transform(np.asarray(frame['state'][side + '_eef_pos']).reshape(-1),
                                 np.asarray(frame['state'][side + '_eef_quat']).reshape(-1))
            error = np.linalg.norm(predicted[:3, 3] - measured[:3, 3])
            angle = Rotation.from_matrix(predicted[:3, :3].T @ measured[:3, :3]).magnitude()
            if not np.isfinite(error + angle) or error > .005 or angle > .03:
                raise SimulationResetError(f'{side} live/FK mismatch: {error:.6f} m, {angle:.6f} rad')
        step = frame.get('latest_step')
        if self.last_step is not None and step is not None and step < self.last_step:
            raise SimulationResetError(f'Simulation step reset: {self.last_step} -> {step}')
        self.last_step = step

    async def check_live_status(self):
        # Use the existing socket: disconnecting a second observer may reset the worker.
        await self.send({'type': 'status'})
        status = await self.receive('status_response')
        (self.directory / 'latest_status.json').write_text(json.dumps(status, indent=2))
        for key in ('scene_id', 'action_layout', 'action_joint_names'):
            if status.get(key) != self.status.get(key):
                raise SimulationResetError(f'Simulator metadata changed: {key}')
        old = self.status['showroom_scene_11_robot_pose']
        new = status['showroom_scene_11_robot_pose']
        a = transform(old['actual_pos'], old['actual_quat_wxyz'])
        b = transform(new['actual_pos'], new['actual_quat_wxyz'])
        if not np.allclose(a, b, atol=.001, rtol=0):
            raise SimulationResetError(f'Robot root changed: {old["actual_pos"]} -> {new["actual_pos"]}')
        if status.get('is_executing') or status.get('queue_length'):
            raise SimulationResetError('Unexpected actions during observation pause')
        if self.last_step is not None and status.get('step') != self.last_step:
            raise SimulationResetError(f'Simulation advanced or reset while paused: {self.last_step} -> {status.get("step")}')

    async def start(self):
        await self.send({'type': 'status'})
        self.status = await self.receive('status_response')
        s = self.status
        (self.directory / 'status.json').write_text(json.dumps(s, indent=2))
        if s['scene_id'] != getattr(self, 'expected_scene', 'showroom_scene_11') or s.get('step_result_subscribed') or s.get('is_executing') or s.get('queue_length'):
            raise RuntimeError('Worker not an idle Scene11 instance')
        self.names = s['action_joint_names']
        pose = s.get('showroom_scene_11_robot_pose')
        self.indices = [s['full_body_joint_names'].index(n) for n in self.names]
        if len(self.names) != s['action_dim']:
            raise RuntimeError('Action metadata mismatch')
        (self.directory / 'status.json').write_text(json.dumps(s, indent=2))
        subscription = {'type': 'subscribe_step_result', 'force_reset': self.root_pose is not None,
                        'empty_queue_policy': 'pause'}
        if self.root_pose is not None:
            subscription['showroom_scene_11_robot_pose'] = self.root_pose
        await self.send(subscription)
        ack = await self.receive('subscribe_step_result_response')
        (self.directory / 'subscribe.json').write_text(json.dumps(ack, indent=2))
        if not ack.get('ok'):
            raise RuntimeError(str(ack))
        if ack.get('showroom_scene_11_robot_pose'):
            pose = ack['showroom_scene_11_robot_pose']
            self.status['showroom_scene_11_robot_pose'] = pose
            (self.directory / 'status.json').write_text(json.dumps(self.status, indent=2))
        if not pose or (self.root_pose is not None and not ack.get('showroom_scene_11_robot_pose')):
            raise RuntimeError('Subscription did not provide the actual robot root pose')
        self.kin = Kinematics(pose['actual_pos'], pose['actual_quat_wxyz'])
        initial = await self.receive('step_result')
        self.record(initial['frames'][-1])
        for side in ('left', 'right'):
            err = np.linalg.norm(self.kin.fk(self.full_q(), side + '_wrist_yaw_link')[:3, 3]
                                 - np.asarray(self.frame['state'][side + '_eef_pos']).reshape(-1))
            if err > .005:
                raise RuntimeError(f'Kinematics calibration failed: {side} error={err}')

    def full_q(self):
        return dict(zip(self.status['full_body_joint_names'],
                        np.asarray(self.frame['state']['joint_position']).reshape(-1)))

    def q(self):
        return np.asarray(self.frame['state']['joint_position']).reshape(-1)[self.indices]

    def observation(self):
        state = self.frame['state']
        return {'joints': dict(zip(self.names, np.round(self.q(), 5).tolist())),
                'eef': {k:v for k,v in state.items() if 'eef' in k},
                'finger_link_positions': {link: self.kin.fk(self.full_q(), link)[:3, 3].round(5).tolist()
                    for link in ['lh_index_distal', 'lh_thumb_distal', 'rh_index_distal', 'rh_thumb_distal']},
                'executed_actions': self.frame.get('executed_actions'),
                'frame_number': self.frame_count - 1}

    async def move(self, targets, *, continuous=False):
        self.validate_frame(self.frame)
        start = self.q()
        target = start.copy()
        if not targets:
            raise ValueError('Specify at least one joint')
        for name, value in targets.items():
            if name not in self.names or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError('Invalid joint or nonfinite target')
            idx = self.names.index(name)
            joint = self.kin.by_name[name]
            if not joint['lower'] <= value <= joint['upper']:
                raise ValueError(f'{name}: target outside joint limits [{joint["lower"]}, {joint["upper"]}]')
            if abs(value - start[idx]) > .35:
                raise ValueError(f'{name}: split motion; max change per tool call is 0.35 rad')
            target[idx] = value
        # Keep contact moves at 0.015 rad/frame; trajectory transit uses 0.0225.
        fast = continuous and self.motion_profile in ("continuous", "trajectory")
        rate = .0225 if fast and self.motion_profile == "trajectory" else .015
        count = max(4 if fast else 12, math.ceil(float(np.max(np.abs(target - start))) / rate))
        hold = 0 if fast else 10
        if isinstance(self.speed,bool) or not isinstance(self.speed,(int,float)) or not math.isfinite(self.speed) or not .25 <= self.speed <= 3.:
            raise ValueError('speed must be finite and within [0.25, 3.0]')
        # Resample the entire move in simulation steps, including dwell time.
        # Keep at least two interpolation steps and four contact settling steps.
        count = max(2, math.ceil(count / self.speed))
        hold = max(4, math.ceil(hold / self.speed)) if hold else 0
        actions = np.linspace(start, target, count + 1)[1:].tolist() + [target.tolist()] * hold
        await self.send({'type': 'submit_actions', 'actions': actions})
        ack = await self.receive('submit_actions_response')
        if not ack.get('ok') or ack.get('accepted_count') != len(actions):
            raise RuntimeError(f'Action submission failed: {ack}')
        self.actions_sent += len(actions)
        self.motion_stats["moves"] += 1
        self.motion_stats["interpolation_frames"] += count
        self.motion_stats["hold_frames"] += hold
        result = await self.receive('step_result', timeout=90)
        for frame in result.get('frames', []):
            self.record(frame)
        if not result.get('frames'):
            raise RuntimeError('No post-action feedback')
        if self.directory is not None:
            with (self.directory / 'motion.jsonl').open('a') as stream:
                stream.write(json.dumps({'phase':getattr(self,'phase','setup'),
                    'speed':self.speed,'interpolation_frames':count,'hold_frames':hold,
                    'end_frame':self.frame_count-1,'frames':len(actions)})+'\n')
        return self.observation()

    async def move_wrist(self, side, position, quat=None):
        pos = np.asarray(position, dtype=float)
        if side not in ('left', 'right') or pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError('Invalid side or position')
        current = np.asarray(self.frame['state'][side + '_eef_pos']).reshape(-1)
        if np.linalg.norm(pos - current) > .18:
            raise ValueError('Split Cartesian move into at most 0.18m per call')
        if quat is None:
            quat = np.asarray(self.frame['state'][side + '_eef_quat']).reshape(-1).tolist()
        from scipy.spatial.transform import Rotation
        goal_rotation = transform([0, 0, 0], quat)[:3, :3]
        previous_error, stalled = float('inf'), 0
        for chunk in range(12):
            target, err = self.kin.solve(self.full_q(), side, pos, quat)
            q = dict(zip(self.names, self.q()))
            ratio = min(1., .34 / max(1e-9, max(abs(v-q[n]) for n,v in target.items())))
            clipped = {n: q[n] + ratio*(v-q[n]) for n,v in target.items()}
            result = await self.move(clipped)
            measured = self.kin.fk(self.full_q(), side + '_wrist_yaw_link')
            position_error = float(np.linalg.norm(measured[:3, 3] - pos))
            orientation_error = float(np.linalg.norm(Rotation.from_matrix(goal_rotation.T @ measured[:3, :3]).as_rotvec()))
            combined_error = position_error + .12 * orientation_error
            stalled = stalled + 1 if combined_error >= previous_error - .0005 else 0
            previous_error = combined_error
            print('WRIST_FEEDBACK', side, chunk, round(position_error, 4), round(orientation_error, 4), flush=True)
            if (position_error < .005 and orientation_error < .035) or stalled >= 3:
                break
        result.update(ik_position_error=err, motion_fraction=ratio,
                      position_error_m=position_error, orientation_error_rad=orientation_error,
                      chunks_executed=chunk+1, stalled=stalled >= 3,
                      note='Bounded feedback-controlled chunks executed toward the requested target; inspect actual errors.')
        return result


async def run(args):
    directory = ROOT / 'runs' / time.strftime('astra_%Y%m%d_%H%M%S')
    directory.mkdir(parents=True)
    print('RUN_DIR', directory, flush=True)
    async with websockets.connect(args.uri, proxy=None, max_size=128*1024*1024,
                                  ping_interval=None, open_timeout=10) as ws:
        sim = Sim(ws, directory)
        await sim.start()
        props = {name: {'type': 'number'} for name in sim.names}
        tools = [
            {'type': 'function', 'name': 'move_wrist',
             'description': 'Move one wrist toward WORLD xyz meters via validated inverse kinematics and up to 12 feedback-controlled bounded chunks. At most 0.18m per call. Optional quaternion wxyz; omitted means preserve orientation. Inspect actual target errors and stalled flag. The wrist is not the fingertips.',
             'parameters': {'type': 'object', 'properties': {
                 'side': {'type': 'string', 'enum': ['left', 'right']},
                 'position': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3},
                 'quat': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4},
                 'note': {'type': 'string'}}, 'required': ['side', 'position', 'note'], 'additionalProperties': False}},
            {'type': 'function', 'name': 'move_joints',
             'description': 'Move named absolute joint targets in radians. Max 0.35 rad change per joint per call; smooth interpolation then fresh image. Unspecified joints hold observed angles.',
             'parameters': {'type': 'object', 'properties': {
                 'targets': {'type': 'object', 'properties': props, 'additionalProperties': False},
                 'note': {'type': 'string'}}, 'required': ['targets', 'note'], 'additionalProperties': False}},
            {'type': 'function', 'name': 'finish', 'description': 'Stop this attempt and report honestly whether the letters were placed.',
             'parameters': {'type': 'object', 'properties': {'success': {'type': 'boolean'}, 'note': {'type': 'string'}},
                            'required': ['success', 'note'], 'additionalProperties': False}},
        ]
        instruction = f'''You control a simulated Unitree G1 with O6 dexterous hands over WebSocket. Task: arrange letter blocks to spell {args.word} left to right in the open near edge of the table, using robot manipulation. Never claim success without visual evidence. You may push/slide blocks if grasping is difficult. You have {args.rounds} tool-call rounds. First establish a useful small motion and assess visual and end-effector feedback, then manipulate. Do not stop just because there is no pretrained policy. All joint targets are absolute radians; moves are interpolated. Fixed base, no locomotion. EEF positions are WORLD meters, at wrist_yaw_link, not fingertips. World +Y is robot-forward, +X is robot-right, +Z up. Robot root approx (0.0033,1.3956,0.76), yaw +90deg. Hand fingers extend beyond the reported wrist. The arm26 layout only controls MCP/CMC; DIP/IP stay near initial settings. Images are actual current simulator renders. Read them carefully. Joint action names: {sim.names}. If you cannot complete, report partial progress honestly. Use tools to execute, not prose plans.'''
        history = [{'role': 'developer', 'content': instruction}]
        history.append({'role': 'developer', 'content': 'A calibrated move_wrist IK tool is also available. Prefer it for arm positioning. You can set WORLD wrist orientation as quaternion wxyz. Joint limits from actual USD are enforced. Finger link positions are articulation frame origins, not contact tips. Use visual feedback to locate block surfaces; do not mistake wrists for fingertips. First achieve moving one target block, then extend to the complete word.'})
        history.append({'role': 'developer', 'content': 'Initial authored scene geometry, only valid until blocks move: table top Z=0.7566; each block 0.04m cube. Centers: A=(-0.00145,1.90351,0.777), C=(0.13870,1.76510,0.777), E=(-0.15373,1.83474,0.777). Suggested final row Y=1.66 with A at X=-0.08, C at X=0, E at X=+0.08. These are asset measurements checked against the initial image, not live object poses. Start with moving C since it is nearest the right hand. First command: raise the right wrist by 0.03m at fixed XY to verify control, then approach C.'})
        if not args.vision:
            history.append({'role': 'developer', 'content': 'For this run no images are sent to the model because the image route is unreliable. Use numeric state feedback and the supplied initial geometry. Frames are saved for an external visual check. Do not claim to have seen images or visually verified success. If needed complete a useful approach/manipulation attempt and report the need for external inspection.'})
        if args.resume:
            previous = []
            for path in sorted(args.resume.glob('model_*.json')):
                previous.extend(item for item in json.loads(path.read_text()).get('output', [])
                                if item.get('type') == 'function_call')
            history.append({'role': 'developer', 'content': 'Continue an interrupted run from CURRENT telemetry; do not repeat calibration.'})
            history.extend(previous[-3:])
        report = {'success_verified': False, 'reason': 'round_limit', 'word': args.word}
        last_feedback = None
        try:
            for turn in range(args.rounds):
                if args.feedback and args.feedback.exists():
                    feedback = args.feedback.read_text()
                    if feedback != last_feedback:
                        history.append({'role': 'user', 'content': 'External visual inspection / operator feedback: ' + feedback})
                        last_feedback = feedback
                im = Image.open(io.BytesIO(image_bytes(sim.frame['image'])))
                im.thumbnail((640, 360))
                encoded = io.BytesIO()
                im.save(encoded, format='JPEG', quality=80)
                jpg = encoded.getvalue()
                observation_content = [{'type': 'input_text', 'text': json.dumps(sim.observation())}]
                if args.vision:
                    observation_content.append({'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(jpg).decode(), 'detail': 'low'})
                history.append({'role': 'user', 'content': observation_content})
                if args.track_letters:
                    from track_letters import Tracker
                    tracking = Tracker(ROOT/'runs/initial.jpg', ROOT/'runs/scene_authored_geometry.json').locate(directory/f'frame_{sim.frame_count-1:04d}.jpg')
                    (directory/f'tracking_{turn:03d}.json').write_text(json.dumps(tracking, indent=2))
                    history.append({'role': 'user', 'content': 'External visual tracking of current frame (XY assumes upright blocks on tabletop, Z unmeasured; missing letters are occluded/uncertain): ' + json.dumps(tracking.get('letters', {}),separators=(',',':'))})
                (directory / 'history.json').write_text(json.dumps(history, ensure_ascii=False))
                print('MODEL_TURN', turn, flush=True)
                result = await asyncio.to_thread(api_call, history, tools)
                (directory / f'model_{turn:03d}.json').write_text(json.dumps(result, ensure_ascii=False))
                output = result.get('output', [])
                history.extend(output)
                calls = [i for i in output if i.get('type') == 'function_call']
                if not calls:
                    print('MODEL_TEXT', result.get('output_text', ''), flush=True)
                    if result.get('status') != 'completed':
                        raise RuntimeError('Model response incomplete')
                    history.append({'role': 'user', 'content': 'Use a motion tool to proceed or finish with an honest result.'})
                    continue
                stopped = False
                for call in calls:
                    params = json.loads(call['arguments'])
                    print('TOOL', call['name'], json.dumps(params, ensure_ascii=False), flush=True)
                    if call['name'] == 'finish':
                        report.update(reason='model_finished', model_claim=params)
                        stopped = True
                        break
                    try:
                        if call['name'] == 'move_wrist':
                            outcome = await sim.move_wrist(params['side'], params['position'], params.get('quat'))
                        elif call['name'] == 'move_joints':
                            outcome = await sim.move(params['targets'])
                        else:
                            raise ValueError('Unknown tool')
                    except ValueError as exc:
                        outcome = {'error': str(exc)}
                    history.append({'type': 'function_call_output', 'call_id': call['call_id'], 'output': json.dumps(outcome)})
                if stopped:
                    break
        except Exception as exc:
            report.update(reason='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            report['actions_sent'] = sim.actions_sent
            report['frames_saved'] = sim.frame_count
            (directory / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
            await sim.send({'type': 'unsubscribe_step_result'})
            print('REPORT', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uri', default='ws://10.19.4.253:8083')
    p.add_argument('--word', default='ACE')
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--vision', action='store_true', help='Send images to the company model; otherwise only telemetry is sent and frames are retained for external inspection')
    p.add_argument('--resume', type=Path, help='Continue from a previous run without resetting the simulation')
    p.add_argument('--feedback', type=Path, help='Read new external visual-inspection notes between model calls')
    p.add_argument('--track-letters', action='store_true', help='Feed conservative classical-vision glyph position estimates to the model')
    asyncio.run(run(p.parse_args()))
