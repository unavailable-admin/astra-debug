"""Scene11 WebSocket transport, feedback validation and bounded trajectories."""

import asyncio
import base64
import io
import json
import math
import time

import numpy as np
from PIL import Image

from .kinematics import Kinematics, transform
from .timing import Timings, server_frame_span


def image_bytes(spec):
    raw = base64.b64decode(spec["data"])
    if spec["encoding"] in ("jpeg_base64", "png_base64"):
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    else:
        h, w, _ = spec["shape"]
        im = Image.frombytes("RGB", (w, h), raw)
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=90)
    return out.getvalue()


class SimulationResetError(RuntimeError):
    """The live simulator no longer matches the observation used for planning."""


class Sim:
    def __init__(self, ws, directory, root_pose=None, timings=None):
        (self.ws, self.directory) = (ws, directory)
        self.root_pose = root_pose
        self.frame = None
        self.frame_count = 0
        self.actions_sent = 0
        self.last_step = None
        self.initial_sim_time = None
        self.speed = 1.0
        self.motion_stats = {"moves": 0, "interpolation_frames": 0, "hold_frames": 0}
        self.timings = timings if timings is not None else Timings()
        self.last_receive = {}

    async def receive(self, kind, timeout=45):

        async def read():
            while True:
                start = time.perf_counter()
                with self.timings.measure("ws_receive_wait"):
                    raw = await self.ws.recv()
                received = time.perf_counter()
                with self.timings.measure("ws_json_decode"):
                    msg = json.loads(raw)
                if msg.get("type") == kind:
                    self.last_receive = {
                        "receive_wait_seconds": received - start,
                        "json_decode_seconds": time.perf_counter() - received,
                        "message_bytes": len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw),
                    }
                    return msg
                if msg.get("ok") is False:
                    raise RuntimeError(str(msg))

        return await asyncio.wait_for(read(), timeout)

    async def send(self, msg):
        with self.timings.measure("ws_json_encode"):
            raw = json.dumps(msg, allow_nan=False)
        with self.timings.measure("ws_send"):
            await self.ws.send(raw)

    def record(self, frame):
        self.frame = frame
        if self.initial_sim_time is None:
            self.initial_sim_time = frame.get("sim_time")
        self.frame_count += 1
        # Keep current feedback in memory; never decode or save motion images.
        self.validate_frame(frame)

    def save_observation(self):
        """Save only the current stereo pair needed by VLM and stereo perception."""
        frame = self.frame
        if frame is None or "right" not in frame.get("images", {}):
            raise ValueError("A current stereo frame is required")
        stem = self.directory / f"observation_{self.frame_count - 1:04d}"
        with self.timings.measure("observation_write"):
            stem.with_suffix(".jpg").write_bytes(image_bytes(frame["image"]))
            stem.with_name(stem.name + "_right").with_suffix(".jpg").write_bytes(
                image_bytes(frame["images"]["right"])
            )
        return stem.with_suffix(".jpg")

    def validate_frame(self, frame):
        from scipy.spatial.transform import Rotation

        q = dict(
            zip(
                self.status["full_body_joint_names"],
                np.asarray(frame["state"]["joint_position"]).reshape(-1),
            )
        )
        for side in ("left", "right"):
            predicted = self.kin.fk(q, side + "_wrist_yaw_link")
            measured = transform(
                np.asarray(frame["state"][side + "_eef_pos"]).reshape(-1),
                np.asarray(frame["state"][side + "_eef_quat"]).reshape(-1),
            )
            error = np.linalg.norm(predicted[:3, 3] - measured[:3, 3])
            angle = Rotation.from_matrix(predicted[:3, :3].T @ measured[:3, :3]).magnitude()
            if not np.isfinite(error + angle) or error > 0.005 or angle > 0.03:
                raise SimulationResetError(
                    f"{side} live/FK mismatch: {error:.6f} m, {angle:.6f} rad"
                )
        step = frame.get("latest_step")
        if self.last_step is not None and step is not None and (step < self.last_step):
            raise SimulationResetError(f"Simulation step reset: {self.last_step} -> {step}")
        self.last_step = step

    async def check_live_status(self):
        await self.send({"type": "status"})
        status = await self.receive("status_response")
        (self.directory / "latest_status.json").write_text(json.dumps(status, indent=2))
        for key in ("scene_id", "action_layout", "action_joint_names"):
            if status.get(key) != self.status.get(key):
                raise SimulationResetError(f"Simulator metadata changed: {key}")
        old = self.status["showroom_scene_11_robot_pose"]
        new = status["showroom_scene_11_robot_pose"]
        a = transform(old["actual_pos"], old["actual_quat_wxyz"])
        b = transform(new["actual_pos"], new["actual_quat_wxyz"])
        if not np.allclose(a, b, atol=0.001, rtol=0):
            raise SimulationResetError(
                f"Robot root changed: {old['actual_pos']} -> {new['actual_pos']}"
            )
        if status.get("is_executing") or status.get("queue_length"):
            raise SimulationResetError("Unexpected actions during observation pause")
        if self.last_step is not None and status.get("step") != self.last_step:
            raise SimulationResetError(
                f"Simulation advanced or reset while paused: {self.last_step} -> {status.get('step')}"
            )

    async def start(self):
        await self.send({"type": "status"})
        self.status = await self.receive("status_response")
        s = self.status
        (self.directory / "status.json").write_text(json.dumps(s, indent=2))
        if (
            s["scene_id"] != "showroom_scene_11_stereo"
            or s.get("step_result_subscribed")
            or s.get("is_executing")
            or s.get("queue_length")
        ):
            raise RuntimeError("Worker not an idle Scene11 instance")
        self.names = s["action_joint_names"]
        pose = s.get("showroom_scene_11_robot_pose")
        self.indices = [s["full_body_joint_names"].index(n) for n in self.names]
        if len(self.names) != s["action_dim"]:
            raise RuntimeError("Action metadata mismatch")
        (self.directory / "status.json").write_text(json.dumps(s, indent=2))
        subscription = {
            "type": "subscribe_step_result",
            "force_reset": self.root_pose is not None,
            "empty_queue_policy": "pause",
        }
        if self.root_pose is not None:
            subscription["showroom_scene_11_robot_pose"] = self.root_pose
        await self.send(subscription)
        ack = await self.receive("subscribe_step_result_response")
        (self.directory / "subscribe.json").write_text(json.dumps(ack, indent=2))
        if not ack.get("ok"):
            raise RuntimeError(str(ack))
        if ack.get("showroom_scene_11_robot_pose"):
            pose = ack["showroom_scene_11_robot_pose"]
            self.status["showroom_scene_11_robot_pose"] = pose
            (self.directory / "status.json").write_text(json.dumps(self.status, indent=2))
        if not pose or (
            self.root_pose is not None and (not ack.get("showroom_scene_11_robot_pose"))
        ):
            raise RuntimeError("Subscription did not provide the actual robot root pose")
        self.kin = Kinematics(pose["actual_pos"], pose["actual_quat_wxyz"], timings=self.timings)
        initial = await self.receive("step_result")
        self.record(initial["frames"][-1])
        for side in ("left", "right"):
            err = np.linalg.norm(
                self.kin.fk(self.full_q(), side + "_wrist_yaw_link")[:3, 3]
                - np.asarray(self.frame["state"][side + "_eef_pos"]).reshape(-1)
            )
            if err > 0.005:
                raise RuntimeError(f"Kinematics calibration failed: {side} error={err}")

    def full_q(self):
        return dict(
            zip(
                self.status["full_body_joint_names"],
                np.asarray(self.frame["state"]["joint_position"]).reshape(-1),
            )
        )

    def q(self):
        return np.asarray(self.frame["state"]["joint_position"]).reshape(-1)[self.indices]

    def observation(self):
        state = self.frame["state"]
        return {
            "joints": dict(zip(self.names, np.round(self.q(), 5).tolist())),
            "eef": {k: v for (k, v) in state.items() if "eef" in k},
            "finger_link_positions": {
                link: self.kin.fk(self.full_q(), link)[:3, 3].round(5).tolist()
                for link in [
                    "lh_index_distal",
                    "lh_thumb_distal",
                    "rh_index_distal",
                    "rh_thumb_distal",
                ]
            },
            "executed_actions": self.frame.get("executed_actions"),
            "frame_number": self.frame_count - 1,
        }

    async def move(self, targets, *, continuous=False):
        self.validate_frame(self.frame)
        start = self.q()
        target = start.copy()
        if not targets:
            raise ValueError("Specify at least one joint")
        for name, value in targets.items():
            if name not in self.names or isinstance(value, bool) or (not math.isfinite(value)):
                raise ValueError("Invalid joint or nonfinite target")
            idx = self.names.index(name)
            joint = self.kin.by_name[name]
            if not joint["lower"] <= value <= joint["upper"]:
                raise ValueError(
                    f"{name}: target outside joint limits [{joint['lower']}, {joint['upper']}]"
                )
            if abs(value - start[idx]) > 0.35:
                raise ValueError(f"{name}: split motion; max change per tool call is 0.35 rad")
            target[idx] = value
        fast = continuous
        rate = 0.0225 if fast else 0.015
        count = max(4 if fast else 12, math.ceil(float(np.max(np.abs(target - start))) / rate))
        hold = 0 if fast else 10
        if (
            isinstance(self.speed, bool)
            or not isinstance(self.speed, (int, float))
            or (not math.isfinite(self.speed))
            or (not 0.25 <= self.speed <= 3.0)
        ):
            raise ValueError("speed must be finite and within [0.25, 3.0]")
        count = max(2, math.ceil(count / self.speed))
        hold = max(4, math.ceil(hold / self.speed)) if hold else 0
        actions = np.linspace(start, target, count + 1)[1:].tolist() + [target.tolist()] * hold
        simulation_start = self.frame.get("sim_time")
        exchange_start = time.perf_counter()
        await self.send({"type": "submit_actions", "actions": actions})
        sent = time.perf_counter()
        ack = await self.receive("submit_actions_response")
        acknowledged = time.perf_counter()
        if not ack.get("ok") or ack.get("accepted_count") != len(actions):
            raise RuntimeError(f"Action submission failed: {ack}")
        self.actions_sent += len(actions)
        self.motion_stats["moves"] += 1
        self.motion_stats["interpolation_frames"] += count
        self.motion_stats["hold_frames"] += hold
        result = await self.receive("step_result", timeout=90)
        received = time.perf_counter()
        frames = result.get("frames", [])
        if not frames:
            raise RuntimeError("No post-action feedback")
        with self.timings.measure("feedback_validation"):
            for frame in frames:
                self.record(frame)
        processed = time.perf_counter()
        span = server_frame_span(frames)
        self.timings.add("action_round_trip", received - exchange_start)
        if span is not None:
            self.timings.add("server_frame_span", span)
        simulation_end = frames[-1].get("sim_time")
        timing = {
            "send_seconds": sent - exchange_start,
            "ack_wait_seconds": acknowledged - sent,
            "result_wait_seconds": received - acknowledged,
            "round_trip_seconds": received - exchange_start,
            "result_message": dict(self.last_receive),
            "feedback_seconds": processed - received,
            "server_frame_span_seconds": span,
            "server_execution_seconds": None,
            "network_transfer_seconds": None,
            "separation_status": "unavailable_without_server_batch_timers",
            "simulation_seconds": (
                simulation_end - simulation_start
                if simulation_start is not None and simulation_end is not None else None
            ),
        }
        if self.directory is not None:
            with (
                self.timings.measure("motion_log_write"),
                (self.directory / "motion.jsonl").open("a") as stream,
            ):
                stream.write(
                    json.dumps(
                        {
                            "phase": getattr(self, "phase", "setup"),
                            "speed": self.speed,
                            "interpolation_frames": count,
                            "hold_frames": hold,
                            "end_frame": self.frame_count - 1,
                            "frames": len(actions),
                            "timing": timing,
                        }
                    )
                    + "\n"
                )
        return self.observation()

    async def move_wrist(self, side, position, quat=None):
        pos = np.asarray(position, dtype=float)
        if side not in ("left", "right") or pos.shape != (3,) or (not np.all(np.isfinite(pos))):
            raise ValueError("Invalid side or position")
        current = np.asarray(self.frame["state"][side + "_eef_pos"]).reshape(-1)
        if np.linalg.norm(pos - current) > 0.18:
            raise ValueError("Split Cartesian move into at most 0.18m per call")
        if quat is None:
            quat = np.asarray(self.frame["state"][side + "_eef_quat"]).reshape(-1).tolist()
        from scipy.spatial.transform import Rotation

        goal_rotation = transform([0, 0, 0], quat)[:3, :3]
        (previous_error, stalled) = (float("inf"), 0)
        for chunk in range(12):
            (target, err) = self.kin.solve(self.full_q(), side, pos, quat)
            q = dict(zip(self.names, self.q()))
            ratio = min(1.0, 0.34 / max(1e-09, max((abs(v - q[n]) for (n, v) in target.items()))))
            clipped = {n: q[n] + ratio * (v - q[n]) for (n, v) in target.items()}
            result = await self.move(clipped)
            measured = self.kin.fk(self.full_q(), side + "_wrist_yaw_link")
            position_error = float(np.linalg.norm(measured[:3, 3] - pos))
            orientation_error = float(
                np.linalg.norm(Rotation.from_matrix(goal_rotation.T @ measured[:3, :3]).as_rotvec())
            )
            combined_error = position_error + 0.12 * orientation_error
            stalled = stalled + 1 if combined_error >= previous_error - 0.0005 else 0
            previous_error = combined_error
            print(
                "WRIST_FEEDBACK",
                side,
                chunk,
                round(position_error, 4),
                round(orientation_error, 4),
                flush=True,
            )
            if position_error < 0.005 and orientation_error < 0.035 or stalled >= 3:
                break
        result.update(
            ik_position_error=err,
            motion_fraction=ratio,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            chunks_executed=chunk + 1,
            stalled=stalled >= 3,
            note="Bounded feedback-controlled chunks executed toward the requested target; inspect actual errors.",
        )
        return result
