"""Cold reset of an idle Scene11 worker; restores full36 before returning."""

import argparse
import asyncio
import json
from pathlib import Path

import websockets

from .paths import DEFAULT_URI


async def recv(ws, kind):
    while True:
        r = json.loads(await asyncio.wait_for(ws.recv(), 15))
        if r.get("type") == kind:
            return r


async def change(layout, uri=DEFAULT_URI):
    async with websockets.connect(
        uri, proxy=None, ping_interval=None, open_timeout=8, close_timeout=2
    ) as ws:
        await ws.send(json.dumps({"type": "status"}))
        r = await recv(ws, "status_response")
        if r.get("step_result_subscribed") or r.get("is_executing") or r.get("queue_length"):
            raise RuntimeError("Worker occupied; stopping")
        if r.get("scene_id") not in ("showroom_scene_11", "showroom_scene_11_stereo"):
            raise RuntimeError("Unexpected scene")
        await ws.send(json.dumps({"type": "switch_action_layout", "action_layout": layout}))
        ack = await recv(ws, "switch_action_layout_response")
        print("SWITCH", ack, flush=True)
        if not ack.get("ok"):
            raise RuntimeError(str(ack))
    for i in range(60):
        await asyncio.sleep(5)
        try:
            async with websockets.connect(
                uri, proxy=None, ping_interval=None, open_timeout=3, close_timeout=1
            ) as ws:
                await ws.send(json.dumps({"type": "status"}))
                r = await recv(ws, "status_response")
                if r.get("action_layout") == layout:
                    print("READY", layout, "step", r.get("step"), flush=True)
                    return r
        except (OSError, TimeoutError, websockets.exceptions.WebSocketException):
            pass
        if i % 6 == 0:
            print("WAIT", layout, flush=True)
    raise TimeoutError("Worker restart timed out")


async def reset(uri=DEFAULT_URI, output=None):
    await change("arm26", uri)
    status = await change("full36", uri)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status, indent=2))
    print("COLD_RESET_COMPLETE", flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(prog="astrabot reset", description=__doc__)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--output", type=Path, help="Optional reset status JSON")
    args = parser.parse_args(argv)
    asyncio.run(reset(args.uri, args.output))
