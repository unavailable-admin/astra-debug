#!/usr/bin/env python3
"""Official OpenAI / EASIM connectivity and function-call probe.

Use a Python environment containing websockets>=15. The official API uses the local HTTP proxy; simulation uses a direct connection.
This script never resets a scene or submits robot actions.
"""

import argparse
import asyncio
import json
from pathlib import Path
import urllib.request
from openai_connection import OfficialOpenAI


ROOT = Path(__file__).resolve().parent


def probe_api(args):
    key = args.key_file.read_text().strip()
    if not key:
        raise ValueError("Empty API key file")
    payload = {
        "model": args.model,
        "input": "Connectivity test only. Call report_probe with ready=true.",
        "reasoning": {"effort": "low"},
        "store": False,
        "tools": [{
            "type": "function", "name": "report_probe",
            "description": "Report function calling readiness; no robot motion.",
            "parameters": {
                "type": "object", "properties": {"ready": {"type": "boolean"}},
                "required": ["ready"], "additionalProperties": False,
            },
            "strict": True,
        }],
        "tool_choice": {"type": "function", "name": "report_probe"},
    }
    opener = OfficialOpenAI().opener
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/responses",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with opener.open(request, timeout=90) as response:
        result = json.load(response)
    calls = [item for item in result.get("output", []) if item.get("type") == "function_call"]
    ready = any(
        item.get("name") == "report_probe"
        and json.loads(item.get("arguments", "{}")) == {"ready": True}
        for item in calls
    )
    return {"ok": result.get("status") == "completed" and ready,
            "model": result.get("model"), "status": result.get("status"),
            "function_calls": calls}


async def probe_worker(uri):
    import websockets

    try:
        async with websockets.connect(
            uri, proxy=None, open_timeout=6, close_timeout=2,
            ping_interval=None, max_size=16 * 1024 * 1024,
        ) as ws:
            await ws.send(json.dumps({"type": "status"}))
            async def receive_status():
                while True:
                    message = json.loads(await ws.recv())
                    if message.get("type") == "status_response":
                        return message
            status = await asyncio.wait_for(receive_status(), timeout=8)
            return {"uri": uri, "ok": True, "status": status}
    except Exception as exc:
        return {"uri": uri, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def main(args):
    async def api():
        try:
            return await asyncio.to_thread(probe_api, args)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    results = await asyncio.gather(api(), *(probe_worker(uri) for uri in args.uri))
    report = {"api": results[0], "workers": results[1:], "robot_actions_sent": 0}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    print(text)
    return 0 if results[0]["ok"] and any(r["ok"] for r in results[1:]) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", action="append", required=True)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--key-file", type=Path, default=ROOT / ".secrets/openai_api_key")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/astra_probe.json")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
