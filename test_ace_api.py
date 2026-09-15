#!/usr/bin/env python3
"""Minimal Responses API smoke test for the official endpoint."""

import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request
from openai_connection import OfficialOpenAI


BASE_URL = "https://api.openai.com/v1"
API_KEY_FILE = Path(__file__).resolve().parent / ".secrets/openai_api_key"


def load_api_key() -> str | None:
    """Load the key from the fixed raw-text key file beside this script."""
    try:
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"无法读取 API key 文件 {API_KEY_FILE}: {exc}", file=sys.stderr)
        return None
    return key or None


def main() -> int:
    # if len(sys.argv) != 3:
    #     print(f'用法: {sys.argv[0]} <模型> <输入>', file=sys.stderr)
    #     print(f'示例: {sys.argv[0]} gpt-5.5 "你好"', file=sys.stderr)
    #     return 2

    # model = sys.argv[1]
    # user_input = sys.argv[2]
    model = "gpt-6-astra"
    user_input = "你觉得RSI和RL有什么本质上的区别么？RSI怎么才能做出差异性？"
    reasoning_effort = "max"
    timeout = 600
    api_key = load_api_key()
    if not api_key:
        print(f"请将 API key 写入 {API_KEY_FILE}", file=sys.stderr)
        return 2

    request = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/responses",
        data=json.dumps(
            {
                "model": model,
                "input": user_input,
                "reasoning": {"effort": reasoning_effort},
                "store": False,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started_at = time.monotonic()
    request_finished = threading.Event()

    def show_elapsed_time() -> None:
        print("等待响应: 0 秒", end="", flush=True)
        while not request_finished.wait(1):
            elapsed = int(time.monotonic() - started_at)
            print(f"\r等待响应: {elapsed} 秒", end="", flush=True)

    timer = threading.Thread(target=show_elapsed_time, daemon=True)
    timer.start()

    try:
        with OfficialOpenAI().opener.open(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except TimeoutError:
        print(f"请求超过 {timeout} 秒仍未完成", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"请求失败: {exc}", file=sys.stderr)
        return 1
    finally:
        request_finished.set()
        timer.join()
        elapsed = time.monotonic() - started_at
        print(f"\r等待响应: {elapsed:.1f} 秒")

    print(f"model: {result.get('model')}")
    print(f"status: {result.get('status')}")
    print(f"output: {result.get('output_text', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
