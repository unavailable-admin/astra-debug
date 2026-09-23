"""Automatic stereo scene preparation; optional manual browser mode, no motion."""

import argparse
import hmac
import json
import secrets
import shlex
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .config import file_digest
from .scene_builder import SceneBuilder


def planning_command(builder):
    """Return the offline planning command using the exported scene paths."""
    return shlex.join(
        [
            "python3",
            "-m",
            "astrabot.robot.trial_plan",
            "--config",
            str(builder.output / "scene-config.json"),
            "--input",
            str(builder.output / "scene-input.json"),
            "--output",
            str(builder.output / "plan.json"),
        ]
    )


def automatic_scene(builder, dimensions, timeout):
    """Make one identification request, compute geometry locally, then export."""
    from .scene_identification import SceneVision, annotation_from_decision

    if not 4 <= len(builder.candidates) <= 192:
        raise ValueError("scene_candidate_count_out_of_range:需要至少四块分散积木；候选过多时改善背景后重拍")
    print(
        (
            f"正在识别目标 {getattr(builder, 'target_letter', 'A')} 和桌面测量点；其他物体由操作者避让，不做障碍确认。"
            if getattr(builder, "operator_cleared_workspace", False)
            else "正在自动识别 A、积木和桌边：最多一次视觉 API 请求，不连接机器人。"
        ),
        flush=True,
    )
    vision = SceneVision(builder.output / "vision", timeout=timeout, glyph_mode="real")
    decision = vision.identify(builder)
    path = builder.output / "identification.json"
    path.write_text(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    annotation = annotation_from_decision(
        decision,
        builder.candidates,
        builder.left.shape,
        dimensions,
        operator_cleared_workspace=getattr(builder, "operator_cleared_workspace", False),
        target_letter=getattr(builder, "target_letter", "A"),
    )
    (builder.output / "annotation.json").write_text(json.dumps(annotation, indent=2, allow_nan=False) + "\n")
    print("识别完成，正在本地计算双目坐标、桌面和障碍。", flush=True)
    summary = builder.build(annotation, identification={"path": str(path), "sha256": file_digest(path)})
    result = builder.export_automatic()
    result.update(summary=summary, planning_command=planning_command(builder))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


def handler(builder, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            # Do not put the local access token into access logs.
            pass

        def reply(self, code, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            provided = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            if not hmac.compare_digest(provided, token):
                self.reply(403, {"error": "请使用终端提供的完整页面地址"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.reply(200, Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/data":
                self.reply(
                    200,
                    {
                        "candidates": builder.candidates,
                        "width": builder.left.shape[1],
                        "height": builder.left.shape[0],
                        "capture": str(builder.capture_dir),
                        "capture_time": datetime.fromtimestamp(builder.meta["wall_time"], timezone.utc).isoformat(),
                    },
                )
            elif path == "/image":
                self.reply(200, (builder.capture_dir / "left.jpg").read_bytes(), "image/jpeg")
            elif path == "/preview" and builder.draft is not None:
                self.reply(200, (builder.output / "scene-review.jpg").read_bytes(), "image/jpeg")
            else:
                self.reply(404, {"error": "not_found"})

        def do_POST(self):
            if not self.authorized():
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.reply(403, {"error": "cross_origin_request_rejected"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("invalid_request_size")
                self.connection.settimeout(5)
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise TypeError("request_must_be_object")
                path = urlsplit(self.path).path
                if path == "/build":
                    result = builder.build(request)
                elif path == "/approve":
                    result = builder.approve(request)
                    result["planning_command"] = planning_command(builder)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                else:
                    self.reply(404, {"error": "not_found"})
                    return
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.reply(400, {"error": str(exc)})
                return
            self.reply(200, result)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True, help="Completed capture directory with robot state")
    parser.add_argument("--output", type=Path, required=True, help="New scene directory")
    parser.add_argument("--manual", action="store_true", help="Use browser annotation instead of the vision API")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--api-timeout", type=float, default=120, help="Single API request timeout in seconds; no retries"
    )
    parser.add_argument("--table-width-mm", type=float, default=1200, help="Automatic mode table width (mm)")
    parser.add_argument("--table-depth-mm", type=float, default=600, help="Automatic mode table depth (mm)")
    parser.add_argument("--table-thickness-mm", type=float, default=10, help="Automatic mode table thickness (mm)")
    args = parser.parse_args(argv)
    if args.manual and not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    if not 1 <= args.api_timeout <= 300:
        parser.error("api-timeout must be between 1 and 300 seconds")
    builder = None
    try:
        from .scene_identification import validate_dimensions

        dimensions = validate_dimensions(args.table_width_mm, args.table_depth_mm, args.table_thickness_mm)
        builder = SceneBuilder(args.config, args.capture, args.output)
        if not args.manual:
            automatic_scene(builder, dimensions, args.api_timeout)
            return 0
        token = secrets.token_urlsafe(24)
        with HTTPServer(("127.0.0.1", args.port), handler(builder, token)) as server:
            print(f"Open on Thor: http://127.0.0.1:{args.port}/?token={token}", flush=True)
            print("本地场景标注，API 调用为 0；不连接执行器、不发送机器人命令。", flush=True)
            print("确认导出后自动退出。Ctrl-C 可取消；尚未确认时不会生成 scene-input.json。", flush=True)
            while not builder.exported:
                server.handle_request()
    except KeyboardInterrupt:
        print("场景标注已停止，未启动机器人。")
        return 1
    except Exception as exc:  # noqa: BLE001 -- persist bounded scene/API failures without motion
        # Include API failures in the same bounded, non-motion CLI failure path.
        if builder is not None and not builder.exported:
            error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "mode": "manual" if args.manual else "automatic",
                "hardware_ready": False,
                "export_complete": False,
                "next_step": "按错误修正后使用新输出目录重试；需要新照片时重新 capture。不会自动重试 API。",
            }
            try:
                (builder.output / "scene-error.json").write_text(json.dumps(error, indent=2, ensure_ascii=False) + "\n")
            except OSError:
                pass
        parser.exit(1, f"Scene build failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
