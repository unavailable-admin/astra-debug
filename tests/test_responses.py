import copy
import io
import json
import tempfile
import unittest
import urllib.response
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from astrabot.api import response_text
from astrabot.vision import AstraVision
from tests import test_api


def completed_response():
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": '{"visible":true}'}],
            },
        ],
    }


def http_response(body, content_type="text/event-stream"):
    headers = Message()
    headers["Content-Type"] = content_type
    return urllib.response.addinfourl(io.BytesIO(body), headers, "https://relay.example/v1/responses", 200)


def event_bytes(event):
    return ("data: " + json.dumps(event) + "\r\n\r\n").encode()


class ResponsesConnection(unittest.TestCase):
    setUp = test_api.OfficialConnection.setUp
    write_config = test_api.OfficialConnection.write_config

    def client(self):
        from astrabot.api import OfficialOpenAI

        (self.root / ".secrets/openai_api_key").write_text("dummy-test-key")
        self.write_config(base_url="https://relay.example/v1")
        return OfficialOpenAI()

    def test_stream_uses_completed_response_not_partial_deltas(self):
        client = self.client()
        result = completed_response()
        stream = b": keepalive\r\nevent: response.output_text.delta\r\n"
        stream += event_bytes({"type": "response.output_text.delta", "delta": "invalid partial text"})
        stream += event_bytes({"type": "response.completed", "response": result})
        with patch.object(client.opener, "open", return_value=http_response(stream)) as call:
            actual = client.responses({"input": [], "model": "ignored", "store": True}, timeout=5)
        self.assertEqual(actual, result)
        self.assertEqual(response_text(actual), '{"visible":true}')
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://relay.example/v1/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer dummy-test-key")
        self.assertEqual(request.get_header("Accept"), "text/event-stream")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "gpt-6-astra")
        self.assertTrue(body["stream"])
        self.assertFalse(body["store"])
        self.assertEqual(call.call_args.kwargs["timeout"], 5)

    def test_json_response_from_compatible_service(self):
        client = self.client()
        result = completed_response()
        with patch.object(
            client.opener, "open", return_value=http_response(json.dumps(result).encode(), "application/json")
        ):
            self.assertEqual(response_text(client.responses({"input": []})), '{"visible":true}')

    def test_stream_errors_or_early_end_never_return_partial_text(self):
        partial = event_bytes({"type": "response.output_text.delta", "delta": '{"visible":true}'})
        endings = [b"", b"data: [DONE]\n\n", b"data: invalid json\n\n"]
        endings += [event_bytes({"type": kind}) for kind in ("error", "response.failed", "response.incomplete")]
        for ending in endings:
            with self.subTest(ending=ending):
                client = self.client()
                with (
                    patch.object(client.opener, "open", return_value=http_response(partial + ending)),
                    self.assertRaises(ValueError),
                ):
                    client.responses({"input": []})

    def test_multiline_sse_data(self):
        client = self.client()
        stream = b'data: {"type":"response.completed",\ndata: "response":'
        stream += json.dumps(completed_response()).encode() + b"}\n\n"
        with patch.object(client.opener, "open", return_value=http_response(stream)):
            self.assertEqual(response_text(client.responses({"input": []})), '{"visible":true}')

    def test_vision_sends_images_and_saves_native_response(self):
        client = self.client()
        image = self.root / "camera.png"
        Image.new("RGB", (12, 10), "white").save(image)
        result = completed_response()
        stream = event_bytes({"type": "response.completed", "response": result})
        with (
            patch("astrabot.vision.OfficialOpenAI", return_value=client),
            patch.object(client.opener, "open", return_value=http_response(stream)) as call,
        ):
            vision = AstraVision(self.root / "vision")
            self.assertEqual(vision._call("test", "Return JSON with visible.", [image]), {"visible": True})
        body = json.loads(call.call_args.args[0].data)
        self.assertNotIn("messages", body)
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["max_output_tokens"], 2048)
        self.assertIn("Report uncertainty explicitly", body["instructions"])
        parts = body["input"][0]["content"]
        self.assertEqual(parts[0], {"type": "input_text", "text": "Return JSON with visible."})
        self.assertEqual(parts[1]["type"], "input_image")
        self.assertTrue(parts[1]["image_url"].startswith("data:image/jpeg;base64,"))
        saved = json.loads(next((self.root / "vision").glob("*.response.json")).read_text())
        self.assertEqual(saved, result)
        metadata = json.loads(next((self.root / "vision").glob("*.request.json")).read_text())
        self.assertEqual(metadata["vision_wire_api"], "responses")
        self.assertNotIn("dummy-test-key", json.dumps(metadata))


class ResponsesOutput(unittest.TestCase):
    def test_incomplete_empty_refused_or_tool_output_is_rejected(self):
        valid = completed_response()
        cases = []
        for status in ("failed", "incomplete", "in_progress", None):
            cases.append({**valid, "status": status})
        cases.extend([{**valid, "output": []}, {**valid, "error": {"code": "error"}}])
        for change in (
            {"status": "incomplete"},
            {"role": "user"},
            {"type": "function_call"},
            {"content": [{"type": "refusal", "refusal": "Cannot identify."}]},
            {"content": [{"type": "output_text", "text": ""}]},
        ):
            result = copy.deepcopy(valid)
            result["output"][1].update(change)
            cases.append(result)
        for result in cases:
            with self.subTest(result=result), self.assertRaises(ValueError):
                response_text(result)

    def test_vision_does_not_save_decision_for_incomplete_response(self):
        with tempfile.TemporaryDirectory() as directory, patch("astrabot.vision.OfficialOpenAI") as factory:
            result = completed_response()
            result["status"] = "incomplete"
            factory.return_value.responses.return_value = result
            factory.return_value.model = "test-model"
            factory.return_value.metadata.return_value = {}
            vision = AstraVision(directory)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                vision._call("test", "Return JSON.", [])
            self.assertFalse(list(Path(directory).glob("*.decision.json")))
            self.assertEqual(len(list(Path(directory).glob("*.error.json"))), 1)


if __name__ == "__main__":
    unittest.main()
