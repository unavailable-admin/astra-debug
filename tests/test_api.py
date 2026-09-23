import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from astrabot.api import ExplicitProxy, OfficialOpenAI


class OfficialConnection(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / ".secrets").mkdir()
        root_patch = patch("astrabot.api.ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        environment_patch = patch.dict(os.environ, {}, clear=True)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

    def write_config(self, **config):
        path = self.root / ".secrets/openai_config.json"
        path.write_text(json.dumps(config))

    def test_company_environment_cannot_select_company_key_or_endpoint(self):
        with patch.dict(
            os.environ,
            {"ASTRA_API_BASE_URL": "http://company.invalid/v1", "ASTRA_MODEL": "old"},
            clear=True,
        ):
            client = OfficialOpenAI()
        self.assertEqual(client.base_url, "https://api.openai.com/v1")
        self.assertEqual(client.key_file, self.root / ".secrets/openai_api_key")
        self.assertEqual(client.model, "gpt-6-astra")
        self.assertEqual(client.metadata()["provider"], "openai_official")

    def test_local_config_selects_compatible_endpoint_and_model(self):
        self.write_config(base_url="https://relay.example/api/openai/", model="relay-model")

        client = OfficialOpenAI()

        self.assertEqual(client.base_url, "https://relay.example/api/openai")
        self.assertEqual(client.model, "relay-model")
        self.assertEqual(client.metadata()["provider"], "openai_compatible")

    def test_environment_overrides_local_config(self):
        self.write_config(base_url="https://relay.example/v1", model="configured-model")
        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "https://override.example:8443/openai/v1/",
                "OPENAI_MODEL": "override-model",
            },
        ):
            client = OfficialOpenAI()

        self.assertEqual(client.base_url, "https://override.example:8443/openai/v1")
        self.assertEqual(client.model, "override-model")

    def test_partial_local_config_keeps_other_defaults(self):
        self.write_config(base_url="https://relay.example/v1")
        self.assertEqual(OfficialOpenAI().model, "gpt-6-astra")

        self.write_config(model="configured-model")
        client = OfficialOpenAI()
        self.assertEqual(client.base_url, "https://api.openai.com/v1")
        self.assertEqual(client.model, "configured-model")
        self.assertEqual(client.metadata()["provider"], "openai_official")

    def test_https_endpoints_allow_custom_paths_and_ports(self):
        for endpoint in (
            "https://relay.example",
            "https://relay.example/v1",
            "https://relay.example/api/openai/v1",
            "https://relay.example:8443/v1",
        ):
            with self.subTest(endpoint=endpoint):
                with patch.dict(os.environ, {"OPENAI_BASE_URL": endpoint}):
                    client = OfficialOpenAI()
                self.assertEqual(client.base_url, endpoint)

    def test_invalid_endpoints_are_rejected_from_environment_and_config(self):
        for endpoint in (
            "http://relay.example/v1",
            "relay.example/v1",
            "https:///v1",
            "https://user@relay.example/v1",
            "https://user:password@relay.example/v1",
            "https://:password@relay.example/v1",
            "https://relay.example/v1?token=value",
            "https://relay.example/v1?",
            "https://relay.example/v1#fragment",
            "https://relay.example/v1#",
            "https://relay.example/v1\n",
            "https://relay.example:invalid/v1",
            "https://relay.example:65536/v1",
            "https://relay.example:-1/v1",
        ):
            with (
                self.subTest(endpoint=endpoint, source="environment"),
                patch.dict(os.environ, {"OPENAI_BASE_URL": endpoint}),
                self.assertRaises(ValueError),
            ):
                OfficialOpenAI()
            with self.subTest(endpoint=endpoint, source="config"):
                self.write_config(base_url=endpoint)
                with self.assertRaises(ValueError):
                    OfficialOpenAI()

    def test_explicit_proxy_preserves_https_tunnel_even_with_no_proxy(self):
        request = urllib.request.Request("https://api.openai.com/v1/models")
        with patch.dict(os.environ, {"NO_PROXY": "*", "no_proxy": "*"}):
            ExplicitProxy().proxy_open(request, "http://127.0.0.1:18888", "https")
        self.assertEqual(request.host, "127.0.0.1:18888")
        self.assertEqual(request._tunnel_host, "api.openai.com")
        self.assertEqual(request.type, "https")

    def test_chat_loads_selected_file_without_storing_key_in_metadata(self):
        path = self.root / "test_key"
        path.write_text("dummy-test-key\n")
        self.write_config(base_url="https://relay.example/api/openai", model="relay-model")
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY_FILE": str(path),
                "OPENAI_API_KEY": "ignored-environment-key",
            },
        ):
            client = OfficialOpenAI()
        with patch.object(client.opener, "open", return_value=io.BytesIO(b'{"choices": []}')) as call:
            self.assertEqual(
                client.chat({"messages": [], "model": "ignored-model", "store": True}),
                {"choices": []},
            )
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://relay.example/api/openai/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer dummy-test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "relay-model")
        self.assertFalse(payload["store"])
        metadata = client.metadata()
        self.assertEqual(metadata["provider"], "openai_compatible")
        self.assertNotIn("dummy-test-key", json.dumps(metadata))
        self.assertNotIn("ignored-environment-key", json.dumps(metadata))

    def test_environment_key_does_not_replace_missing_key_file(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "ignored-environment-key"}):
            client = OfficialOpenAI()
            with (
                patch.object(client.opener, "open") as call,
                self.assertRaises(FileNotFoundError),
            ):
                client.chat({"messages": []})
        call.assert_not_called()

    def test_opener_rejects_redirect_without_sending_a_second_request(self):
        requests = []

        class MemoryHTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(self, request):
                requests.append(request)
                headers = Message()
                headers["Location"] = "https://redirect.example/chat/completions"
                response = urllib.response.addinfourl(
                    io.BytesIO(b'{"choices": []}'),
                    headers,
                    request.full_url,
                    302 if len(requests) == 1 else 200,
                )
                response.msg = "Found" if len(requests) == 1 else "OK"
                return response

        (self.root / ".secrets/openai_api_key").write_text("dummy-test-key")
        self.write_config(base_url="https://relay.example/v1")
        with patch("urllib.request.HTTPSHandler", MemoryHTTPSHandler):
            client = OfficialOpenAI()
        with self.assertRaises(urllib.error.HTTPError) as raised:
            client.chat({"messages": []})
        self.assertEqual(raised.exception.code, 302)
        raised.exception.close()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].full_url, "https://relay.example/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
