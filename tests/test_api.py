import io
import json
import os
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from astrabot.api import ExplicitProxy, OfficialOpenAI


class OfficialConnection(unittest.TestCase):
    def test_company_environment_cannot_select_company_key_or_endpoint(self):
        with patch.dict(
            os.environ,
            {"ASTRA_API_BASE_URL": "http://company.invalid/v1", "ASTRA_MODEL": "old"},
            clear=True,
        ):
            client = OfficialOpenAI()
        self.assertEqual(client.base_url, "https://api.openai.com/v1")
        self.assertEqual(client.key_file.name, "openai_api_key")
        self.assertEqual(client.model, "gpt-6-astra")

    def test_explicit_proxy_preserves_https_tunnel_even_with_no_proxy(self):
        request = urllib.request.Request("https://api.openai.com/v1/models")
        with patch.dict(os.environ, {"NO_PROXY": "*", "no_proxy": "*"}):
            ExplicitProxy().proxy_open(request, "http://127.0.0.1:18888", "https")
        self.assertEqual(request.host, "127.0.0.1:18888")
        self.assertEqual(request._tunnel_host, "api.openai.com")
        self.assertEqual(request.type, "https")

    def test_chat_loads_selected_file_without_storing_key_in_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_key"
            path.write_text("dummy-test-key")
            with patch.dict(os.environ, {"OPENAI_API_KEY_FILE": str(path)}, clear=True):
                client = OfficialOpenAI()
            with patch.object(
                client.opener, "open", return_value=io.BytesIO(b'{"choices": []}')
            ) as call:
                self.assertEqual(client.chat({"messages": []}), {"choices": []})
            request = call.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer dummy-test-key")
            self.assertFalse(json.loads(request.data)["store"])
            self.assertNotIn("dummy-test-key", json.dumps(client.metadata()))


if __name__ == "__main__":
    unittest.main()
