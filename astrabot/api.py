"""Official OpenAI connection; never falls back to the company credential."""

import json
import os
import ssl
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .paths import ROOT


class ExplicitProxy(urllib.request.ProxyHandler):
    """Honor the configured proxy even when the shell has NO_PROXY entries."""

    def proxy_open(self, req, proxy, type):
        parsed = urlparse(proxy)
        if parsed.scheme != "http" or not parsed.hostname or parsed.username:
            raise ValueError("OPENAI_PROXY must be an HTTP proxy without embedded credentials")
        req.set_proxy(parsed.netloc, "http")
        return None


class OfficialOpenAI:
    def __init__(self):
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or parsed.hostname != "api.openai.com" or parsed.path != "/v1":
            raise ValueError("Official client requires https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-6-astra")
        self.key_file = Path(
            os.environ.get("OPENAI_API_KEY_FILE", str(ROOT / ".secrets/openai_api_key"))
        )
        self.proxy = os.environ.get("OPENAI_PROXY", "http://127.0.0.1:18888")
        system_ca = "/etc/ssl/certs/ca-certificates.crt"
        self.ca_bundle = os.environ.get(
            "OPENAI_CA_BUNDLE", system_ca if Path(system_ca).exists() else None
        )
        context = ssl.create_default_context(cafile=self.ca_bundle)
        self.opener = urllib.request.build_opener(
            ExplicitProxy({"https": self.proxy}) if self.proxy else urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=context),
        )

    def metadata(self):
        return {
            "provider": "openai_official",
            "model": self.model,
            "base_url": self.base_url,
            "proxy": self.proxy,
            "key_file": str(self.key_file),
            "ca_bundle": self.ca_bundle,
        }

    def chat(self, body, timeout=180):
        key = self.key_file.read_text().strip()
        if not key:
            raise ValueError("Official API key file is empty")
        payload = {**body, "model": self.model, "store": False}
        if self.model.startswith(("gpt-4.", "gpt-4o")):
            payload.pop("reasoning_effort", None)
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        with self.opener.open(req, timeout=timeout) as response:
            return json.load(response)
