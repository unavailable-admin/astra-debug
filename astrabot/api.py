"""OpenAI-compatible connection using an explicitly selected endpoint and key file."""

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


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    """Keep the API credential on the configured endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OfficialOpenAI:
    """Responses and legacy Chat Completions client for the configured service."""

    def __init__(self):
        config_file = ROOT / ".secrets/openai_config.json"
        config = json.loads(config_file.read_text()) if config_file.exists() else {}
        if not isinstance(config, dict):
            raise TypeError("OpenAI config must be a JSON object")
        self.base_url = os.environ.get("OPENAI_BASE_URL", config.get("base_url", "https://api.openai.com/v1"))
        if not isinstance(self.base_url, str):
            raise TypeError("OPENAI_BASE_URL must be a valid HTTPS URL")
        self.base_url = self.base_url.rstrip("/")
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "?" in self.base_url
            or "#" in self.base_url
            or any(character.isspace() or ord(character) < 32 for character in self.base_url)
        ):
            raise ValueError("OPENAI_BASE_URL must be HTTPS without credentials, query or fragment")
        # Accessing port also validates malformed and out-of-range values.
        _ = parsed.port
        self.model = os.environ.get("OPENAI_MODEL", config.get("model", "gpt-6-astra"))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("OPENAI_MODEL must be a nonempty string")
        self.key_file = Path(os.environ.get("OPENAI_API_KEY_FILE", str(ROOT / ".secrets/openai_api_key")))
        self.proxy = os.environ.get("OPENAI_PROXY", "")
        system_ca = "/etc/ssl/certs/ca-certificates.crt"
        self.ca_bundle = os.environ.get("OPENAI_CA_BUNDLE", system_ca if Path(system_ca).exists() else None)
        context = ssl.create_default_context(cafile=self.ca_bundle)
        self.opener = urllib.request.build_opener(
            (ExplicitProxy({"https": self.proxy}) if self.proxy else urllib.request.ProxyHandler({})),
            urllib.request.HTTPSHandler(context=context),
            RejectRedirect(),
        )

    def metadata(self):
        return {
            "provider": ("openai_official" if self.base_url == "https://api.openai.com/v1" else "openai_compatible"),
            "model": self.model,
            "base_url": self.base_url,
            "proxy": self.proxy,
            "key_file": str(self.key_file),
            "ca_bundle": self.ca_bundle,
            "vision_wire_api": "responses",
        }

    def chat(self, body, timeout=180):
        """Send a legacy Chat Completions request; vision uses responses instead."""
        payload = dict(body)
        if self.model.startswith(("gpt-4.", "gpt-4o")):
            payload.pop("reasoning_effort", None)
        with self._post("chat/completions", payload, timeout) as response:
            return json.load(response)

    def responses(self, body: dict, timeout: float = 180) -> dict:
        """Read a complete Responses result, rejecting interrupted SSE streams."""
        payload = {**body, "stream": True}
        if self.model.startswith(("gpt-4.", "gpt-4o")):
            payload.pop("reasoning", None)
        with self._post("responses", payload, timeout) as response:
            if response.headers.get_content_type() == "application/json":
                return json.load(response)
            data = []
            for raw in response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                elif not line and data:
                    encoded = "\n".join(data)
                    data = []
                    if encoded == "[DONE]":
                        break
                    event = json.loads(encoded)
                    kind = event.get("type")
                    if kind == "response.completed":
                        return event["response"]
                    if kind in ("error", "response.failed", "response.incomplete"):
                        raise ValueError(f"Responses API did not complete: {kind}")
        raise ValueError("Responses stream ended without response.completed")

    def _post(self, endpoint: str, body: dict, timeout: float):
        key = self.key_file.read_text().strip()
        if not key:
            raise ValueError("OpenAI API key file is empty")
        payload = {**body, "model": self.model, "store": False}
        req = urllib.request.Request(
            self.base_url + "/" + endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if body.get("stream") else "application/json",
            },
        )
        return self.opener.open(req, timeout=timeout)


def response_text(response: dict) -> str:
    """Extract only completed assistant text; partial or refused output is unusable."""
    if response.get("status") != "completed" or response.get("error"):
        raise ValueError("Incomplete model response")
    text = []
    for item in response.get("output", []):
        if item.get("type") == "reasoning":
            continue
        if item.get("type") != "message" or item.get("role") != "assistant" or item.get("status") != "completed":
            raise ValueError("Unexpected or incomplete model output")
        for part in item.get("content", []):
            if part.get("type") != "output_text":
                raise ValueError("Model response refused or contains non-text output")
            text.append(part["text"])
    if not text or not "".join(text).strip():
        raise ValueError("Empty model response")
    return "".join(text)
