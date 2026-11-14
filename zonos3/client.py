"""Minimal HTTP client for the ZONOS3 server API (servers/api/openapi.yaml in Zyphra/ZONOS3-Mini).

Standard library only, so the node pack adds no Python dependencies to ComfyUI.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np


class Zonos3Error(RuntimeError):
    pass


class Zonos3Client:
    def __init__(self, url: str, token: str = "", timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.token = token.strip()
        self.timeout = timeout

    # ------------------------------------------------------------------ plumbing
    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None):
        data = None
        headers = {"Accept": "*/*"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.read(), dict(resp.headers.items())
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                detail = json.loads(raw).get("detail", raw.decode("utf-8", "replace"))
            except Exception:
                detail = raw.decode("utf-8", "replace")
            if isinstance(detail, list):  # 422: [{"loc", "msg", "type"}]
                detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
            raise Zonos3Error(f"ZONOS3 server {e.code} on {path}: {detail}") from None
        except urllib.error.URLError as e:
            raise Zonos3Error(
                f"Can't reach the ZONOS3 server at {self.url} ({e.reason}). "
                "Start it, or give the ZONOS3 Server node its folder so it starts itself."
            ) from None

    def _json(self, method: str, path: str, body: dict | None = None, timeout: float | None = None):
        raw, _ = self._request(method, path, body, timeout)
        return json.loads(raw)

    # ------------------------------------------------------------------ routes
    def healthy(self, timeout: float = 2.0) -> bool:
        try:
            return self._json("GET", "/v2/health", timeout=timeout).get("status") == "ok"
        except Exception:
            return False

    def capabilities(self) -> dict:
        return self._json("GET", "/v2/capabilities", timeout=30)

    def speakers(self) -> list[dict]:
        return self._json("GET", "/v2/speakers", timeout=30)["speakers"]

    def generate(self, request: dict) -> tuple[np.ndarray, dict]:
        """Buffered native generation. Returns mono float32 samples at 44.1 kHz and the response headers."""
        body = dict(request)
        body["stream"] = False
        body["format"] = "pcm"
        raw, headers = self._request("POST", "/v2/generate", body)
        h = {k.lower(): v for k, v in headers.items()}
        if h.get("x-audio-format", "float32") != "float32":
            raise Zonos3Error(f"Unexpected audio format {h.get('x-audio-format')}")
        audio = np.frombuffer(raw, dtype="<f4").copy()
        return audio, h
