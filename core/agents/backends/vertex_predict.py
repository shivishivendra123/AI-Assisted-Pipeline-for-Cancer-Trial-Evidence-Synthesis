from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
from urllib.parse import urlparse
import json, socket, requests

from agents.backends.base import ChatBackend
from agents.utils.types import Message, GenParams, ChatResult
from agents.utils.auth import google_access_token


def _endpoint_resource(project_id: str, location: str, endpoint_id: str) -> str:
    return f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"


def _host_from_any(s: str) -> str:
    s = s.strip()
    if s.startswith("http://") or s.startswith("https://"):
        host = (urlparse(s).netloc or "").strip()
    else:
        host = s
    if ".endpoints.vertexai.goog" in host:
        host = host.replace(".endpoints.vertexai.goog", ".prediction.vertexai.goog")
    if not host or ".prediction.vertexai.goog" not in host:
        raise ValueError(f"Invalid DNS/URL: '{s}' (need *.prediction.vertexai.goog)")
    return host


def _predict_url(dns_or_url: str, project_id: str, location: str, endpoint_id: str) -> str:
    s = dns_or_url.strip()
    if s.startswith("http://") or s.startswith("https://"):
        parsed = urlparse(s)
        host = parsed.netloc
        if ".prediction.vertexai.goog" not in host or not s.endswith(":predict"):
            raise ValueError("URL must be a valid predict URL ending with ':predict'.")
        url = s
    else:
        host = _host_from_any(s)
        url = f"https://{host}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"

    # DNS preflight (clear error if wrong)
    socket.getaddrinfo(host, 443)
    return url


def _extract_text(pred0: Any) -> str:
    if isinstance(pred0, str):
        return pred0
    if isinstance(pred0, dict):
        cand = pred0.get("candidates")
        if isinstance(cand, list) and cand:
            first = cand[0]
            if isinstance(first, dict):
                msg = first.get("message") or {}
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str):
                        return c
                for k in ("content", "text", "generated_text", "output", "response"):
                    v = first.get(k)
                    if isinstance(v, str):
                        return v
        for k in ("content", "text", "generated_text", "output", "response"):
            v = pred0.get(k)
            if isinstance(v, str):
                return v
        ch = pred0.get("choices")
        if isinstance(ch, list) and ch:
            item = ch[0]
            if isinstance(item, dict):
                msg = item.get("message", {})
                if isinstance(msg, dict):
                    v = msg.get("content")
                    if isinstance(v, str):
                        return v
    return json.dumps(pred0, ensure_ascii=False)


@dataclass
class VertexPredictBackend(ChatBackend):
    """Vertex Predict backend using @requestFormat='chatCompletions'."""
    project_id: str
    location: str
    endpoint_id: str
    dedicated_dns_or_predict_url: str
    model: Optional[str] = None  # human-readable; not required by predict
    provider: str = field(default="vertex", init=False)
    timeout: int = 300
    default_stop: Optional[list[str]] = field(default_factory=lambda: ["<end_of_turn>"])

    def __post_init__(self):
        self._predict_url = _predict_url(
            self.dedicated_dns_or_predict_url, self.project_id, self.location, self.endpoint_id
        )
        self._endpoint_resource = _endpoint_resource(self.project_id, self.location, self.endpoint_id)

    def _headers(self):
        return {
            "Authorization": f"Bearer {google_access_token()}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Message], params: GenParams) -> ChatResult:
        inst = {
            "@requestFormat": "chatCompletions",
            "messages": messages,
            "max_tokens": int(params.max_tokens),
            "temperature": float(params.temperature),
        }
        if params.top_p is not None:
            inst["top_p"] = float(params.top_p)
        if params.top_k is not None:
            inst["top_k"] = int(params.top_k)
        # Prefer explicit params.stop, fallback to backend default_stop
        stops = params.stop or self.default_stop
        if stops:
            inst["stop"] = list(stops)

        # Backend-specific extras if any
        if params.extra:
            inst.update(params.extra)

        payload = {"instances": [inst]}
        r = requests.post(self._predict_url, headers=self._headers(),
                          data=json.dumps(payload), timeout=self.timeout)
        if r.status_code != 200:
            try:
                details = r.json()
            except Exception:
                details = r.text
            raise RuntimeError(f"Predict failed: {r.status_code} {details}")

        data = r.json()
        preds = data.get("predictions")
        text = _extract_text(preds[0]) if isinstance(preds, list) and preds else json.dumps(data)
        # Basic usage/finish info if present (not guaranteed)
        usage = None
        finish_reason = None
        try:
            usage = data.get("predictions", [{}])[0].get("usage")
            ch0 = data.get("predictions", [{}])[0].get("choices", [{}])[0]
            finish_reason = ch0.get("finish_reason")
        except Exception:
            pass

        return ChatResult(text=text.strip(), raw=data, finish_reason=finish_reason, usage=usage)

    # Optional streaming surface (not supported by this route)
    def chat_stream(self, messages: List[Message], params: GenParams):
        raise NotImplementedError("Vertex Predict streaming is not implemented in this backend.")
