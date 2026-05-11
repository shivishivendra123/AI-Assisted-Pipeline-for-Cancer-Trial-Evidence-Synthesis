"""
Vertex AI Generative Models backend (Gemini family).

Distinct from VertexPredictBackend — that one calls `:predict` on a *deployed*
model endpoint (e.g. medgemma deployed to a private endpoint). This one calls
the Gemini API on Vertex (`vertexai.generative_models.GenerativeModel`).

Auth: Application Default Credentials. Run once locally:
    gcloud auth application-default login

Or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from agents.backends.base import ChatBackend
from agents.utils.types import Message, GenParams, ChatResult


@dataclass
class VertexGenerativeBackend(ChatBackend):
    project_id: str
    location: str = "us-central1"
    model: str = "gemini-2.5-flash"
    provider: str = "vertex_generative"

    def __post_init__(self):
        # Lazy import — keeps the rest of the codebase working even if the
        # google-cloud-aiplatform SDK isn't fully available at import time.
        import vertexai
        vertexai.init(project=self.project_id, location=self.location)

    def chat(self, messages: List[Message], params: GenParams) -> ChatResult:
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_parts   = [m["content"] for m in messages if m["role"] != "system"]

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        model = (
            GenerativeModel(self.model, system_instruction=system_instruction)
            if system_instruction
            else GenerativeModel(self.model)
        )

        prompt = "\n\n".join(user_parts) if user_parts else ""

        gen_config_kwargs: dict[str, Any] = {
            "max_output_tokens": params.max_tokens,
            "temperature":       params.temperature,
        }
        if params.top_p is not None:
            gen_config_kwargs["top_p"] = params.top_p
        if params.top_k is not None:
            gen_config_kwargs["top_k"] = params.top_k
        if params.stop:
            gen_config_kwargs["stop_sequences"] = params.stop

        resp = model.generate_content(
            prompt, generation_config=GenerationConfig(**gen_config_kwargs)
        )

        # response.text raises if the candidate had no text part — guard it.
        try:
            text = resp.text or ""
        except Exception:
            text = ""

        try:
            raw = resp.to_dict()  # type: ignore[attr-defined]
        except Exception:
            raw = str(resp)

        finish = None
        try:
            finish = str(resp.candidates[0].finish_reason)
        except Exception:
            pass

        return ChatResult(text=text.strip(), raw=raw, finish_reason=finish)
