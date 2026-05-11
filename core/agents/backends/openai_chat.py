## CODE IN THIS FILE IS NOT TESTED AND MIGHT BE INCOMPLETE

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from openai import OpenAI

from agents.backends.base import ChatBackend
from agents.utils.types import Message, GenParams, ChatResult, StreamChunk


@dataclass
class OpenAIChatBackend(ChatBackend):
    """OpenAI-style backend (e.g., GPT-5 later)."""
    api_key: str
    model: str = "gpt-5"
    base_url: str = "https://api.openai.com/v1"
    provider: str = "openai"

    def chat(self, messages: List[Message], params: GenParams) -> ChatResult:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            **({"top_p": params.top_p} if params.top_p is not None else {}),
            **({"stop": params.stop} if params.stop else {}),
        )
        txt = (resp.choices[0].message.content or "").strip()
        return ChatResult(text=txt, raw=resp.to_dict(), finish_reason=getattr(resp.choices[0], "finish_reason", None))

    # Optional streaming — supported here for future convenience
    def chat_stream(self, messages: List[Message], params: GenParams):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            stream=True,
            **({"top_p": params.top_p} if params.top_p is not None else {}),
            **({"stop": params.stop} if params.stop else {}),
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield StreamChunk(text=delta.content, raw=chunk.to_dict())
