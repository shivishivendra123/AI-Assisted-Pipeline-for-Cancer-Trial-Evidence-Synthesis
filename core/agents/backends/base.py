from __future__ import annotations

from typing import Iterable, List, Optional, Protocol, runtime_checkable
from agents.utils.types import Message, GenParams, ChatResult, StreamChunk


@runtime_checkable
class ChatBackend(Protocol):
    """
    Provider-agnostic backend contract.
    Implementations (Vertex, OpenAI, etc.) plug in behind this interface.
    """

    provider: str               # e.g., "vertex", "openai"
    model: Optional[str]        # human-readable model identifier (optional)

    def chat(self, messages: List[Message], params: GenParams) -> ChatResult:
        """
        Single non-streaming completion.
        Must return a normalized ChatResult.
        """
        ...

    def chat_stream(self, messages: List[Message], params: GenParams) -> Iterable[StreamChunk]:
        """
        Optional streaming interface. If unsupported, backends may raise NotImplementedError.
        """
        ...
