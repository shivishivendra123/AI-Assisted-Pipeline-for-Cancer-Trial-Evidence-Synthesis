from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict, Literal


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenParams:
    """
    Provider-agnostic generation parameters.
    Backends may honor a subset and can read extra backend-specific knobs from `extra`.
    """
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResult:
    """
    Normalized chat result.
    `raw` carries the provider's full response for debugging/metrics if needed.
    """
    text: str
    raw: Any
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


@dataclass
class StreamChunk:
    """
    Optional streaming chunk shape. Only used if a backend supports streaming.
    """
    text: str
    raw: Any
