from __future__ import annotations

from typing import Callable, List, Optional
from agents.backends.base import ChatBackend
from agents.utils.types import Message, GenParams, ChatResult


class Agent:
    """
    Provider-agnostic chat agent:
      - Holds system prompt + history
      - Delegates generation to a pluggable ChatBackend
      - Applies an optional cleaner (provider-specific quirks stay out of here)
    """

    def __init__(
        self,
        backend: ChatBackend,
        *,
        default_params: Optional[GenParams] = None,
        cleaner: Optional[Callable[[str], str]] = None,
    ):
        self.backend = backend
        self.default_params = default_params or GenParams()
        self.cleaner = cleaner
        self.system: Optional[str] = None
        self.history: List[Message] = []

    # ---------- utilities ----------

    def set_system(self, text: str) -> None:
        self.system = text

    def reset(self) -> None:
        self.history.clear()
        self.system = None

    def _with_system(self, messages: List[Message]) -> List[Message]:
        return ([{"role": "system", "content": self.system}] if self.system else []) + messages

    def _apply_cleaner(self, text: str) -> str:
        return self.cleaner(text) if self.cleaner else text

    def _merge_params(self, **overrides) -> GenParams:
        merged = GenParams(**self.default_params.__dict__)  # shallow copy
        for k, v in overrides.items():
            if hasattr(merged, k) and v is not None:
                setattr(merged, k, v)
        return merged

    # ---------- main API ----------

    def complete(self, prompt: str, **overrides) -> str:
        params = self._merge_params(**overrides)
        msgs = self._with_system([{"role": "user", "content": prompt}])
        result: ChatResult = self.backend.chat(msgs, params)
        return self._apply_cleaner(result.text)

    def say(self, user_text: str, **overrides) -> str:
        params = self._merge_params(**overrides)
        msgs = self._with_system(self.history + [{"role": "user", "content": user_text}])
        result: ChatResult = self.backend.chat(msgs, params)
        reply = self._apply_cleaner(result.text)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        return reply

    # Optional: expose streaming if backend supports it.
    def say_stream(self, user_text: str, **overrides):
        """
        Yields text chunks if the backend implements chat_stream; otherwise raises NotImplementedError.
        Note: no history mutation until the stream finishes (caller decides how to assemble).
        """
        if not hasattr(self.backend, "chat_stream"):
            raise NotImplementedError("This backend does not support streaming.")

        params = self._merge_params(**overrides)
        msgs = self._with_system(self.history + [{"role": "user", "content": user_text}])

        for chunk in self.backend.chat_stream(msgs, params):
            yield self._apply_cleaner(chunk.text)
