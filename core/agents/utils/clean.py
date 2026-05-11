from __future__ import annotations
import re

_END_MARKERS = [r"<\s*end_of_turn\s*>", r"<\|end_of_turn\|>", r"<\|eot_id\|>"]
_END_RX = re.compile("|".join(_END_MARKERS), flags=re.IGNORECASE)
_END_TAIL_RX = re.compile(r"(?:\s*(?:%s)\s*)+$" % "|".join(_END_MARKERS), re.IGNORECASE)

def clean_reply(text: str) -> str:
    if not isinstance(text, str):
        return text
    m = _END_RX.search(text)
    if m:
        text = text[:m.start()]
    text = _END_TAIL_RX.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
