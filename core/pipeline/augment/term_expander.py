from __future__ import annotations
from typing import List, Optional, Dict, Any
import os, re
import orjson
from json_repair import repair_json

from agents.factory import build_vertex_agent
from schemas.pico import PICO, AugmentedField, AugmentedPICO
from configs.env_config import config

# ---------- vertex + parsing helpers (mirror Day 1 style) ----------

def _agent():
    return build_vertex_agent(
        project_id=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION,
        endpoint_id=config.GCP_ENDPOINT_ID,
        dedicated_dns_or_predict_url=config.GCP_DEDICATED_DNS,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )

def _loads_or_repair(s: str) -> Dict[str, Any]:
    try:
        return orjson.loads(s)
    except Exception:
        fixed = repair_json(s)
        return orjson.loads(fixed)

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)
def _strip_fences(text: str) -> str:
    m = _CODE_FENCE_RE.match(text.strip())
    return m.group(1) if m else text

def _unwrap_vertex(resp: Any) -> str:
    """
    Works for:
    - dict with predictions.choices[0].message.content (your current shape)
    - dict with candidates[0].content.parts[*].text
    - string (already text or stringified JSON of the dict)
    """
    if isinstance(resp, str):
        t = resp.strip()
        if t.startswith("{") and ("\"predictions\"" in t or "'predictions'" in t):
            try:
                resp = _loads_or_repair(t)
            except Exception:
                return resp
        else:
            return resp

    if isinstance(resp, dict):
        preds = resp.get("predictions")
        if isinstance(preds, dict):
            ch = preds.get("choices") or []
            if ch and isinstance(ch[0], dict):
                msg = ch[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        cands = resp.get("candidates") or []
        if cands and isinstance(cands[0], dict):
            parts = ((cands[0].get("content") or {}).get("parts")) or []
            texts = [p.get("text","") for p in parts if isinstance(p, dict) and p.get("text")]
            merged = "\n".join([t for t in texts if t.strip()])
            if merged.strip():
                return merged

    return str(resp)

def _parse_json_text(text: str) -> Dict[str, Any]:
    inner = _strip_fences(text).strip()
    return _loads_or_repair(inner)

# ---------- normalization helpers (tiny + predictable) ----------

_WS_RE = re.compile(r"\s+")
def _norm(t: str) -> str:
    return _WS_RE.sub(" ", t.strip().lower())

def _local_variants(term: str) -> List[str]:
    """Generate a few deterministic local variants (hyphen↔space, simple plural)."""
    t = term.strip()
    out = set()
    # hyphen <-> space
    out.add(t.replace("-", " "))
    out.add(t.replace(" ", "-"))
    # naive plural if one token and endswith consonant
    parts = t.split()
    if len(parts) == 1 and len(t) >= 3 and not t.endswith(("s","x","z","ch","sh")):
        out.add(t + "s")
    return [v for v in out if v and v != t]

def _dedupe_keep(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        n = _norm(s)
        if not n or n in seen: 
            continue
        seen.add(n); out.append(n)
    return out

def _clean_synonyms(value: str, syns: List[str], cap: int = 10) -> List[str]:
    """
    Normalize, add a few local variants, limit length and words.
    """
    base = _norm(value)
    extra = []
    for s in list(syns):
        extra.extend(_local_variants(s))
    all_candidates = syns + extra

    cleaned = []
    for s in all_candidates:
        n = _norm(s)
        if not n or n == base:
            continue
        # keep short phrases only
        if len(n.split()) > 4 or len(n) > 40:
            continue
        cleaned.append(n)

    return _dedupe_keep(cleaned)[:cap]

# ---------- main expansion ----------

def expand_terms_with_llm(pico: PICO, prompt_template: str) -> AugmentedPICO:
    agent = _agent()
    agent.set_system("You expand clinical phrases into synonym lists. You ONLY return valid JSON. No thoughts. No explanations. No reasoning. Your response MUST start with { and end with }.")

    comp_json = f"\"{pico.Comparator}\"" if pico.Comparator else "null"
    outs_json = orjson.dumps(pico.Outcomes).decode()

    prompt = (prompt_template
        .replace("{{POP}}", pico.Population)
        .replace("{{INT}}", pico.Intervention)
        .replace("{{COMP_JSON}}", comp_json)
        .replace("{{OUTS_JSON}}", outs_json)
    )

    resp = agent.say(prompt)
    content = _unwrap_vertex(resp)
    data = _parse_json_text(content)

    def mk(field_obj: Optional[Dict[str, Any]]) -> Optional[AugmentedField]:
        if field_obj is None: return None
        value = str(field_obj.get("value", "")).strip()
        syns  = field_obj.get("synonyms") or []
        syns  = [s for s in syns if isinstance(s, str)]
        cleaned = _clean_synonyms(value, syns)
        return AugmentedField(value=_norm(value), synonyms=cleaned)

    pop = mk(data.get("Population"))
    itv = mk(data.get("Intervention"))
    comp = mk(data.get("Comparator")) if data.get("Comparator") is not None else None

    outs = []
    for o in data.get("Outcomes", []):
        if not isinstance(o, dict): 
            # tolerate outputs like ["mortality", ...]
            value = str(o)
            outs.append(AugmentedField(value=_norm(value), synonyms=_clean_synonyms(value, [])))
            continue
        item = mk(o)
        if item: outs.append(item)

    return AugmentedPICO(Population=pop, Intervention=itv, Comparator=comp, Outcomes=outs)
