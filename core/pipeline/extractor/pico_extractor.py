from typing import Tuple, Dict, Any, Optional
import os
import re
import orjson
from json_repair import repair_json
from pydantic import ValidationError

# Pipeline LLM is Gemini 2.5 Flash on Vertex AI.
# This _agent() factory is shared across PICO extraction, screening,
# eligibility building, study-characteristics extraction, outcome extraction,
# and evidence synthesis — every downstream stage that needs an LLM imports
# _agent from this module. Swapping the backend here propagates everywhere.
from agents.factory import build_vertex_gemini_agent
from schemas.pico import PICO, QuoteBacks
from configs.env_config import config


def _agent():
    """Build the shared pipeline LLM agent (Gemini 2.5 Flash on Vertex AI).

    Reads model + region from .env (GEMINI_MODEL, GEMINI_LOCATION) so the
    same code can run against gemini-2.5-flash, gemini-2.5-pro, etc., by
    flipping a single env var.

    Returns:
        Agent: A configured agent ready for `agent.say(prompt)` calls.
    """
    return build_vertex_gemini_agent(
        project_id=config.GCP_PROJECT_ID,
        location=config.GEMINI_LOCATION,
        model=config.GEMINI_MODEL,
        temperature=config.LLM_TEMPERATURE,
        # max_tokens=12000 to accommodate the synthesis stage, which produces
        # multi-paragraph narrative_markdown over many studies. Smaller stages
        # (PICO, screening, single-study extraction) only use 200-1500 tokens
        # and aren't billed for unused budget. Empirical truncations were
        # observed at 4000 when summarising 10+ studies.
        max_tokens=max(config.LLM_MAX_TOKENS, 12000),
    )


# ---------- parsing helpers ----------

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)

def _strip_code_fences(s: str) -> str:
    m = _CODE_FENCE_RE.match(s.strip())
    return m.group(1) if m else s

def _loads_or_repair(text: str) -> Dict:
    """
    Parse JSON str; if it fails, repair and parse again.
    """
    s = text.strip()
    try:
        return orjson.loads(s)
    except Exception:
        fixed = repair_json(s)
        return orjson.loads(fixed)

def _extract_vertex_content(obj: Any) -> Optional[str]:
    """
    Accepts either:
      - stringified JSON of Vertex response (your case),
      - dict Vertex response,
      - or already-plain text.
    Returns the assistant text (could be fenced).
    """
    # If it's already plain text, return it
    if isinstance(obj, str):
        # If it looks like a JSON object for Vertex, try to decode then drill in
        trimmed = obj.strip()
        if trimmed.startswith("{") and ("\"predictions\"" in trimmed or "'predictions'" in trimmed):
            try:
                obj = _loads_or_repair(trimmed)
            except Exception:
                # It's plain text, just return it
                return obj
        else:
            return obj

    if isinstance(obj, dict):
        # Your response shape
        preds = obj.get("predictions")
        if isinstance(preds, dict):
            choices = preds.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content

        # Alternate Vertex shape (candidates / parts)
        cands = obj.get("candidates") or []
        if cands and isinstance(cands[0], dict):
            parts = ((cands[0].get("content") or {}).get("parts")) or []
            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
            merged = "\n".join([t for t in texts if t.strip()])
            if merged.strip():
                return merged

    # Fallback: nothing found
    return None


def _parse_pico_from_text(text: str) -> Dict:
    """
    Strip code fences, then parse/repair into a dict.
    Handles LLM responses that include reasoning before JSON.
    """
    # First try to strip code fences
    inner = _strip_code_fences(text).strip()

    # If no code fences found and text contains reasoning, extract just the JSON
    if inner == text.strip() and ('{' in text and '}' in text):
        # Find the last JSON object in the text (in case there's reasoning before it)
        # Look for patterns like ```json{...}``` or just {...}
        import re

        # Try to find JSON in code fence first
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, re.IGNORECASE)
        if json_match:
            inner = json_match.group(1).strip()
        else:
            # Look for the last complete JSON object
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                inner = json_match.group(0).strip()

    parsed = _loads_or_repair(inner)

    # Handle case where LLM returns a list instead of dict
    if isinstance(parsed, list):
        if len(parsed) > 0 and isinstance(parsed[0], dict):
            # Take the first element if it's a list of dicts
            return parsed[0]
        else:
            # Return empty dict if list is empty or doesn't contain dicts
            return {}

    return parsed


# ---------- main entry ----------

def extract_pico(question: str, prompt_template: str) -> Tuple[PICO, QuoteBacks, Dict]:
    """
    Single LLM call -> unwrap Vertex -> strip fences -> parse/repair -> validate.
    Returns (pico_valid, quotes, parsed_pico_dict).
    """
    agent = _agent()
    agent.set_system("You extract PICO from questions. You ONLY return valid JSON. No thoughts. No explanations. No reasoning. Your response MUST start with { and end with }. ABSOLUTELY NO THINKING STEPS.")

    prompt = prompt_template.replace("{{QUESTION}}", question)
    resp = agent.say(prompt)  # may be a raw dict OR a JSON string of the dict

    # 1) Get the assistant text out of the Vertex envelope
    content = _extract_vertex_content(resp)
    if not isinstance(content, str) or not content.strip():
        # Last resort: treat the whole thing as text and try to parse
        content = str(resp)
    
    print("---- PICO ---- ")
    print(content)
    print("----/PICO------")
    # 2) Now parse the actual PICO JSON that was inside the content
    parsed = _parse_pico_from_text(content)

    # 3) Validate/coerce to clean PICO model
    try:
        pico = PICO.model_validate({
            "Population":   parsed.get("Population"),
            "Intervention": parsed.get("Intervention"),
            "Comparator":   parsed.get("Comparator"),
            "Outcomes":     parsed.get("Outcomes", []),
        })
        
    except ValidationError:
        pop = str(parsed.get("Population", "")).strip() or "unknown"
        itv = str(parsed.get("Intervention", "")).strip() or "unknown"
        comp_raw = parsed.get("Comparator")
        comp = None if comp_raw in [None, ""] else (str(comp_raw).strip() or None)
        outs = parsed.get("Outcomes") or []
        outs = [o for o in outs if isinstance(o, str) and o.strip()]
        pico = PICO(Population=pop, Intervention=itv, Comparator=comp, Outcomes=outs)

    quotes = QuoteBacks.model_validate(parsed.get("_quote_backs", {}))

    # Return the parsed inner JSON (not the outer Vertex payload)
    return pico, quotes, parsed
