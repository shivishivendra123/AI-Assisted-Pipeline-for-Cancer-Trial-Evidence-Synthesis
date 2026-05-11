from pipeline.extractor.pico_extractor import _agent,_extract_vertex_content
import json
from pathlib import Path
import ast
import re
from datetime import datetime


def _read(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def xml_to_plain(xml_str: str) -> str:
    # Remove tags like <...>
    no_tags = re.sub(r"<[^>]+>", " ", xml_str)
    # Collapse whitespace
    return re.sub(r"\s+", " ", no_tags).strip()

import re

def extract_json_block(text: str) -> str:
    """
    Extract what looks like a JSON object ({...}) from an LLM response.
    Handles optional ```json ... ``` fences and surrounding text.
    """
    # Remove code fences first if present
    cleaned = text.strip()

    # Pattern 1: ```json\n{...}\n```
    if cleaned.startswith("```"):
        # Find the end of first fence marker
        fence_end = cleaned.find("\n", 3)
        if fence_end == -1:
            # No newline, might be ```json{...}``` format
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]  # Remove ```json
            elif cleaned.startswith("```JSON"):
                cleaned = cleaned[7:]  # Remove ```JSON
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]  # Remove ```
        else:
            # Has newline after fence
            cleaned = cleaned[fence_end+1:]

        # Remove trailing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

    # Fallback: just grab from first '{' to last '}' in the whole response
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON-like object found in LLM response")

    block = cleaned[start : end + 1].strip()
    return block



import csv
from pathlib import Path

def save_screening_result_to_csv(result: dict[str, any], csv_path: Path) -> None:
    """
    Append one screening/extraction result (JSON dict) as a row into a CSV file.
    Creates the file and writes the header if it does not exist yet.
    """
    # Check if file already exists
    file_exists = csv_path.exists()

    # Open in append mode
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))

        # If file is new (or empty), write header first
        if not file_exists or csv_path.stat().st_size == 0:
            writer.writeheader()

        # Then append the row
        writer.writerow(result)

    print(f"Saved row to {csv_path}")


import csv
import json
from pathlib import Path
from typing import Dict, Any, List

def parse_screening_result(content: str) -> dict:
    """
    Parse the LLM screening output into a Python dict.
    Tries JSON first; if that fails, falls back to Python-literal parsing.
    """
    block = extract_json_block(content)

    try:
        # Preferred: strict JSON
        return json.loads(block)
    except json.JSONDecodeError as e:
        print("[WARN] json.loads failed, trying ast.literal_eval")
        print("[WARN] JSONDecodeError:", e)
        # Fallback for Python-dict style outputs: {'key': 'value'}
        data = ast.literal_eval(block)
        # `data` is now a normal Python dict/list structure
        return data
    
def join_list(lst: List[Any]) -> str:
    """Join list items into a single string with ' | ' separator."""
    if not lst:
        return ""
    return " | ".join(str(x) for x in lst)


def screening_json_to_row(result: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten one screening JSON object (for a single study)
    into a flat dict suitable for writing as a CSV row.
    """
    cm = result.get("criteria_match", {})

    pop = cm.get("population", {}) or {}
    itv = cm.get("intervention", {}) or {}
    comp = cm.get("comparator", {}) or {}
    out = cm.get("outcomes", {}) or {}
    sd = cm.get("study_design", {}) or {}
    other = cm.get("other_criteria", {}) or {}

    # Flatten reasons_for_exclusion
    reasons = result.get("reasons_for_exclusion", []) or []
    reasons_domains = [r.get("domain", "") for r in reasons]
    reasons_codes = [r.get("reason_code", "") for r in reasons]
    reasons_texts = [r.get("reason_text", "") for r in reasons]

    row = {
    # Basic
    "study_id": result.get("study_id"),
    "decision": result.get("decision"),
    "cumulative_score": result.get("cumulative_score", 0),
    "overall_notes": result.get("overall_notes", ""),

    # Reasons
    "reasons_domains": join_list(reasons_domains),
    "reasons_codes": join_list(reasons_codes),
    "reasons_texts": join_list(reasons_texts),

    # Population
    "population_matched_must_include": join_list(
        pop.get("matched_must_include", [])
    ),
    "population_missing_must_include": join_list(
        pop.get("missing_must_include", [])
    ),
    "population_exclude_flags_triggered": join_list(
        pop.get("exclude_flags_triggered", [])
    ),

    # Intervention
    "intervention_matched_must_include_any_of": join_list(
        itv.get("matched_must_include_any_of", [])
    ),
    "intervention_exclude_flags_triggered": join_list(
        itv.get("exclude_flags_triggered", [])
    ),

    # Comparator
    "comparator_required": comp.get("comparator_required"),
    "comparator_accept_single_arm": comp.get("accept_single_arm"),
    "comparator_matched_must_include_any_of": join_list(
        comp.get("matched_must_include_any_of", [])
    ),
    "comparator_missing": comp.get("comparator_missing"),
    "comparator_notes": comp.get("notes", ""),

    # Outcomes
    "outcomes_matched_required_any_of": join_list(
        out.get("matched_required_any_of", [])
    ),
    "outcomes_exclude_if_only_triggered": out.get(
        "exclude_if_only_triggered"
    ),
    "outcomes_notes": out.get("notes", ""),

    # Study design
    "study_design_classified_as": sd.get("design_classified_as"),
    "study_design_matches_include_list": sd.get("matches_include_list"),
    "study_design_matches_exclude_list": sd.get("matches_exclude_list"),

    # Other criteria
    "other_estimated_sample_size": other.get("estimated_sample_size"),
    "other_sample_size_ok": other.get("sample_size_ok"),
    "other_year_of_publication": other.get("year_of_publication"),
    "other_within_year_range": other.get("within_year_range"),
    "other_language": other.get("language"),
    "other_language_allowed": other.get("language_allowed"),
}

    return row


def extract_study_char_outcomes(art, ftext, pico_json, rel_studies):


    PICO = pico_json["pico_valid"]
    P, I, C, O = (
        PICO["Population"],
        PICO["Intervention"],
        PICO["Comparator"],
        PICO["Outcomes"],
    )

    # Check if there are studies to process
    if not rel_studies:
        print("[WARN] No relevant studies to extract characteristics from")
        return None

    # Deduplicate within this run so each pm_id is screened only once
    seen_pmids = set()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(f"artifacts_day6/study_char_{ts}.csv")

    # Ensure directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for rel in rel_studies:
        article = art[rel]
        agent = _agent()
        agent.set_system("You are a systematic review data extraction expert. You ONLY return valid JSON. No thoughts. No explanations. No reasoning. Your response MUST start with { and end with }.")
        prompt_template = "prompts/study_characteristics.txt"


        # if max_studies <= 0:
        #     break

        # # max_studies -= 1

        title = article["title"]
        abstract = article["abstract"]

        # ----- FULL TEXT HANDLING (fix pmc_xml bug) -----
        record = ftext.get(rel)  # might be None
        full_text_plain = "null"   # default if nothing available
        pmc_xml = None             # reset every iteration

        if record:
            pmc_xml = record.get("pmc_full_xml")
        if pmc_xml:
            if(True):
                full_text_plain = xml_to_plain(pmc_xml)

        # ----- BUILD PROMPT -----
        prompt = _read(prompt_template)

        replacements = {
            "{{DOC_TITLE}}": title,
            "{{DOC_ABSTRACT_OR_NULL}}": abstract or "null",
            "{{DOC_FULL_TEXT_OR_NULL}}": full_text_plain,
            "{{STUDY_ID}}": rel,
            "{{QUESTION_ID}}": pico_json["qid"],
            "{{QUESTION_TEXT}}": pico_json["question"],
            "{{P}}": P,
            "{{I}}": I,
            "{{C}}": C,
            "{{O}}": O,
        }
        # print(replacements)

        for placeholder, value in replacements.items():
            # Turn dicts/lists into JSON, everything else into str
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif value is None:
                value = "null"
            else:
                value = str(value)

            prompt = prompt.replace(placeholder, value)

        # print("[INFO] ================")
        # print(prompt)
        # print("[INFO] ================")

        # ----- CALL LLM -----
        resp = agent.say(prompt)
        content = _extract_vertex_content(resp)
        if not isinstance(content, str) or not content.strip():
            # Last resort: treat the whole thing as text and try to parse
            content = str(resp)

        # ----- PARSE AND SAVE RESULT -----
        print(content)
        
        try:
            screening_result = parse_screening_result(content)
            save_screening_result_to_csv(screening_result, csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to parse/save screening result for {rel}: {e}")
            # Optional: debug snippet
            # print(content[:500])
            continue

    # Verify CSV was created and has content
    if csv_path.exists():
        # Convert Path to string for compatibility with Gradio FileData
        return str(csv_path)
    else:
        print(f"[WARN] CSV file was not created at {csv_path}")
        return None

    