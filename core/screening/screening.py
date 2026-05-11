from pipeline.extractor.pico_extractor import _agent,_extract_vertex_content
import json
from pathlib import Path
import ast
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from configs.env_config import config

def _read(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()




def xml_to_plain(xml_str: str) -> str:
    # Remove tags like <...>
    no_tags = re.sub(r"<[^>]+>", " ", xml_str)
    # Collapse whitespace
    return re.sub(r"\s+", " ", no_tags).strip()


# def extract_json_block(text: str) -> str:
#     """
#     Take an LLM response and return the first {...} JSON block as a string.
#     Handles ```json ... ``` and plain text around it.
#     """
#     # Strip code fences if present
#     if "```" in text:
#         # Keep everything inside the first fenced block
#         parts = text.split("```")
#         # parts might be like: ["", "json\n{...}", ""]
#         for part in parts:
#             part = part.strip()
#             if part.startswith("json"):
#                 # remove leading 'json' line
#                 part = part[len("json"):].strip()
#                 text = part
#                 break

#     # Now text SHOULD just be JSON or at least contain a { ... } block
#     start = text.find("{")
#     end = text.rfind("}")
#     if start == -1 or end == -1:
#         raise ValueError("No JSON object found in LLM response")

#     return text[start : end + 1]

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



def save_screening_result_to_csv(result: dict[str, any], csv_path: Path, lock: Lock = None) -> None:
    """
    Append one screening result (JSON dict) as a row into a CSV file.
    Creates the file and writes the header if it does not exist yet.
    Thread-safe when lock is provided.
    """
    row = screening_json_to_row(result)

    # Ensure parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Use lock for thread-safe file access
    if lock:
        lock.acquire()

    try:
        file_exists = csv_path.exists()

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    finally:
        if lock:
            lock.release()

import csv
import json
from pathlib import Path
from typing import Dict, Any, List

def parse_screening_result(content: str) -> dict:
    """
    Parse the LLM screening output into a Python dict.
    Uses 4-level fallback strategy for maximum reliability:
    1. Strict JSON parsing
    2. JSON repair for malformed JSON
    3. Python literal evaluation
    4. Valid fallback structure
    """
    try:
        block = extract_json_block(content)
    except ValueError as e:
        print(f"[ERROR] Failed to extract JSON block: {e}")
        # Return minimal valid screening result
        return {
            "study_id": "",
            "decision": "error",
            "cumulative_score": 0,
            "overall_notes": "Failed to extract JSON from LLM response",
            "reasons_for_exclusion": [{
                "domain": "error",
                "reason_code": "json_extraction_failed",
                "reason_text": str(e)
            }]
        }

    # Strategy 1: Try strict JSON parsing
    try:
        return json.loads(block)
    except json.JSONDecodeError as json_error:
        print(f"[WARN] Strategy 1 (json.loads) failed: {json_error}")

        # Strategy 2: Try json-repair for malformed JSON
        try:
            from json_repair import repair_json
            print("[INFO] Attempting JSON repair...")
            repaired = repair_json(block)
            result = json.loads(repaired)
            print("[SUCCESS] JSON repair successful!")
            return result
        except Exception as repair_error:
            print(f"[WARN] Strategy 2 (json-repair) failed: {repair_error}")

            # Strategy 3: Try Python literal evaluation
            try:
                print("[INFO] Attempting ast.literal_eval...")
                result = ast.literal_eval(block)
                print("[SUCCESS] ast.literal_eval successful!")
                return result
            except Exception as literal_error:
                print(f"[WARN] Strategy 3 (ast.literal_eval) failed: {literal_error}")

                # Strategy 4: Return valid fallback structure
                print("[WARN] All parsing strategies failed, using fallback structure")
                return {
                    "study_id": "",
                    "decision": "error",
                    "cumulative_score": 0,
                    "overall_notes": f"LLM response could not be parsed. Original error: {json_error}",
                    "reasons_for_exclusion": [{
                        "domain": "error",
                        "reason_code": "json_parsing_failed",
                        "reason_text": f"Failed to parse LLM response with all strategies. Last error: {literal_error}"
                    }],
                    "criteria_match": {}
                }
    
def join_list(lst: List[Any]) -> str:
    """Join list items into a single string with ' | ' separator."""
    if not lst:
        return ""
    return " | ".join(str(x) for x in lst)


def screening_json_to_row(result: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten one screening JSON object (for a single study)
    into a flat dict suitable for writing as a CSV row.
    Handles missing fields gracefully with defaults.
    """
    # Ensure result has basic structure
    if not isinstance(result, dict):
        result = {}

    cm = result.get("criteria_match", {})
    if not isinstance(cm, dict):
        cm = {}

    pop = cm.get("population", {}) or {}
    itv = cm.get("intervention", {}) or {}
    comp = cm.get("comparator", {}) or {}
    out = cm.get("outcomes", {}) or {}
    sd = cm.get("study_design", {}) or {}
    other = cm.get("other_criteria", {}) or {}

    # Flatten reasons_for_exclusion
    reasons = result.get("reasons_for_exclusion", [])
    if not isinstance(reasons, list):
        reasons = []

    reasons_domains = [r.get("domain", "") if isinstance(r, dict) else "" for r in reasons]
    reasons_codes = [r.get("reason_code", "") if isinstance(r, dict) else "" for r in reasons]
    reasons_texts = [r.get("reason_text", "") if isinstance(r, dict) else "" for r in reasons]

    # row = {
    #     # Basic
    #     "study_id": result.get("study_id"),
    #     "decision": result.get("decision"),
    #     "overall_notes": result.get("overall_notes", ""),

    #     # Reasons
    #     "reasons_domains": join_list(reasons_domains),
    #     "reasons_codes": join_list(reasons_codes),
    #     "reasons_texts": join_list(reasons_texts),

    #     # Population
    #     "population_matched_must_include": join_list(pop.get("matched_must_include", [])),
    #     "population_missing_must_include": join_list(pop.get("missing_must_include", [])),
    #     "population_exclude_flags_triggered": join_list(pop.get("exclude_flags_triggered", [])),

    #     # Intervention
    #     "intervention_matched_must_include_any_of": join_list(
    #         itv.get("matched_must_include_any_of", [])
    #     ),
    #     "intervention_exclude_flags_triggered": join_list(
    #         itv.get("exclude_flags_triggered", [])
    #     ),

    #     # Comparator
    #     "comparator_required": comp.get("comparator_required"),
    #     "comparator_accept_single_arm": comp.get("accept_single_arm"),
    #     "comparator_matched_must_include_any_of": join_list(
    #         comp.get("matched_must_include_any_of", [])
    #     ),
    #     "comparator_missing": comp.get("comparator_missing"),
    #     "comparator_notes": comp.get("notes", ""),

    #     # Outcomes
    #     "outcomes_matched_required_any_of": join_list(
    #         out.get("matched_required_any_of", [])
    #     ),
    #     "outcomes_exclude_if_only_triggered": out.get("exclude_if_only_triggered"),
    #     "outcomes_notes": out.get("notes", ""),

    #     # Study design
    #     "study_design_classified_as": sd.get("design_classified_as"),
    #     "study_design_matches_include_list": sd.get("matches_include_list"),
    #     "study_design_matches_exclude_list": sd.get("matches_exclude_list"),

    #     # Other criteria
    #     "other_estimated_sample_size": other.get("estimated_sample_size"),
    #     "other_sample_size_ok": other.get("sample_size_ok"),
    #     "other_year_of_publication": other.get("year_of_publication"),
    #     "other_within_year_range": other.get("within_year_range"),
    #     "other_language": other.get("language"),
    #     "other_language_allowed": other.get("language_allowed"),
    # }

    row = {
    # Basic - use empty string for missing required fields
    "study_id": result.get("study_id", ""),
    "decision": result.get("decision", "unknown"),
    "cumulative_score": result.get("cumulative_score", 0),
    "overall_notes": result.get("overall_notes", ""),

    # Reasons
    "reasons_domains": join_list(reasons_domains),
    "reasons_codes": join_list(reasons_codes),
    "reasons_texts": join_list(reasons_texts),

    # Population
    "population_matched_must_include": join_list(
        pop.get("matched_must_include", []) if isinstance(pop, dict) else []
    ),
    "population_missing_must_include": join_list(
        pop.get("missing_must_include", []) if isinstance(pop, dict) else []
    ),
    "population_exclude_flags_triggered": join_list(
        pop.get("exclude_flags_triggered", []) if isinstance(pop, dict) else []
    ),

    # Intervention
    "intervention_matched_must_include_any_of": join_list(
        itv.get("matched_must_include_any_of", []) if isinstance(itv, dict) else []
    ),
    "intervention_exclude_flags_triggered": join_list(
        itv.get("exclude_flags_triggered", []) if isinstance(itv, dict) else []
    ),

    # Comparator
    "comparator_required": comp.get("comparator_required", "") if isinstance(comp, dict) else "",
    "comparator_accept_single_arm": comp.get("accept_single_arm", "") if isinstance(comp, dict) else "",
    "comparator_matched_must_include_any_of": join_list(
        comp.get("matched_must_include_any_of", []) if isinstance(comp, dict) else []
    ),
    "comparator_missing": comp.get("comparator_missing", "") if isinstance(comp, dict) else "",
    "comparator_notes": comp.get("notes", "") if isinstance(comp, dict) else "",

    # Outcomes
    "outcomes_matched_required_any_of": join_list(
        out.get("matched_required_any_of", []) if isinstance(out, dict) else []
    ),
    "outcomes_exclude_if_only_triggered": out.get("exclude_if_only_triggered", "") if isinstance(out, dict) else "",
    "outcomes_notes": out.get("notes", "") if isinstance(out, dict) else "",

    # Study design
    "study_design_classified_as": sd.get("design_classified_as", "") if isinstance(sd, dict) else "",
    "study_design_matches_include_list": sd.get("matches_include_list", "") if isinstance(sd, dict) else "",
    "study_design_matches_exclude_list": sd.get("matches_exclude_list", "") if isinstance(sd, dict) else "",

    # Other criteria
    "other_estimated_sample_size": other.get("estimated_sample_size", "") if isinstance(other, dict) else "",
    "other_sample_size_ok": other.get("sample_size_ok", "") if isinstance(other, dict) else "",
    "other_year_of_publication": other.get("year_of_publication", "") if isinstance(other, dict) else "",
    "other_within_year_range": other.get("within_year_range", "") if isinstance(other, dict) else "",
    "other_language": other.get("language", "") if isinstance(other, dict) else "",
    "other_language_allowed": other.get("language_allowed", "") if isinstance(other, dict) else "",
}

    return row


def validate_and_adjust_screening_score(screening_result: dict, eligibility_criteria: dict) -> dict:
    """
    Validate screening result against eligibility criteria and adjust score if needed.
    Enforces gating criteria (e.g., required comparator) that MUST result in low scores.

    Args:
        screening_result: The screening result dict from LLM
        eligibility_criteria: The eligibility criteria dict

    Returns:
        Modified screening_result with validated/adjusted score and decision
    """
    print(screening_result)
    try:
        criteria_match = screening_result.get("criteria_match", {})
        comparator_match = criteria_match.get("comparator", {})
        eligibility_comp = eligibility_criteria.get("comparator", {})

        # Check if comparator is required
        comparator_required = eligibility_comp.get("required", False)
        accept_single_arm = eligibility_comp.get("accept_single_arm", True)
        comparator_missing = comparator_match.get("comparator_missing", False)
        matched_comparators = comparator_match.get("matched_must_include_any_of", [])

        # GATING RULE: If comparator is required and missing/not matched, cap score at 2
        if comparator_required and not accept_single_arm:
            if comparator_missing or not matched_comparators:
                original_score = screening_result.get("cumulative_score", 0)
                original_decision = screening_result.get("decision", "unknown")

                # Cap score at 2 maximum
                if original_score > 2:
                    screening_result["cumulative_score"] = 2
                    print(f"[VALIDATION] Capped score from {original_score} to 2 (missing required comparator)")

                # Force decision to exclude or unclear
                if original_decision == "include":
                    screening_result["decision"] = "exclude"
                    print(f"[VALIDATION] Changed decision from 'include' to 'exclude' (missing required comparator)")

                # Add exclusion reason if not present
                reasons = screening_result.get("reasons_for_exclusion", [])
                has_comparator_reason = any(
                    r.get("domain") == "comparator" for r in reasons if isinstance(r, dict)
                )

                if not has_comparator_reason:
                    reasons.append({
                        "domain": "comparator",
                        "reason_code": "missing_required_comparator",
                        "reason_text": "Study lacks required comparator arm and accept_single_arm is false"
                    })
                    screening_result["reasons_for_exclusion"] = reasons
                    print(f"[VALIDATION] Added missing_required_comparator exclusion reason")

        # Check population requirements
        pop_match = criteria_match.get("population", {})
        eligibility_pop = eligibility_criteria.get("population", {})
        must_include_pop = eligibility_pop.get("must_include", [])
        missing_must_include = pop_match.get("missing_must_include", [])

        # GATING RULE: If critical population criteria are missing, cap score at 3
        if must_include_pop and missing_must_include:
            if len(missing_must_include) >= len(must_include_pop) / 2:  # Missing > 50% of required
                original_score = screening_result.get("cumulative_score", 0)
                if original_score > 3:
                    screening_result["cumulative_score"] = 3
                    print(f"[VALIDATION] Capped score to 3 (missing critical population criteria)")

        return screening_result

    except Exception as e:
        print(f"[WARN] Validation error (continuing with original result): {e}")
        return screening_result


def screen_single_study(pm_id: str, article: dict, ftext: dict, elig: str, pico_json: dict,
                        csv_path: Path, csv_lock: Lock, ftmode: bool = True) -> dict:
    """
    Screen a single study and save the result to CSV.
    Returns the screening result dict with success status.
    Always writes a row to CSV (even on failure).
    """
    try:
        agent = _agent()
        agent.set_system("You are a systematic review screening expert. You ONLY return valid JSON. No thoughts. No explanations. No reasoning. Your response MUST start with { and end with }.")
        prompt_template = "prompts/screening.txt"

        PICO = pico_json["pico_valid"]
        P, I, C, O = (
            PICO["Population"],
            PICO["Intervention"],
            PICO["Comparator"],
            PICO["Outcomes"],
        )

        title = article.get("title", "")
        abstract = article.get("abstract", "")

        # ----- FULL TEXT HANDLING -----
        record = ftext.get(pm_id)
        full_text_plain = "null"
        pmc_xml = None

        if record:
            pmc_xml = record.get("pmc_full_xml")
        if pmc_xml:
            if ftmode:
                full_text_plain = xml_to_plain(pmc_xml)

        # ----- BUILD PROMPT -----
        prompt = _read(prompt_template)

        replacements = {
            "{{DOC_TITLE}}": title or "Not available",
            "{{DOC_ABSTRACT_OR_NULL}}": abstract or "null",
            "{{DOC_FULL_TEXT_OR_NULL}}": full_text_plain,
            "{{ELIGIBILITY_JSON}}": elig,
            "{{STUDY_ID}}": pm_id,
            "{{QUESTION_ID}}": pico_json["qid"],
            "{{QUESTION_TEXT}}": pico_json["question"],
            "{{P}}": P or "Not specified",
            "{{I}}": I or "Not specified",
            "{{C}}": C or "Not specified",
            "{{O}}": O if O else "Not specified",
        }

        for placeholder, value in replacements.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif value is None:
                value = "null"
            else:
                value = str(value)
            prompt = prompt.replace(placeholder, value)

        # ----- CALL LLM -----
        print(f"[INFO] Screening {pm_id}...")
        resp = agent.say(prompt)
        content = _extract_vertex_content(resp)
        if not isinstance(content, str) or not content.strip():
            content = str(resp)

        # ----- PARSE AND SAVE RESULT -----
        screening_result = parse_screening_result(content)

        # Ensure study_id is in result
        if "study_id" not in screening_result:
            screening_result["study_id"] = pm_id

        # ----- VALIDATE AND ADJUST SCORE -----
        # Parse eligibility criteria from JSON string
        try:
            eligibility_criteria = json.loads(elig) if isinstance(elig, str) else elig
        except Exception as parse_error:
            print(f"[WARN] Could not parse eligibility criteria for validation: {parse_error}")
            eligibility_criteria = {}

        # Apply validation rules to enforce gating criteria
        screening_result = validate_and_adjust_screening_score(screening_result, eligibility_criteria)

        save_screening_result_to_csv(screening_result, csv_path, csv_lock)

        decision = screening_result.get('decision', 'unknown')
        score = screening_result.get('cumulative_score', 0)
        print(f"[SUCCESS] {pm_id}: {decision} (score: {score})")

        return {"success": True, "pmid": pm_id, "result": screening_result}

    except Exception as e:
        print(f"[ERROR] Failed to screen study {pm_id}: {e}")
        import traceback
        print(traceback.format_exc())

        # Create a minimal error result and write to CSV
        error_result = {
            "study_id": pm_id,
            "decision": "error",
            "cumulative_score": 0,
            "overall_notes": f"Screening failed: {str(e)}",
            "reasons_for_exclusion": [{"domain": "error", "reason_code": "screening_error", "reason_text": str(e)}]
        }

        try:
            save_screening_result_to_csv(error_result, csv_path, csv_lock)
        except Exception as save_error:
            print(f"[ERROR] Could not save error result for {pm_id}: {save_error}")

        return {"success": False, "pmid": pm_id, "error": str(e)}


def screen_studies(art, ftext, elig, pico_json, max_studies=5, score_threshold=1 , ftmode = True, max_workers=5):
    """
    Screen studies in parallel using multithreading.
    Ensures ALL screening tasks complete before returning.

    Args:
        art: Dictionary of articles keyed by pmid
        ftext: Full text data dictionary
        elig: Eligibility criteria as JSON string
        pico_json: PICO extraction result
        max_studies: Maximum number of studies to screen
        score_threshold: Not used in current implementation
        ftmode: Whether to use full text when available
        max_workers: Number of parallel threads (default: 5)

    Returns:
        Path to the CSV file with screening results
    """
    import json
    from pathlib import Path

    print("\n" + "="*70)
    print("STAGE 5: PARALLEL STUDY SCREENING")
    print("="*70)

    # Deduplicate within this run so each pm_id is screened only once
    seen_pmids = set()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(f"{config.ARTIFACTS_SCREENING_DIR}/screening_results_{ts}.csv")

    # Ensure directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Create thread-safe lock for CSV writing
    csv_lock = Lock()

    # Collect studies to screen (respecting max_studies and deduplication)
    studies_to_screen = []
    for a in art.values():
        if len(studies_to_screen) >= max_studies:
            break

        pm_id = a.get("pmid")
        if not pm_id:
            print(f"[WARN] Skipping article with no PMID")
            continue

        if pm_id in seen_pmids:
            print(f"[WARN] Skipping duplicate PMID: {pm_id}")
            continue
        seen_pmids.add(pm_id)

        studies_to_screen.append((pm_id, a))

    if not studies_to_screen:
        print("[ERROR] No studies to screen!")
        return None

    print(f"\n[INFO] Screening {len(studies_to_screen)} studies using {max_workers} parallel workers")
    print(f"[INFO] Results will be saved to: {csv_path}")
    print("-"*70)

    # Track results
    successful_screenings = []
    failed_screenings = []
    all_results = {}

    # Process studies in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all screening tasks
        future_to_pmid = {
            executor.submit(
                screen_single_study,
                pm_id,
                article,
                ftext,
                elig,
                pico_json,
                csv_path,
                csv_lock,
                ftmode
            ): (pm_id, article)
            for pm_id, article in studies_to_screen
        }

        print(f"[INFO] Submitted {len(future_to_pmid)} screening tasks to thread pool")
        print(f"[INFO] Waiting for all tasks to complete...")
        print()

        # Collect results as they complete - this blocks until ALL futures are done
        completed = 0
        for future in as_completed(future_to_pmid):
            pmid_article = future_to_pmid[future]
            pm_id = pmid_article[0]
            completed += 1

            try:
                result = future.result(timeout=300)  # 5 minute timeout per study
                all_results[pm_id] = result

                if result and result.get("success"):
                    successful_screenings.append(pm_id)
                    print(f"[{completed}/{len(studies_to_screen)}] ✅ {pm_id}")
                else:
                    failed_screenings.append(pm_id)
                    error = result.get("error", "Unknown error") if result else "No result returned"
                    print(f"[{completed}/{len(studies_to_screen)}] ❌ {pm_id} - {error}")

            except Exception as e:
                failed_screenings.append(pm_id)
                all_results[pm_id] = {"success": False, "pmid": pm_id, "error": str(e)}
                print(f"[{completed}/{len(studies_to_screen)}] ❌ {pm_id} - Exception: {e}")

    # Ensure the thread pool is fully shut down before continuing
    print()
    print("-"*70)
    print("SCREENING COMPLETE - All parallel tasks finished")
    print("-"*70)

    # Print summary
    print(f"\n📊 SCREENING SUMMARY:")
    print(f"   Total studies:     {len(studies_to_screen)}")
    print(f"   ✅ Successful:     {len(successful_screenings)}")
    print(f"   ❌ Failed:         {len(failed_screenings)}")
    print(f"   Success rate:      {len(successful_screenings)/len(studies_to_screen)*100:.1f}%")

    if failed_screenings:
        print(f"\n⚠️  Failed PMIDs: {', '.join(failed_screenings)}")

    # Verify CSV file exists and has data
    if csv_path.exists():
        import csv as csv_module
        with open(csv_path, 'r') as f:
            row_count = sum(1 for row in csv_module.reader(f)) - 1  # Subtract header
        print(f"\n✅ CSV file created: {csv_path}")
        print(f"   Rows written: {row_count}")

        if row_count != len(studies_to_screen):
            print(f"   ⚠️  WARNING: Expected {len(studies_to_screen)} rows but got {row_count}")
    else:
        print(f"\n❌ ERROR: CSV file was not created at {csv_path}")
        return None

    print("\n" + "="*70)
    print("✅ SCREENING STAGE COMPLETE - Ready for next stage")
    print("="*70 + "\n")

    # Convert Path to string for compatibility with Gradio FileData
    return str(csv_path)