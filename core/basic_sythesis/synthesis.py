from pathlib import Path
import json
import ast
import re
from json_repair import repair_json

from pipeline.extractor.pico_extractor import _agent, _extract_vertex_content
from configs.env_config import config


def extract_json_block(text: str) -> str:
    """
    Extract what looks like a JSON object ({...}) from an LLM response.
    Handles optional ```json ... ``` fences and surrounding text.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Empty or invalid text provided")

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

    # Now extract the JSON object from { to }
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        # Try to find if there's any JSON-like content
        print("[DEBUG] LLM Response (first 500 chars):")
        print(text[:500])
        print("[DEBUG] LLM Response (last 500 chars):")
        print(text[-500:])
        raise ValueError("No JSON-like object found in LLM response")

    block = cleaned[start : end + 1].strip()
    return block


def parse_synthesis_result(content: str) -> dict:
    """
    Parse the LLM synthesis output into a Python dict.
    Tries JSON first; if that fails, repairs it or falls back to literal eval.
    """
    try:
        block = extract_json_block(content)
    except ValueError as e:
        print(f"[ERROR] Could not extract JSON block: {e}")
        print(f"[DEBUG] First 500 chars of LLM response: {content[:500]}")
        print("[INFO] LLM returned plain text instead of JSON, using as narrative...")

        # If no JSON found, use the plain text as the narrative
        # This handles cases where LLM just writes a synthesis without JSON wrapper
        return {
            "narrative_markdown": content if content else "Synthesis failed - no response from LLM",
            "summary": {
                "one_sentence": "Evidence synthesis completed (plain text format).",
                "key_findings": ["Synthesis provided in narrative format"],
                "limitations": ["Structured summary not available - LLM returned plain text"]
            }
        }

    try:
        # Try strict JSON first
        return json.loads(block)
    except json.JSONDecodeError as e:
        print(f"[WARN] json.loads failed: {e}")
        print("[INFO] Attempting JSON repair...")

        try:
            # Try to repair the JSON
            fixed = repair_json(block)
            return json.loads(fixed)
        except Exception as repair_error:
            print(f"[WARN] JSON repair failed: {repair_error}")
            print("[INFO] Attempting ast.literal_eval...")

            try:
                # Fallback to Python dict parsing
                data = ast.literal_eval(block)
                return data
            except Exception as ast_error:
                print(f"[ERROR] All parsing attempts failed: {ast_error}")

                # Last resort: return minimal valid structure
                return {
                    "narrative_markdown": block[:1000] if block else "Parsing failed",
                    "summary": {
                        "one_sentence": "Evidence synthesis completed but response parsing failed.",
                        "key_findings": ["Raw response available in logs"],
                        "limitations": ["Response format could not be parsed"]
                    }
                }

def _read(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def run_evidence_synthesis(
    pico_json: dict,
    chars_csv_path: Path,
    outcomes_csv_path: Path,
    prompt_template_path: Path = Path("prompts/evidence_synthesis.txt"),
) -> dict:
    """
    Run LLM-based narrative evidence synthesis using:
      - PICO info (pico_json)
      - Study characteristics CSV (Table 1)
      - Outcomes CSV (Table 2)

    Returns a dict:
      {
        "narrative_synthesis_markdown": "...",
        "summary": { ... }
      }
    and also saves it as JSON in artifacts_day5/evidence_synthesis_<qid>.json
    """
    question_text = pico_json["question"]
    pico_valid = pico_json["pico_valid"]
    qid = pico_json["qid"]

    # Handle both string and Path objects for CSV paths
    if chars_csv_path is None or chars_csv_path == "":
        raise ValueError("chars_csv_path cannot be None or empty - no study characteristics data available")
    if outcomes_csv_path is None or outcomes_csv_path == "":
        raise ValueError("outcomes_csv_path cannot be None or empty - no study outcomes data available")

    # Convert string paths to Path objects
    if not isinstance(chars_csv_path, Path):
        try:
            chars_csv_path = Path(str(chars_csv_path))
        except Exception as e:
            raise ValueError(f"Invalid chars_csv_path: {chars_csv_path} - {e}")

    if not isinstance(outcomes_csv_path, Path):
        try:
            outcomes_csv_path = Path(str(outcomes_csv_path))
        except Exception as e:
            raise ValueError(f"Invalid outcomes_csv_path: {outcomes_csv_path} - {e}")

    # Check if files exist
    if not chars_csv_path.exists():
        raise FileNotFoundError(f"Study characteristics file not found: {chars_csv_path}")
    if not outcomes_csv_path.exists():
        raise FileNotFoundError(f"Study outcomes file not found: {outcomes_csv_path}")

    # Read tables as raw CSV text
    chars_csv_text = chars_csv_path.read_text(encoding="utf-8")
    outcomes_csv_text = outcomes_csv_path.read_text(encoding="utf-8")

    # Load prompt template
    template = _read(str(prompt_template_path))

    # Build replacements
    replacements = {
        "{{QUESTION_TEXT}}": question_text,
        "{{PICO_JSON}}": json.dumps(pico_valid, ensure_ascii=False),
        "{{TABLE1_CSV}}": chars_csv_text,
        "{{TABLE2_CSV}}": outcomes_csv_text,
    }

    prompt = template
    for ph, val in replacements.items():
        prompt = prompt.replace(ph, str(val))

    # Call LLM
    agent = _agent()
    agent.set_system("You are an evidence synthesis expert. You ONLY return valid JSON. No thoughts. No explanations. No reasoning. Your response MUST start with { and end with }.")

    print("[INFO] Calling LLM for evidence synthesis...")

    try:
        resp = agent.say(prompt)
        content = _extract_vertex_content(resp)

        if not isinstance(content, str) or not content.strip():
            print("[WARN] No valid content extracted, using raw response")
            content = str(resp)

        print(f"[DEBUG] LLM response length: {len(content)} characters")
        print(f"[DEBUG] First 200 chars of raw response: {content[:200]}")
        print(f"[DEBUG] Response starts with: {repr(content[:50])}")

        # Parse the response
        result = parse_synthesis_result(content)

        # Ensure required fields exist
        if "narrative_markdown" not in result:
            result["narrative_markdown"] = result.get("narrative_synthesis_markdown",
                                                      result.get("synthesis",
                                                                "No narrative synthesis available"))

        if "summary" not in result:
            result["summary"] = {
                "one_sentence": "Summary not available",
                "key_findings": [],
                "limitations": []
            }

        # Save for later
        out_dir = Path(config.ARTIFACTS_SYNTHESIS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"evidence_synthesis_{qid}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[INFO] Evidence synthesis saved to {out_path}")

        # Return the JSON string (as original code did)
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        print(f"[ERROR] Evidence synthesis failed: {e}")
        import traceback
        print(traceback.format_exc())

        # Return minimal valid response
        error_result = {
            "narrative_markdown": f"Evidence synthesis failed: {str(e)}",
            "summary": {
                "one_sentence": "Synthesis could not be completed.",
                "key_findings": ["Error during synthesis"],
                "limitations": [f"Error: {str(e)}"]
            }
        }

        return json.dumps(error_result, ensure_ascii=False)
