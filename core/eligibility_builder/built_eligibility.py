from pipeline.extractor.pico_extractor import _agent,_extract_vertex_content
import json

def _read(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def extract_json_block(text: str) -> str:
    """
    Take an LLM response and return the first {...} JSON block as a string.
    Handles ```json ... ``` and plain text around it.
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

    # Now extract the JSON object from { to }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")

    return cleaned[start : end + 1]

def build_eligibility(question: str , PICO ,qid,constraints_or_null=''):

    print(question)
    print(PICO)
    agent = _agent()
    agent.set_system("You ONLY return valid JSON. No thoughts. No explanations. No reasoning. Your response MUST start with { and end with }.")
    prompt_template = "prompts/eligiblity.txt"
    P , I ,C , O = PICO['Population'] , PICO['Intervention'] , PICO['Comparator'] , PICO['Outcomes']
    replacements = {
        "{{question_text}}": question,
        "{{P}}": P,
        "{{I}}": I,
        "{{C}}": C,
        "{{O}}": O[0],
        "{{constraints_or_null}}": constraints_or_null, 
    }

    # print(replacements)

    prompt = _read(prompt_template)

    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)

    # print(prompt)
    resp = agent.say(prompt)  # may be a raw dict OR a JSON string of the dict

    # 1) Get the assistant text out of the Vertex envelope
    content = _extract_vertex_content(resp)
    if not isinstance(content, str) or not content.strip():
        # Last resort: treat the whole thing as text and try to parse
        content = str(resp)

    print('----- CONTENT -------')
    print(content)
    print('----- CONTENT -------')

    output_pretty = f"artifacts_day5/eligiblity_{qid}.json"

    with open(output_pretty, "w", encoding="utf-8") as f:
        f.write(json.dumps(json.loads(extract_json_block(content)), indent=2))

    return json.loads(extract_json_block(content))
