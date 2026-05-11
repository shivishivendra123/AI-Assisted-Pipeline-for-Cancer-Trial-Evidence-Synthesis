import argparse, json, os
from typing import Optional
from schemas.pico import PICO, AugmentedPICO
from pipeline.augment.term_expander import expand_terms_with_llm
from utils.io import qid, write_jsonl, now_iso

def _read(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def _load_pico_from_day1(path: str) -> PICO:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    d = doc.get("pico_valid") or {}
    return PICO.model_validate(d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, help="If provided, we will re-run Day 1 implicitly (not required).")
    ap.add_argument("--pico-file", type=str, help="Path to Day 1 JSON (pico_<hash>.json).")
    ap.add_argument("--syn-prompt", default="prompts/synonym_prompt.txt")
    ap.add_argument("--outdir", default="artifacts_day2")
    args = ap.parse_args()

    if not args.pico_file and not args.question:
        ap.error("Provide either --pico-file (preferred) or --question.")

    if args.pico_file:
        pico = _load_pico_from_day1(args.pico_file)
        qhash = qid(json.dumps(pico.model_dump(), sort_keys=True))
        question = f"[from file] {os.path.basename(args.pico_file)}"
    else:
        # lightweight inline Day-1 run via your main if you want; for now assume PICO constructed elsewhere
        ap.error("For the lean Day-2 CLI, please pass --pico-file from Day 1.")

    syn_prompt = _read(args.syn_prompt)
    augmented: AugmentedPICO = expand_terms_with_llm(pico, syn_prompt)

    os.makedirs(args.outdir, exist_ok=True)
    out_jsonl = os.path.join(args.outdir, "pico_aug_traces.jsonl")
    out_pretty = os.path.join(args.outdir, f"aug_{qhash}.json")

    row = {
        "qid": qhash,
        "created_at": now_iso(),
        "pico_valid": json.loads(pico.model_dump_json()),
        "augmented": json.loads(augmented.model_dump_json()),
        "model": "gemma-3 via Vertex",
        "temperature": 0.2
    }
    write_jsonl(out_jsonl, [row])
    with open(out_pretty, "w", encoding="utf-8") as f:
        f.write(json.dumps(row, indent=2))

    print("=== AUGMENTED PICO ===")
    print(augmented.model_dump_json(indent=2))
    print(f"\nSaved:\n- {out_jsonl}\n- {out_pretty}")

if __name__ == "__main__":
    main()
