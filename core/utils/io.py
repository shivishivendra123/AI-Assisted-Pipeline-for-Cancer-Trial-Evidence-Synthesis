import hashlib, json, os, time
from typing import Any, Dict, List

def qid(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:12]

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
