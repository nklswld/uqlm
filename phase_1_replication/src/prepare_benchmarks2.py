# scripts/prepare_benchmarks.py
# Final version: builds unified {id, prompt, gold} JSONL files for ARC-C and TruthfulQA-MC
# Supports varied source schemas (lists, dicts, list[dict{'label','text'}], etc.)

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Union

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LETTER_RE_SINGLE = re.compile(r"^[A-Za-z]$")

# ---------- IO helpers ----------

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(rows)} rows -> {p}")

# ---------- choice + gold normalization ----------

def normalize_choices(raw) -> List[str]:
    """
    Normalize answer choices to a list[str] in A,B,C,... order.
    Supports:
      - list[str]
      - dict{A: "...", B: "..."} (also lowercase)
      - list[dict] with keys like {"label":"A","text":"..."} or {"label":"A","choice":"..."}
    """
    # Case 1: already list[str]
    if isinstance(raw, list) and (len(raw) == 0 or isinstance(raw[0], str)):
        return [str(x) for x in raw]

    # Case 2: dict keyed by letters
    if isinstance(raw, dict):
        out = []
        # upper-case keys first
        for L in LETTERS:
            if L in raw:
                out.append(str(raw[L]))
        # fallback: lower-case keys
        if not out and any(k in raw for k in ["a", "b", "c", "d"]):
            for L in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                if L in raw:
                    out.append(str(raw[L]))
        if out:
            return out

    # Case 3: list of dicts with labels
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        items = []
        for d in raw:
            lab = d.get("label") or d.get("letter") or d.get("id")
            txt = d.get("text") or d.get("choice") or d.get("content") or d.get("value")
            if lab is None and "index" in d:
                try:
                    lab = LETTERS[int(d["index"])]
                except Exception:
                    pass
            if lab is not None and txt is not None:
                items.append((str(lab).upper().strip(), str(txt)))
        # sort by A,B,C,...
        items.sort(key=lambda x: LETTERS.index(x[0]) if x[0] in LETTERS else 999)
        if items:
            return [t for _, t in items]

    # Fallback (prevents crashes; consider logging a warning in production)
    return []

def normalize_gold(gold: Union[str, int, None], choices: List[str]) -> str:
    """
    Normalize gold label to a single letter A/B/C/...
    Accepts index, single letter, or textual match to one of the choices.
    """
    if gold is None:
        return "A"
    # index -> letter
    if isinstance(gold, int):
        return LETTERS[gold] if 0 <= gold < len(choices) else "A"

    g = str(gold).strip()
    # already a single letter?
    if LETTER_RE_SINGLE.fullmatch(g):
        return g.upper()

    # textual equality to a choice
    try:
        idx = [c.strip() for c in choices].index(g)
        return LETTERS[idx]
    except Exception:
        pass

    # extract letter from text like "(C)" or "Answer: C"
    m = re.search(r"\b([A-H])\b", g, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return "A"

# ---------- prompt builder ----------

def build_mc_prompt(question: str, choices: List[str]) -> str:
    letters = LETTERS[:len(choices)]
    opts = "\n".join(f"{L}) {txt}" for L, txt in zip(letters, choices))
    return (
        "You are a careful reasoning assistant.\n"
        "Choose exactly one option (just the letter).\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        "Answer (A/B/C/D/... only):"
    )

# ---------- dataset-specific unifiers ----------

def unify_arc(path_in: str, path_out: str) -> None:
    """
    ARC-C typical fields:
      - question (or query/prompt)
      - choices: list[{'label':'A','text':'...'}, ...]  OR list[str] OR dict{A:...}
      - answerKey: 'C' (or label/text/index)
    """
    rows: List[Dict[str, Any]] = []
    with open(path_in, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            q = ex.get("question") or ex.get("query") or ex.get("prompt") or ""
            raw_choices = ex.get("choices") or ex.get("answers") or ex.get("options")
            choices = normalize_choices(raw_choices)
            if not choices:
                choices = ["Option A", "Option B", "Option C", "Option D"]

            gold_raw = ex.get("answerKey") or ex.get("label") or ex.get("gold")
            gold = normalize_gold(gold_raw, choices)

            rid = ex.get("id") or ex.get("uid") or str(i)
            prompt = build_mc_prompt(q, choices)
            rows.append({"id": rid, "prompt": prompt, "gold": gold})
    write_jsonl(path_out, rows)

def unify_truthfulqa_mc(path_in: str, path_out: str) -> None:
    """
    TruthfulQA-MC common fields:
      - question (or prompt)
      - choices/options/mc1_targets: list[str] (sometimes other shapes)
      - correct/label/mc1_targets_correct: index/letter/text
    """
    rows: List[Dict[str, Any]] = []
    with open(path_in, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            q = ex.get("question") or ex.get("prompt") or ""
            raw_choices = ex.get("mc1_targets") or ex.get("choices") or ex.get("options")
            choices = normalize_choices(raw_choices)
            if not choices:
                choices = ["Option A", "Option B", "Option C", "Option D"]

            gold_raw = ex.get("mc1_targets_correct") or ex.get("correct") or ex.get("label")
            gold = normalize_gold(gold_raw, choices)

            rid = ex.get("id") or ex.get("uid") or str(i)
            prompt = build_mc_prompt(q, choices)
            rows.append({"id": rid, "prompt": prompt, "gold": gold})
    write_jsonl(path_out, rows)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Prepare unified JSONL for ARC-C and TruthfulQA-MC.")
    ap.add_argument("--arc_in", default="phase_1_replication/benchmarks/arc_c/test.jsonl")
    ap.add_argument("--arc_out", default="phase_1_replication/benchmarks/arc_c/test_unified.jsonl")
    ap.add_argument("--tqa_in", default="phase_1_replication/benchmarks/truthfulqa_mc/validation.jsonl")
    ap.add_argument("--tqa_out", default="phase_1_replication/benchmarks/truthfulqa_mc/validation_unified.jsonl")
    args = ap.parse_args()

    unify_arc(args.arc_in, args.arc_out)
    unify_truthfulqa_mc(args.tqa_in, args.tqa_out)

if __name__ == "__main__":
    main()
