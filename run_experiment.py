#!/usr/bin/env python3
"""
Quantization experiment runner.

Sends the same MMLU multiple-choice items to one model at several quantization
levels (via a local Ollama server) and logs whether each answer is correct.

Design notes
- Items are sampled once from a fixed shuffled order (seed), so every level sees
  exactly the same questions -> paired design (McNemar / Cochran's Q).
- The first --pilot-size items in that order are reserved for the pilot; the main
  run draws from the items after them, so pilot data never leaks into the final test.
- temperature=0 + fixed seed + constrained JSON output -> deterministic, single-letter
  answers. The item is the unit of replication.
- Results are appended to a JSONL file, so an interrupted run resumes where it left off.

Usage
  # pilot: 50 items, just Q4 vs Q8
  python run_experiment.py --stage pilot --levels q4_K_M q8_0

  # main run: all four levels, n chosen from your power analysis
  python run_experiment.py --stage main --n 600
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset

OLLAMA = "http://localhost:11434"
DEFAULT_BASE = "llama3.1:8b-instruct"
DEFAULT_LEVELS = ["q2_K", "q3_K_M", "q4_K_M", "q8_0"]
LETTERS = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "You are taking a multiple-choice exam. Choose the single best answer "
    "and respond with only its letter."
)

# Constrains the model's output to exactly one of the four letters.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": LETTERS}},
    "required": ["answer"],
}


# ---------------------------------------------------------------- data

def load_items(stage: str, n: int, pilot_size: int, seed: int) -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)

    if stage == "pilot":
        chosen = order[:pilot_size]
    else:
        chosen = order[pilot_size : pilot_size + n]

    items = []
    for i in chosen:
        row = ds[i]
        items.append(
            {
                "item_id": int(i),
                "subject": row["subject"],
                "question": row["question"],
                "choices": list(row["choices"]),
                "gold": LETTERS[row["answer"]],
            }
        )
    return items


def build_prompt(item: dict) -> str:
    lines = [item["question"].strip(), ""]
    for letter, choice in zip(LETTERS, item["choices"]):
        lines.append(f"{letter}. {choice}")
    lines += ["", 'Respond as JSON: {"answer": "<letter>"}']
    return "\n".join(lines)


# ---------------------------------------------------------------- ollama

def check_server() -> None:
    try:
        requests.get(f"{OLLAMA}/api/tags", timeout=5).raise_for_status()
    except requests.RequestException:
        sys.exit(f"Can't reach Ollama at {OLLAMA}. Start it with `ollama serve` (or open the app).")


def ensure_model(model: str) -> None:
    tags = requests.get(f"{OLLAMA}/api/tags", timeout=10).json().get("models", [])
    if any(m.get("name") == model or m.get("model") == model for m in tags):
        return
    print(f"  pulling {model} (one-time download)...", flush=True)
    r = requests.post(f"{OLLAMA}/api/pull", json={"model": model, "stream": False}, timeout=None)
    r.raise_for_status()


def unload(model: str) -> None:
    requests.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)


def ask(model: str, item: dict, num_ctx: int) -> tuple[str | None, str, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(item)},
        ],
        "stream": False,
        "format": ANSWER_SCHEMA,
        "options": {"temperature": 0, "seed": 0, "num_predict": 20, "num_ctx": num_ctx},
        "keep_alive": "10m",
    }
    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    latency = time.perf_counter() - t0

    raw = r.json()["message"]["content"]
    try:
        pred = str(json.loads(raw)["answer"]).strip().upper()
    except (json.JSONDecodeError, KeyError, TypeError):
        pred = None
    if pred not in LETTERS:
        pred = None  # counted as incorrect; analyze.py reports how many
    return pred, raw, latency


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["pilot", "main"], required=True)
    ap.add_argument("--n", type=int, help="number of items for the main run")
    ap.add_argument("--pilot-size", type=int, default=50)
    ap.add_argument("--base", default=DEFAULT_BASE, help="Ollama model name without the quant suffix")
    ap.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    ap.add_argument("--seed", type=int, default=510)
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    if args.stage == "main" and not args.n:
        ap.error("--n is required for --stage main (take it from the power analysis)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    results_path = out_dir / f"{args.stage}_results.jsonl"
    items_path = out_dir / f"{args.stage}_items.json"

    check_server()
    items = load_items(args.stage, args.n or 0, args.pilot_size, args.seed)
    items_path.write_text(json.dumps([{k: it[k] for k in ("item_id", "subject", "gold")} for it in items], indent=1))
    print(f"{args.stage}: {len(items)} items x {len(args.levels)} levels -> {results_path}")

    done = set()
    if results_path.exists():
        with results_path.open() as f:
            for line in f:
                rec = json.loads(line)
                done.add((rec["level"], rec["item_id"]))
        print(f"resuming: {len(done)} responses already logged")

    with results_path.open("a") as out:
        for level in args.levels:
            model = f"{args.base}-{level}"
            todo = [it for it in items if (level, it["item_id"]) not in done]
            if not todo:
                print(f"[{level}] already complete")
                continue

            print(f"[{level}] {model}: {len(todo)} items")
            ensure_model(model)
            t_start = time.perf_counter()
            n_correct = 0

            for k, item in enumerate(todo, 1):
                pred, raw, latency = ask(model, item, args.num_ctx)
                correct = int(pred == item["gold"])
                n_correct += correct
                out.write(
                    json.dumps(
                        {
                            "stage": args.stage,
                            "level": level,
                            "model": model,
                            "item_id": item["item_id"],
                            "subject": item["subject"],
                            "gold": item["gold"],
                            "pred": pred,
                            "correct": correct,
                            "valid": pred is not None,
                            "latency_s": round(latency, 3),
                            "raw": raw,
                        }
                    )
                    + "\n"
                )
                out.flush()
                if k % 25 == 0 or k == len(todo):
                    elapsed = time.perf_counter() - t_start
                    print(f"  {k}/{len(todo)}  acc so far {n_correct / k:.3f}  ({elapsed / k:.2f}s/item)", flush=True)

            unload(model)  # free memory before loading the next level

    print("done. next: python analyze.py " + str(results_path))


if __name__ == "__main__":
    main()