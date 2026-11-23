"""Put the earlier 160 MuSiQue reader cases in exact question-last order."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "qa-grounding"
sys.path.insert(0, str(PRIOR))

from prepare_comparison import reorder
from prepare_data import MAX_PROMPT, digest, load_prior, verify_fingerprints
from transformers import AutoTokenizer

FIELDS = ("id", "group", "type", "complete", "article_complete", "answer", "aliases")


def read_panel(directory, archived):
    manifest = json.loads((archived / "preparation.json").read_text())
    if json.loads((directory / "manifest.json").read_text())["test_sha256"] != manifest["test_sha256"]:
        raise ValueError("Prepared input manifest differs from the archived panel")
    source = directory / "test.jsonl"
    if digest(source) != manifest["test_sha256"]:
        raise ValueError("Prepared test inputs differ from the archived panel")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    prior_output = archived / "cases.jsonl"
    earlier = [json.loads(line) for line in prior_output.read_text().splitlines()]
    if len(rows) != len(earlier) or len(rows) != manifest["num_test"]:
        raise ValueError("Archived and prepared case counts differ")
    for row, scored in zip(rows, earlier, strict=True):
        for key in ("id", "group", "type", "complete", "article_complete"):
            if row[key] != scored[key]:
                raise ValueError(f"Archive mismatch for {key}")
        prompt = row["prompt_ids"]
        prompt_sha = hashlib.sha256(",".join(map(str, prompt)).encode()).hexdigest()
        if (len(prompt), prompt_sha) != (scored["prompt_tokens"], scored["prompt_sha256"]):
            raise ValueError("Archived prompt count or fingerprint differs")
    for key in ("id", "group"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {key} in panel")
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("remaining", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh temporary destination")

    earlier, reference = read_panel(args.earlier, PRIOR / "musique")
    remaining, held = read_panel(args.remaining, HERE / "transfer")
    assert len(earlier) == 160 and len(remaining) == 169
    for key in ("id", "group"):
        assert {row[key] for row in earlier}.isdisjoint(row[key] for row in remaining)

    lock = verify_fingerprints(load_prior(), args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Exact token offsets require the fast tokenizer")

    args.output.mkdir(parents=True)
    output = args.output / "test.jsonl"
    lengths = []
    counts = Counter()
    question_types = Counter()
    with output.open("x", encoding="utf-8") as stream:
        for row in earlier:
            prompt_ids = reorder(tokenizer, row["prompt_ids"])
            assert len(row["prompt_ids"]) == len(prompt_ids) <= MAX_PROMPT
            assert Counter(row["prompt_ids"]) == Counter(prompt_ids)
            case = {key: row[key] for key in FIELDS}
            case["prompt_ids"] = prompt_ids
            lengths.append(len(prompt_ids))
            counts.update({
                "complete": row["complete"],
                "article_complete": row["article_complete"],
                "cases_with_aliases": int(bool(row["aliases"])),
                "truncated": int(row["truncated"]),
            })
            question_types[row["type"]] += 1
            stream.write(json.dumps(case, separators=(",", ":")) + "\n")

    manifest = {
        "dataset": "MuSiQue-Ans development, earlier reader panel with question-last prompt order",
        "source_sha256": reference["source_sha256"],
        "script_sha256": digest(Path(__file__)),
        "base_model_fingerprints": lock["model"],
        "prompt_ordering_sha256": digest(PRIOR / "prepare_comparison.py"),
        "original_reader_panel_sha256": reference["test_sha256"],
        "disjoint_musique_panel_sha256": held["test_sha256"],
        "test_sha256": digest(output),
        "num_test": len(earlier),
        "max_prompt_tokens": MAX_PROMPT,
        "actual_max_prompt_tokens": max(lengths),
        "mean_prompt_tokens": statistics.mean(lengths),
        "prompt_order": "instruction, passages, question, answer cue",
        "counts": dict(counts),
        "question_types": dict(question_types),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
