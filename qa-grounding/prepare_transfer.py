"""Prepare a MuSiQue answer-transfer panel without running model inference."""

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import resource
import statistics

from prepare_data import (
    HERE, MAX_PROMPT, SPLITS, digest, formatted_prompt, load_prior, selected_sentences,
    title_keys, verify_fingerprints,
)
from transformers import AutoTokenizer

PANEL_SIZE = 160
SALT = "musique-qwen-transfer-23nov2026-v1"


def normalized_title(title, retrieval):
    return " ".join(retrieval.words(title))


def fitting_titles(qa, hotpot_path, config):
    groups = qa.representatives(qa.load(hotpot_path))
    fitting = groups[SPLITS["train"][0]:SPLITS["dev"][1]]
    gold = title_keys(fitting, qa)
    acquired = {
        normalized_title(title, qa.retrieval)
        for _, _, row in fitting
        for title, _, _ in selected_sentences(row, qa.retrieval, config)[0]
    }
    return gold, acquired


def eligible_panel(musique_path, retrieval, excluded_titles):
    if digest(musique_path) != retrieval.DATA_SHA:
        raise ValueError("Unexpected MuSiQue source checksum")
    with musique_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != 2417 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Unexpected MuSiQue row count or duplicate IDs")
    by_component = defaultdict(list)
    for row, (role, group) in zip(rows, retrieval.make_split(rows), strict=True):
        if role == "test":
            by_component[group].append(row)

    selected = []
    excluded = 0
    for group, cases in by_component.items():
        source = min(cases, key=lambda case: retrieval.stable(f'{SALT}:row:{case["id"]}'))
        gold = {
            normalized_title(p["title"], retrieval)
            for p in source["paragraphs"] if p["is_supporting"]
        }
        if gold & excluded_titles:
            excluded += 1
        else:
            selected.append((group, source))
    order = sorted(selected, key=lambda item: retrieval.stable(f"{SALT}:group:{item[0]}"))
    if len(order) < PANEL_SIZE:
        raise ValueError("Too few eligible held-out components")
    return order[:PANEL_SIZE], len(by_component), excluded


def make_row(tokenizer, source, group, retrieval, config):
    # Freeze text acquisition first, with no support or answer labels exposed.
    query_row = {
        "id": source["id"],
        "question": source["question"],
        "paragraphs": [{**p, "is_supporting": False} for p in source["paragraphs"]],
    }
    query, views, tails, no_labels, ids = retrieval.inputs(query_row, 24)
    decision = retrieval.run_one(
        retrieval.Index(query, views), views, tails, no_labels, "words", 320, "reveal_bridge", config,
    )
    chosen = {ids[i] for i in decision["selection"]}
    units = [
        (p["title"], p["idx"], p["paragraph_text"])
        for p in source["paragraphs"] if p["idx"] in chosen and p["paragraph_text"].strip()
    ]
    full_count = len(units)
    prompt = formatted_prompt(tokenizer, source["question"], units)
    while prompt is None and units:
        units.pop()
        prompt = formatted_prompt(tokenizer, source["question"], units)
    if prompt is None:
        raise ValueError("Question alone exceeds token limit")
    prompt_ids, positions = prompt
    gold = {p["idx"] for p in source["paragraphs"] if p["is_supporting"]}
    shown = {i for _, i, _ in units}
    gold_titles = {p["title"] for p in source["paragraphs"] if p["is_supporting"]}
    answer_ids = tokenizer.encode(source["answer"], add_special_tokens=False)
    if not answer_ids:
        raise ValueError("Answer has no tokens")
    return {
        "id": source["id"],
        "group": group,
        "type": f'{len(source["question_decomposition"])}hop',
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "answer": source["answer"],
        "aliases": list(dict.fromkeys(
            a for a in source["answer_aliases"] if a.strip() and a != source["answer"]
        )),
        "positions": positions,
        "support_labels": [int(i in gold) for _, i, _ in units],
        "complete": int(gold <= shown),
        "article_complete": int(gold_titles <= {title for title, _, _ in units}),
        "truncated": len(units) != full_count,
        "spent_words": decision["words"],
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("hotpot", type=Path)
    cli.add_argument("musique", type=Path)
    cli.add_argument("model", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh temporary destination")
    qa = load_prior()
    lock = verify_fingerprints(qa, args.model)
    config = json.loads((HERE.parent / "FROZEN.json").read_text())["selection"]["reveal_bridge"]["config"]
    fitting_gold, fitting_acquired = fitting_titles(qa, args.hotpot, config)
    gc.collect()
    panel, candidate_count, excluded_count = eligible_panel(
        args.musique, qa.retrieval, fitting_gold | fitting_acquired,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("The fast tokenizer is required for exact offsets")
    args.output.mkdir(parents=True)
    output = args.output / "test.jsonl"
    counts = Counter()
    types = Counter()
    lengths = []
    with output.open("x", encoding="utf-8") as handle:
        for group, source in panel:
            row = make_row(tokenizer, source, group, qa.retrieval, config)
            counts["complete"] += row["complete"]
            counts["article_complete"] += row["article_complete"]
            counts["truncated"] += int(row["truncated"])
            counts["cases_with_aliases"] += bool(row["aliases"])
            types[row["type"]] += 1
            lengths.append(len(row["prompt_ids"]))
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "dataset": "MuSiQue-Ans development, retrieval-test components only",
        "source_sha256": qa.retrieval.DATA_SHA,
        "hotpot_sha256": qa.DATA_DIGEST,
        "script_sha256": digest(Path(__file__)),
        "prompt_builder_sha256": digest(HERE / "prepare_data.py"),
        "base_model_fingerprints": lock["model"],
        "test_sha256": digest(output),
        "group_salt": SALT,
        "num_test": PANEL_SIZE,
        "retrieval_test_components": candidate_count,
        "excluded_due_to_hotpot_fitting_title_overlap": excluded_count,
        "disjoint_from_hotpot_fitting_gold_titles": len(fitting_gold),
        "disjoint_from_hotpot_fitting_acquired_titles": len(fitting_acquired),
        "max_prompt_tokens": MAX_PROMPT,
        "actual_max_prompt_tokens": max(lengths),
        "mean_prompt_tokens": statistics.mean(lengths),
        "counts": dict(counts),
        "question_types": dict(types),
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
