"""Prepare the untouched MuSiQue transfer components for question-last inference."""

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "qa-grounding"
sys.path.insert(0, str(PRIOR))

from prepare_comparison import reorder
from prepare_data import MAX_PROMPT, digest, load_prior, verify_fingerprints
from prepare_transfer import (
    PANEL_SIZE as EARLIER_SIZE,
    SALT,
    eligible_panel,
    fitting_titles,
    make_row,
    normalized_title,
)
from transformers import AutoTokenizer


def fresh_panel(musique_path, retrieval, excluded_titles):
    earlier, component_count, excluded_count = eligible_panel(
        musique_path, retrieval, excluded_titles,
    )
    earlier_groups = [group for group, _ in earlier]
    with musique_path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    components = defaultdict(list)
    for row, (role, group) in zip(rows, retrieval.make_split(rows), strict=True):
        if role == "test":
            components[group].append(row)

    eligible = []
    for group, members in components.items():
        representative = min(members, key=lambda row: retrieval.stable(f'{SALT}:row:{row["id"]}'))
        gold = {
            normalized_title(p["title"], retrieval)
            for p in representative["paragraphs"] if p["is_supporting"]
        }
        if not gold & excluded_titles:
            eligible.append((group, representative))
    eligible.sort(key=lambda item: retrieval.stable(f"{SALT}:group:{item[0]}"))
    assert len(components) == component_count
    assert component_count - len(eligible) == excluded_count
    assert eligible[:EARLIER_SIZE] == earlier, "Rebuilt prefix does not match the published panel"
    panel = eligible[EARLIER_SIZE:]
    earlier_set = set(earlier_groups)
    groups = [group for group, _ in panel]
    assert len(groups) == len(set(groups)) and earlier_set.isdisjoint(groups)
    return panel, earlier_groups, component_count, excluded_count


def write_panel(destination, panel, tokenizer, retrieval, config):
    counts = Counter()
    types = Counter()
    lengths = []
    with destination.open("x", encoding="utf-8") as handle:
        for group, source in panel:
            row = make_row(tokenizer, source, group, retrieval, config)
            original = row["prompt_ids"]
            ordered = reorder(tokenizer, original)
            assert len(original) == len(ordered) <= MAX_PROMPT
            assert Counter(original) == Counter(ordered)
            counts.update({
                "complete": row["complete"],
                "article_complete": row["article_complete"],
                "truncated": int(row["truncated"]),
                "cases_with_aliases": int(bool(row["aliases"])),
            })
            types[row["type"]] += 1
            lengths.append(len(ordered))
            case = {
                key: row[key] for key in
                ("id", "group", "type", "complete", "article_complete", "answer", "aliases")
            }
            case["prompt_ids"] = ordered
            handle.write(json.dumps(case, separators=(",", ":")) + "\n")
    return dict(counts), dict(types), lengths


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
    panel, earlier_groups, component_count, excluded_count = fresh_panel(
        args.musique, qa.retrieval, fitting_gold | fitting_acquired,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("The fast tokenizer is required for exact offsets")
    args.output.mkdir(parents=True)
    output = args.output / "test.jsonl"
    counts, types, lengths = write_panel(output, panel, tokenizer, qa.retrieval, config)
    manifest = {
        "dataset": "MuSiQue-Ans development, unused retrieval-test components",
        "source_sha256": qa.retrieval.DATA_SHA,
        "hotpot_sha256": qa.DATA_DIGEST,
        "script_sha256": digest(Path(__file__)),
        "prior_panel_builder_sha256": digest(PRIOR / "prepare_transfer.py"),
        "prompt_builder_sha256": digest(PRIOR / "prepare_data.py"),
        "prompt_ordering_sha256": digest(PRIOR / "prepare_comparison.py"),
        "base_model_fingerprints": lock["model"],
        "test_sha256": digest(output),
        "group_salt": SALT,
        "earlier_panel_components": len(earlier_groups),
        "earlier_groups_sha256": hashlib.sha256("\n".join(earlier_groups).encode()).hexdigest(),
        "num_test": len(panel),
        "retrieval_test_components": component_count,
        "excluded_due_to_hotpot_fitting_title_overlap": excluded_count,
        "disjoint_from_hotpot_fitting_gold_titles": len(fitting_gold),
        "disjoint_from_hotpot_fitting_acquired_titles": len(fitting_acquired),
        "max_prompt_tokens": MAX_PROMPT,
        "actual_max_prompt_tokens": max(lengths),
        "mean_prompt_tokens": statistics.mean(lengths),
        "prompt_order": "instruction, passages, question, answer cue",
        "counts": counts,
        "question_types": types,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    print("Peak process RSS (bytes, macOS)", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == "__main__":
    main()
