"""Prepare a fresh HotpotQA panel for the exact earlier question-last prompt."""

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import resource
import statistics
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "qa-grounding"
sys.path.insert(0, str(PRIOR))

from prepare_data import MAX_PROMPT, SPLITS, build_row, digest, load_prior, selected_sentences
from prepare_data import title_keys, verify_fingerprints
from prepare_comparison import MARK_PASSAGES, MARK_QUESTION, reorder
from transformers import AutoTokenizer


TEST_START = 5304
TEST_SIZE = 320


def prepare_case(tokenizer, source, group, retrieval, config):
    row = build_row(tokenizer, source, group, retrieval, config)
    original = row["prompt_ids"]
    ordered = reorder(tokenizer, original)

    # The previous formatter places evidence sites in the passages block. Move
    # each site by exactly the same displacement as its surrounding tokens.
    decoded = tokenizer.decode(original, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    head = decoded.index(MARK_QUESTION)
    evidence = decoded.index(MARK_PASSAGES, head + len(MARK_QUESTION))
    offsets = tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    starts = [start for start, _ in offsets]
    question_index = bisect_right(starts, head + 2) - 1
    passage_index = bisect_right(starts, evidence + 2) - 1
    question_length = passage_index - question_index
    if question_length <= 0 or any(pos < passage_index for pos in row["positions"]):
        raise ValueError(f"Invalid evidence offsets for {row['id']}")
    row["positions"] = [pos - question_length for pos in row["positions"]]
    row["prompt_ids"] = ordered
    assert len(ordered) == len(original) <= MAX_PROMPT
    assert all(0 <= pos < len(ordered) for pos in row["positions"])
    return row


def choose_groups(groups, qa, config):
    train_start, train_end = SPLITS["train"]
    dev_start, dev_end = SPLITS["dev"]
    train = list(enumerate(groups[train_start:train_end], train_start))
    dev = list(enumerate(groups[dev_start:dev_end], dev_start))
    gold = title_keys(groups[train_start:dev_end], qa)
    acquired = {
        " ".join(qa.retrieval.words(title))
        for _, _, source in groups[train_start:dev_end]
        for title, _, _ in selected_sentences(source, qa.retrieval, config)[0]
    }
    forbidden = gold | acquired
    selected = []
    excluded = 0
    for index, (group, size, source) in enumerate(groups[TEST_START:], TEST_START):
        supporting_titles = {
            " ".join(qa.retrieval.words(title)) for title, _ in source["supporting_facts"]
        }
        if supporting_titles & forbidden:
            excluded += 1
            continue
        selected.append((index, (group, size, source)))
        if len(selected) == TEST_SIZE:
            break
    if len(selected) != TEST_SIZE:
        raise ValueError("Insufficient fresh support-title components")
    assert selected[0][0] >= TEST_START
    assert len({rec[1][0] for rec in train + dev + selected}) == len(train) + len(dev) + TEST_SIZE
    return {"train": train, "dev": dev, "test": selected}, excluded


def write_split(batch, path, tokenizer, qa, config):
    counts = Counter()
    lengths = []
    with path.open("x", encoding="utf-8") as out:
        for _, (group, _, source) in batch:
            row = prepare_case(tokenizer, source, group, qa.retrieval, config)
            counts.update({
                "case_count": 1,
                "complete": row["complete"],
                "article_complete": row["article_complete"],
                "truncated": row["truncated"],
                "support_labels": len(row["positions"]),
                "positive_labels": sum(row["support_labels"]),
            })
            lengths.append(len(row["prompt_ids"]))
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        "group_slice": [batch[0][0], batch[-1][0] + 1],
        "component_indices": [index for index, _ in batch],
        "sha256": digest(path),
        "min_prompt": min(lengths),
        "mean_prompt": statistics.mean(lengths),
        "max_prompt": max(lengths),
        **counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh temporary destination")
    qa = load_prior()
    lock = verify_fingerprints(qa, args.model)
    groups = qa.representatives(qa.load(args.data))
    config = json.loads((HERE.parent / "FROZEN.json").read_text())["selection"]["reveal_bridge"]["config"]
    splits, excluded = choose_groups(groups, qa, config)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("This verification needs token offsets")
    args.output.mkdir(parents=True)
    summary = {
        name: write_split(rows, args.output / f"{name}.jsonl", tokenizer, qa, config)
        for name, rows in splits.items()
    }
    manifest = {
        "corpus_sha256": qa.DATA_DIGEST,
        "script_sha256": digest(Path(__file__)),
        "prompt_builder_sha256": digest(PRIOR / "prepare_data.py"),
        "prompt_ordering_sha256": digest(PRIOR / "prepare_comparison.py"),
        "base_model_fingerprints": lock["model"],
        "max_prompt_tokens": MAX_PROMPT,
        "eos_token_id": tokenizer.eos_token_id,
        "prompt_order": "instruction, passages, question, answer cue",
        "test_component_lower_bound": TEST_START,
        "test_candidates_excluded_for_fitting_title_overlap": excluded,
        "splits": summary,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, details in summary.items():
        print(name, {k: v for k, v in details.items() if k != "component_indices"}, flush=True)
    print("Peak process RSS (bytes, macOS)", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)


if __name__ == "__main__":
    main()
