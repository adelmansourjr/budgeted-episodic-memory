"""Replay split membership, exact prompts, and supervised spans without inference."""

import argparse
import json
from pathlib import Path

from prepare_data import (
    HERE, SPLITS, build_row, digest, load_prior, selected_sentences, title_keys, verify_fingerprints,
)
from transformers import AutoTokenizer


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("data", type=Path)
    cli.add_argument("model", type=Path)
    cli.add_argument("prepared", type=Path)
    args = cli.parse_args()
    qa = load_prior()
    verify_fingerprints(qa, args.model)
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    assert manifest["corpus_sha256"] == qa.DATA_DIGEST
    assert manifest["script_sha256"] == digest(HERE / "prepare_data.py")
    groups = qa.representatives(qa.load(args.data))
    test_gold = title_keys(groups[slice(*SPLITS["test"])], qa)
    seen = title_keys(groups[:3268], qa)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    config = json.loads((HERE.parent / "FROZEN.json").read_text())["selection"]["reveal_bridge"]["config"]
    for split, (start, end) in SPLITS.items():
        path = args.prepared / f"{split}.jsonl"
        assert manifest["splits"][split]["sha256"] == digest(path)
        batch = groups[start:end]
        support = title_keys(batch, qa)
        assert seen.isdisjoint(support)
        seen.update(support)
        shown_test_titles = set()
        shown_count = 0
        with path.open() as stream:
            for record, (group, _, row) in zip(stream, batch, strict=True):
                expected = json.loads(record)
                recovered = build_row(tokenizer, row, group, qa.retrieval, config)
                assert recovered == expected, recovered["id"]
                if split == "test":
                    continue
                sentences, _ = selected_sentences(row, qa.retrieval, config)
                seen_count = len(recovered["positions"])
                overlap = {
                    " ".join(qa.retrieval.words(title))
                    for title, _, _ in sentences[:seen_count]
                } & test_gold
                if overlap:
                    shown_count += 1
                    shown_test_titles.update(overlap)
        print(
            split,
            len(batch),
            "exactly regenerated cases;",
            shown_count,
            "examples showing later test support titles as distractors across",
            len(shown_test_titles),
            "distinct titles",
            flush=True,
        )


if __name__ == "__main__":
    main()
