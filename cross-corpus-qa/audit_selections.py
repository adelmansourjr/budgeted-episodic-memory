"""Check the frozen question selection and evidence accounting without a model."""

import argparse
import json
from pathlib import Path

import run as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = study.load(args.source)
    expected = study.representatives(source)[8:200]
    policy = json.loads((study.PARENT / "FROZEN.json").read_text())
    result_dir = study.ROOT / "results"
    archived = [
        json.loads(line) for line in (result_dir / "raw.jsonl").read_text().splitlines()
    ]
    assert len(archived) == len(expected) == 192

    physical_calls = 0
    support_titles = set()
    for (group, size, row), outcome in zip(expected, archived, strict=True):
        assert (outcome["group"], outcome["group_size"], outcome["id"]) == (
            group, size, row["_id"]
        )
        docs, selected = study.prepare(row, policy)
        title_keys = {
            tuple(study.retrieval.words(title)) for title, _ in row["supporting_facts"]
        }
        assert title_keys.isdisjoint(support_titles)
        support_titles.update(title_keys)
        gold = {title for title, _ in row["supporting_facts"]}
        prompts_by_evidence = {}
        for method, record in outcome["arms"].items():
            ids, spent, complete = selected[method]
            assert record["ids"] == ids and record["spent"] == spent
            if method != "closed_book":
                assert gold.issubset(docs[i][0] for i in ids) == bool(complete)
            assert record["complete"] == complete
            assert record["gold_terms"] == len(study.normalize(row["answer"]).split())
            assert 0 <= record["generation_tokens"] <= 20
            prompt = tuple(record[field] for field in (
                "prompt_sha256", "prompt_tokens", "prediction_sha256", "generation_tokens"
            ))
            assert prompts_by_evidence.setdefault(tuple(ids), prompt) == prompt
        physical_calls += len({p[0] for p in prompts_by_evidence.values()})
    summary = json.loads((result_dir / "summary.json").read_text())
    assert physical_calls == summary["physical_calls"]
    print(f"Recomputed {len(archived)} fixed questions and all evidence selections.")
    print("Verified independent support-title groups and exact within-item prompt reuse.")


if __name__ == "__main__":
    main()
