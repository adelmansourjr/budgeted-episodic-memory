"""Replay whole-article masks and token accounting without loading weights."""

import argparse
import importlib.util
import json
from pathlib import Path

from transformers import AutoTokenizer

path = Path(__file__).with_name("study.py")
spec = importlib.util.spec_from_file_location("article_study", path)
article = importlib.util.module_from_spec(spec)
spec.loader.exec_module(article)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    method = article.sentence
    prior = json.loads((method.OLD / "LOCK.json").read_text())
    assert all(article.qa.sha(method.OLD / k) == v for k, v in prior["files"].items())
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    retrieval = json.loads((article.ROOT.parent / "FROZEN.json").read_text())
    classifier = json.loads((article.BASE / "FIT.json").read_text())
    fitted = json.loads((article.ROOT / "FIT.json").read_text())
    groups = article.qa.representatives(article.qa.load(args.data))
    used = {
        tuple(method.qa.retrieval.words(t))
        for _, _, row in groups[:3076] for t, _ in row["supporting_facts"]
    }
    archived = [
        json.loads(line) for line in (article.ROOT / "results/rows.jsonl").read_text().splitlines()
    ]
    for (group, _, row), saved in zip(groups[3076:3268], archived, strict=True):
        assert group == saved["group"] and row["_id"] == saved["id"]
        gold = {tuple(pair) for pair in row["supporting_facts"]}
        gold_titles = {tuple(method.qa.retrieval.words(t)) for t, _ in gold}
        assert gold_titles.isdisjoint(used)
        used.update(gold_titles)
        triples, index, features, _ = article.prepare(row, retrieval, classifier)
        probs = method.scores(features, fitted["fitted"])
        for arm in article.KINDS:
            keep = article.selected(index, probs, arm, fitted["settings"].get(arm))
            v = saved["arms"][arm]
            assert v["selected"] == keep
            assert all(all(i in keep for i in entries) or all(i not in keep for i in entries)
                       for entries in index.values())
            assert len(method.encode(tokenizer, row, triples, keep)) == v["prompt_tokens"]
            assert all(v[k] == value for k, value in method.evidence(triples, keep, gold).items())
    print(f"Verified {len(archived)} whole-article masks, token counts and disjoint evidence groups.")


if __name__ == "__main__":
    main()
