"""Replay acquisition and tokenization with the pinned tokenizer, without inference."""

import argparse
import importlib.util
import json
from pathlib import Path

from transformers import AutoTokenizer

module = Path(__file__).with_name("study.py")
spec = importlib.util.spec_from_file_location("sentence_study", module)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def title_keys(rows):
    return {
        tuple(study.qa.retrieval.words(title))
        for _, _, row in rows for title, _ in row["supporting_facts"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    library = study.qa
    parent_lock = json.loads((study.OLD/"LOCK.json").read_text())
    assert all(library.sha(study.OLD/name)==value for name,value in parent_lock["files"].items())
    inputs = library.representatives(library.load(args.data))
    prior, train, dev, test = inputs[:200], inputs[200:1600], inputs[1600:1760], inputs[1760:2016]
    seen = set()
    for split in (prior, train, dev, test):
        titles = title_keys(split)
        assert titles.isdisjoint(seen)
        seen.update(titles)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    lock = json.loads((study.ROOT/"FIT.json").read_text())
    source = json.loads((study.ROOT.parent/"FROZEN.json").read_text())
    scored = [
        json.loads(line) for line in (study.ROOT/"results/rows.jsonl").read_text().splitlines()
    ]
    for (group, _, row), archived in zip(test, scored, strict=True):
        assert row["_id"] == archived["id"] and group == archived["group"]
        triples, words, rarity, matrix = study.candidates(row, source)
        gold, _ = study.labels(triples, row)
        selections = study.format_scores(tokenizer, row, triples, words, rarity,
                                        study.scores(matrix, lock["params"]), lock)
        for arm, (chosen, tokens) in selections.items():
            observed = archived["arms"][arm]
            assert observed["selected"] == chosen
            assert observed["prompt_tokens"] == len(tokens)
            assert observed["prompt_sha256"] == library.stable(",".join(map(str,tokens)))
            for k,v in study.evidence(triples,chosen,gold).items():
                assert observed[k] == v
    print(f"Verified {len(scored)} disjoint support groups and all exact input tokens.")


if __name__=="__main__":
    main()
