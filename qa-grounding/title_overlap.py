"""Test how exposed distractor titles affect the numeric HotpotQA contrasts."""

import argparse
import importlib.util
import json
from pathlib import Path

import prepare_data as prep

PATH=Path(__file__).with_name("study.py")
SPEC=importlib.util.spec_from_file_location("grounding_study",PATH)
study=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def normalized(title, retrieval):
    return " ".join(retrieval.words(title))


def exposed_titles(source, locked_input, retrieval, config):
    shown=set()
    for src, line in zip(source, locked_input.read_text().splitlines(), strict=True):
        record=json.loads(line)
        group, _, row=src
        assert row["_id"]==record["id"] and group==record["group"]
        sentences,_=prep.selected_sentences(row,retrieval,config)
        for title, _, _ in sentences[:len(record["positions"])]:
            shown.add(normalized(title,retrieval))
    return shown


def evaluate(args):
    qa=prep.load_prior()
    group_list=qa.representatives(qa.load(args.source))
    config=json.loads((prep.HERE.parent/"FROZEN.json").read_text())["selection"]["reveal_bridge"]["config"]
    titles=set()
    for name in ("train","dev"):
        start,end=prep.SPLITS[name]
        titles|=exposed_titles(group_list[start:end],args.prepared/(name+".jsonl"),qa.retrieval,config)
    start,end=prep.SPLITS["test"]
    exposed_groups={
        group for group, _, row in group_list[start:end]
        if any(normalized(title,qa.retrieval) in titles for title,_ in row["supporting_facts"])
    }
    print(f"{len(exposed_groups)} final support groups overlap an acquired Hotpot training or development title.")
    if not args.scores:
        return
    rows=[json.loads(line) for line in (args.scores/"cases.jsonl").read_text().splitlines()]
    retained=[r for r in rows if r["group"] not in exposed_groups]
    assert len(rows)==320
    report={"excluded_ids":[r["id"] for r in rows if r["group"] in exposed_groups],
            "retained_count":len(retained),"descriptive_analysis":study.summarize(retained)}
    if args.report:
        with args.report.open("x") as stream:
            json.dump(report,stream,indent=2)
    print(json.dumps(report["descriptive_analysis"]["paired"]["support_vs_answer_only"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("prepared",type=Path)
    parser.add_argument("--scores",type=Path)
    parser.add_argument("--report",type=Path)
    evaluate(parser.parse_args())


if __name__=="__main__":
    main()
