"""Independently reconcile numeric ledgers with prepared reader inputs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


ARMS=("frozen","answer_only","support")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    with path.open("rb") as fh:
        return hashlib.file_digest(fh,"sha256").hexdigest()


def pair(rows,left,right):
    selected=[(r["arms"][left],r["arms"][right]) for r in rows]
    values=[a["f1"]-b["f1"] for a,b in selected]
    return {
        "groups":len(rows),
        "mean_f1_difference":statistics.mean(values) if rows else None,
        "wins":sum(d>0 for d in values),
        "ties":sum(d==0 for d in values),
        "losses":sum(d<0 for d in values),
        "identical_normalized_answers":sum(a["prediction_sha256"]==b["prediction_sha256"] for a,b in selected),
        "exact_wins":sum(a["exact"]>b["exact"] for a,b in selected),
        "exact_losses":sum(a["exact"]<b["exact"] for a,b in selected),
    }


def run(data,result,training_path):
    test=data/"test.jsonl"
    result_file=result/"cases.jsonl"
    ledger=read_jsonl(result_file)
    prepared=read_jsonl(test)
    report=json.loads((result/"summary.json").read_text())
    training_record=json.loads(training_path.read_text())
    assert digest(training_path)==report["training_sha256"]
    transfer="prepared_test_sha256" in report
    if transfer:
        assert digest(test)==report["prepared_test_sha256"]
        assert digest(data/"manifest.json")==report["preparation_sha256"]
        assert digest(result_file)==report["result_sha256"]
    else:
        assert digest(test)==training_record["inputs"]["data"]["test.jsonl"]
        assert digest(data/"manifest.json")==training_record["inputs"]["data"]["manifest.json"]
        assert digest(result_file)==report["test_sha256"]
    prompt_histogram=Counter()
    for src,r in zip(prepared,ledger,strict=True):
        for key in ("id","group","type","complete","article_complete"):
            assert src[key]==r[key],(r["id"],key)
        tokens=src["prompt_ids"]
        assert len(tokens)==r["prompt_tokens"]
        assert hashlib.sha256(",".join(map(str,tokens)).encode()).hexdigest()==r["prompt_sha256"]
        assert 0<len(tokens)<=1024
        assert set(r["arms"])==set(ARMS)
        prompt_histogram[len(tokens)]+=1
    assert len(ledger)==(160 if transfer else 320)
    groups=set(r["group"] for r in ledger)
    assert len(groups)==len(ledger)
    audits={
        "case_count":len(ledger),
        "ids_unique":len({r["id"] for r in ledger})==len(ledger),
        "prompt_tokens":dict(min=min(prompt_histogram),max=max(prompt_histogram),
                             mean=statistics.mean(r["prompt_tokens"] for r in ledger)),
        "empty_normalized_answers":{a:sum(r["arms"][a]["answer_terms"]==0 for r in ledger) for a in ARMS},
        "mean_answer_terms":{a:statistics.mean(r["arms"][a]["answer_terms"] for r in ledger) for a in ARMS},
        "all":{f"{left}_vs_{right}":pair(ledger,left,right) for left,right in (
            ("support","answer_only"),("support","frozen"),("answer_only","frozen")
        )},
        "conditional_descriptive":{}
    }
    for name,subgroup in (
        ("all_support", [r for r in ledger if r["complete"]]),
        ("missing_support", [r for r in ledger if not r["complete"]]),
        *[(q_type,[r for r in ledger if r["type"]==q_type]) for q_type in sorted({r["type"] for r in ledger})],
    ):
        audits["conditional_descriptive"][name]=pair(subgroup,"support","answer_only")
    return audits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("result",type=Path)
    parser.add_argument("training_record",type=Path)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    audit=run(args.data,args.result,args.training_record)
    result=json.dumps(audit,indent=2)+"\n"
    if args.report:
        with args.report.open("x") as f:
            f.write(result)
    print(f"Reconstructed exact prompt identities and token counts for {audit['case_count']} scored cases.")


if __name__=="__main__":
    main()
