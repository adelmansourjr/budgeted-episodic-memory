"""Replay private equal-length token traces with a local CPU tokenizer."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

import score_equal_decode as experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--transfer",type=Path,required=True)
    parser.add_argument("--replication",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--private-tokens",type=Path,required=True)
    args=parser.parse_args()
    split=experiment.panels(args)
    public=experiment.json_rows(args.output/"cases.jsonl")
    trace=experiment.json_rows(args.private_tokens)
    summary=json.loads((args.output/"summary.json").read_text())
    assert experiment.sha(args.private_tokens)==summary["private_tokens_sha256"]
    assert experiment.sha(args.output/"cases.jsonl")==summary["ledger_sha256"]
    experiment.transfer.validate_model(args.model)
    cfg=json.loads((args.model/"config.json").read_text())
    eos=cfg.get("eos_token_id",[])
    if isinstance(eos,int):
        eos=[eos]
    hf=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
    wrapper=SimpleNamespace(_tokenizer=hf,eos_token_ids=eos or [])
    allowed=experiment.token_mask(wrapper,cfg["vocab_size"])
    kept=[str(i) for i,yes in enumerate(allowed) if yes]
    assert summary["allowed_mask_sha256"]==experiment.study.qa.stable(",".join(kept))
    prepared=[(name,case) for name,rows in split.items() for case in rows]
    for (origin,case),row,data in zip(prepared,public,trace,strict=True):
        assert data["panel"]==origin and data["id"]==case["id"]
        assert set(data["tokens"])==set(experiment.ARMS)
        assert all(len(ids)==experiment.LIMIT and all(allowed[i] for i in ids)
                   for ids in data["tokens"].values())
        assert row==experiment.score_row(case,hf,data["tokens"],data["first_blocked"],origin)
    assert summary["analysis"]==experiment.summarize(public)
    print(f"CPU replay verified {len(public)*2} emitted streams, the vocabulary mask and all prefix scores.")


if __name__=="__main__":
    main()
