"""Apply the three selected readers to MuSiQue without retraining."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import resource
import time

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("grounding_study",ROOT/"study.py")
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def score_aliases(output, canonical, alternates):
    # MuSiQue takes the best match over answer aliases without a special yes/no rule.
    first = next((line.strip() for line in output.splitlines() if line.strip()), "")
    prediction = study.qa.normalize(re.sub(r"^answer\s*:\s*", "", first, flags=re.I))
    terms = Counter(prediction.split())
    variants = []
    for target in (canonical, *alternates):
        gold = study.qa.normalize(target)
        gt = Counter(gold.split())
        overlap = sum((terms & gt).values())
        variants.append({
            "f1": 2 * overlap / max(1, sum(terms.values()) + sum(gt.values())),
            "exact": int(prediction == gold),
            "answer_terms": sum(terms.values()),
            "gold_terms": sum(gt.values()),
            "overlap": overlap,
            "incompatible": False,
            "prediction_sha256": study.qa.stable(prediction),
        })
    best = max(variants, key=lambda x: (x["f1"], x["exact"]))
    return dict(best, exact=max(v["exact"] for v in variants))


def run(args):
    mx,_,_,flatten,_,unflatten,load,stream,_,sampler_fn,make_lora=study.gpu()
    training_path=args.adapters/"training.json"
    training=json.loads(training_path.read_text())
    for key,value in training["inputs"]["sources"].items():
        assert study.qa.sha(ROOT/key)==value
    prior=json.loads((study.CROSS/"LOCK.json").read_text())
    assert all(study.qa.sha(args.model/k)==v for k,v in prior["model"].items())
    assert all(study.qa.sha(args.adapters/(k+".safetensors"))==v for k,v in training["adapters"].items())
    source=args.data/"test.jsonl"
    data=[json.loads(line) for line in source.read_text().splitlines()]
    assert len(data)==len({r["group"] for r in data})==160
    args.output.mkdir()
    model,tokenizer=load(str(args.model))
    model.freeze()
    make_lora(model,study.LAYERS,study.LORA)
    states={"frozen":dict(flatten(model.trainable_parameters())),
            **{a:mx.load(str(args.adapters/(a+".safetensors"))) for a in study.ADAPTED}}
    assert all(set(v)==set(states["frozen"]) for v in states.values())
    assert all(not bool(mx.any(v!=0).item()) for k,v in states["frozen"].items() if k.endswith("lora_b"))
    sampler=sampler_fn(temp=0.0)
    ledgers=[]
    started=time.perf_counter()
    with (args.output/"cases.jsonl").open("w") as f:
        for i,row in enumerate(data,1):
            prompt=row["prompt_ids"]
            assert len(prompt)<=1024
            meta={k:row[k] for k in ("id","group","type","complete","article_complete")}
            meta.update(prompt_tokens=len(prompt),prompt_sha256=study.qa.stable(",".join(map(str,prompt))),arms={})
            for name in sorted(study.ARMS,key=lambda x:study.qa.stable(row["group"]+x)):
                model.update(unflatten(list(states[name].items())))
                parts=[]
                generated=0
                for response in stream(model,tokenizer,prompt,sampler=sampler,max_tokens=study.MAX_ANSWER):
                    parts.append(response.text)
                    generated=response.generation_tokens
                    if "\n" in "".join(parts).strip():
                        break
                meta["arms"][name]=dict(score_aliases("".join(parts),row["answer"],row.get("aliases",row.get("answer_aliases",[]))),
                                        generation_tokens=generated)
            ledgers.append(meta)
            f.write(json.dumps(meta,sort_keys=True)+"\n")
            f.flush()
            if i%16==0:
                print("transfer",i,"/160",round(time.perf_counter()-started),flush=True)
    outputs=dict(analysis=study.summarize(ledgers),preparation_sha256=study.qa.sha(args.data/"manifest.json"),
                 prepared_test_sha256=study.qa.sha(source),training_sha256=study.qa.sha(training_path),
                 seconds=time.perf_counter()-started,peak_mlx_bytes=mx.get_peak_memory(),
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 result_sha256=study.qa.sha(args.output/"cases.jsonl"),
                 configuration_sha256={n:study.qa.sha(ROOT/n) for n in
                                       ("TRANSFER_PROTOCOL.md","prepare_transfer.py","score_transfer.py")})
    study.write(args.output/"summary.json",outputs)
    print(json.dumps(outputs,indent=2),flush=True)


def verify(args):
    test=args.output/"cases.jsonl"
    rows=[json.loads(line) for line in test.read_text().splitlines()]
    summary=json.loads((args.output/"summary.json").read_text())
    assert len(rows)==len({r["group"] for r in rows})==160
    assert all(r["prompt_tokens"]<=1024 for r in rows)
    assert all(study.qa.sha(ROOT/k)==v for k,v in summary["configuration_sha256"].items())
    assert all(
        set(r["arms"])==set(study.ARMS)
        and all(0<=a["generation_tokens"]<=study.MAX_ANSWER
                and a["f1"]==(0 if a["incompatible"] else 2*a["overlap"]/max(1,a["answer_terms"]+a["gold_terms"]))
                for a in r["arms"].values())
        for r in rows
    )
    assert study.qa.sha(test)==summary["result_sha256"]
    assert study.summarize(rows)==summary["analysis"]
    print("Verified 160 transfer cases, input counts, answer scores, and paired intervals.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("score","verify"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",type=Path)
    parser.add_argument("--data",type=Path)
    parser.add_argument("--adapters",type=Path)
    args=parser.parse_args()
    (run if args.mode=="score" else verify)(args)


if __name__=="__main__":
    main()
