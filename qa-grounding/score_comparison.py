"""Evaluate the prespecified comparison-question panel with matched token counts."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("grounding_study", ROOT/"study.py")
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
READER_ARMS=(*study.ARMS, "frozen_first")


def summarize(rows):
    result=study.summarize(rows)
    delta=np.array([r["arms"]["frozen"]["f1"]-r["arms"]["frozen_first"]["f1"] for r in rows])
    draws=np.random.default_rng(95531).integers(len(rows),size=(20000,len(rows)))
    result["question_last_vs_first_f1"]=[
        float(delta.mean()),*np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist()
    ]
    result["frozen_first"]={
        field:float(np.mean([row["arms"]["frozen_first"][field] for row in rows]))
        for field in ("f1","exact","generation_tokens")
    }
    return result


def consume(responses):
    pieces=[]
    produced=0
    try:
        for response in responses:
            pieces.append(response.text)
            produced=response.generation_tokens
            if "\n" in "".join(pieces).strip():
                break
    finally:
        responses.close()
    return "".join(pieces),produced


def run(args):
    from mlx_lm.models.cache import make_prompt_cache

    mx,_,_,flatten,_,unflatten,load,stream,mask,sampler_fn,make_lora=study.gpu()
    training_path=args.adapters/"training.json"
    training=json.loads(training_path.read_text())
    assert training["inputs"]==study.lock_inputs(args.model,args.hotpot_data)
    assert all(study.qa.sha(args.adapters/(k+".safetensors"))==v for k,v in training["adapters"].items())
    source=args.data/"test.jsonl"
    panel=[json.loads(line) for line in source.read_text().splitlines()]
    assert len(panel)==len({r["group"] for r in panel})==192
    manifest=json.loads((args.data/"manifest.json").read_text())
    assert manifest["test_sha256"]==study.qa.sha(source)
    dev=study.data_file(args.hotpot_data,"dev")
    args.output.mkdir()
    model,tokenizer=load(str(args.model))
    model.freeze()
    make_lora(model,study.LAYERS,study.LORA)
    states={"frozen":dict(flatten(model.trainable_parameters())),
            **{name:mx.load(str(args.adapters/(name+".safetensors"))) for name in study.ADAPTED}}
    assert all(set(state)==set(states["frozen"]) for state in states.values())
    assert all(not bool(mx.any(v!=0).item()) for k,v in states["frozen"].items() if k.endswith("lora_b"))
    sampler=sampler_fn(temp=0.0)
    shared_depth=len(model.model.layers)-study.LAYERS

    def ordinary(prompt):
        return consume(stream(model,tokenizer,prompt,max_tokens=study.MAX_ANSWER,sampler=sampler))

    def prefill(prompt):
        caches=make_prompt_cache(model)
        prefix=mx.array([prompt[:-1]],dtype=mx.int32)
        h=model.model.embed_tokens(prefix)
        causal=mask(h,caches[0])
        lower=caches[:shared_depth]
        for block,cache in zip(model.model.layers[:shared_depth],lower):
            h=block(h,causal,cache)
        mx.eval(h,[cache.state for cache in lower])
        return h,lower

    def cached(prompt,prepared):
        h,lower=prepared
        suffix=make_prompt_cache(model)[shared_depth:]
        causal=mask(h,suffix[0])
        for block,cache in zip(model.model.layers[shared_depth:],suffix):
            h=block(h,causal,cache)
        mx.eval([cache.state for cache in suffix])
        caches=lower+suffix
        value=consume(stream(model,tokenizer,[prompt[-1]],max_tokens=study.MAX_ANSWER,
                             sampler=sampler,prompt_cache=caches))
        end=len(prompt)-1
        for cache in lower:
            extra=cache.offset-end
            if extra<0 or cache.trim(extra)!=extra:
                raise ValueError("Unable to restore the common prompt prefix")
        return value

    exact=True
    for row in dev[:6]:
        prompt=row["prompt_ids"]
        context=prefill(prompt)
        for name in (*study.ADAPTED, "frozen"):
            model.update(unflatten(list(states[name].items())))
            exact &= ordinary(prompt)==cached(prompt,context)
    print("development prefix replay", "exact" if exact else "fallback to standard",flush=True)

    ledger=[]
    started=time.perf_counter()
    with (args.output/"cases.jsonl").open("w") as f:
        for i,row in enumerate(panel,1):
            prompt=row["prompt_ids"]
            later=row["question_last_ids"]
            assert len(prompt)==len(later)<=1024 and Counter(prompt)==Counter(later)
            assert row["type"]=="comparison"
            context=prefill(prompt) if exact else None
            item={key:row[key] for key in ("id","group","type","complete","article_complete")}
            item.update(prompt_tokens=len(prompt),
                        prompt_sha256=study.qa.stable(",".join(map(str,prompt))),
                        frozen_prompt_sha256=study.qa.stable(",".join(map(str,later))),
                        arms={})
            for name in sorted(READER_ARMS,key=lambda mode:study.qa.stable(row["group"]+mode)):
                weights = states["frozen"] if name == "frozen_first" else states[name]
                model.update(unflatten(list(weights.items())))
                if name=="frozen":
                    out,tokens=ordinary(later)
                elif exact:
                    out,tokens=cached(prompt,context)
                else:
                    out,tokens=ordinary(prompt)
                item["arms"][name]=dict(study.qa.answer_score(out,row["answer"]),generation_tokens=tokens)
            ledger.append(item)
            f.write(json.dumps(item,sort_keys=True)+"\n")
            f.flush()
            if i%16==0:
                print("comparison",i,"/192",round(time.perf_counter()-started),flush=True)
    summary=dict(analysis=summarize(ledger),training_sha256=study.qa.sha(training_path),
                 result_sha256=study.qa.sha(args.output/"cases.jsonl"),
                 prepared_test_sha256=study.qa.sha(source),
                 preparation_sha256=study.qa.sha(args.data/"manifest.json"),
                 exact_development_prefix_replay=bool(exact),seconds=time.perf_counter()-started,
                 peak_mlx_bytes=mx.get_peak_memory(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 fingerprints={k:study.qa.sha(ROOT/k) for k in
                               ("COMPARISON_PROTOCOL.md","prepare_comparison.py","score_comparison.py")})
    study.write(args.output/"summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


def verify(args):
    summary=json.loads((args.output/"summary.json").read_text())
    inputs=[json.loads(row) for row in (args.data/"test.jsonl").read_text().splitlines()]
    scored=args.output/"cases.jsonl"
    records=[json.loads(row) for row in scored.read_text().splitlines()]
    assert len(records)==len({r["group"] for r in records})==192
    assert summary["result_sha256"]==study.qa.sha(scored)
    assert summary["prepared_test_sha256"]==study.qa.sha(args.data/"test.jsonl")
    assert all(study.qa.sha(ROOT/k)==v for k,v in summary["fingerprints"].items())
    for x,y in zip(inputs,records,strict=True):
        assert (x["id"],x["group"])==(y["id"],y["group"])
        assert Counter(x["prompt_ids"])==Counter(x["question_last_ids"])
        assert len(x["prompt_ids"])==len(x["question_last_ids"])==y["prompt_tokens"]
        for key,field in (("prompt_ids","prompt_sha256"),("question_last_ids","frozen_prompt_sha256")):
            assert y[field]==study.qa.stable(",".join(map(str,x[key])))
        assert all(arm["f1"]==(0 if arm["incompatible"] else 2*arm["overlap"]/max(1,arm["answer_terms"]+arm["gold_terms"]))
                   for arm in y["arms"].values())
    assert all(set(record["arms"])==set(READER_ARMS) for record in records)
    assert summary["analysis"]==summarize(records)
    print("Verified exact token budget and all paired comparison-panel outcomes.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("score","verify"))
    parser.add_argument("--model",type=Path)
    parser.add_argument("--adapters",type=Path)
    parser.add_argument("--hotpot-data",type=Path)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    (run if args.mode=="score" else verify)(args)


if __name__=="__main__":
    main()
