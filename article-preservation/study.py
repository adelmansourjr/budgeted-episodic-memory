"""Prospective comparison of whole-article input selection policies."""

import argparse
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/"sentence-bottleneck"
SPEC=importlib.util.spec_from_file_location("sentence_study",BASE/"study.py")
sentence=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sentence)
qa=sentence.qa
KINDS=("full","fixed","adaptive")
THRESHOLDS=(0,.015,.03,.06,.12,.18,.24,.30,1.0)
DEPTHS=(2,3,4,5,6,10)


def prepare(row, retrieval_lock, sentence_lock):
    triples, _, _, features = sentence.candidates(row,retrieval_lock)
    sentence_probs=sentence.scores(features,sentence_lock["params"])
    query=sentence.terms(row["question"])
    ids={}
    for i,(title,_,_) in enumerate(triples):
        ids.setdefault(title,[]).append(i)
    title_keys=list(ids)
    matrix=[]
    gold={title for title,_ in row["supporting_facts"]}
    for title in title_keys:
        group=ids[title]
        p=np.sort(sentence_probs[group])[::-1]
        title_overlap=sentence.terms(title)&query
        matrix.append([p[0],p[:2].sum(),p[:3].sum(),p.mean(),p.sum(),float(len(p)),
                       len(title_overlap),len(title_overlap)/max(1,len(query)),
                       max(features[group,8]),max(features[group,3])])
    return triples, ids, np.asarray(matrix).reshape(-1,10), np.asarray([int(t in gold) for t in ids])


def kept(prob,mode,setting):
    ranking=sorted(range(len(prob)),key=lambda i:(-prob[i],i))
    if mode=="full":
        return ranking
    if mode=="fixed":
        return ranking[:setting]
    n=min(2,len(prob))
    while n<len(prob) and sum(prob[i] for i in ranking[n:])>setting:
        n+=1
    return ranking[:n]


def selected(ids, prob, mode, setting):
    groups=list(ids.values())
    return sorted(i for j in kept(prob,mode,setting) for i in groups[j])


def report(rows):
    estimates={}
    for name in KINDS:
        estimates[name]={k:float(np.mean([r["arms"][name][k] for r in rows]))
                         for k in ("f1","exact","complete","prompt_tokens","articles_in_prompt")}
    sampler=np.random.default_rng(38713266)
    draw=sampler.integers(len(rows),size=(20_000,len(rows)))
    contrast={}
    for new,ref in (("fixed","full"),("adaptive","full"),("adaptive","fixed")):
        summary={}
        for measure in ("f1","exact","complete"):
            delta=np.asarray([r["arms"][new][measure]-r["arms"][ref][measure] for r in rows])
            summary[measure]=[float(delta.mean()), *np.quantile(delta[draw].mean(axis=1),[.025,.975]).tolist()]
        summary["token_reduction"]=1-estimates[new]["prompt_tokens"]/estimates[ref]["prompt_tokens"]
        contrast[f"{new}_vs_{ref}"]=summary
    main=contrast["adaptive_vs_full"]
    return dict(arms=estimates,contrasts=contrast,passes=bool(main["f1"][1]>-.04 and main["token_reduction"]>=.30))


def dev_select(development, fit, tokenizer, retrieval_lock, sentence_lock):
    scores={(m,param): [] for m,params in (("full",(0,)),("fixed",DEPTHS),("adaptive",THRESHOLDS)) for param in params}
    for _,_,row in development:
        trs,ids,x,_=prepare(row,retrieval_lock,sentence_lock)
        probs=sentence.scores(x,fit)
        gold={tuple(pair) for pair in row["supporting_facts"]}
        for (arm,value),ledger in scores.items():
            chosen=selected(ids,probs,arm,value)
            tok=sentence.encode(tokenizer,row,trs,chosen)
            ev=sentence.evidence(trs,chosen,gold)
            ledger.append((len(tok),ev["complete"]))
    dev={k:[float(np.mean([t for t,c in rows])),int(sum(c for t,c in rows))] for k,rows in scores.items()}
    critical=.95*dev["full",0][1]
    best={}
    for arm in KINDS[1:]:
        valid=[(m,s) for m,s in dev if m==arm and dev[m,s][1]>=critical]
        if not valid:
            raise ValueError("No retention-preserving comparator setting")
        best[arm]=min(valid,key=lambda k:(dev[k][0],-dev[k][1],k[1]))[1]
    return best,{f"{m}_{s}":v for (m,s),v in dev.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("fit","score","verify"))
    parser.add_argument("data",type=Path,nargs="?")
    parser.add_argument("model",type=Path,nargs="?")
    parser.add_argument("output",type=Path,nargs="?")
    args=parser.parse_args()
    locked_files=(ROOT/"study.py", ROOT/"PROTOCOL.md", BASE/"study.py", BASE/"FIT.json", sentence.OLD/"LOCK.json")
    fingerprint={str(path.relative_to(ROOT.parent)):qa.sha(path) for path in locked_files}
    frozen_file=ROOT/"FIT.json"
    if args.mode=="verify":
        saved=json.loads(frozen_file.read_text())
        result_dir=ROOT/"results"
        summary=json.loads((result_dir/"summary.json").read_text())
        rows=[json.loads(line) for line in (result_dir/"rows.jsonl").read_text().splitlines()]
        assert saved["fingerprint"]==fingerprint and len(rows)==len({r["group"] for r in rows})==192
        assert report(rows)==summary["outcomes"]
        assert qa.sha(result_dir/"rows.jsonl")==summary["raw_sha256"]
        assert qa.sha(frozen_file)==summary["fit_sha256"]
        for row in rows:
            for arm in KINDS:
                v=row["arms"][arm]
                expected=0 if v["incompatible"] else 2*v["overlap"]/max(1,v["answer_terms"]+v["gold_terms"])
                assert expected==v["f1"]
        print("Verified 192 prospective paired outcomes, frozen parameters and input counts")
        return
    qlock=json.loads((sentence.OLD/"LOCK.json").read_text())
    assert all(qa.sha(args.model/name)==sha for name,sha in qlock["model"].items())
    retrieval_lock=json.loads((ROOT.parent/"FROZEN.json").read_text())
    sentence_lock=json.loads((BASE/"FIT.json").read_text())
    groups=qa.representatives(qa.load(args.data))
    if args.mode=="fit":
        from transformers import AutoTokenizer
        examples=[prepare(row,retrieval_lock,sentence_lock)[2:] for _,_,row in groups[2016:2916]]
        fitted=sentence.fit(examples)
        tokenizer=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
        settings,dev=dev_select(groups[2916:3076],fitted,tokenizer,retrieval_lock,sentence_lock)
        with frozen_file.open("x") as f:
            json.dump(dict(fingerprint=fingerprint,fitted=fitted,settings=settings,dev=dev,
                           dev_fields=["mean_prompt_tokens","complete_support_count"]),f,indent=2)
        print(settings,dev,flush=True)
        return

    import mlx.core as mx
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler

    lock=json.loads(frozen_file.read_text())
    assert lock["fingerprint"]==fingerprint
    sample=groups[3076:3268]
    del groups
    args.output.mkdir()
    mx.set_cache_limit(128*1024**2)
    start=time.perf_counter()
    model,tokenizer=load(str(args.model))
    sampler=make_sampler(temp=0)
    ledger=[]
    calls=0
    with (args.output/"rows.jsonl").open("w") as handle:
        for ix,(group,_,row) in enumerate(sample,1):
            triples,ids,x,_=prepare(row,retrieval_lock,sentence_lock)
            p=sentence.scores(x,lock["fitted"])
            gold={tuple(pair) for pair in row["supporting_facts"]}
            cache={}
            record={"group":group,"id":row["_id"],"arms":{}}
            for arm in sorted(KINDS,key=lambda key:qa.stable(group+key)):
                choose=selected(ids,p,arm,lock["settings"].get(arm))
                tokens=tuple(sentence.encode(tokenizer,row,triples,choose))
                if tokens not in cache:
                    parts=[]
                    out=0
                    for item in stream_generate(model,tokenizer,list(tokens),max_tokens=24,sampler=sampler):
                        parts.append(item.text)
                        out=item.generation_tokens
                        if "\n" in "".join(parts).strip():
                            break
                    cache[tokens]=dict(qa.answer_score("".join(parts),row["answer"]),generation_tokens=out)
                    calls+=1
                evidence=sentence.evidence(triples,choose,gold)
                record["arms"][arm]=dict(cache[tokens],**evidence,prompt_tokens=len(tokens),
                                        articles_in_prompt=len({triples[i][0] for i in choose}),
                                        selected=choose)
            ledger.append(record)
            handle.write(json.dumps(record,sort_keys=True)+"\n")
            handle.flush()
            if ix%16==0:
                print(f"{ix}/192 ({calls} distinct generations)",flush=True)
    summary=dict(outcomes=report(ledger),physical_calls=calls,peak_mlx_bytes=mx.get_peak_memory(),
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 duration_seconds=time.perf_counter()-start,fit_sha256=qa.sha(frozen_file),
                 raw_sha256=qa.sha(args.output/"rows.jsonl"))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
