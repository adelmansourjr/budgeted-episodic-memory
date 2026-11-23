"""Compare frozen and adapted MuSiQue answers at exact emitted token counts."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("transfer_scoring", HERE / "score_transfer.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
study = transfer.study
sha = transfer.sha
ARMS = transfer.ARMS
LENGTHS = (2, 4, 6, 8, 12)
PRIMARY = 6
LIMIT = max(LENGTHS)
PANELS = ("transfer", "replication")
SEEDS = {"transfer": 288194, "replication": 288195, "pooled": 288196}
BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def panels(args):
    cases = {
        "transfer": transfer.read_data(args.transfer),
        "replication": json_rows(args.replication / "test.jsonl"),
    }
    prior_results = HERE.parent / "qa-grounding/musique/cases.jsonl"
    earlier = [(r["group"], r["id"]) for r in json_rows(prior_results)]
    assert len(earlier) == 160
    replication = [(r["group"], r["id"]) for r in cases["replication"]]
    assert replication == earlier
    manifest = json.loads((args.replication / "manifest.json").read_text())
    assert manifest["test_sha256"] == sha(args.replication / "test.jsonl")
    assert len({r["group"] for rows in cases.values() for r in rows}) == 329
    assert len({r["id"] for rows in cases.values() for r in rows}) == 329
    for rows in cases.values():
        for case in rows:
            assert 0 < len(case["prompt_ids"]) <= 1024
            assert all(study.qa.normalize(target) for target in (case["answer"], *case["aliases"]))
    return cases


def source_locks(args):
    files = ("EQUAL_DECODE_PROTOCOL.md", "score_equal_decode.py", "prepare_equal_replication.py",
             "prepare_transfer.py", "score_transfer.py", "study.py")
    return dict(
        code={filename: sha(HERE / filename) for filename in files},
        checkpoint=transfer.adapter_fingerprints(),
        panels={name:{field:sha(getattr(args,name)/field) for field in ("manifest.json","test.jsonl")}
                for name in PANELS},
    )


def token_mask(tokenizer, vocab_size):
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    excluded = set(hf.all_special_ids)
    excluded.update(getattr(tokenizer,"eos_token_ids",[]))
    excluded.update(k for k,v in hf.added_tokens_decoder.items() if v.special)
    ids = set(hf.get_vocab().values()) - excluded
    allowed = np.zeros(vocab_size, dtype=bool)
    backend = hf.backend_tokenizer
    for i in ids:
        if 0 <= i < vocab_size:
            text = backend.decode([i], skip_special_tokens=False)
            has_visible = any(char.isprintable() and not char.isspace() for char in text)
            if has_visible and BREAKS.isdisjoint(text):
                allowed[i] = True
    if not allowed.any():
        raise ValueError("No ordinary tokens remain.")
    return allowed


def emit(reader, prompt, allowed, limit, prepared=None):
    from mlx_lm.generate import generate_step

    mx=reader.mx
    if prepared is None:
        tokens=mx.array(prompt,dtype=mx.int32)
        cache=None
    else:
        hidden,lower=prepared
        suffix=reader.make_cache(reader.model)[reader.depth:]
        causal=reader.mask(hidden,suffix[0])
        for block,entry in zip(reader.model.model.layers[reader.depth:],suffix):
            hidden=block(hidden,causal,entry)
        mx.eval([entry.state for entry in suffix])
        cache=lower+suffix
        tokens=mx.array([prompt[-1]],dtype=mx.int32)

    raw_choices=[]
    def constrain(_, logits):
        logits=logits.astype(mx.float32)
        raw_choices.append(mx.argmax(logits,axis=-1).reshape(-1)[0])
        return mx.where(allowed,logits,-mx.inf)

    generated=[token for token,_ in generate_step(tokens,reader.model,max_tokens=limit,
                                                  prompt_cache=cache,sampler=reader.sampler,
                                                  logits_processors=[constrain])]
    assert len(raw_choices)>=limit and len(generated)==limit
    raw=[int(t.item()) for t in raw_choices[:limit]]
    for step,(token,unmasked) in enumerate(zip(generated,raw),1):
        if not bool(allowed[token].item()):
            raise ValueError(f"A forbidden token was emitted at step {step}.")
        if bool(allowed[unmasked].item()) and token!=unmasked:
            raise ValueError("Unexpected difference from the unconstrained greedy choice.")
    if prepared is not None:
        for entry in lower:
            excess=entry.offset-(len(prompt)-1)
            if excess<0 or entry.trim(excess)!=excess:
                raise ValueError("Unable to restore the shared prefix cache.")
    first_blocked=next((i for i,(actual,unmasked) in enumerate(zip(generated,raw),1) if actual!=unmasked),None)
    return generated,first_blocked


def whole_prefix(tokenizer, ids, case):
    answer = tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if not BREAKS.isdisjoint(answer):
        raise ValueError("A single-line response contains a line break.")
    normalized=study.qa.normalize(answer)
    tokens=Counter(normalized.split())
    measures=[]
    for target in (case["answer"],*case["aliases"]):
        gold=study.qa.normalize(target)
        reference=Counter(gold.split())
        overlap=sum((tokens&reference).values())
        f1=2*overlap/(sum(tokens.values())+sum(reference.values()))
        measures.append(dict(f1=f1,exact=int(normalized==gold),overlap=overlap,
                             answer_words=sum(tokens.values()),gold_words=sum(reference.values())))
    best=max(measures,key=lambda x:(x["f1"],x["exact"]))
    return dict(best,exact=max(x["exact"] for x in measures))


def score_row(case,tokenizer,sequences,blocked,origin):
    row={k:case[k] for k in ("id","group","type","complete","article_complete")}
    row.update(panel=origin,prompt_tokens=len(case["prompt_ids"]),
               prompt_sha256=study.qa.stable(",".join(map(str,case["prompt_ids"]))),arms={})
    preblock=min((b-1 if b is not None else LIMIT) for b in blocked.values())
    row["common_unforced_prefix_tokens"]=preblock
    for name,ids in sequences.items():
        assert len(ids)==LIMIT
        row["arms"][name]=dict(
            emitted_tokens=len(ids),
            sequence_sha256=study.qa.stable(",".join(map(str,ids))),
            first_blocked_step=blocked[name],
            budgets={str(k):whole_prefix(tokenizer,ids[:k],case) for k in LENGTHS},
            common_unforced_prefix=whole_prefix(tokenizer,ids[:preblock],case),
        )
    return row


def summarize(rows):
    by_panel={name:[row for row in rows if row["panel"]==name] for name in PANELS}
    report={}
    for panel,group in (*by_panel.items(),("pooled",rows)):
        values=np.array([
            [[row["arms"][arm]["budgets"][str(k)]["f1"] for k in LENGTHS] for arm in ARMS]
            for row in group
        ])
        delta=values[:,1,:]-values[:,0,:]
        exact=np.array([[row["arms"][a]["budgets"][str(PRIMARY)]["exact"] for a in ARMS] for row in group])
        preblock_delta=np.array([row["arms"][ARMS[1]]["common_unforced_prefix"]["f1"]-
                                 row["arms"][ARMS[0]]["common_unforced_prefix"]["f1"] for row in group])
        seeds=np.random.default_rng(SEEDS[panel])
        matrices=[]
        if panel=="pooled":
            strata=[np.array([i for i,row in enumerate(group) if row["panel"]==p]) for p in PANELS]
        else:
            strata=[np.arange(len(group))]
        joined=np.column_stack((delta,exact[:,1]-exact[:,0],preblock_delta))
        for offset in range(0,20000,500):
            n=min(500,20000-offset)
            total=np.zeros((n,joined.shape[1]))
            for s in strata:
                pick=s[seeds.integers(len(s),size=(n,len(s)))]
                total+=joined[pick].sum(axis=1)
            matrices.append(total/len(group))
        draws=np.concatenate(matrices)
        means=joined.mean(axis=0)
        intervals=np.quantile(draws,[.025,.975],axis=0)
        half=float(np.quantile(np.max(np.abs(draws[:,:len(LENGTHS)]-means[:len(LENGTHS)]),axis=1),.95))
        curve={}
        for i,k in enumerate(LENGTHS):
            curve[str(k)]=dict(f1={arm:float(values[:,j,i].mean()) for j,arm in enumerate(ARMS)},
                              delta=float(means[i]),ci=intervals[:,i].tolist(),
                              simultaneous_ci=[float(means[i]-half),float(means[i]+half)])
        p=curve[str(PRIMARY)]
        d=delta[:,LENGTHS.index(PRIMARY)]
        report[panel]=dict(
            count=len(group),curve=curve,
            passes=bool(p["delta"]>=.03 and p["ci"][0]>0),
            exact_match={a:float(exact[:,i].mean()) for i,a in enumerate(ARMS)},
            exact_delta=[float(means[-2]),*intervals[:,-2].tolist()],
            wins_ties_losses=[int((d>0).sum()),int((d==0).sum()),int((d<0).sum())],
            blocked_by_six={arm:sum(
                row["arms"][arm]["first_blocked_step"] is not None
                and row["arms"][arm]["first_blocked_step"]<=PRIMARY for row in group
            ) for arm in ARMS},
            common_unforced_prefix=dict(mean_tokens=float(np.mean(
                [row["common_unforced_prefix_tokens"] for row in group]
            )),delta=[float(means[-1]),*intervals[:,-1].tolist()]),
        )
    report["replicated"]=bool(report["transfer"]["passes"] and report["replication"]["passes"])
    return report


def run(args):
    split=panels(args)
    locked=source_locks(args)
    transfer.validate_model(args.model)
    dev=study.cases(args.hotpot_data,"dev")
    trained=json.loads((HERE/"adapter/training.json").read_text())
    assert sha(args.hotpot_data/"dev.jsonl")==trained["inputs"]["prepared"]["dev.jsonl"]
    args.output.mkdir()
    reader,frozen=study.load_reader(args.model,dev[0]["prompt_ids"])
    tensor=reader.mx.load(str(HERE/"adapter/question_last.safetensors"))
    assert set(tensor)==set(frozen)
    states={"frozen":frozen,"question_last_adapter":tensor}
    allowed_np=token_mask(reader.tokenizer,reader.model.args.vocab_size)
    allowed=reader.mx.array(allowed_np)
    cached=True
    for row in dev[:4]:
        prompt=row["prompt_ids"]
        ready=reader.prefill(prompt)
        for state in states.values():
            reader.set_weights(state)
            ordinary=emit(reader,prompt,allowed,LIMIT)
            cutoff=emit(reader,prompt,allowed,PRIMARY)
            assert ordinary[0][:PRIMARY]==cutoff[0]
            cached &= (ordinary==emit(reader,prompt,allowed,LIMIT,ready))
    print("exact six-token standalone replay true; exact shared-prefix replay",cached,flush=True)
    started=time.perf_counter()
    ledger=[]
    main_log=args.output/"cases.jsonl"
    token_log=args.output/"private_tokens.jsonl"
    with main_log.open("x") as public,token_log.open("x") as private:
        for name,rows in split.items():
            for i,case in enumerate(rows,1):
                prompt=case["prompt_ids"]
                ready=reader.prefill(prompt) if cached else None
                sequences={}
                blocked={}
                for arm in sorted(ARMS,key=lambda key:study.qa.stable(case["group"]+key)):
                    reader.set_weights(states[arm])
                    sequences[arm],blocked[arm]=emit(reader,prompt,allowed,LIMIT,ready)
                row=score_row(case,reader.tokenizer,sequences,blocked,name)
                ledger.append(row)
                public.write(json.dumps(row,sort_keys=True)+"\n")
                private.write(json.dumps(dict(panel=name,id=case["id"],tokens=sequences,first_blocked=blocked),sort_keys=True)+"\n")
                public.flush()
                private.flush()
                if i%16==0 or i==len(rows):
                    print(name,i,"/",len(rows),"seconds",round(time.perf_counter()-started),flush=True)
    result=dict(analysis=summarize(ledger),paired_cases=len(ledger),lengths=list(LENGTHS),
                primary_tokens=PRIMARY,emitted_tokens_per_arm=LIMIT,
                validated_prefix=cached,validated_standalone_six=True,frozen_logits_verified=True,
                allowed_token_ids=int(allowed_np.sum()),allowed_mask_sha256=study.qa.stable(
                    ",".join(map(str,np.flatnonzero(allowed_np).tolist()))),
                locks=locked,ledger_sha256=sha(main_log),private_tokens_sha256=sha(token_log),
                seconds=time.perf_counter()-started,peak_mlx_bytes=reader.mx.get_peak_memory(),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    study.ground.write(args.output/"summary.json",result)
    print(json.dumps(result,indent=2),flush=True)


def verify(args):
    split=panels(args)
    records=json_rows(args.output/"cases.jsonl")
    report=json.loads((args.output/"summary.json").read_text())
    assert report["locks"]==source_locks(args) and report["ledger_sha256"]==sha(args.output/"cases.jsonl")
    assert report["analysis"]==summarize(records)
    n=sum(len(rows) for rows in split.values())
    assert n==report["paired_cases"]==len(records)
    prepared=[(name,case) for name,rows in split.items() for case in rows]
    for (name,case),result in zip(prepared,records,strict=True):
        assert name==result["panel"]
        assert all(result[k]==case[k] for k in ("id","group","type","complete","article_complete"))
        assert result["prompt_tokens"]==len(case["prompt_ids"])
        assert result["prompt_sha256"]==study.qa.stable(",".join(map(str,case["prompt_ids"])))
        for a in result["arms"].values():
            assert a["emitted_tokens"]==LIMIT
            for scores in a["budgets"].values():
                assert scores["f1"]==2*scores["overlap"]/(scores["answer_words"]+scores["gold_words"])
    if args.private_tokens:
        assert sha(args.private_tokens)==report["private_tokens_sha256"]
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
        mask_size=json.loads((args.model/"config.json").read_text())["vocab_size"]
        allowed=token_mask(tokenizer,mask_size)
        assert report["allowed_mask_sha256"]==study.qa.stable(",".join(map(str,np.flatnonzero(allowed).tolist())))
        for (name,case),row,trace in zip(prepared,records,json_rows(args.private_tokens),strict=True):
            assert trace["panel"]==name and trace["id"]==case["id"]
            assert all(len(ids)==LIMIT and all(allowed[i] for i in ids) for ids in trace["tokens"].values())
            assert row==score_row(case,tokenizer,trace["tokens"],trace["first_blocked"],name)
    print(f"Verified {n} pairs, equal prompt and generated token counts, score components and bootstrap results.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    phases=p.add_subparsers(dest="mode",required=True)
    for mode in ("score","verify"):
        c=phases.add_parser(mode)
        c.add_argument("--transfer",type=Path,required=True)
        c.add_argument("--replication",type=Path,required=True)
        c.add_argument("--output",type=Path,required=True)
        c.add_argument("--model",type=Path,required=mode=="score")
        if mode=="score":
            c.add_argument("--hotpot-data",type=Path,required=True)
        else:
            c.add_argument("--private-tokens",type=Path)
    args=p.parse_args()
    if args.mode=="verify" and args.private_tokens and not args.model:
        p.error("--private-tokens requires --model for exact decoding")
    (run if args.mode=="score" else verify)(args)


if __name__=="__main__":
    main()
