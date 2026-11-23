"""Answer-only QLoRA trained and scored with the established question-last prompt."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"qa-grounding"
SPEC=importlib.util.spec_from_file_location("grounding_baseline", PREVIOUS/"study.py")
ground=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ground)
qa=ground.qa
SEED=1921106
LR=.0002
ACCUM=4
UPDATES=(40,80,160)
GAINS=(.5,1.0)
ARMS=("frozen","question_first_adapter","question_last_adapter")
COUNT={"train":640,"dev":64,"test":320}


def locked_inputs(model,data):
    prior=json.loads((ground.CROSS/"LOCK.json").read_text())
    assert all(qa.sha(ground.CROSS/k)==v for k,v in prior["files"].items())
    assert all(qa.sha(model/k)==v for k,v in prior["model"].items())
    previous=json.loads((PREVIOUS/"adapters/training.json").read_text())
    assert qa.sha(PREVIOUS/"adapters/answer_only.safetensors")==previous["adapters"]["answer_only"]
    return {
        "code":{k:qa.sha(HERE/k) for k in ("PROTOCOL.md","prepare.py","study.py")},
        "prepared":{k:qa.sha(data/k) for k in ("train.jsonl","dev.jsonl","test.jsonl","manifest.json")},
        "base_lock_sha256":qa.sha(ground.CROSS/"LOCK.json"),
        "old_adapter_sha256":previous["adapters"]["answer_only"],
    }


def cases(data,part):
    rows=[json.loads(line) for line in (data/(part+".jsonl")).read_text().splitlines()]
    assert len(rows)==len({row["group"] for row in rows})==COUNT[part]
    assert all(0<len(r["prompt_ids"])<=1024 and r["answer_ids"] for r in rows)
    return rows


def read_output(iterator):
    chunks=[]
    count=0
    try:
        for response in iterator:
            chunks.append(response.text)
            count=response.generation_tokens
            if "\n" in "".join(chunks).strip():
                break
    finally:
        iterator.close()
    return "".join(chunks),count


class Reader:
    """Shared frozen prefill for matched prompts, with a standard inference fallback."""

    def __init__(self, model, tokenizer, tools):
        from mlx_lm.models.cache import make_prompt_cache
        self.model=model
        self.tokenizer=tokenizer
        self.mx,self.nn,self.optim,self.flatten,self.map_tree,self.unflatten,_,self.stream,self.mask,make_sampler,_=tools
        self.make_cache=make_prompt_cache
        self.sampler=make_sampler(temp=0)
        self.depth=len(model.model.layers)-ground.LAYERS

    def weights(self):
        return dict(self.flatten(self.model.trainable_parameters()))

    def set_weights(self,state):
        self.model.update(self.unflatten(list(state.items())))

    def ordinary(self, prompt):
        return read_output(self.stream(self.model,self.tokenizer,prompt,max_tokens=ground.MAX_ANSWER,sampler=self.sampler))

    def prefill(self, prompt):
        lower=self.make_cache(self.model)[:self.depth]
        prefix=self.mx.array([prompt[:-1]],dtype=self.mx.int32)
        h=self.model.model.embed_tokens(prefix)
        causal=self.mask(h,lower[0])
        for block,cache in zip(self.model.model.layers[:self.depth],lower):
            h=block(h,causal,cache)
        self.mx.eval(h,[c.state for c in lower])
        return h,lower

    def cached(self,prompt,prepared):
        h,lower=prepared
        suffix=self.make_cache(self.model)[self.depth:]
        causal=self.mask(h,suffix[0])
        for block,cache in zip(self.model.model.layers[self.depth:],suffix):
            h=block(h,causal,cache)
        self.mx.eval([c.state for c in suffix])
        result=read_output(self.stream(self.model,self.tokenizer,[prompt[-1]],prompt_cache=lower+suffix,
                                      max_tokens=ground.MAX_ANSWER,sampler=self.sampler))
        for cache in lower:
            excess=cache.offset-(len(prompt)-1)
            if excess<0 or cache.trim(excess)!=excess:
                raise ValueError("Failed to restore common prefill")
        return result

    def exact_replay(self,dev,states):
        for row in dev[:4]:
            prompt=row["prompt_ids"]
            common=self.prefill(prompt)
            for state in states:
                self.set_weights(state)
                if self.ordinary(prompt)!=self.cached(prompt,common):
                    return False
        return True


def load_reader(model_path, parity_prompt=None):
    tools=ground.gpu()
    mx,_,_,_,_,_,load,_,_,_,make_lora=tools
    mx.random.seed(SEED)
    model,tokenizer=load(str(model_path))
    def final_logits():
        h=model.model(mx.array([parity_prompt],dtype=mx.int32))[:,-1:,:]
        logits=model.model.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)
        mx.eval(logits)
        return logits

    reference=final_logits() if parity_prompt is not None else None
    model.freeze()
    make_lora(model,ground.LAYERS,ground.LORA)
    if reference is not None:
        assert bool(mx.all(reference==final_logits()).item()), "Zero-gain LoRA altered frozen logits"
        print("frozen logits equal with zero-gain adapters",flush=True)
    reader=Reader(model,tokenizer,tools)
    frozen=reader.weights()
    assert len(frozen)==24 and all("lora_" in k for k in frozen)
    assert all(not bool(mx.any(v!=0).item()) for k,v in frozen.items() if k.endswith("lora_b"))
    return reader,frozen


def scaled(reader,state,gain):
    out={k:(v*gain if k.endswith("lora_b") else v) for k,v in state.items()}
    reader.mx.eval(out)
    return out


def validate(reader, dev, alternatives, cached):
    outcomes={k:[] for k in alternatives}
    for row in dev:
        prompt=row["prompt_ids"]
        prefill=reader.prefill(prompt) if cached else None
        for gain,weights in alternatives.items():
            reader.set_weights(weights)
            answer,gen=reader.cached(prompt,prefill) if cached else reader.ordinary(prompt)
            outcomes[gain].append(dict(qa.answer_score(answer,row["answer"]),generation_tokens=gen))
    return {gain:{key:float(np.mean([r[key] for r in values]))
                  for key in ("f1","exact","generation_tokens")}
            for gain,values in outcomes.items()}


def train(args):
    started=time.perf_counter()
    lock=locked_inputs(args.model,args.data)
    train_rows=cases(args.data,"train")
    dev=cases(args.data,"dev")
    assert {r["group"] for r in train_rows}.isdisjoint(r["group"] for r in dev)
    args.output.mkdir()
    ground.write(args.output/"inputs.json",lock)
    reader,frozen=load_reader(args.model)
    model=reader.model
    mx,nn,optim=reader.mx,reader.nn,reader.optim
    optimizer=optim.AdamW(learning_rate=LR,weight_decay=.01)

    def loss(m,h,start,answer):
        attention_mask=reader.mask(h,None)
        for block in m.model.layers[reader.depth:]:
            h=block(h,attention_mask,None)
        hidden=m.model.norm(h)[:,start:start+answer.shape[1]]
        logits=m.model.embed_tokens.as_linear(hidden) if m.args.tie_word_embeddings else m.lm_head(hidden)
        return nn.losses.cross_entropy(logits.astype(mx.float32),answer,reduction="mean")

    def batch(row):
        prompt=row["prompt_ids"]
        target=[*row["answer_ids"],reader.tokenizer.eos_token_id][:ground.MAX_ANSWER]
        x=mx.array([prompt+target[:-1]],dtype=mx.int32)
        h=model.model.embed_tokens(x)
        causal=reader.mask(h,None)
        for block in model.model.layers[:reader.depth]:
            h=block(h,causal,None)
        mx.eval(h)
        return h,len(prompt)-1,mx.array([target],dtype=mx.int32)

    vg=nn.value_and_grad(model,loss)
    sequence=np.random.default_rng(SEED).permutation(len(train_rows))
    cached=False
    baseline=None
    chosen=None
    dev_log=[]
    train_log=[]
    acc=None
    n=args.smoke or len(sequence)
    for i,j in enumerate(sequence[:n],1):
        result,grad=vg(model,*batch(train_rows[int(j)]))
        acc=grad if acc is None else reader.map_tree(lambda a,b:a+b,acc,grad)
        value=float(result.item())
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite answer loss")
        train_log.append(value)
        if i%ACCUM==0:
            mean_grad=reader.map_tree(lambda t:t/ACCUM,acc)
            clipped,norm=optim.clip_grad_norm(mean_grad,1.)
            optimizer.update(model,clipped)
            mx.eval(model.trainable_parameters(),optimizer.state,norm)
            if not math.isfinite(float(norm.item())):
                raise FloatingPointError("Non-finite gradient norm")
            acc=None
        else:
            mx.eval(acc)
        if mx.get_peak_memory()>5*1024**3:
            raise MemoryError("Exceeded the 5 GiB MLX allocation budget")
        if i%32==0 or i==n:
            print("training",i,"of",n,"loss",round(float(np.mean(train_log[-32:])),4),
                  "seconds",round(time.perf_counter()-started,1),"peak_mib",round(mx.get_peak_memory()/1024**2),flush=True)
        step=i//ACCUM
        if not args.smoke and i%ACCUM==0 and step in UPDATES:
            working=reader.weights()
            alternatives={g:scaled(reader,working,g) for g in GAINS}
            if baseline is None:
                cached=reader.exact_replay(dev,[frozen,alternatives[1.0]])
                alternatives={0.0:frozen,**alternatives}
            report=validate(reader,dev,alternatives,cached)
            for gain,data in report.items():
                rec=dict(updates=step,gain=gain,**data)
                if gain==0.0:
                    baseline=data
                    continue
                dev_log.append(rec)
                key=lambda r:(r["f1"],-r["updates"],-r["gain"])
                if chosen is None or key(rec)>key(chosen):
                    chosen=rec
                    mx.save_safetensors(str(args.output/"question_last.safetensors"),alternatives[gain])
            print("development",step,report,"shared_prefix",cached,flush=True)
            reader.set_weights(working)
    if args.smoke:
        print("engineering smoke passed",flush=True)
        return
    result=dict(inputs=lock,learning_rate=LR,examples=len(train_rows),updates=len(train_rows)//ACCUM,
                seed=SEED,trainable_parameters=int(sum(v.size for v in frozen.values())),
                frozen_development=baseline,development=dev_log,selected=chosen,train_nll=train_log,
                training_cache_equivalent=cached,adapter_sha256=qa.sha(args.output/"question_last.safetensors"),
                seconds=time.perf_counter()-started,peak_mlx_bytes=mx.get_peak_memory(),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    ground.write(args.output/"training.json",result)


def summarize(rows):
    metrics={}
    for arm in ARMS:
        records=[r["arms"][arm] for r in rows]
        metrics[arm]={key:float(np.mean([r[key] for r in records])) for key in ("f1","exact","generation_tokens")}
        metrics[arm]["prompt_tokens"]=float(np.mean([r["prompt_tokens"] for r in rows]))
        metrics[arm]["at_generation_cap"]=sum(r["generation_tokens"]>=ground.MAX_ANSWER for r in records)
    draws=np.random.default_rng(53191277).integers(len(rows),size=(20_000,len(rows)))
    comparisons={}
    for a,b in (("question_last_adapter","frozen"),("question_last_adapter","question_first_adapter"),
                ("question_first_adapter","frozen")):
        values={}
        for field in ("f1","exact","generation_tokens"):
            delta=np.array([r["arms"][a][field]-r["arms"][b][field] for r in rows])
            values[field]=[float(delta.mean()),*np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist()]
            if field=="f1":
                values["wins_ties_losses"]=[int((delta>0).sum()),int((delta==0).sum()),int((delta<0).sum())]
        comparisons[f"{a}_vs_{b}"]=values
    point,lo,_=comparisons["question_last_adapter_vs_frozen"]["f1"]
    return dict(arms=metrics,paired=comparisons,passes=bool(point>=.03 and lo>0))


def score(args):
    t0=time.perf_counter()
    training_file=args.adapter/"training.json"
    training=json.loads(training_file.read_text())
    assert training["inputs"]==locked_inputs(args.model,args.data)
    assert training["adapter_sha256"]==qa.sha(args.adapter/"question_last.safetensors")
    test=cases(args.data,"test")
    dev=cases(args.data,"dev")
    assert {r["group"] for r in test}.isdisjoint(r["group"] for r in dev)
    args.output.mkdir()
    reader,frozen=load_reader(args.model, dev[0]["prompt_ids"])
    states={"frozen":frozen,
            "question_first_adapter":reader.mx.load(str(PREVIOUS/"adapters/answer_only.safetensors")),
            "question_last_adapter":reader.mx.load(str(args.adapter/"question_last.safetensors"))}
    assert all(set(v)==set(frozen) for v in states.values())
    cached=reader.exact_replay(dev,list(states.values()))
    print("development exact prefill replay",cached,flush=True)
    ledger=[]
    with (args.output/"cases.jsonl").open("w") as stream:
        for i,r in enumerate(test,1):
            prompt=r["prompt_ids"]
            prepared=reader.prefill(prompt) if cached else None
            item={k:r[k] for k in ("id","group","type","complete","article_complete")}
            item.update(prompt_tokens=len(prompt),prompt_sha256=qa.stable(",".join(map(str,prompt))),arms={})
            for name in sorted(ARMS,key=lambda mode:qa.stable(r["group"]+mode)):
                reader.set_weights(states[name])
                output,tokens=reader.cached(prompt,prepared) if cached else reader.ordinary(prompt)
                item["arms"][name]=dict(qa.answer_score(output,r["answer"]),generation_tokens=tokens)
            ledger.append(item)
            stream.write(json.dumps(item,sort_keys=True)+"\n")
            stream.flush()
            if i%16==0:
                print("scored",i,"/",len(test),"seconds",round(time.perf_counter()-t0),flush=True)
    result=dict(analysis=summarize(ledger),test_count=len(ledger),max_output_tokens=ground.MAX_ANSWER,
                test_sha256=qa.sha(args.output/"cases.jsonl"),training_sha256=qa.sha(training_file),
                prepared_test_sha256=qa.sha(args.data/"test.jsonl"),exact_prefill_replay=cached,frozen_logits_verified=True,
                seconds=time.perf_counter()-t0,peak_mlx_bytes=reader.mx.get_peak_memory(),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    ground.write(args.output/"summary.json",result)
    print(json.dumps(result,indent=2),flush=True)


def verify(args):
    r=args.output
    summary=json.loads((r/"summary.json").read_text())
    rows=[json.loads(line) for line in (r/"cases.jsonl").read_text().splitlines()]
    record=args.adapter/"training.json"
    fitted=json.loads(record.read_text())
    assert fitted["adapter_sha256"]==qa.sha(args.adapter/"question_last.safetensors")
    assert fitted["inputs"]["old_adapter_sha256"]==qa.sha(PREVIOUS/"adapters/answer_only.safetensors")
    assert all(qa.sha(HERE/k)==v for k,v in fitted["inputs"]["code"].items())
    assert summary["training_sha256"]==qa.sha(record) and summary["test_sha256"]==qa.sha(r/"cases.jsonl")
    assert len(rows)==summary["test_count"]==len({row["group"] for row in rows})==COUNT["test"]
    assert summarize(rows)==summary["analysis"]
    if args.data:
        source=cases(args.data,"test")
        assert qa.sha(args.data/"test.jsonl")==summary["prepared_test_sha256"]==fitted["inputs"]["prepared"]["test.jsonl"]
        for row,original in zip(rows,source,strict=True):
            assert all(row[k]==original[k] for k in ("id","group","type","complete","article_complete"))
            ids=original["prompt_ids"]
            assert row["prompt_sha256"]==qa.stable(",".join(map(str,ids))) and row["prompt_tokens"]==len(ids)
    for row in rows:
        assert set(row["arms"])==set(ARMS) and 0<row["prompt_tokens"]<=1024
        for a in row["arms"].values():
            assert 0<=a["generation_tokens"]<=ground.MAX_ANSWER
            assert a["f1"]==(0 if a["incompatible"] else 2*a["overlap"]/max(1,a["answer_terms"]+a["gold_terms"]))
    print("Verified 320 case pairs, exact prompt budgets and all paired intervals.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    for name in ("train","score","verify"):
        cmd=sub.add_parser(name)
        cmd.add_argument("--output",type=Path,required=True)
        cmd.add_argument("--data",type=Path,required=name!="verify")
        if name!="verify":
            cmd.add_argument("--model",type=Path,required=True)
        if name=="train":
            cmd.add_argument("--smoke",type=int)
        else:
            cmd.add_argument("--adapter",type=Path,required=True)
    args=parser.parse_args()
    {"train":train,"score":score,"verify":verify}[args.action](args)


if __name__=="__main__":
    main()
