"""Matched-budget Qwen adaptation with a training-only support objective."""

import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
CROSS=ROOT.parent/"cross-corpus-qa"
SPEC=importlib.util.spec_from_file_location("qa_source",CROSS/"run.py")
qa=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)

SEED=1921106
LAYERS=4
LORA={"rank":8,"scale":2.0,"dropout":0.0,
      "keys":["self_attn.q_proj","self_attn.v_proj","mlp.down_proj"]}
ARMS=("frozen","answer_only","support")
ADAPTED=ARMS[1:]
CHECKPOINTS=(80,160)
ACCUM=4
WEIGHT=.35
LR=.0004
MAX_ANSWER=24
SHAPES={"train":640,"dev":64,"test":320}


def write(path,payload):
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")


def data_file(data,name):
    path=data/(name+".jsonl")
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==SHAPES[name] and len({r["group"] for r in rows})==len(rows)
    for r in rows:
        assert 0<len(r["prompt_ids"])<=1024 and r["answer_ids"]
        assert 0<len(r["positions"])==len(r["support_labels"])
        assert all(0<=n<len(r["prompt_ids"]) for n in r["positions"])
    return rows


def lock_inputs(model,data):
    prior=json.loads((CROSS/"LOCK.json").read_text())
    assert all(qa.sha(CROSS/k)==v for k,v in prior["files"].items())
    assert all(qa.sha(model/k)==v for k,v in prior["model"].items())
    return dict(
        sources={n:qa.sha(ROOT/n) for n in ("study.py","prepare_data.py","PROTOCOL.md")},
        data={n:qa.sha(data/n) for n in ("train.jsonl","dev.jsonl","test.jsonl","manifest.json")},
        base_checkpoint_lock=qa.sha(CROSS/"LOCK.json"),
    )


def gpu():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_lm import load, stream_generate
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers

    mx.set_cache_limit(128*1024**2)
    mx.set_memory_limit(5*1024**3)
    return mx,nn,optim,tree_flatten,tree_map,tree_unflatten,load,stream_generate,create_attention_mask,make_sampler,linear_to_lora_layers


def train(args):
    mx,nn,optim,flatten,map_tree,unflatten,load,_,mask,_,make_lora=gpu()

    class Reader(nn.Module):
        def __init__(self,base):
            super().__init__()
            self.backbone=base
            self.support_head=nn.Linear(base.args.hidden_size,1)

        def __call__(self,initial,start,answers,positions,labels,supervise):
            h=initial
            attention_mask=mask(h,None)
            for block in self.backbone.model.layers[-LAYERS:]:
                h=block(h,attention_mask,None)
            h=self.backbone.model.norm(h)
            answer=h[:,start:start+answers.shape[1]]
            lm=self.backbone
            raw=(lm.model.embed_tokens.as_linear(answer) if lm.args.tie_word_embeddings else lm.lm_head(answer))
            nll=nn.losses.cross_entropy(raw.astype(mx.float32),answers,reduction="mean")
            if not supervise:
                return nll
            logits=self.support_head(h[0,positions]).squeeze(-1).astype(mx.float32)
            losses=mx.logaddexp(mx.zeros_like(logits),logits)-logits*labels
            positive=labels
            negative=1-labels
            # Equal total weight for each class present within a question.
            balance=positive/mx.maximum(positive.sum(),1)+negative/mx.maximum(negative.sum(),1)
            objective=(balance*losses).sum()/mx.maximum((positive.sum()>0)+(negative.sum()>0).astype(mx.float32),1)
            return nll+WEIGHT*objective

    started=time.perf_counter()
    lock=lock_inputs(args.model,args.data)
    training=data_file(args.data,"train")
    development=data_file(args.data,"dev")
    assert {x["group"] for x in training}.isdisjoint(x["group"] for x in development)
    args.output.mkdir()
    write(args.output/"inputs.json",lock)
    mx.random.seed(SEED)
    model,tok=load(str(args.model))
    model.freeze()
    make_lora(model,LAYERS,LORA)
    net=Reader(model)
    state=dict(flatten(net.trainable_parameters()))
    mx.eval(state)
    start_state={arm:dict(state) for arm in ADAPTED}
    current=start_state.copy()
    optimizers={name:optim.AdamW(learning_rate=LR,weight_decay=.01) for name in ADAPTED}
    value_grad=nn.value_and_grad(net,lambda m,*x:m(*x))
    assert all("lora_" in key or key.startswith("support_head.") for key in state)
    trainable=sum(v.size for k,v in state.items() if "lora_" in k)
    grad_sums={}
    metrics={arm:[] for arm in ADAPTED}
    histories=[]
    rng=np.random.default_rng(SEED)

    def batch(row):
        prompt=row["prompt_ids"]
        answer=[*row["answer_ids"],tok.eos_token_id][:MAX_ANSWER]
        tokens=mx.array([prompt+answer[:-1]],dtype=mx.int32)
        start=len(prompt)-1
        h=model.model.embed_tokens(tokens)
        mm=mask(h,None)
        for block in model.model.layers[:-LAYERS]:
            h=block(h,mm,None)
        mx.eval(h)
        return (h,start,mx.array([answer],dtype=mx.int32),
                mx.array(row["positions"],dtype=mx.int32),mx.array(row["support_labels"],dtype=mx.float32))

    def apply(arm):
        net.update(unflatten(list(current[arm].items())))

    def validate(step):
        dev_scores={arm:[] for arm in ADAPTED}
        for row in development:
            args_each=batch(row)
            for arm in ADAPTED:
                apply(arm)
                loss=net(*args_each,False)
                dev_scores[arm].append(float(loss.item()))
        for arm in ADAPTED:
            dev_mean=float(np.mean(dev_scores[arm]))
            metrics[arm].append({"updates":step,"nll":dev_mean})
            if dev_mean<=min(r["nll"] for r in metrics[arm]):
                adapter={k.removeprefix("backbone."):v for k,v in current[arm].items() if k.startswith("backbone.") and "lora_" in k}
                mx.save_safetensors(str(args.output/(arm+".safetensors")),adapter)
            print("development",arm,step,round(dev_mean,5),flush=True)

    indices=rng.permutation(len(training))
    steps=args.smoke if args.smoke else len(training)
    for count,j in enumerate(indices[:steps],start=1):
        x=batch(training[int(j)])
        logs={}
        for arm in ADAPTED:
            apply(arm)
            loss,grads=value_grad(net,*x,arm=="support")
            history=grad_sums.get(arm)
            grads=grads if history is None else map_tree(lambda g,h:g+h,grads,history)
            logs[arm]=float(loss.item())
            if not np.isfinite(logs[arm]):
                raise FloatingPointError("Non-finite training objective")
            if count%ACCUM==0:
                averaged=map_tree(lambda v:v/ACCUM,grads)
                clipped,norm=optim.clip_grad_norm(averaged,1.0)
                optimizers[arm].update(net,clipped)
                mx.eval(net.trainable_parameters(),optimizers[arm].state,norm)
                if not np.isfinite(float(norm.item())):
                    raise FloatingPointError("Non-finite gradients")
                current[arm]=dict(flatten(net.trainable_parameters()))
                grad_sums.pop(arm,None)
            else:
                mx.eval(grads)
                grad_sums[arm]=grads
        histories.append(logs)
        if count%32==0 or count==steps:
            last=histories[-32:]
            print("training",count,{arm:round(float(np.mean([row[arm] for row in last])),4) for arm in ADAPTED},
                  "seconds",round(time.perf_counter()-started,1),"peak_mib",round(mx.get_peak_memory()/1024**2),flush=True)
        if mx.get_peak_memory()>5*1024**3:
            raise MemoryError("Exceeded the fitted 5 GiB MLX allocation guard")
        if not args.smoke and count%ACCUM==0 and count//ACCUM in CHECKPOINTS:
            validate(count//ACCUM)
    if args.smoke:
        print("engineering smoke only, no held-out scoring",flush=True)
        return
    selected={arm:min(metrics[arm],key=lambda r:(r["nll"],-r["updates"])) for arm in ADAPTED}
    result=dict(inputs=lock,seed=SEED,num_train=len(training),trainable_lora_parameters=trainable,
                auxiliary_parameters=int(sum(v.size for k,v in state.items() if k.startswith("support_head."))),
                train_log=histories,validation=metrics,selected=selected,training_seconds=time.perf_counter()-started,
                peak_mlx_bytes=mx.get_peak_memory(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                adapters={n:qa.sha(args.output/(n+".safetensors")) for n in ADAPTED})
    write(args.output/"training.json",result)


def summarize(rows):
    summary={}
    for name in ARMS:
        values=[r["arms"][name] for r in rows]
        summary[name]={k:float(np.mean([s[k] for s in values])) for k in ("f1","exact","generation_tokens")}
        summary[name]["prompt_tokens"]=float(np.mean([r["prompt_tokens"] for r in rows]))
        summary[name]["clipped_outputs"]=sum(s["generation_tokens"]>=MAX_ANSWER for s in values)
    draws=np.random.default_rng(291701).integers(len(rows),size=(20000,len(rows)))
    pairs={}
    for name,control in (("support","answer_only"),("support","frozen"),("answer_only","frozen")):
        result={}
        for key in ("f1","exact","generation_tokens"):
            delta=np.array([r["arms"][name][key]-r["arms"][control][key] for r in rows])
            result[key]=[float(delta.mean()),*np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist()]
            if key=="f1":
                result["f1_wins_ties_losses"]=[int((delta>0).sum()),int((delta==0).sum()),int((delta<0).sum())]
        pairs[f"{name}_vs_{control}"]=result
    p=pairs["support_vs_answer_only"]["f1"]
    return dict(arms=summary,paired=pairs,passes=bool(p[0]>=.03 and p[1]>0))


def score(args):
    mx,nn,_,flatten,_,unflatten,load,stream,_,sampler_fn,make_lora=gpu()
    training_file=args.adapters/"training.json"
    training=json.loads(training_file.read_text())
    assert training["inputs"]==lock_inputs(args.model,args.data)
    for arm,digest in training["adapters"].items():
        assert qa.sha(args.adapters/(arm+".safetensors"))==digest
    test=data_file(args.data,"test")
    args.output.mkdir()
    mx.random.seed(SEED)
    model,tokenizer=load(str(args.model))
    model.freeze()
    make_lora(model,LAYERS,LORA)
    arms={"frozen":dict(flatten(model.trainable_parameters())),
          **{name:mx.load(str(args.adapters/(name+".safetensors"))) for name in ADAPTED}}
    sampler=sampler_fn(temp=0.0)
    assert all(set(arms[k])==set(arms["frozen"]) for k in ADAPTED)
    for key,value in arms["frozen"].items():
        if key.endswith("lora_b"):
            assert not bool(mx.any(value!=0).item())
    results=[]
    started=time.perf_counter()
    with (args.output/"cases.jsonl").open("w") as f:
        for i,row in enumerate(test,1):
            prompt=row["prompt_ids"]
            result=dict(id=row["id"],group=row["group"],type=row["type"],article_complete=row["article_complete"],
                        complete=row["complete"],prompt_tokens=len(prompt),prompt_sha256=qa.stable(",".join(map(str,prompt))),arms={})
            for arm in sorted(ARMS,key=lambda key:qa.stable(row["group"]+key)):
                model.update(unflatten(list(arms[arm].items())))
                out=[]
                generation=0
                for token in stream(model,tokenizer,prompt,max_tokens=MAX_ANSWER,sampler=sampler):
                    out.append(token.text)
                    generation=token.generation_tokens
                    if "\n" in "".join(out).strip():
                        break
                result["arms"][arm]=dict(qa.answer_score("".join(out),row["answer"]),generation_tokens=generation)
            results.append(result)
            f.write(json.dumps(result,sort_keys=True)+"\n")
            f.flush()
            if i%16==0:
                print("finished",i,"/",len(test),"seconds",round(time.perf_counter()-started),flush=True)
    report=dict(analysis=summarize(results),test_count=len(results),test_sha256=qa.sha(args.output/"cases.jsonl"),
                training_sha256=qa.sha(training_file),max_prompt_tokens=1024,max_generation_tokens=MAX_ANSWER,
                seconds=time.perf_counter()-started,peak_mlx_bytes=mx.get_peak_memory(),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(args.output/"summary.json",report)
    print(json.dumps(report,indent=2),flush=True)


def verify(args):
    rows=[json.loads(line) for line in (args.output/"cases.jsonl").read_text().splitlines()]
    report=json.loads((args.output/"summary.json").read_text())
    fit_file=args.adapters/"training.json"
    training=json.loads(fit_file.read_text())
    assert qa.sha(fit_file)==report["training_sha256"]
    for key,value in training["inputs"]["sources"].items():
        assert qa.sha(ROOT/key)==value
    for arm,digest in training["adapters"].items():
        assert qa.sha(args.adapters/(arm+".safetensors"))==digest
    assert len(rows)==len({r["group"] for r in rows})==SHAPES["test"]
    assert qa.sha(args.output/"cases.jsonl")==report["test_sha256"]
    for row in rows:
        assert set(row["arms"])==set(ARMS) and row["prompt_tokens"]<=report["max_prompt_tokens"]
        for r in row["arms"].values():
            assert 0<=r["generation_tokens"]<=report["max_generation_tokens"]
            assert r["f1"]==(0.0 if r["incompatible"] else 2*r["overlap"]/max(1,r["answer_terms"]+r["gold_terms"]))
    assert summarize(rows)==report["analysis"]
    print(f"Verified exact per-case token accounting and paired statistics for {len(rows)} groups.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("train","score","verify"):
        part=sub.add_parser(command)
        part.add_argument("--output",type=Path,required=True)
        if command!="verify":
            part.add_argument("--model",type=Path,required=True)
            part.add_argument("--data",type=Path,required=True)
        if command!="train":
            part.add_argument("--adapters",type=Path,required=True)
        else:
            part.add_argument("--smoke",type=int)
    args=parser.parse_args()
    {"train":train,"score":score,"verify":verify}[args.command](args)


if __name__=="__main__":
    main()
