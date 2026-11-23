"""Frozen cross-corpus short-answer transfer study with a local Qwen reader."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import string
import sys
import time
import urllib.request

os.environ.update({"OPENBLAS_NUM_THREADS":"1", "VECLIB_MAXIMUM_THREADS":"1",
                   "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"})
import numpy as np

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
import study as retrieval

DATA_DIGEST = "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea"
URL = ("https://huggingface.co/datasets/RAGLAB/data/resolve/c33b09b/"
       "eval_datasets/HotPotQA/hotpot_dev_distractor_v1.json")
SALT = "hotpot-cross-corpus-23nov26-v1"
CONDITIONS = ("preview_bridge", "reveal_bridge", "iterative", "closed_book", "gold_articles")
LABELLED = {"yes", "no", "noanswer"}
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")
PROMPT = ("Answer the question in a short phrase or with yes/no. Use the passages if supplied. "
          "If you cannot answer reliably, say unknown. Output one answer on the first line.\n\n")


def sha(path):
    return retrieval.digest(Path(path))


def stable(value):
    return retrieval.stable(value)


def load(data):
    if sha(data) != DATA_DIGEST:
        raise ValueError("HotpotQA source checksum differs")
    with data.open(encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) != 7405 or len(set(r["_id"] for r in rows)) != len(rows):
        raise ValueError("Unexpected archived data dimensions")
    return rows


def representatives(rows):
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for title, _ in row["supporting_facts"]:
            key = " ".join(retrieval.words(title))
            old = seen.setdefault(key, i)
            parents[find(i)] = find(old)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    order = []
    for group in groups.values():
        key = stable(SALT + "|" + "|".join(sorted(r["_id"] for r in group)))
        selected = min(group, key=lambda r: stable(SALT + "|item|" + r["_id"]))
        order.append((key, len(group), selected))
    order.sort(key=lambda item: item[0])
    return order


def prepare(row, original_lock):
    gold = {t for t, _ in row["supporting_facts"]}
    docs = [(title, " ".join(sentences)) for title, sentences in row["context"]]
    if not gold or not gold <= {t for t,_ in docs}:
        raise ValueError("Evidence titles do not match candidate documents")
    mapped = {"id": row["_id"], "question": row["question"],
              "paragraphs": [{"idx": i, "title": t, "paragraph_text": txt, "is_supporting": t in gold}
                             for i, (t,txt) in enumerate(docs)]}
    query, views, tails, support, indices = retrieval.inputs(mapped, 24)
    index = retrieval.Index(query, views)
    selection = {}
    for name in CONDITIONS[:3]:
        results = retrieval.run_one(index, views, tails, support, "words", 320,
                                    name, original_lock["selection"][name]["config"])
        # Identical article sets yield identical model input across policies.
        ids = [indices[i] for i in sorted(results["selection"])]
        selection[name] = (ids, results["words"], bool(results["complete"]))
    selection["closed_book"] = ([], 0, False)
    g = [i for i in range(len(indices)) if support[i]]
    selection["gold_articles"] = ([indices[i] for i in g], sum(views[i].cost for i in g), True)
    return docs, selection


def normalize(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def answer_score(output, gold):
    first = next((x.strip() for x in output.splitlines() if x.strip()), "")
    pred = normalize(re.sub(r"^answer\s*:\s*", "", first, flags=re.I))
    actual = normalize(gold)
    m = sum((Counter(pred.split()) & Counter(actual.split())).values())
    incompatible = pred != actual and (pred in LABELLED or actual in LABELLED)
    return {"exact": int(pred == actual), "answer_terms": len(pred.split()),
            "gold_terms": len(actual.split()), "overlap": m, "incompatible": incompatible,
            "f1": 0.0 if incompatible else 2*m/max(1,len(pred.split())+len(actual.split())),
            "prediction_sha256": stable(pred)}


def analysis(rows):
    assert len({r["group"] for r in rows})==len(rows)
    report={}
    for arm in CONDITIONS:
        report[arm]={}
        for k in ("f1","exact","complete","spent","prompt_tokens","generation_tokens"):
            report[arm][k] = float(np.mean([r["arms"][arm][k] for r in rows]))
    rng = np.random.default_rng(9021126)
    draw = rng.integers(0, len(rows), size=(20000,len(rows)))
    comparisons={}
    for base in ("preview_bridge", "iterative"):
        comparisons[base]={}
        for metric in ("f1","exact","complete","prompt_tokens"):
            pair=np.array([r["arms"]["reveal_bridge"][metric] - r["arms"][base][metric] for r in rows])
            comparisons[base][metric]=[float(pair.mean()),*np.quantile(pair[draw].mean(axis=1),[.025,.975]).tolist()]
    d,lo,_ = comparisons["preview_bridge"]["f1"]
    return {"arms":report,"comparisons":comparisons,"primary_pass":bool(d>=.02 and lo>0)}


def make_prompt(row, docs, ids):
    body = "\n\n".join(f"Passage {j}: {docs[i][0]}\n{docs[i][1]}" for j,i in enumerate(ids,1))
    return PROMPT + f"Passages:\n{body if body else '(none)'}\n\nQuestion: {row['question']}\nAnswer:"


def run(args):
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    lock=json.loads((ROOT/"LOCK.json").read_text())
    for name,digest in lock["files"].items():
        if sha(ROOT / name)!=digest:
            raise ValueError(f"Changed locked study file: {name}")
    for name,digest in lock["model"].items():
        if sha(args.model/name)!=digest:
            raise ValueError(f"Unexpected model file: {name}")
    if args.output.exists():
        raise ValueError("Choose a fresh output directory")
    args.output.mkdir(parents=True)
    data = order = None
    data=load(args.data)
    order=representatives(data)
    count_groups=len(order)
    records=order[:8] if args.smoke else order[8:200]
    assert len(records)==(8 if args.smoke else 192)
    del data, order
    start=time.perf_counter()
    mx.set_cache_limit(128*1024**2)
    model, tokenizer = mlx_lm.load(str(args.model))
    sampler=make_sampler(temp=0.0)
    original_lock=json.loads((PARENT/"FROZEN.json").read_text())
    physical=0
    scored=[]
    with (args.output/"raw.jsonl").open("w",encoding="utf-8") as handle:
        for index,(group,size,row) in enumerate(records):
            docs,selection=prepare(row,original_lock)
            previous={}
            arms={}
            for method in sorted(CONDITIONS,key=lambda m:stable(row["_id"]+m)):
                ids,spent,full = selection[method]
                msg=[{"role":"user","content":make_prompt(row,docs,ids)}]
                token_ids=tokenizer.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False)
                assert len(token_ids)<4096
                key=tuple(token_ids)
                if key not in previous:
                    start_call=time.perf_counter()
                    parts=[]
                    n_output=0
                    for response in mlx_lm.stream_generate(model,tokenizer,token_ids,max_tokens=20,sampler=sampler):
                        parts.append(response.text)
                        n_output=response.generation_tokens
                        if "\n" in "".join(parts).strip():
                            break
                    previous[key]=(answer_score("".join(parts),row["answer"]),n_output,time.perf_counter()-start_call)
                    physical+=1
                score,n_output,cold_seconds=previous[key]
                arms[method]={**score,"ids":ids,"spent":spent,"complete":int(full),
                              "prompt_tokens":len(token_ids),"generation_tokens":n_output,
                              "prompt_sha256":stable(",".join(map(str,token_ids))),"scoring_seconds":cold_seconds}
            line={"id":row["_id"],"group":group,"group_size":size,"type":row["type"],"arms":arms}
            scored.append(line)
            handle.write(json.dumps(line,sort_keys=True)+"\n")
            handle.flush()
            if (index+1)%12==0 or index+1==len(records):
                print(f"{index+1}/{len(records)} rows ({physical} physical generations)",flush=True)
    result={"smoke":args.smoke,"n":len(scored),"source_group_count":count_groups,"physical_calls":physical,
            "logical_calls":len(scored)*len(CONDITIONS),"readout":analysis(scored),
            "runtime_seconds":time.perf_counter()-start,"peak_mlx_bytes":mx.get_peak_memory(),
            "peak_process_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "raw_sha256":sha(args.output/"raw.jsonl"),"locked_sha256":sha(ROOT/"LOCK.json"),
            "runtime":{"python":sys.version.split()[0],"mlx_lm":mlx_lm.__version__}}
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


def verify():
    output=ROOT/"results"
    summary=json.loads((output/"summary.json").read_text())
    assert summary["locked_sha256"]==sha(ROOT/"LOCK.json")
    assert summary["raw_sha256"]==sha(output/"raw.jsonl")
    rows=[json.loads(x) for x in (output/"raw.jsonl").read_text().splitlines()]
    assert len(rows)==summary["n"]==192
    for row in rows:
        for method,s in row["arms"].items():
            assert 0<=s["overlap"]<=min(s["answer_terms"],s["gold_terms"])
            expected=0.0 if s["incompatible"] else 2*s["overlap"]/max(1,s["answer_terms"]+s["gold_terms"])
            assert expected==s["f1"]
            if method in CONDITIONS[:3]:
                assert s["spent"]<=320
    assert analysis(rows)==summary["readout"]
    print("Verified 192 distinct topics, paired outcomes, acquisition caps and archived summary.")


def main():
    parser=argparse.ArgumentParser()
    cli=parser.add_subparsers(dest="mode",required=True)
    fetch=cli.add_parser("fetch")
    fetch.add_argument("data",type=Path)
    lock=cli.add_parser("freeze")
    lock.add_argument("model",type=Path)
    score=cli.add_parser("score")
    score.add_argument("data",type=Path)
    score.add_argument("model",type=Path)
    score.add_argument("output",type=Path)
    score.add_argument("--smoke",action="store_true")
    cli.add_parser("verify")
    args=parser.parse_args()
    if args.mode=="freeze":
        with (ROOT/"LOCK.json").open("x") as f:
            json.dump({"files":{name:sha(ROOT/name) for name in
                               ("run.py","PROTOCOL.md","../FROZEN.json","../study.py")},
                       "model":{name:sha(args.model/name) for name in MODEL_FILES}},f,indent=2)
            f.write("\n")
    elif args.mode=="score":
        run(args)
    elif args.mode=="verify":
        verify()
    elif args.mode=="fetch":
        if args.data.exists():
            if sha(args.data)==DATA_DIGEST:
                return
            raise ValueError("Existing path has a different checksum")
        urllib.request.urlretrieve(URL,args.data)
        if sha(args.data)!=DATA_DIGEST:
            raise ValueError("Unexpected downloaded data digest")


if __name__=="__main__":
    main()
