"""Paired sentence compression with a frozen Qwen reader."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "cross-corpus-qa"
spec = importlib.util.spec_from_file_location("prior_qa", OLD / "run.py")
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
ARMS = ("full", "independent", "bridge")
STOP = set("a an the of in on to for at as from that is are was were with by and or its it this which what who how did be have had do does than but their he she they his her not then these also first after into over during one two".split())
PROMPT = "Answer using the passages. Give only the shortest answer phrase or yes/no.\n\n"


def terms(text):
    return set(qa.retrieval.words(text)) - STOP


def candidates(row, source_lock):
    docs, chosen = qa.prepare(row, source_lock)
    index = chosen["reveal_bridge"][0]
    triples = [
        (title, j, text) for i, (title, sentences) in enumerate(row["context"])
        if i in index for j, text in enumerate(sentences) if text.strip()
    ]
    sets = [terms(text) for _, _, text in triples]
    counts = Counter(word for s in sets for word in s)
    query = terms(row["question"])
    title_terms = {title: terms(title) for title, _, _ in triples}
    source_count = len(sets)
    distinctive = {word: math.log1p(source_count / n) for word, n in counts.items()}

    def mass(words):
        return sum(distinctive[w] for w in sorted(words))

    features = []
    for (title, j, text), words in zip(triples, sets):
        overlap = query & words
        other_titles = [m for t, m in title_terms.items() if t != title and m]
        linked = max((len(words & m) / len(m) for m in other_titles), default=0)
        other_query = max((len(query & m) for m in other_titles), default=0)
        features.append([
            len(overlap), mass(overlap), len(overlap) / max(1, len(query)),
            mass(overlap) / math.sqrt(max(1, len(words))), len(title_terms[title] & query),
            1 / (1 + j), min(10, j), math.log1p(len(words)),
            linked, other_query, len(title_terms[title] & words),
            float(text[:1].isupper()), float(any(w.isdigit() for w in words)),
        ])
    matrix = np.array(features, dtype=float).reshape(-1, 13)
    return triples, sets, distinctive, matrix


def labels(triples, row):
    gold = set(map(tuple, row["supporting_facts"]))
    values = np.array([int((title, j) in gold) for title, j, _ in triples])
    return gold, values


def fit(examples):
    matrix = np.vstack([x for x, _ in examples])
    target = np.concatenate([y for _, y in examples])
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 1e-6)
    design = np.column_stack((np.ones(len(matrix)), (matrix - mean) / scale))
    beta = np.zeros(design.shape[1])
    penalty = np.diag([1e-6] + [0.02] * (len(beta) - 1))
    for _ in range(12):
        p = 1 / (1 + np.exp(-np.clip(design @ beta, -35, 35)))
        delta = design.T @ (p - target) / len(p) + penalty @ beta
        hessian = (design.T * (p * (1 - p))) @ design / len(p) + penalty
        beta -= np.linalg.solve(hessian, delta)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "beta": beta.tolist()}


def scores(matrix, params):
    x = (matrix - params["mean"]) / params["scale"]
    beta = params["beta"]
    return 1 / (1 + np.exp(-np.clip(x @ beta[1:] + beta[0], -35, 35)))


def order(triples, words, rarity, probabilities, pair_weight):
    ranking = sorted(range(len(triples)), key=lambda i: (-probabilities[i], i))
    if pair_weight is None:
        return ranking
    best, anchor = -math.inf, ()
    for m, i in enumerate(ranking[:14]):
        for j in ranking[m+1:14]:
            if triples[i][0] == triples[j][0]:
                continue
            common = words[i] & words[j]
            bridge = sum(rarity[w] for w in sorted(common)) / max(1, len(words[i] | words[j]))
            value = probabilities[i] + probabilities[j] + pair_weight * bridge
            if value > best:
                best, anchor = value, (i, j)
    return [*anchor, *(i for i in ranking if i not in anchor)]


def messages(row, triples, chosen):
    docs = {}
    for i in sorted(chosen):
        title, _, sentence = triples[i]
        docs.setdefault(title, []).append(sentence)
    evidence = "\n\n".join(f"{title}\n{' '.join(sents)}" for title, sents in docs.items())
    return [{"role": "user", "content": PROMPT + f"Passages:\n{evidence}\n\nQuestion: {row['question']}\nAnswer:"}]


def encode(tokenizer, row, triples, chosen):
    token_ids = tokenizer.apply_chat_template(messages(row, triples, chosen), tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False,
                                              return_dict=False)
    if not isinstance(token_ids, list) or not token_ids or not isinstance(token_ids[0], int):
        raise TypeError("Expected a nonempty, flat list of chat-template token IDs")
    return token_ids


def compact(tokenizer, row, triples, ranking, cap):
    limit = len(encode(tokenizer, row, triples, [])) + cap
    chosen = []
    encoded = encode(tokenizer, row, triples, [])
    for i in ranking:
        proposal = [*chosen, i]
        tokens = encode(tokenizer, row, triples, proposal)
        if len(tokens) <= limit:
            chosen, encoded = proposal, tokens
    return chosen, encoded


def evidence(triples, chosen, gold):
    pairs = {(triples[i][0], triples[i][1]) for i in chosen}
    return {"support_hits": len(pairs & gold), "support_total": len(gold),
            "complete": int(gold <= pairs),
            "articles": int({t for t,_ in gold} <= {t for t,_ in pairs})}


def summarize(rows):
    readout = {}
    for arm in ARMS:
        readout[arm] = {k: float(np.mean([r["arms"][arm][k] for r in rows]))
                        for k in ("f1", "exact", "complete", "articles", "prompt_tokens", "generation_tokens")}
        readout[arm]["support_recall"] = float(np.mean([
            r["arms"][arm]["support_hits"] / r["arms"][arm]["support_total"] for r in rows
        ]))
    rng = np.random.default_rng(2026112354)
    draws = rng.integers(len(rows), size=(20_000,len(rows)))
    comparisons = {}
    for control in ("full", "independent"):
        for method in (("independent", "bridge") if control == "full" else ("bridge",)):
            d = {}
            for metric in ("f1", "exact", "complete", "articles"):
                delta = np.array([r["arms"][method][metric]-r["arms"][control][metric] for r in rows])
                d[metric] = [float(delta.mean()), *np.quantile(delta[draws].mean(axis=1), [.025,.975]).tolist()]
            d["token_reduction"] = 1-readout[method]["prompt_tokens"]/readout[control]["prompt_tokens"]
            comparisons[f"{method}_vs_{control}"] = d
    p = comparisons["bridge_vs_full"]
    return dict(arms=readout, comparisons=comparisons, pass_primary=bool(p["f1"][1]>-.04 and p["token_reduction"]>=.40))


def format_scores(tokenizer, row, triples, words, rarity, probability, lock):
    result = {"full": (list(range(len(triples))), encode(tokenizer, row, triples, range(len(triples))))}
    for name, weight in (("independent", None), ("bridge", lock["pair_weight"])):
        ranking = order(triples, words, rarity, probability, weight)
        result[name] = compact(tokenizer, row, triples, ranking, lock["cap"])
    return result


def find_policy(dev, model, source_lock, fitted):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    # The same Qwen fast tokenizer and chat template are used for development and inference.
    cases = []
    for _, _, row in dev:
        t, w, z, x = candidates(row, source_lock)
        cases.append((row, t, w, z, scores(x, fitted), labels(t,row)[0]))

    def measure(cap, weight):
        values = []
        for row,t,w,z,p,gold in cases:
            selected, _ = compact(tokenizer, row, t, order(t,w,z,p,weight), cap)
            values.append(evidence(t,selected,gold))
        return [float(np.mean([v["complete"] for v in values])),
                float(np.mean([v["support_hits"]/v["support_total"] for v in values]))]

    caps = {c:measure(c,None) for c in (192,256,320)}
    cap = next(c for c,(complete,recall) in caps.items() if recall >= .92*caps[320][1])
    pair = {w:measure(cap,w) for w in (0,.25,.5)}
    weight = max(pair, key=lambda w: (*pair[w], -w))
    return cap, weight, caps, pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "score", "verify"))
    parser.add_argument("data", type=Path, nargs="?")
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    lock_path = ROOT/"FIT.json"
    code = {"study":qa.sha(ROOT/"study.py"), "protocol":qa.sha(ROOT/"PROTOCOL.md"),
            "old_study":qa.sha(OLD/"run.py"), "old_lock":qa.sha(OLD/"LOCK.json")}
    if args.mode == "verify":
        lock = json.loads(lock_path.read_text())
        assert lock["code"] == code
        rows = [json.loads(line) for line in (ROOT/"results/rows.jsonl").read_text().splitlines()]
        report = json.loads((ROOT/"results/summary.json").read_text())
        assert len(rows)==len({r["group"] for r in rows})==256
        assert summarize(rows)==report["outcomes"]
        assert qa.sha(ROOT/"results/rows.jsonl")==report["raw_sha256"]
        assert qa.sha(lock_path)==report["fit_sha256"]
        for row in rows:
            for arm in ARMS:
                z = row["arms"][arm]
                expected = 0 if z["incompatible"] else 2*z["overlap"]/max(1,z["answer_terms"]+z["gold_terms"])
                assert expected==z["f1"]
                if arm != "full":
                    assert z["prompt_tokens"] <= row["empty_tokens"]+lock["cap"]
        print("Verified 256 paired cases, scores, compact token limits and frozen summary")
        return

    prior = json.loads((OLD/"LOCK.json").read_text())
    assert all(qa.sha(args.model/name)==digest for name,digest in prior["model"].items())
    sections = qa.representatives(qa.load(args.data))
    source_lock = json.loads((ROOT.parent/"FROZEN.json").read_text())
    if args.mode=="fit":
        training = []
        for _, _, row in sections[200:1600]:
            triples, _, _, matrix = candidates(row,source_lock)
            training.append((matrix,labels(triples,row)[1]))
        params=fit(training)
        cap, weight, caps, pair = find_policy(sections[1600:1760],args.model,source_lock,params)
        with lock_path.open("x") as f:
            json.dump(dict(code=code, params=params, cap=cap, pair_weight=weight, rank_dev=caps,
                           pair_dev=pair, training_sentences=sum(len(y) for _,y in training)),f,indent=2)
        print(cap,weight,caps,pair,flush=True)
        return

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_cache_limit(128*1024**2)
    lock = json.loads(lock_path.read_text())
    assert lock["code"] == code
    args.output.mkdir()
    sample=sections[1760:2016]
    del sections
    started=time.perf_counter()
    model,tokenizer=load(str(args.model))
    sampler=make_sampler(temp=0)
    saved=[]
    calls=0
    with (args.output/"rows.jsonl").open("w") as handle:
        for ix,(group, _, row) in enumerate(sample,1):
            t,w,z,x=candidates(row,source_lock)
            gold,_=labels(t,row)
            planned=format_scores(tokenizer,row,t,w,z,scores(x,lock["params"]),lock)
            cache={}
            result=dict(id=row["_id"],group=group,empty_tokens=len(encode(tokenizer,row,t,[])),arms={})
            for arm in sorted(ARMS,key=lambda a:qa.stable(group+a)):
                chosen,token_ids=planned[arm]
                key=tuple(token_ids)
                if key not in cache:
                    generated=[]
                    out=0
                    for response in stream_generate(model,tokenizer,token_ids,max_tokens=24,sampler=sampler):
                        generated.append(response.text)
                        out=response.generation_tokens
                        if "\n" in "".join(generated).strip():
                            break
                    cache[key]=dict(qa.answer_score("".join(generated),row["answer"]),generation_tokens=out)
                    calls+=1
                result["arms"][arm]=dict(cache[key],**evidence(t,chosen,gold),prompt_tokens=len(key),
                                         selected=chosen,prompt_sha256=qa.stable(",".join(map(str,key))))
            saved.append(result)
            handle.write(json.dumps(result,sort_keys=True)+"\n")
            handle.flush()
            if ix % 16==0:
                print(f"{ix}/256 ({calls} distinct generations)",flush=True)
    report=dict(outcomes=summarize(saved), physical_calls=calls,runtime_seconds=time.perf_counter()-started,
                peak_mlx_bytes=mx.get_peak_memory(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                fit_sha256=qa.sha(lock_path),raw_sha256=qa.sha(args.output/"rows.jsonl"))
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
