"""Recompute coverage, timing, and cluster intervals from locked choices."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from study import BUDGETS, LOCK, POLICIES, digest, inputs, load, make_split


def open_rows(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def load_audited(path: Path):
    lock = json.loads(LOCK.read_text())
    data = load()
    src = {row["id"]:(row, split, group)
           for row, (split, group) in zip(data, make_split(data))}
    raw_digest = hashlib.sha256()
    choices = {}
    lookup = {}
    with open_rows(path) as records:
        for line in records:
            raw_digest.update(line.encode())
            result = json.loads(line)
            episode = result["id"]
            p = result["preview_words"]
            method = result["method"]
            label = result["budget"]
            key = (episode,p,label,method)
            if key in choices:
                raise ValueError(f"duplicate row: {key}")
            row, split, group = src[episode]
            assert p in lock["previews"] and method in POLICIES and label in BUDGETS, key
            assert result["split"] == split and result["group"] == group, key
            assert split != "development" and (split != "topic_stress" or p == 24), key
            if (episode,p) not in lookup:
                _, views, _, truth, ids = inputs(row,p)
                lookup[episode,p] = (
                    {j:v.cost for v,j in zip(views,ids)},
                    {j for j, gold in zip(ids,truth) if gold},
                    sum(len(v.title)+len(v.preview) for v in views),
                    sum(len(v.title)+len(v.preview)+v.cost-1 for v in views))
            costs, support, index_words, corpus_words = lookup[episode,p]
            order = result["selection"]
            chosen = set(order)
            assert chosen <= costs.keys() and len(chosen) == len(order), key
            cost = sum(costs[i] for i in chosen)
            assert cost == result["words"] and len(chosen) == result["reads"], key
            assert result["coverage"] == len(chosen & support), key
            assert support == set(result["gold"]) and len(support) == result["total_support"], key
            assert result["complete"] == int(support <= chosen), key
            assert result["index_words"] == index_words and result["corpus_words"] == corpus_words, key
            mode, budget = BUDGETS[label]
            assert (cost if mode == "words" else len(chosen)) <= budget, key
            # Independently check maximum support achievable under the declared budget.
            tally = 0
            feasible = 0
            for gold_cost in sorted(costs[i] if mode == "words" else 1 for i in support):
                tally += gold_cost
                feasible += tally <= budget
            assert result["oracle"] == feasible, key
            assert feasible >= result["coverage"], key
            choices[key] = result

    expected = sum((len(lock["previews"]) if split == "test" else split == "topic_stress")
                   * len(BUDGETS) * len(POLICIES) for _,split,_ in src.values())
    assert len(choices) == expected
    data_path = path.parent/"metadata.json"
    if data_path.exists():
        metadata = json.loads(data_path.read_text())
        assert metadata["row_sha256"] == raw_digest.hexdigest(), "Trace hash mismatch"
        assert metadata["lock_sha256"] == digest(LOCK), "Lock hash mismatch"
    return choices, raw_digest.hexdigest()


def contrast(a, b, seed=110327, draws=5000):
    aa = {x["id"]:x for x in a}
    bb = {x["id"]:x for x in b}
    if aa.keys() != bb.keys():
        raise ValueError("Unpaired episodes")
    diffs = defaultdict(list)
    for ident in sorted(aa):
        x, y = aa[ident], bb[ident]
        assert x["group"] == y["group"]
        diffs[x["group"]].append(x["complete"] - y["complete"])
    clusters = np.array([[sum(values), len(values)] for _,values in sorted(diffs.items())])
    totals, sizes = clusters.T
    ratios = totals / sizes
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sizes), (draws,len(sizes)))
    weighted = totals[idx].sum(axis=1)/sizes[idx].sum(axis=1)
    unweighted = ratios[idx].mean(axis=1)
    def interval(replays):
        return (np.quantile(replays,(.025,.975))*100).tolist() if len(sizes) > 1 else None
    paired = [y for values in diffs.values() for y in values]
    return {"questions":len(aa),"clusters":len(diffs),"effect_pp":float(100*totals.sum()/sizes.sum()),
            "ci95_pp":interval(weighted),"cluster_weighted_effect_pp":100*float(ratios.mean()),
            "cluster_weighted_ci95_pp":interval(unweighted),
            "wins":paired.count(1),"losses":paired.count(-1),"ties":paired.count(0)}


def describe(rows):
    return {
        "n": len(rows),
        "complete": statistics.mean(r["complete"] for r in rows),
        "recall": statistics.mean(r["coverage"]/r["total_support"] for r in rows),
        "read_words": statistics.mean(r["words"] for r in rows),
        "reads": statistics.mean(r["reads"] for r in rows),
        "index_words": statistics.mean(r["index_words"] for r in rows),
        "corpus_words": statistics.mean(r["corpus_words"] for r in rows),
        "cpu_ms": statistics.mean((r["setup_cpu_ns"] + r["cpu_ns"])/1e6 for r in rows),
        "oracle": statistics.mean(r["oracle"]==r["total_support"] for r in rows),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows",type=Path)
    parser.add_argument("--write",type=Path,required=True)
    args=parser.parse_args()
    lock=json.loads(LOCK.read_text())
    choices, raw_sha=load_audited(args.rows)
    groups=defaultdict(lambda:defaultdict(list))
    for row in choices.values():
        p,b,m,s=row["preview_words"],row["budget"],row["method"],row["split"]
        groups[s,p,b,"all"][m].append(row)
        if (p,b)==(24,"words320"):
            groups[s,p,b,f"{row['id'][0]}hop"][m].append(row)
    comparator=lock["best_comparator"]
    output={}
    for condition, methods in sorted(groups.items()):
        responses=methods["reveal_bridge"]
        metric={"methods":{m:describe(r) for m,r in methods.items()},
                "contrast":contrast(responses,methods[comparator],lock["bootstrap_seed"],lock["bootstrap_draws"]),
                "reveal_minus_iterative":contrast(responses,methods["iterative"],lock["bootstrap_seed"],lock["bootstrap_draws"])}
        output["/".join(map(str,condition))]=metric
    primary=output["test/24/words320/all"]["contrast"]
    gate=primary["effect_pp"] >= lock["primary_minimum_pp"] and primary["ci95_pp"][0] > 0
    result={"rows":len(choices),"trace_sha256":raw_sha,"comparator":comparator,
            "primary_gate":gate,"conditions":output}
    args.write.write_text(json.dumps(result,indent=2)+"\n")
    print("Primary contrast",primary, "practical gate",gate)


if __name__ == "__main__":
    main()
