"""Score matched whole-token prefixes from naturally stopping MuSiQue readers."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prior_equal_decode", HERE / "score_equal_decode.py")
equal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(equal)
study, transfer, sha = equal.study, equal.transfer, equal.sha
ARMS = equal.ARMS
PANELS = equal.PANELS
CAP = 24
SEEDS = {"transfer": 831641, "replication": 831642, "pooled": 831643, "both_eos": 831644}
READOUTS = ("matched_eos", "matched_line", "native_full")


def locks(args):
    return dict(
        code={k: sha(HERE / k) for k in ("score_natural_stop.py", "NATURAL_STOP_PROTOCOL.md")},
        prior=equal.source_locks(args),
    )


def decode(reader, prompt, prepared=None):
    cache = None
    tokens = prompt
    if prepared is not None:
        hidden, lower = prepared
        suffix = reader.make_cache(reader.model)[reader.depth:]
        mask = reader.mask(hidden, suffix[0])
        for block, entry in zip(reader.model.model.layers[reader.depth:], suffix):
            hidden = block(hidden, mask, entry)
        reader.mx.eval([entry.state for entry in suffix])
        cache = lower + suffix
        tokens = [prompt[-1]]

    ids = []
    reason = None
    eos = None
    steps = 0
    for response in reader.stream(reader.model, reader.tokenizer, tokens,
                                  max_tokens=CAP, sampler=reader.sampler, prompt_cache=cache):
        steps = response.generation_tokens
        reason = response.finish_reason
        if reason == "stop":
            eos = int(response.token)
            assert eos in reader.tokenizer.eos_token_ids
        else:
            ids.append(int(response.token))
    assert reason in ("stop", "length")
    assert steps == len(ids) + (reason == "stop") and steps <= CAP
    assert reason != "length" or steps == CAP

    if prepared is not None:
        for entry in lower:
            extra = entry.offset - (len(prompt) - 1)
            if extra < 0 or entry.trim(extra) != extra:
                raise ValueError("Failed to restore the common prompt cache.")
    return dict(ids=ids, reason=reason, eos=eos, steps=steps)


def text_of(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def line_limit(tokenizer, ids):
    for n in range(1, len(ids) + 1):
        for line in text_of(tokenizer, ids[:n]).splitlines(keepends=True):
            if line and line[-1] in equal.BREAKS and line.strip():
                return n - 1
    return len(ids)


def evaluate(text, case):
    normalized = study.qa.normalize(text)
    actual = Counter(normalized.split())
    variants = []
    for answer in (case["answer"], *case["aliases"]):
        target = study.qa.normalize(answer)
        gold = Counter(target.split())
        overlap = sum((actual & gold).values())
        variants.append(dict(
            f1=2 * overlap / (sum(actual.values()) + sum(gold.values())),
            exact=int(normalized == target), overlap=overlap,
            answer_words=sum(actual.values()), gold_words=sum(gold.values()),
        ))
    best = max(variants, key=lambda item: (item["f1"], item["exact"]))
    return dict(best, exact=max(item["exact"] for item in variants))


def paired_case(panel, case, traces, tokenizer, specials):
    ids = {arm: traces[arm]["ids"] for arm in ARMS}
    limits = {arm: line_limit(tokenizer, ids[arm]) for arm in ARMS}
    eos_k = min(map(len, ids.values()))
    line_k = min(limits.values())
    row = {key: case[key] for key in ("id", "group", "type", "complete", "article_complete")}
    row.update(
        panel=panel, prompt_tokens=len(case["prompt_ids"]),
        prompt_sha256=study.qa.stable(",".join(map(str, case["prompt_ids"]))),
        matched_eos_tokens=eos_k, matched_line_tokens=line_k, arms={},
    )
    for arm in ARMS:
        sequence = ids[arm]
        generation = traces[arm]
        row["arms"][arm] = dict(
            finish_reason=generation["reason"], eos_id=generation["eos"], steps=generation["steps"],
            answer_tokens=len(sequence), first_line_tokens=limits[arm],
            special_tokens_in_matched_eos=sum(t in specials for t in sequence[:eos_k]),
            sequence_sha256=study.qa.stable(",".join(map(str, sequence))),
            matched_eos=evaluate(text_of(tokenizer, sequence[:eos_k]), case),
            matched_line=evaluate(text_of(tokenizer, sequence[:line_k]), case),
            native_full=evaluate(text_of(tokenizer, sequence), case),
        )
    return row


def bootstrap(columns, strata, seed):
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(40):
        combined = np.zeros((500, columns.shape[1]))
        for indices in strata:
            take = indices[rng.integers(len(indices), size=(500, len(indices)))]
            combined += columns[take].sum(axis=1)
        boot.append(combined / len(columns))
    return np.quantile(np.concatenate(boot), [.025, .975], axis=0)


def summarize_group(rows, strata, seed):
    result = {"count": len(rows), "readouts": {}}
    columns = np.array([
        [r["arms"][ARMS[1]][metric][key] - r["arms"][ARMS[0]][metric][key]
         for metric in READOUTS for key in ("f1", "exact")] for r in rows
    ])
    intervals = bootstrap(columns, strata, seed)
    for i, metric in enumerate(READOUTS):
        d = columns[:, 2*i]
        result["readouts"][metric] = dict(
            arms={arm: {key: float(np.mean([r["arms"][arm][metric][key] for r in rows]))
                        for key in ("f1", "exact")} for arm in ARMS},
            f1=[float(d.mean()), *intervals[:, 2*i].tolist()],
            exact=[float(columns[:, 2*i+1].mean()), *intervals[:, 2*i+1].tolist()],
            wins_ties_losses=[int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())],
        )
    point, lower, _ = result["readouts"]["matched_eos"]["f1"]
    result["passes"] = bool(point >= .03 and lower > 0)
    for mode in ("eos", "line"):
        lengths = [r["matched_" + mode + "_tokens"] for r in rows]
        result["matched_" + mode + "_lengths"] = dict(
            mean=float(np.mean(lengths)), histogram={str(k): v for k, v in sorted(Counter(lengths).items())}
        )
    result["native_lengths"] = {
        arm: dict(
            mean_answer_tokens=float(np.mean([r["arms"][arm]["answer_tokens"] for r in rows])),
            mean_steps=float(np.mean([r["arms"][arm]["steps"] for r in rows])),
            cap_finishes=sum(r["arms"][arm]["finish_reason"] == "length" for r in rows),
            special_tokens_in_matched_eos=sum(r["arms"][arm]["special_tokens_in_matched_eos"] for r in rows),
        )
        for arm in ARMS
    }
    return result


def summarize(rows):
    report = {}
    for panel in PANELS:
        group = [r for r in rows if r["panel"] == panel]
        report[panel] = summarize_group(group, [np.arange(len(group))], SEEDS[panel])
    strata = [np.asarray([i for i,r in enumerate(rows) if r["panel"] == p]) for p in PANELS]
    report["pooled"] = summarize_group(rows, strata, SEEDS["pooled"])
    both = [r for r in rows if all(r["arms"][arm]["finish_reason"] == "stop" for arm in ARMS)]
    if both:
        groups = [np.asarray([i for i,r in enumerate(both) if r["panel"] == p]) for p in PANELS]
        report["both_eos"] = summarize_group(both, [g for g in groups if len(g)], SEEDS["both_eos"])
    report["replicated"] = all(report[p]["passes"] for p in PANELS)
    return report


def run(args):
    prepared = equal.panels(args)
    inputs = locks(args)
    transfer.validate_model(args.model)
    dev = study.cases(args.hotpot_data, "dev")
    checkpoint = json.loads((HERE/"adapter/training.json").read_text())
    assert sha(args.hotpot_data/"dev.jsonl") == checkpoint["inputs"]["prepared"]["dev.jsonl"]
    args.output.mkdir()
    reader, frozen = study.load_reader(args.model, dev[0]["prompt_ids"])
    adapter = reader.mx.load(str(HERE/"adapter/question_last.safetensors"))
    assert set(adapter) == set(frozen)
    states = {"frozen": frozen, "question_last_adapter": adapter}
    special = frozenset(reader.tokenizer.all_special_ids)
    cached = True
    for row in dev[:4]:
        prompt = row["prompt_ids"]
        prefix = reader.prefill(prompt)
        for state in states.values():
            reader.set_weights(state)
            cached &= decode(reader, prompt) == decode(reader, prompt, prefix)
    print("Exact development token and stop replay", cached, flush=True)

    ledger = []
    t0 = time.perf_counter()
    numeric = args.output/"cases.jsonl"
    private = args.output/"private_tokens.jsonl"
    with numeric.open("x") as f, private.open("x") as g:
        for panel, cases in prepared.items():
            for i, case in enumerate(cases, 1):
                prompt = case["prompt_ids"]
                prefix = reader.prefill(prompt) if cached else None
                traces = {}
                for arm in sorted(ARMS, key=lambda a: study.qa.stable(case["group"]+a)):
                    reader.set_weights(states[arm])
                    traces[arm] = decode(reader, prompt, prefix)
                row = paired_case(panel, case, traces, reader.tokenizer, special)
                ledger.append(row)
                f.write(json.dumps(row, sort_keys=True)+"\n")
                g.write(json.dumps(dict(id=case["id"], panel=panel, traces=traces), sort_keys=True)+"\n")
                f.flush()
                g.flush()
                if i % 16 == 0 or i == len(cases):
                    print(panel, i, "/", len(cases), "seconds", round(time.perf_counter()-t0), flush=True)
    result = dict(
        analysis=summarize(ledger), paired_cases=len(ledger), locks=inputs, cap=CAP,
        eos_ids=sorted(reader.tokenizer.eos_token_ids), exact_cache_replay=cached,
        frozen_logits_verified=True, ledger_sha256=sha(numeric), private_tokens_sha256=sha(private),
        peak_mlx_bytes=reader.mx.get_peak_memory(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        seconds=time.perf_counter()-t0,
    )
    study.ground.write(args.output/"summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def verify(args):
    prepared = equal.panels(args)
    rows = equal.json_rows(args.output/"cases.jsonl")
    report = json.loads((args.output/"summary.json").read_text())
    assert report["locks"] == locks(args) and report["ledger_sha256"] == sha(args.output/"cases.jsonl")
    assert report["analysis"] == summarize(rows)
    assert report["paired_cases"] == len(rows) == 329
    ordered = [(panel,case) for panel,cases in prepared.items() for case in cases]
    for (panel,case),row in zip(ordered,rows,strict=True):
        assert panel == row["panel"] and case["id"] == row["id"] and case["group"] == row["group"]
        assert row["prompt_tokens"] == len(case["prompt_ids"])
        assert row["prompt_sha256"] == study.qa.stable(",".join(map(str,case["prompt_ids"])))
        for mode, length in (("eos","answer_tokens"),("line","first_line_tokens")):
            assert row["matched_"+mode+"_tokens"] == min(row["arms"][a][length] for a in ARMS)
        for arm in row["arms"].values():
            assert arm["steps"] <= CAP
            assert arm["steps"] == arm["answer_tokens"] + (arm["finish_reason"] == "stop")
            assert (arm["eos_id"] in report["eos_ids"]) == (arm["finish_reason"] == "stop")
            for metric in READOUTS:
                x = arm[metric]
                assert x["f1"] == 2*x["overlap"]/(x["answer_words"]+x["gold_words"])
    if args.private_tokens:
        from transformers import AutoTokenizer
        transfer.validate_model(args.model)
        assert report["private_tokens_sha256"] == sha(args.private_tokens)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
        special = frozenset(tokenizer.all_special_ids)
        for (panel,case),row,raw in zip(ordered,rows,equal.json_rows(args.private_tokens),strict=True):
            assert raw["panel"] == panel and raw["id"] == case["id"]
            assert all(set(t["ids"]).isdisjoint(report["eos_ids"]) for t in raw["traces"].values())
            assert row == paired_case(panel,case,raw["traces"],tokenizer,special)
    print("Verified 329 paired prompts, natural stop lengths, prefix scores and bootstrap intervals.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    for mode in ("score","verify"):
        cmd=sub.add_parser(mode)
        cmd.add_argument("--transfer",type=Path,required=True)
        cmd.add_argument("--replication",type=Path,required=True)
        cmd.add_argument("--output",type=Path,required=True)
        cmd.add_argument("--model",type=Path,required=mode=="score")
        if mode=="score":
            cmd.add_argument("--hotpot-data",type=Path,required=True)
        else:
            cmd.add_argument("--private-tokens",type=Path)
    args=parser.parse_args()
    if args.mode=="verify" and args.private_tokens and not args.model:
        parser.error("--private-tokens requires the local model tokenizer")
    (run if args.mode=="score" else verify)(args)


if __name__=="__main__":
    main()
