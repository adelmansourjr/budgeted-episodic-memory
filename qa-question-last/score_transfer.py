"""Score the exact saved question-last adapter on a fresh MuSiQue panel."""

import argparse
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


study = module(HERE / "study.py", "question_last_study")
aliases = module(HERE.parent / "qa-grounding/score_transfer.py", "prior_musique_scoring")
sha = study.qa.sha
ARMS = ("frozen", "question_last_adapter")
N = 169
SEED = 944351


def read_data(data):
    input_file = data / "test.jsonl"
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["test_sha256"] != sha(input_file) or manifest["num_test"] != N:
        raise ValueError("MuSiQue prompts differ from the frozen preparation manifest.")
    if manifest["script_sha256"] != sha(HERE / "prepare_transfer.py"):
        raise ValueError("The prepared inputs were built with a different dataset script.")
    cases = [json.loads(line) for line in input_file.read_text().splitlines()]
    if len(cases) != N or len({case["group"] for case in cases}) != N:
        raise ValueError("Expected exactly 169 distinct MuSiQue support components.")
    if any(not 0 < len(case["prompt_ids"]) <= 1024 for case in cases):
        raise ValueError("A prompt falls outside the input token limits.")
    prior_panel = HERE.parent / "qa-grounding/musique/cases.jsonl"
    prior_groups = {json.loads(line)["group"] for line in prior_panel.read_text().splitlines()}
    if any(case["group"] in prior_groups for case in cases):
        raise ValueError("This transfer panel overlaps the prior MuSiQue reader benchmark.")
    for case in cases:
        for target in (case["answer"], *case["aliases"]):
            if not study.qa.normalize(target):
                raise ValueError("Expected a nonempty normalized answer or alias.")
    return cases


def summarize(records):
    arms = {}
    for name in ARMS:
        arm = [row["arms"][name] for row in records]
        arms[name] = {
            key: float(np.mean([row[key] for row in arm]))
            for key in ("f1", "exact", "generation_tokens")
        }
        arms[name]["prompt_tokens"] = float(np.mean([row["prompt_tokens"] for row in records]))
        arms[name]["at_generation_cap"] = sum(
            row["generation_tokens"] >= study.ground.MAX_ANSWER for row in arm
        )
    draws = np.random.default_rng(SEED).integers(len(records), size=(20_000, len(records)))
    paired = {}
    for field in ("f1", "exact", "generation_tokens"):
        delta = np.array([
            row["arms"]["question_last_adapter"][field] - row["arms"]["frozen"][field]
            for row in records
        ])
        interval = np.quantile(delta[draws].mean(axis=1), [.025, .975])
        paired[field] = [float(delta.mean()), *interval.tolist()]
        if field == "f1":
            paired["wins_ties_losses"] = [int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())]
    point, lower, _ = paired["f1"]
    return dict(arms=arms, paired=paired, passes=bool(point >= .03 and lower > 0))


def source_fingerprints():
    return {
        name: sha(HERE / name) for name in
        ("TRANSFER_PROTOCOL.md", "prepare_transfer.py", "score_transfer.py", "study.py",
         "../qa-grounding/score_transfer.py")
    }


def adapter_fingerprints():
    directory = HERE / "adapter"
    config = json.loads((directory / "training.json").read_text())
    weights = directory / "question_last.safetensors"
    if sha(weights) != config["adapter_sha256"]:
        raise ValueError("The saved adapter does not match its HotpotQA training record.")
    return {
        "weights": sha(weights),
        "training_record": sha(directory / "training.json"),
        "base_lock": sha(study.ground.CROSS / "LOCK.json"),
    }


def validate_model(model):
    lock = json.loads((study.ground.CROSS / "LOCK.json").read_text())
    if any(sha(model / name) != digest for name, digest in lock["model"].items()):
        raise ValueError("The local base model differs from the frozen HotpotQA checkpoint.")


def score(args):
    docs = source_fingerprints()
    adapter = adapter_fingerprints()
    validate_model(args.model)
    cases = read_data(args.data)
    dev = study.cases(args.hotpot_data, "dev")
    trained_record = json.loads((HERE / "adapter/training.json").read_text())
    if sha(args.hotpot_data / "dev.jsonl") != trained_record["inputs"]["prepared"]["dev.jsonl"]:
        raise ValueError("The prefix replay must use the original development prompts.")
    args.output.mkdir()
    reader, frozen = study.load_reader(args.model, dev[0]["prompt_ids"])
    weights = reader.mx.load(str(HERE / "adapter/question_last.safetensors"))
    if set(weights) != set(frozen):
        raise ValueError("The adapter tensor layout does not match the frozen reader.")
    states = dict(frozen=frozen, question_last_adapter=weights)
    cached = reader.exact_replay(dev, list(states.values()))
    print("exact development prefix replay", cached, flush=True)
    started = time.perf_counter()
    ledger = []
    result_file = args.output / "cases.jsonl"
    with result_file.open("w") as f:
        for i, case in enumerate(cases, 1):
            prompt = case["prompt_ids"]
            prepared = reader.prefill(prompt) if cached else None
            item = {key: case[key] for key in ("id", "group", "type", "complete", "article_complete")}
            item.update(prompt_tokens=len(prompt), prompt_sha256=study.qa.stable(",".join(map(str, prompt))), arms={})
            for name in sorted(ARMS, key=lambda arm: study.qa.stable(case["group"] + arm)):
                reader.set_weights(states[name])
                text, tokens = reader.cached(prompt, prepared) if cached else reader.ordinary(prompt)
                item["arms"][name] = dict(
                    aliases.score_aliases(text, case["answer"], case["aliases"]), generation_tokens=tokens
                )
            ledger.append(item)
            f.write(json.dumps(item, sort_keys=True) + "\n")
            f.flush()
            if i % 16 == 0 or i == len(cases):
                print("transfer scored", i, "/", len(cases), "seconds", round(time.perf_counter() - started), flush=True)
    record = dict(
        analysis=summarize(ledger),
        paired_groups=len(cases),
        exact_prefill_replay=cached,
        frozen_logits_verified=True,
        max_output_tokens=study.ground.MAX_ANSWER,
        input_sha256=sha(args.data / "test.jsonl"),
        preparation_sha256=sha(args.data / "manifest.json"),
        result_sha256=sha(result_file),
        code=docs,
        frozen_checkpoint=adapter,
        elapsed_seconds=time.perf_counter() - started,
        peak_mlx_bytes=reader.mx.get_peak_memory(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    study.ground.write(args.output / "summary.json", record)
    print(json.dumps(record, indent=2), flush=True)


def verify(args):
    source = read_data(args.data)
    result_file = args.output / "cases.jsonl"
    ledger = [json.loads(line) for line in result_file.read_text().splitlines()]
    report = json.loads((args.output / "summary.json").read_text())
    assert report["code"] == source_fingerprints()
    assert report["frozen_checkpoint"] == adapter_fingerprints()
    assert report["input_sha256"] == sha(args.data / "test.jsonl")
    assert report["preparation_sha256"] == sha(args.data / "manifest.json")
    assert report["result_sha256"] == sha(result_file)
    assert report["paired_groups"] == len(ledger) == N
    for case, row in zip(source, ledger, strict=True):
        assert all(case[k] == row[k] for k in ("id", "group", "type", "complete", "article_complete"))
        assert row["prompt_tokens"] == len(case["prompt_ids"])
        assert row["prompt_sha256"] == study.qa.stable(",".join(map(str, case["prompt_ids"])))
        assert set(row["arms"]) == set(ARMS)
        for arm in row["arms"].values():
            expected = 2 * arm["overlap"] / max(1, arm["answer_terms"] + arm["gold_terms"])
            assert expected == arm["f1"]
            assert 0 <= arm["generation_tokens"] <= study.ground.MAX_ANSWER
    assert report["analysis"] == summarize(ledger)
    print(f"Verified {N} paired input IDs, exact prompt tokens, artifact hashes and confidence intervals.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("score", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--data", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if name == "score":
            cmd.add_argument("--model", type=Path, required=True)
            cmd.add_argument("--hotpot-data", type=Path, required=True)
    args = parser.parse_args()
    (score if args.action == "score" else verify)(args)


if __name__ == "__main__":
    main()
