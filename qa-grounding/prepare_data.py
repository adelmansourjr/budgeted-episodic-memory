"""Rebuild disjoint HotpotQA prompts and causal sentence-supervision masks."""

import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import time

os.environ.update({
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "cross-corpus-qa"
SPLITS = {"train": (3268, 3908), "dev": (3908, 3972), "test": (3972, 4292)}
MAX_PROMPT = 1024
INSTRUCTION = "Answer using the passages. Give only the shortest answer phrase or yes/no."


def load_prior():
    spec = importlib.util.spec_from_file_location("grounding_prior_qa", PRIOR / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_fingerprints(qa, model):
    lock = json.loads((PRIOR / "LOCK.json").read_text())
    for directory, entries in ((PRIOR, lock["files"]), (model, lock["model"])):
        for name, sha in entries.items():
            if digest(directory / name) != sha:
                raise ValueError(f"Unexpected fingerprint for {directory / name}")
    return lock


def selected_sentences(row, retrieval, config):
    # The selector receives no support labels, answer spans or gold article IDs.
    mapped = {
        "id": row["_id"],
        "question": row["question"],
        "paragraphs": [
            {"idx": i, "title": title, "paragraph_text": " ".join(sentences), "is_supporting": False}
            for i, (title, sentences) in enumerate(row["context"])
        ],
    }
    query, views, tails, support, indices = retrieval.inputs(mapped, 24)
    selected = retrieval.run_one(
        retrieval.Index(query, views), views, tails, support, "words", 320, "reveal_bridge", config,
    )
    ids = {indices[i] for i in selected["selection"]}
    sentences = [
        (title, j, text)
        for i, (title, texts) in enumerate(row["context"]) if i in ids
        for j, text in enumerate(texts) if text.strip()
    ]
    return sentences, selected["words"]


def format_content(question, sentences):
    # Sentence hidden states must be able to condition on the question.
    text = f"{INSTRUCTION}\n\nQuestion: {question}\n\nPassages:\n"
    endpoints = []
    previous = None
    for title, _, sentence in sentences:
        if title != previous:
            text += ("\n\n" if previous is not None else "") + f"{title}\n"
        else:
            text += " "
        text += sentence
        endpoints.append(len(text.rstrip()))
        previous = title
    return text + "\n\nAnswer:", endpoints


def formatted_prompt(tokenizer, question, sentences):
    content, endpoints = format_content(question, sentences)
    messages = [{"role": "user", "content": content}]
    options = {"add_generation_prompt": True, "enable_thinking": False}
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    origin = rendered.find(content)
    if origin < 0 or rendered.count(content) != 1:
        raise ValueError("Could not locate the unmodified user turn in the Qwen chat template")
    tokens = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = tokens["input_ids"]
    if len(ids) > MAX_PROMPT:
        return None
    canonical = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options)
    if ids != canonical or not ids:
        raise ValueError("Offset tokenization differs from canonical Qwen chat input")
    offsets = tokens["offset_mapping"]
    starts = [start for start, _ in offsets]
    assert starts == sorted(starts), "Token offset starts expected to be monotonic"
    positions = []
    for endpoint in endpoints:
        absolute = origin + endpoint
        index = bisect_left(starts, absolute) - 1
        start, end = offsets[index]
        if not 0 <= index < len(ids) or not start < absolute <= end:
            raise ValueError("A support label did not align with its sentence endpoint")
        positions.append(index)
    if positions != sorted(set(positions)) or len(positions) != len(sentences):
        raise ValueError("Sentence supervision did not align with distinct causal positions")
    return ids, positions


def build_row(tokenizer, source, group, retrieval, config):
    sentences, spent = selected_sentences(source, retrieval, config)
    acquired_count = len(sentences)
    prompt = formatted_prompt(tokenizer, source["question"], sentences)
    while prompt is None and sentences:
        sentences.pop()
        prompt = formatted_prompt(tokenizer, source["question"], sentences)
    if prompt is None:
        raise ValueError("Question alone exceeds token cap")
    ids, positions = prompt
    gold = {tuple(pair) for pair in source["supporting_facts"]}
    shown = {(title, i) for title, i, _ in sentences}
    shown_articles = {t for t, _ in shown}
    answer_ids = tokenizer.encode(source["answer"], add_special_tokens=False)
    if not answer_ids:
        raise ValueError("Answer has no tokens")
    return {
        "id": source["_id"],
        "group": group,
        "type": source["type"],
        "prompt_ids": ids,
        "answer_ids": answer_ids,
        "answer": source["answer"],
        "positions": positions,
        "support_labels": [int((title, i) in gold) for title, i, _ in sentences],
        "complete": int(gold <= shown),
        "article_complete": int({t for t, _ in gold} <= shown_articles),
        "truncated": acquired_count != len(sentences),
        "spent_words": spent,
    }


def title_keys(rows, qa):
    return {
        " ".join(qa.retrieval.words(title))
        for _, _, source in rows for title, _ in source["supporting_facts"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("Choose a fresh temporary output directory")
    qa = load_prior()
    model_lock = verify_fingerprints(qa, args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("The fast tokenizer is required for source span offsets")
    config = json.loads((HERE.parent / "FROZEN.json").read_text())["selection"]["reveal_bridge"]["config"]
    groups = qa.representatives(qa.load(args.data))
    seen = title_keys(groups[:SPLITS["train"][0]], qa)
    test_gold = title_keys(groups[slice(*SPLITS["test"])], qa)
    args.output.mkdir(parents=True)
    summary = {}
    for split, (start, end) in SPLITS.items():
        batch = groups[start:end]
        keys = title_keys(batch, qa)
        if len(batch) != end-start or not seen.isdisjoint(keys):
            raise ValueError("Unexpected split size or overlap in supporting article titles")
        seen.update(keys)
        path = args.output / f"{split}.jsonl"
        counts = Counter()
        token_counts = []
        with path.open("x", encoding="utf-8") as output:
            for group, _, source in batch:
                example = build_row(tokenizer, source, group, qa.retrieval, config)
                counts["case_count"] += 1
                counts["complete"] += example["complete"]
                counts["article_complete"] += example["article_complete"]
                counts["truncated"] += int(example["truncated"])
                counts["support_labels"] += len(example["positions"])
                counts["positive_labels"] += sum(example["support_labels"])
                counts["examples_with_test_title_in_candidates"] += int(
                    split != "test" and any(
                        " ".join(qa.retrieval.words(title)) in test_gold for title, _ in source["context"]
                    )
                )
                token_counts.append(len(example["prompt_ids"]))
                output.write(json.dumps(example, separators=(",", ":")) + "\n")
        summary[split] = {
            "group_slice": [start, end],
            "sha256": digest(path),
            "min_prompt": min(token_counts),
            "mean_prompt": statistics.mean(token_counts),
            "max_prompt": max(token_counts),
            **counts,
        }
        print(split, summary[split], flush=True)
    manifest = {
        "corpus_sha256": qa.DATA_DIGEST,
        "script_sha256": digest(Path(__file__)),
        "base_model_fingerprints": model_lock["model"],
        "max_prompt_tokens": MAX_PROMPT,
        "eos_token_id": tokenizer.eos_token_id,
        "splits": summary,
        "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "build_wall_seconds": time.monotonic()-started,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
