"""Freeze an untouched HotpotQA comparison panel at exact matched input budgets."""

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import resource
import statistics

from prepare_data import HERE, MAX_PROMPT, build_row, digest, load_prior, verify_fingerprints
from prepare_transfer import fitting_titles, normalized_title
from transformers import AutoTokenizer

PANEL_START = 4292
PANEL_SIZE = 192
MARK_QUESTION = "\n\nQuestion: "
MARK_PASSAGES = "\n\nPassages:\n"
MARK_ANSWER = "\n\nAnswer:"


def reorder(tokenizer, ids):
    decode = lambda values: tokenizer.decode(
        values, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    prompt = decode(ids)
    retokenized = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if retokenized["input_ids"] != ids:
        raise ValueError("Decoded prompt did not round-trip to the original IDs")
    q_char = prompt.index(MARK_QUESTION)
    p_char = prompt.index(MARK_PASSAGES, q_char + len(MARK_QUESTION))
    a_char = prompt.rindex(MARK_ANSWER)
    assert 0 <= q_char < p_char < a_char
    chars = (q_char + 2, p_char + 2, a_char + 2)
    offsets = retokenized["offset_mapping"]
    starts = [start for start, _ in offsets]
    assert starts == sorted(starts)
    q, p, a = [bisect_right(starts, char) - 1 for char in chars]
    for idx, char in zip((q,p,a), chars, strict=True):
        start, end = offsets[idx]
        assert start <= char < end
        # Label can share leading whitespace, but may not share prior semantic content.
        assert prompt[start:char].strip() == ""
    assert 0 <= q < p < a < len(ids)
    permuted = ids[:q] + ids[p:a] + ids[q:p] + ids[a:]
    assert Counter(permuted) == Counter(ids) and len(permuted) == len(ids)
    after = decode(permuted)
    verify_reordered(prompt, after)
    return permuted


def verify_reordered(before, after):
    # Check semantic body preservation independently from token-level invariants.
    prefix, text = before.split(MARK_QUESTION, maxsplit=1)
    question, text = text.split(MARK_PASSAGES, maxsplit=1)
    evidence, suffix = text.rsplit(MARK_ANSWER, maxsplit=1)
    prefix_after, text = after.split(MARK_PASSAGES, maxsplit=1)
    evidence_after, text = text.rsplit(MARK_QUESTION, maxsplit=1)
    question_after, suffix_after = text.rsplit(MARK_ANSWER, maxsplit=1)
    if (prefix, question, evidence, suffix) != (prefix_after, question_after, evidence_after, suffix_after):
        raise ValueError("Question-last permutation changed prompt contents")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("hotpot", type=Path)
    cli.add_argument("model", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh temporary destination")
    qa = load_prior()
    lock = verify_fingerprints(qa, args.model)
    config = json.loads((HERE.parent / "FROZEN.json").read_text())["selection"]["reveal_bridge"]["config"]
    fitting_gold, fitting_acquired = fitting_titles(qa, args.hotpot, config)
    forbidden = fitting_gold | fitting_acquired
    groups = qa.representatives(qa.load(args.hotpot))
    eligible = []
    omitted = 0
    for component_index, (group, _, row) in enumerate(groups[PANEL_START:], PANEL_START):
        if row["type"] != "comparison":
            continue
        if any(normalized_title(t, qa.retrieval) in forbidden for t, _ in row["supporting_facts"]):
            omitted += 1
            continue
        eligible.append((component_index, group, row))
    panel = eligible[:PANEL_SIZE]
    if len(panel) < PANEL_SIZE:
        raise ValueError("Too few eligible comparison components")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("This audit needs token offsets")
    args.output.mkdir(parents=True)
    output = args.output / "test.jsonl"
    case_counts, prompt_lengths = Counter(), []
    with output.open("x", encoding="utf-8") as handle:
        for _, group, source in panel:
            row = build_row(tokenizer, source, group, qa.retrieval, config)
            row["question_last_ids"] = reorder(tokenizer, row["prompt_ids"])
            lengths = len(row["prompt_ids"]), len(row["question_last_ids"])
            assert lengths[0] == lengths[1] and lengths[0] <= MAX_PROMPT
            prompt_lengths.append(lengths[0])
            case_counts["complete"] += row["complete"]
            case_counts["article_complete"] += row["article_complete"]
            case_counts["truncated"] += row["truncated"]
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "panel": "HotpotQA comparison representatives",
        "prospective_range_start": PANEL_START,
        "last_included_component_index": panel[-1][0],
        "num_test": PANEL_SIZE,
        "eligible_components": len(eligible),
        "excluded_for_fitting_title_overlap": omitted,
        "prompt_lengths_equal": True,
        "prompt_token_multisets_equal": True,
        "normalized_gold_title_disjoint_from_fitting_gold": True,
        "normalized_gold_title_disjoint_from_fitting_acquired": True,
        "counts": dict(case_counts),
        "mean_prompt_tokens": statistics.mean(prompt_lengths),
        "max_prompt_tokens": max(prompt_lengths),
        "source_sha256": qa.DATA_DIGEST,
        "script_sha256": digest(Path(__file__)),
        "prompt_builder_sha256": digest(HERE / "prepare_data.py"),
        "fitting_title_builder_sha256": digest(HERE / "prepare_transfer.py"),
        "base_model_fingerprints": lock["model"],
        "test_sha256": digest(output),
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
