"""Budgeted reading of a compact index for multi-hop evidence retrieval."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
import re
import resource
import sys
import time
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "musique_ans_v1.0_dev.jsonl"
URL = "https://huggingface.co/datasets/voidful/MuSiQue/resolve/a7d9f9adf6191604fc67cde318ee1a86fcf7babc/musique_ans_v1.0_dev.jsonl"
DATA_SHA = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
LOCK = ROOT / "FROZEN.json"
WORD = re.compile(r"[a-zA-Z0-9]+")
STOP = frozenset("""
a an and are as at be been by can could did do does during for from had has have how i if in into
is it its just may more most of on one or other our out over said she so some than that the their
them then there these they this those through to too under up us very was we were what when where
whether which while who whose why will with would you your he his her were about after before also
""".split())

BUDGETS = {
    "words192": ("words", 192),
    "words320": ("words", 320),
    "words512": ("words", 512),
    "reads4": ("reads", 4),
}
PRIMARY = "words320"
SPLIT_SALT = "musique-memory-23nov2026-v1"
POLICIES = ("bm25", "tfidf", "mmr", "coverage", "iterative", "preview_bridge", "reveal_bridge")
OPTIONS = {
    "bm25": ({},),
    "tfidf": ({},),
    "mmr": tuple({"weight": w} for w in (.15, .35, .55, .75)),
    "coverage": tuple({"weight": w} for w in (.15, .35, .55, .75)),
    "iterative": tuple({"weight": w, "expansions": k} for w, k in itertools.product((.25, .5, .75), (6, 16))),
    "preview_bridge": tuple({"weight": w, "title_share": k} for w, k in itertools.product((.25, .5, .75), (.33, .67))),
    "reveal_bridge": tuple({"weight": w, "title_share": k} for w, k in itertools.product((.25, .5, .75), (.33, .67))),
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stable(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def informative(tokens) -> set[str]:
    return {term for term in tokens if term not in STOP and (len(term) > 2 or term.isdigit())}


def scale(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    return values / maximum if maximum > 0 else values


def download() -> None:
    if DATA.exists() and digest(DATA) == DATA_SHA:
        print("Verified local data")
        return
    DATA.parent.mkdir(exist_ok=True)
    stage = DATA.with_suffix(".tmp")
    urllib.request.urlretrieve(URL, stage)
    if digest(stage) != DATA_SHA:
        stage.unlink()
        raise ValueError("Unexpected dataset digest")
    stage.replace(DATA)


def load() -> list[dict]:
    if digest(DATA) != DATA_SHA:
        raise ValueError("Unexpected dataset digest")
    with DATA.open(encoding="utf-8") as records:
        rows = [json.loads(line) for line in records]
    if len(rows) != 2417 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Unexpected corpus size or duplicate IDs")
    return rows


def make_split(rows: list[dict]) -> list[tuple[str, str]]:
    """Split only. Support labels and source question ids never reach search."""
    parent = list(range(len(rows)))
    seen: dict[tuple[str, str], int] = {}

    def find(i: int) -> int:
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(rows):
        markers = [
            ("source", str(item["id"])) for item in row["question_decomposition"]
        ]
        markers += [
            ("title", " ".join(words(p["title"])))
            for p in row["paragraphs"] if p["is_supporting"]
        ]
        markers += [
            ("passage", stable(" ".join(words(p["paragraph_text"]))))
            for p in row["paragraphs"] if p["is_supporting"]
        ]
        for marker in markers:
            prev = seen.setdefault(marker, i)
            parent[find(i)] = find(prev)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    largest = max(groups.values(), key=len)
    split = [None] * len(rows)
    for indices in groups.values():
        group_id = stable("|".join(sorted(rows[i]["id"] for i in indices)))
        assignment = "topic_stress" if indices is largest else (
            "development" if int(stable(SPLIT_SALT + group_id)[:8], 16) % 5 == 0 else "test")
        for i in indices:
            split[i] = (assignment, group_id)
    return split


@dataclass(frozen=True)
class View:
    title: tuple[str, ...]
    preview: tuple[str, ...]
    cost: int


@dataclass
class ReadLedger:
    """The only object that can reveal a passage tail. Reads are charged here."""

    views: tuple[View, ...]
    mode: str
    budget: int
    _tails: tuple[tuple[str, ...], ...]
    selected: list[int]
    words_spent: int = 0

    def affordable(self) -> list[int]:
        if self.mode == "reads" and len(self.selected) >= self.budget:
            return []
        return [
            i for i, view in enumerate(self.views)
            if i not in self.selected
            and (self.mode == "reads" or self.words_spent + view.cost <= self.budget)
        ]

    def read(self, i: int) -> tuple[str, ...]:
        if i not in self.affordable():
            raise ValueError("Repeated, invalid, or unaffordable read")
        self.selected.append(i)
        self.words_spent += self.views[i].cost
        return self._tails[i]


def inputs(row: dict, preview_words: int) -> tuple:
    order = sorted(
        range(len(row["paragraphs"])),
        key=lambda i: stable(f'{SPLIT_SALT}:{row["id"]}:{i}'),
    )
    views, tails, support, original_ids = [], [], [], []
    for i in order:
        paragraph = row["paragraphs"][i]
        body = tuple(words(paragraph["paragraph_text"]))
        tail = body[preview_words:]
        # One word of overhead for a read, even if the fixed preview is complete.
        views.append(View(tuple(words(paragraph["title"])), body[:preview_words], 1 + len(tail)))
        tails.append(tail)
        support.append(bool(paragraph["is_supporting"]))
        original_ids.append(int(paragraph["idx"]))
    return words(row["question"]), tuple(views), tuple(tails), tuple(support), tuple(original_ids)


class Index:
    """Compact, query-local lexical search. Its constructor never sees tails."""

    def __init__(self, query: list[str], views: tuple[View, ...]):
        self.query = informative(query)
        self.views = views
        self.titles = [informative(v.title) for v in views]
        self.counts = []
        for view in views:
            keys = informative((*view.title, *view.preview))
            self.counts.append(Counter(x for x in (*view.title, *view.title, *view.preview) if x in keys))
        self.termsets = [set(counts) for counts in self.counts]
        dfs = Counter(x for tokens in self.termsets for x in tokens)
        size = len(views)
        self.idf = {x: math.log(1 + (size - df + .5) / (df + .5)) for x, df in dfs.items()}
        self.lengths = np.array([sum(counts.values()) for counts in self.counts], dtype=float)
        mean_length = max(1., self.lengths.mean())
        self.norm = 1.2 * (.25 + .75 * self.lengths / mean_length)
        self.bm25_base = scale(self.bm25({x: 1. for x in sorted(self.query)}))
        self.tfidf_base = scale(self.tfidf_cosine())
        self._similarity: np.ndarray | None = None

    def bm25(self, weights: dict[str, float]) -> np.ndarray:
        scores = []
        terms = self.idf
        for counts, norm in zip(self.counts, self.norm):
            score = 0.
            for x, w in weights.items():
                tf = counts.get(x, 0)
                if tf:
                    score += w * terms[x] * 2.2 * tf / (tf + norm)
            scores.append(score)
        return np.array(scores)

    def tfidf_cosine(self) -> np.ndarray:
        q_weights = {x: self.idf[x] for x in sorted(self.query) if x in self.idf}
        q_norm = max(1e-12, math.sqrt(sum(w*w for w in q_weights.values())))
        scores = []
        for counts in self.counts:
            doc = {x: self.idf[x] * (1 + math.log(tf)) for x, tf in counts.items()}
            norm = max(1e-12, math.sqrt(sum(w*w for w in doc.values())))
            scores.append(sum(doc.get(x, 0.) * w for x,w in q_weights.items()) / (q_norm * norm))
        return np.array(scores)

    def similarity(self) -> np.ndarray:
        if self._similarity is None:
            weighted = [{x: self.idf[x] * (1 + math.log(tf)) for x,tf in c.items()} for c in self.counts]
            norms = [max(1e-12, math.sqrt(sum(w*w for w in row.values()))) for row in weighted]
            n = len(weighted)
            sims = np.zeros((n,n))
            for i in range(n):
                for j in range(i):
                    s = sum(w * weighted[j].get(x, 0.) for x,w in weighted[i].items())
                    sims[i,j] = sims[j,i] = s / (norms[i] * norms[j])
            self._similarity = sims
        return self._similarity

    def bridge(self, opened: list[tuple[int, tuple[str, ...]]], title_share: float) -> np.ndarray:
        """Rank new candidates by uncommon cross-passage links from paid reads."""
        revealed = {}
        for age, (i, tokens) in enumerate(reversed(opened)):
            anchor_weight = (1. + self.bm25_base[i]) / (1. + .2 * age)
            for term in sorted(informative(tokens) - self.query):
                if term in self.idf:
                    revealed[term] = max(revealed.get(term, 0.), anchor_weight * self.idf[term])
        if not revealed:
            return np.zeros(len(self.views))
        bm25_link = scale(self.bm25(revealed))
        title_link = scale(np.array([
            sum(revealed[x] for x in sorted(titles & revealed.keys())) / math.sqrt(max(1, len(titles)))
            for titles in self.titles
        ]))
        return title_share * title_link + (1. - title_share) * bm25_link


def select(index: Index, ledger: ReadLedger, method: str, config: dict[str, float]) -> None:
    history: list[tuple[int, tuple[str, ...]]] = []
    seen_query_words: set[str] = set()
    while available := ledger.affordable():
        weight = config.get("weight", 0.)
        if method == "bm25":
            scores = index.bm25_base
        elif method == "tfidf":
            scores = index.tfidf_base
        elif method == "mmr":
            penalty = index.similarity()[:, ledger.selected].max(axis=1) if history else 0.
            scores = (1. - weight) * index.bm25_base - weight * penalty
        elif method == "coverage":
            missing = index.query - seen_query_words
            novelty = scale(index.bm25({x: 1. for x in sorted(missing)}))
            scores = (1. - weight) * index.bm25_base + weight * novelty
        elif method == "iterative":
            if history:
                # Classic exact pseudo-relevance feedback over purchased passages.
                full = Counter(x for _, tokens in history for x in tokens if x in index.idf and x not in index.query)
                weighted = sorted(full, key=lambda x: (-(1 + math.log(full[x])) * index.idf[x], x))
                expand = scale(index.bm25({x: index.idf[x] for x in weighted[:config["expansions"]]}))
                scores = (1. - weight) * index.bm25_base + weight * expand
            else:
                scores = index.bm25_base
        elif method in {"preview_bridge", "reveal_bridge"}:
            link = index.bridge(history, config["title_share"])
            scores = (1. - weight) * index.bm25_base + weight * link
        else:
            raise ValueError(method)
        winner = max(available, key=lambda i: (scores[i], -i))
        tail = ledger.read(winner)
        doc = index.views[winner]
        visible = doc.title + doc.preview
        history.append((winner, visible if method == "preview_bridge" else visible + tail))
        seen_query_words.update(index.query & index.termsets[winner])


def run_one(index: Index, views: tuple[View, ...], tails: tuple[tuple[str, ...], ...],
            support: tuple[bool, ...], mode: str, budget: int, method: str, config: dict) -> dict:
    ledger = ReadLedger(views, mode, budget, tails, [])
    begin = time.process_time_ns()
    select(index, ledger, method, config)
    cpu_ns = time.process_time_ns() - begin
    support_total = sum(support)
    found = sum(support[i] for i in ledger.selected)
    return {
        "coverage": found,
        "complete": int(found == support_total),
        "words": ledger.words_spent,
        "reads": len(ledger.selected),
        "selection": ledger.selected,
        "cpu_ns": cpu_ns,
    }


def dev() -> None:
    rows = load()
    jobs = [(m, i, c) for m, configs in OPTIONS.items() for i, c in enumerate(configs)]
    scores = {key: [] for key in [(m,i) for m,i,_ in jobs]}
    start = time.perf_counter()
    for row, (split, group) in zip(rows, make_split(rows)):
        if split != "development":
            continue
        q, views, tails, support, ids = inputs(row, 24)
        index = Index(q, views)
        for method, i, config in jobs:
            result = run_one(index, views, tails, support, *BUDGETS[PRIMARY], method, config)
            scores[(method,i)].append((result["complete"], result["coverage"], -result["words"]))
    outcomes = [
        {"method": m, "i": i, "config": c, "n": len(scores[(m,i)]),
         "score": np.mean(scores[(m,i)], axis=0).tolist()}
        for m,i,c in jobs
    ]
    best = {
        method: max((r for r in outcomes if r["method"] == method),
                    key=lambda r: (tuple(r["score"]), -r["i"]))
        for method in POLICIES
    }
    comparator = max((r for m,r in best.items() if m != "reveal_bridge"),
                     key=lambda r: tuple(r["score"]))
    root_out = ROOT / "results"
    root_out.mkdir(exist_ok=True)
    (root_out / "development.json").write_text(json.dumps(
        {"selection": best, "best_comparator": comparator["method"],
         "trials": outcomes, "wall_seconds": time.perf_counter()-start}, indent=2)+"\n")
    print("Development configuration", best, "best comparator", comparator["method"])


def oracle(support, views, mode, budget):
    chosen = sorted((i for i, gold in enumerate(support) if gold), key=lambda i: views[i].cost)
    if mode == "reads":
        return min(len(chosen), budget)
    spend = count = 0
    for i in chosen:
        spend += views[i].cost
        if spend > budget:
            break
        count += 1
    return count


def evaluate(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    lock = json.loads(LOCK.read_text())
    for name, expected in lock["sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    rows = load()
    started = time.perf_counter()
    destination.mkdir(parents=True)
    with (destination/"rows.jsonl").open("w") as output:
        for row, (split, group) in zip(rows, make_split(rows)):
            if split == "development":
                continue
            for preview_words in lock["previews"]:
                if preview_words != 24 and split == "topic_stress":
                    continue
                begin = time.process_time_ns()
                q, views, tails, support, ids = inputs(row, preview_words)
                index = Index(q, views)
                setup_ns = time.process_time_ns() - begin
                for budget_name, (mode, budget) in BUDGETS.items():
                    # Deterministic order rotation keeps timing comparing cold/warm policies fair.
                    methods = sorted(POLICIES, key=lambda m: stable(row["id"] + m))
                    for method in methods:
                        result = run_one(index, views, tails, support, mode, budget,
                                         method, lock["selection"][method]["config"])
                        result["selection"] = [ids[i] for i in result["selection"]]
                        output.write(json.dumps(
                            {"id":row["id"], "split":split, "group":group,
                             "preview_words":preview_words,"budget":budget_name,
                             "method":method,"total_support":sum(support),
                             "gold": [ids[i] for i, v in enumerate(support) if v],
                             "oracle":oracle(support, views, mode, budget),
                             "index_words":sum(len(v.title)+len(v.preview) for v in views),
                             "corpus_words":sum(len(v.title)+len(v.preview)+v.cost-1 for v in views),
                             "setup_cpu_ns":setup_ns,**result},
                            separators=(",", ":"))+"\n")
    meta = {"data_sha256": DATA_SHA,"lock_sha256":digest(LOCK),
            "row_sha256":digest(destination/"rows.jsonl"),
            "wall_seconds":time.perf_counter()-started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform=="darwin" else 1024),
            "platform":platform.platform(),"python":platform.python_version(),"numpy":np.__version__}
    (destination/"metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("development")
    ev = sub.add_parser("run")
    ev.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        download()
    elif args.command == "development":
        if LOCK.exists():
            raise ValueError("Development is closed after freezing")
        dev()
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
