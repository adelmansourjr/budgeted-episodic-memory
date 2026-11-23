# Budgeted episodic memory

**Question-last follow-up:** [Matched-budget Qwen evaluation](qa-question-last/) tests an adapter trained on the stronger prompt. It scores **42.27** versus **29.47** for frozen Qwen on 320 fresh HotpotQA questions.

**Reader adaptation:** [Fixed-budget Qwen answer quality](qa-grounding/) improved HotpotQA answer F1 by **+15.99 points** over frozen Qwen at identical input tokens. The controlled support-supervision test is reported alongside it.

**Compression stress tests:** [Sentence selection](sentence-bottleneck/) saved 58.6% of prompt tokens but lost 8.0 F1 points. [Conservative whole-article selection](article-preservation/) preserved the quality tolerance at a 2.1% token reduction. The two tests identify a bottleneck between retrieval coverage and reliable answer efficiency.

**Cross-corpus follow-up:** [HotpotQA with a local Qwen reader](cross-corpus-qa/) replicated the complete-evidence gain (+12.0 points against preview-only retrieval). Answer F1 improved by an uncertain +2.24 points.

**How much is it worth reopening an event after seeing a clue in another one?** This project combines an exact fixed-batch benchmark for a finite routing problem and a held-out text retrieval experiment based on real multi-hop questions. The text experiment uses a compact memory consisting of article titles and partial passages. A policy can buy the hidden remainder of each passage under a fixed reading budget.

On **1,332 held-out MuSiQue questions** at the predeclared 24-word preview and 320-word budget, a ranker that used newly purchased evidence retrieved all annotated support paragraphs in **54.50%** of episodes. Its strongest development-selected control, the same bridge policy limited to the original previews, reached **50.30%**. The paired improvement was **4.20 percentage points** (95% component-bootstrap interval **1.55 to 7.05**). Mean paid volume was **312.46 vs 312.56 words**. Four equal reads yielded an inconclusive result. The advantage almost disappeared with 48-word previews.

This is a test of evidence acquisition in a closed set of text passages. Answer generation, video inference and embedding or checkpoint compression are separate questions.

## How the policies work

1. Index each title and the first few words of each passage. Use the question to identify a promising first read.
2. Purchase the hidden remainder at a cost of one plus the number of newly visible words.
3. Re-rank by query relevance and uncommon terms connecting what has been read to other candidates, with a separate contribution for matches to article titles.
4. Repeat until the next purchase would exceed the budget.

The only difference in the main comparison is whether step 3 may use words revealed by earlier purchases. We tuned both variants on the same six-setting grid before evaluating either on the test set. Development selected identical settings. Five other controls include BM25, TF-IDF cosine, coverage-aware retrieval, MMR, and iterative BM25 feedback from purchased passages.

The data include language that triggers distractor matches. The method remains lexical and cannot recognise all paraphrases. Titles, previews and cost metadata are supplied to every policy. The candidate set and evidence annotations derive from a published multi-hop benchmark.

## Fixed protocol and results

| Policy at 24-word preview and 320-word cap | Complete support sets | Support recall | Mean paid words | CPU ms per episode |
|---|---:|---:|---:|---:|
| BM25 | 29.73% | 63.61% | 311.79 | 0.414 |
| TF-IDF cosine | 30.56% | 64.38% | 311.96 | 0.414 |
| MMR | 27.40% | 62.70% | 311.73 | 0.441 |
| Residual query coverage | 29.73% | 63.97% | 311.89 | 0.482 |
| Iterative BM25 | 44.74% | 71.31% | 312.36 | 0.700 |
| Preview-based bridge | 50.30% | 74.59% | 312.56 | 0.972 |
| Reveal-based bridge | **54.50%** | **75.41%** | 312.46 | 1.092 |
| Budget-feasible oracle | 92.42% | — | — | — |

There were 110 gains, 54 losses and 1,168 ties against the preview bridge. Giving each of the 330 components equal weight yielded a smaller gain of **2.32 points** (95% interval **0.08 to 4.62**). The independent iterative-BM25 comparison was **+9.76 points** (95% interval **5.60 to 13.63**).

Costs include a shared index of **539.17 words** on average, in addition to the reported paid volume. The full candidate corpus averaged **1,675.23 words** including titles. The unit measures word access. It cannot be converted directly into model tokens or FLOPs. CPU time includes local parsing, indexing, ranking and read accounting, with one shared index charged to each method. The adaptive bridge buys better evidence at a small CPU premium on this hardware. The measured loop includes bookkeeping for the synthetic ledger, but excludes loading the source file and JSON output.

### When does reopening matter?

| Setting | Iterative BM25 | Preview bridge | Reveal bridge | Reveal minus preview (95% CI) |
|---|---:|---:|---:|---:|
| 8-word preview, 320-word cap | 43.32% | 24.70% | 45.05% | +20.35 pp [15.10, 26.80] |
| 24-word preview, 192-word cap | 32.96% | 38.14% | 39.86% | +1.73 pp [−0.08, 3.83] |
| 24-word preview, 320-word cap | 44.74% | 50.30% | 54.50% | +4.20 pp [1.55, 7.05] |
| 24-word preview, 512-word cap | 57.73% | 66.14% | 68.09% | +1.95 pp [−0.72, 4.97] |
| 24-word preview, exactly 4 reads allowed | 34.16% | 43.09% | 45.35% | +2.25 pp [−1.27, 6.26] |
| 48-word preview, 320-word cap | 59.61% | 70.80% | 71.17% | +0.38 pp [−1.16, 2.22] |
| Largest connected topic, 24/320 | 19.71% | 19.28% | 23.04% | +3.77 pp, descriptive |

Prior access changes the result: under very short previews, content feedback of either kind accounts for much of the gain. The large 8-word contrast is against the preview-only control. In that condition the gain over iterative BM25 is just 1.73 points. A richer index increases the information shared upfront, so the 48-word result uses a larger total word budget. These sensitivity contrasts do not establish a universal advantage. The largest topic is one cluster and is shown without an across-topic interval.

### Separate exact result

For arbitrary finite route trees and integer read prices, [the exact allocation proof](THEORY.md) gives the best fixed batch at every budget. It also gives the optimal adaptive success probability if successive events reveal exact next-event pointers. Exhaustive tests check a weighted example. This proof uses a known tree-knapsack reduction and an explicit information model. The MuSiQue effect comes from the experiment above, independently of that model.

## Reproduce

Use Python 3.10+ and NumPy 2.3.5. The run used one CPU process and no model files. Stored results contain 131,208 method-condition-episode records. The experiment finished in 44.1 seconds and its peak RSS was 102.5 MB on the recorded machine.

```shell
python -m pip install -r requirements.txt
python study.py download
python -m unittest -v
python routing_bound.py
python analyse.py results/final/rows.jsonl.gz --write /tmp/episodic-summary.json
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python study.py run --output /tmp/episodic-regen
python replay.py /tmp/episodic-regen/rows.jsonl
```

Data and code digests pin the frozen run. `analyse.py` independently recomputes support metrics, paid costs, oracle counts and confidence intervals from stored selections. `replay.py` checks every new selection against the original, allowing timing differences. The [data notes](data/README.md) contain licensing and provenance. The [protocol](PROTOCOL.md), [lock file](FROZEN.json), [prior work notes](PAPERS.md), and [complete results](results/final/summary.json) preserve the exact scope and the failed settings.
