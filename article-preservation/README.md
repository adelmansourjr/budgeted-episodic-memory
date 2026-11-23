# Protecting whole articles gives up most of the compression

In the sentence experiment, dropping most of the input harmed quality and often removed support inside retained articles. Here a learned article scorer prunes with a conservative development-selected threshold and keeps every sentence of the articles it retains. The resulting policy preserved the predefined F1 tolerance, but saved only **2.11%** of prompt tokens. It failed the joint compression-and-quality criterion.

| Reader input | Input tokens | Mean articles | Complete support | Answer F1 | Exact match |
|---|---:|---:|---:|---:|---:|
| Full acquired context | 736.55 | 6.06 | 89.06% | 32.00 | 25.52% |
| Fixed depth six | 660.63 | 5.34 | 86.98% | 29.86 | 23.96% |
| Confidence-dependent depth | 720.98 | 5.97 | 88.54% | 31.48 | 25.00% |

On **192 previously unused support-title groups**, adaptive depth changed the article set on just 17 cases and changed F1 on one. Its difference from full context was **−0.52 points [−1.56, 0.00]**. The fixed-depth control reduced tokens by **10.31%**, with a difference of **−2.14 F1 points [−5.80, +1.40]**. That interval crosses the prespecified −4-point tolerance. The adaptive policy's F1 comparison against fixed depth was **+1.62 points [−1.79, +5.14]** and it spent 9.14% more prompt tokens. No policy passed the combined gate.

The variants share the acquired text corpus, ranking scores, fixed local Qwen3-1.7B checkpoint, decoding and prompt. The operating points are selected on article relevance and prompt length from 160 separate development groups. Full input contained all support on 145 development questions. Adaptive retained 143 and fixed retained 138. The score mass used for pruning is a heuristic, without a distribution-free guarantee about support recall. These results concern the costs of one reader under a limited preparation budget. The paired intervals are over cases, and only a few cases differ under the most conservative policy.

[Protocol](PROTOCOL.md) · [Frozen article scorer and selection](FIT.json) · [Scoring ledger](results/rows.jsonl) · [Outcomes](results/summary.json) · [Sentence bottleneck](../sentence-bottleneck/)

## Reproduce

Use the separately acquired checksum-pinned HotpotQA artifact and matching local checkpoint. No answer or Wikipedia text is included in the committed ledgers. The input audit verifies that each selection is a union of complete acquired articles and recalculates all counts with the pinned local tokenizer.

```sh
python article-preservation/study.py verify
python article-preservation/audit_inputs.py /tmp/hotpot-dev-original.json /path/to/Qwen3-1.7B-4bit
python article-preservation/study.py score /tmp/hotpot-dev-original.json /path/to/Qwen3-1.7B-4bit /tmp/hotpot-article-replay
```

The measured run took 214 seconds, with 285 distinct generations, 1.47 GB maximum process RSS and a separately reported 1.82 GB peak from the MLX allocator.
