# Does better evidence retrieval produce better answers?

I transferred the MuSiQue retrieval policies to HotpotQA without retuning them and gave their selected passages to a fixed, local Qwen3-1.7B reader. Reopening evidence recovered **both supporting articles on 91.15% of questions**, up from **79.17%** for the preview-only bridge at the same reading cap. This was a **+11.98-point** paired gain [95% interval **+6.25 to +17.71**]. The answer-F1 gain of **+2.24 points** [**−1.17 to +5.75**] was inconclusive and did not pass the prespecified success criterion.

## Design

HotpotQA distractor development has 7,405 examples with ten supplied candidate articles. Grouping the dataset by transitive overlap of normalized gold support titles yielded 6,471 components. I used one deterministic hash-selected question from each of 192 distinct held-out components. Eight separate cases served as an engineering smoke check. All three algorithms reused the same code, development-selected settings, 24-word previews and 320-unit cap from the MuSiQue pilot. Each acquisition costs one unit plus the number of newly visible words. The reader received the selected articles in the same deterministic order for every method. The gold titles were used for grouping, evaluation and the separate annotation-supplied diagnostic.

I held the local quantized checkpoint, prompt and greedy decoding fixed. I asked for the short answer or an explicit unknown; the scored answer was the first nonempty line. The primary outcome was token-overlap answer F1 with the HotpotQA yes/no normalization rules. The frozen criterion was a point estimate of at least +2 F1 points and a positive lower 95% paired bootstrap endpoint, compared with the preview-only bridge. The intervals resample distinct retained support components. This estimand weights support components equally. [Frozen protocol](PROTOCOL.md) · [Code and model fingerprints](LOCK.json)

| Input selection | Full support | Answer F1 | Exact match | Mean paid units | Mean Qwen input tokens |
|---|---:|---:|---:|---:|---:|
| Preview-only bridge | 79.17% | 32.77 | 25.00% | 291.32 | 775.89 |
| Reveal bridge | 91.15% | 35.00 | 26.56% | 291.51 | 776.34 |
| Iterative lexical feedback | 88.02% | 33.72 | 25.52% | 291.68 | 771.01 |
| No articles | — | 2.13 | 1.56% | — | 77.99 |
| Annotated support articles | 100% | 46.10 | 31.77% | — | 305.28 |

The reveal bridge's comparison with iterative feedback was unresolved for full support (+3.13 points [−1.56, +7.81]) and answer F1 (+1.28 [−2.03, +4.64]). Against preview only, 29 cases gained complete support and six lost it. The gain cases contributed most of the overall F1 change descriptively. Those subsets depend on the retrieval outputs. The annotation-supplied diagnostic uses fewer tokens because it can select just the two relevant articles and omit distractors. Both the label access and shorter context affect that comparison.

This distinguishes cross-corpus evidence acquisition from reliable answer improvement. Both datasets use Wikipedia and possible pretraining overlap remains unmeasured. The sample is small, uses a single reader and compares lexical methods inside a closed candidate set. Inference time, video memory, learned compression and multi-model generality remain open.

## Reproduce

The original HotpotQA release is available through the [project site](https://hotpotqa.github.io/) and this experiment's source hash is pinned to the original distractor file. The dataset and model are fetched or supplied separately. The archive publishes no article or answer text. The evidence audit uses Python and NumPy; end-to-end replay requires the existing matching local Qwen checkpoint, MLX and mlx-lm 0.32.0. No model download is performed by the study.

```sh
python cross-corpus-qa/run.py verify
python cross-corpus-qa/run.py fetch /tmp/hotpot-dev-original.json
python cross-corpus-qa/audit_selections.py /tmp/hotpot-dev-original.json
python cross-corpus-qa/run.py score /tmp/hotpot-dev-original.json /path/to/local/Qwen3-1.7B-4bit /tmp/hotpot-qa-replay
```

The final run evaluated 960 logical calls and executed 773 distinct within-question generations. Peak process RSS was 1.46 GB and peak MLX allocations were 1.81 GB, as reported by separate instruments. Total elapsed time was 435 seconds. Deduplication and the shared device make runtime unsuitable for comparing policy latency. [Archived scoring ledger](results/raw.jsonl) · [Summary](results/summary.json)

[HotpotQA](https://aclanthology.org/D18-1259/) and [MuSiQue](https://aclanthology.org/2022.tacl-1.31/) supply the evidence tasks. [IRCoT](https://aclanthology.org/2023.acl-long.557/) interleaves reasoning with retrieval using an LLM. Here the retrieval policies remain lightweight and fixed before generation starts. [MARC](https://proceedings.iclr.cc/paper_files/paper/2026/hash/16049e0c3f47899091ac46f8b3afb178-Abstract-Conference.html) motivates asking what evidence a compact representation should preserve. It evaluates video reasoning and a trained student, beyond the scope of the text transfer here.
