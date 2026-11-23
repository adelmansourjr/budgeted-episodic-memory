# Where sentence compression loses multi-hop evidence

A learned sentence bottleneck cut local Qwen input tokens by **58.64%** at a fixed upstream retrieval policy. Answer F1 fell by **8.02 points** [95% paired interval **−12.65, −3.58**]. This missed the prespecified quality criterion. Adding a cross-article preference to the same learned ranker changed F1 by **+0.13 points [−1.43, +1.69]** at the same cap.

| Input | Mean Qwen input tokens | Both support articles represented | All support sentences retained | Answer F1 | Exact match |
|---|---:|---:|---:|---:|---:|
| Full retrieved articles | 738.29 | 87.50% | 87.50% | 29.97 | 24.22% |
| Independent sentence scores | 305.67 | 71.48% | 46.88% | 21.81 | 17.97% |
| Bridge-first sentence scores | 305.38 | 71.48% | 46.09% | 21.94 | 17.97% |

The estimate pertains to the existing local Qwen3-1.7B checkpoint and a fixed closed candidate set. We fit a 13-feature sentence-support ranker on 1,400 support-title groups, chose the 256-additional-token cap and bridge coefficient using 160 separate groups, then evaluated both methods on 256 new groups. Scoring uses greedily generated short answers with identical prompts and templates. The independent selector fits the same ranker and shares the cap. A development-only tokenizer check caught an API return type that incorrectly counted a mapping's keys. The final lock was generated after explicitly requiring a flat list of token IDs. No held-out text was scored before that correction. [Protocol](PROTOCOL.md) · [Fit lock](FIT.json)

## Diagnostic

Of the 224 cases for which the full input contained all annotated support, 118 retained all annotated sentences after bridge-first selection. These had an average F1 change of −2.81 points. In 65 cases, both gold article titles were still represented while at least one support sentence was removed. These averaged −13.47 points. The 41 that lost a support article averaged −15.77 points. These groups are defined by the method's outputs and provide a descriptive accounting. They do not isolate the causal effect of deleting a sentence: retained cases can also be easier questions. Article coverage substantially overstated sentence preservation.

[Paired scoring ledger](results/rows.jsonl) · [Complete estimates](results/summary.json) · [Whole-article follow-up](../article-preservation/)

## Reproduce

Use the hash-pinned [HotpotQA file](https://hotpotqa.github.io/) and the local matching checkpoint from the [preceding experiment](../cross-corpus-qa/). Run from the repository root. The tokenization verifier checks all 768 exact prompt sequences, case groups and evidence masks with the pinned local tokenizer without running model inference.

```sh
python cross-corpus-qa/run.py fetch /tmp/hotpot-dev-original.json
python sentence-bottleneck/study.py verify
python sentence-bottleneck/verify_inputs.py /tmp/hotpot-dev-original.json /path/to/Qwen3-1.7B-4bit
python sentence-bottleneck/study.py score /tmp/hotpot-dev-original.json /path/to/Qwen3-1.7B-4bit /tmp/hotpot-sentence-replay
```

The inference replay requires MLX on Apple Silicon. The run used a single active sequence, 538 physical generations and about 316 seconds. Peak process RSS was 1.47 GB and the MLX allocator recorded a separate 1.80 GB peak. The archive contains identifiers, token-accounting fields and normalized overlap measurements.
