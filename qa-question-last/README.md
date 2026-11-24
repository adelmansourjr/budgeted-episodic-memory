# Question-last training for budgeted Qwen answers

The earlier fixed-budget evaluation found that frozen Qwen with the question after the evidence had the highest observed score on a comparison-question panel. Here all three readers use that same token ordering. The test asks whether adaptation can improve generated answers against that control on previously unused, mixed HotpotQA questions.

The new reader starts from the same local 4-bit Qwen3-1.7B checkpoint as the frozen model. Its rank-8 adapters train on 640 questions with answer labels. On a separate 64-question development set, it selects a checkpoint and adapter gain using generated answer F1. The previous question-first answer-only adapter provides a secondary comparator. That control shares the evaluation prompt, while the earlier training and selection recipe also used a different learning rate and criterion.

Every held-out case supplies exactly the same prompt token IDs to all three models. Every reader has the same 24-token generation cap, fixed greedy decoding and first-line short-answer scorer. Realized answer lengths are separate measurements. All 320 held-out support-title groups are new to the reader experiments. Three candidate groups with support-title overlap with fitting inputs were excluded before scoring.

[Prespecified comparison](PROTOCOL.md) · [Prompt/split preparation](DATA.md) · [Earlier stronger-prompt control](../qa-grounding/comparison/summary.json)

## Held-out results

All arms used the same prompt token IDs for each case, averaging **729.86 input tokens**. Each had a 24-token generation cap.

| Reader | Answer F1 | Exact match | Mean generated tokens |
|---|---:|---:|---:|
| Frozen question-last Qwen | 29.47 | 22.50% | 4.41 |
| Earlier question-first adapter | 37.22 | 26.25% | 4.22 |
| New question-last adapter | 42.27 | 27.50% | 5.61 |

The new adapter versus frozen Qwen changed answer F1 by **+12.81 F1 points [95% paired interval +8.16, +17.58]**. It met the prespecified three-point gain criterion and the positive lower confidence bound.

The paired F1 differences had 86 wins, 212 ties and 22 losses. The new versus earlier adapter changed F1 by **+5.05 F1 points [95% paired interval +1.00, +9.12]**.

Development selected update 160 at adapter gain 1.0. Its generated development F1 was 38.68, and frozen development F1 was 21.82. Each tuned checkpoint had 491,520 low-rank parameters.

Peak MLX allocation was 2.25 GiB in training and 1.74 GiB in held-out evaluation. Evaluation used one loaded checkpoint and generated one sequence at a time. The shared-prefix replay matched stock inference exactly on the development check.

[Training and development record](adapter/training.json) · [Paired summary](results/summary.json) · [Numerical case ledger](results/cases.jsonl) · [Input fingerprints](results/preparation.json)

## MuSiQue transfer

The saved adapter scored 24.50 F1 versus 17.81 for frozen Qwen on 169 fresh groups, a paired difference of +6.69 points [95% interval +2.65, +11.06]. The transfer met the prespecified minimum three-point gain and positive lower confidence bound. [Transfer report](TRANSFER.md)

## Equal generated lengths

The two readers each emitted twelve ordinary tokens, scored at matched prefix lengths on disjoint MuSiQue panels. The pooled six-token F1 was 7.55 for frozen Qwen and 16.71 for the adapter. [Exact-length results and panel-specific intervals](EQUAL_DECODE.md)

## Equal prefixes before natural stopping

With unrestricted greedy decoding, the equal pre-EOS prefixes scored 21.23 F1 for the saved adapter and 17.00 for frozen Qwen across 329 questions. The paired gain was +4.23 [+1.65 to +6.85] points. [Natural-stop report](NATURAL_STOP.md)

## Reproduce

The recorded run used Apple Silicon, Python 3.12.8, NumPy 2.5.4, MLX 0.32.4, mlx-lm 0.32.0, Transformers 5.12.1 and Tokenizers 0.22.2. The earlier [model lock](../cross-corpus-qa/LOCK.json) identifies the matching local checkpoint. The trained adapter and numeric case ledger are included. The raw dataset and tokenized prompts are separate inputs.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
model=/path/to/Qwen3-1.7B-4bit
scratch=$(mktemp -d)
python cross-corpus-qa/run.py fetch "$scratch/hotpot.json"
python qa-question-last/prepare.py "$scratch/hotpot.json" "$model" "$scratch/prepared"
python qa-question-last/study.py verify --adapter qa-question-last/adapter --data "$scratch/prepared" --output qa-question-last/results
python qa-question-last/study.py score --model "$model" --data "$scratch/prepared" --adapter qa-question-last/adapter --output "$scratch/replay"
python qa-question-last/study.py train --model "$model" --data "$scratch/prepared" --output "$scratch/refit"
```

The original [HotpotQA paper](https://aclanthology.org/D18-1259/) provides the benchmark and token-overlap scoring. [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) motivates controlling token order. [MARC](https://proceedings.iclr.cc/paper_files/paper/2026/hash/16049e0c3f47899091ac46f8b3afb178-Abstract-Conference.html) motivates testing whether a compact reader can make better use of the same retrieved evidence. This experiment examines that question in text.
