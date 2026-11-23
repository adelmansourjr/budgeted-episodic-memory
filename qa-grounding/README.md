# Training a small Qwen reader to use supporting evidence

A budgeted retriever can bring a small model the right articles without improving the answer. This experiment holds the retrieved text fixed and evaluates whether support annotations can improve reading with the same inference token budget.

The same local Qwen3-1.7B quantized checkpoint supplies three readers. One uses frozen weights. Both adapted arms train low-rank weights on the same 640 HotpotQA questions. The support-supervised arm also classifies annotated supporting sentences at their causal endpoints after seeing the question. A small external head provides that training loss and is discarded before scoring. Development selects the checkpoint by answer likelihood from the same two scheduled checkpoints in each adapted arm. The primary and transfer evaluations give all three readers exactly the same input token sequence for every paired question. Every arm can generate up to 24 greedy tokens. EOS and the first-line answer instruction can change realized generation length.

The primary contrast isolates the value of support supervision against an answer-only adapter. The separate comparison to the frozen reader measures the value of task adaptation overall. The 320 held-out HotpotQA questions each represent a different supporting-title component. The 160-question MuSiQue panel tests the same weights with its released answer aliases. A further 192 previously unused comparison questions test the scoped supervision hypothesis and the effect of moving the question without changing the token multiset. Training and development used HotpotQA only. The panels and intervals remain conditional on the selected checkpoint, the split and the matched training seed.

[Primary design](PROTOCOL.md) · [Exact token preparation](DATA.md) · [Transfer design](TRANSFER_PROTOCOL.md) · [Transfer data](TRANSFER_DATA.md) · [Comparison design](COMPARISON_PROTOCOL.md) · [Comparison data](COMPARISON_DATA.md)

## Results

Every paired input was identical across readers in the primary and transfer evaluations. The mean prompt was **729.29 tokens** on HotpotQA and **833.16 tokens** on MuSiQue. Every reader had a 24-token generation cap.

| Case panel | Reader | Answer F1 | Exact match | Mean generated tokens |
|---|---|---:|---:|---:|
| HotpotQA (320) | Frozen Qwen | 9.91 | 8.12% | 3.55 |
| HotpotQA (320) | Answer-only adapter | 25.90 | 19.69% | 4.06 |
| HotpotQA (320) | Support-supervised adapter | 25.95 | 19.69% | 4.56 |
| MuSiQue (160) | Frozen Qwen | 1.96 | 0.62% | 3.61 |
| MuSiQue (160) | Answer-only adapter | 6.41 | 3.12% | 2.84 |
| MuSiQue (160) | Support-supervised adapter | 9.51 | 4.38% | 4.06 |
| Comparison (192) | Frozen question-last | 32.99 | 28.12% | 4.47 |
| Comparison (192) | Answer-only adapter | 24.91 | 21.35% | 3.18 |
| Comparison (192) | Support-supervised adapter | 29.49 | 23.96% | 4.08 |
| Comparison (192) | Frozen question-first | 27.40 | 23.44% | 3.80 |

Adding support supervision changed HotpotQA F1 by **+0.05 points [95% paired interval -3.32, +3.35]** compared with the answer-only adapter. The support objective missed the prespecified three-point and positive-interval criterion. The paired groups had 28 wins, 271 ties and 21 losses.

Against the frozen model, the answer-only adapter changed HotpotQA F1 by **+15.99 points [95% paired interval +11.28, +20.76]** and the support-supervised adapter by **+16.04 points [95% paired interval +11.19, +20.91]**. These comparisons measure learning the answer task from labeled examples with the same inference inputs.

The support versus answer-only contrast on MuSiQue was **+3.10 points [95% paired interval +0.35, +6.05]**. The transfer estimate also cleared those two numerical thresholds.

The fresh comparison-question panel used 759.59 mean prompt tokens per arm. Support versus answer-only changed F1 by **+4.58 points [95% paired interval -1.35, +10.47]**. This missed the comparison-panel gain criterion.

The question-last frozen control changed F1 by **+5.59 points [95% paired interval -0.63, +11.98]** relative to the same frozen model and exact token multiset with the question first. Support-supervised Qwen versus the question-last control scored **-3.50 points [95% paired interval -12.18, +5.28]**. The input comparison tests token order on 192 comparison questions.

The title-overlap sensitivity removed 4 HotpotQA test groups whose support included a title seen as a distractor in training or checkpoint selection. The remaining 316 groups gave **+0.05 points [95% paired interval -3.30, +3.39]** for support versus answer-only. The original 320-group analysis remains primary.

### Training and verification

Each fitted adapter had 491,520 trainable low-rank parameters. Development selected update 80 for answer-only and update 80 for support-supervised, with mean answer NLL 1.846 and 1.587, respectively.

Training the two adapters in one process took 14.3 minutes. Peak MLX allocation was 2.26 GiB during training, 1.73 GiB during the primary evaluation, 1.73 GiB during transfer, and 1.82 GiB during the comparison panel. Each evaluation generated one sequence at a time.

The input audits checked all 672 case IDs, prompt hashes and token counts. On HotpotQA it counted 0, 0 and 0 empty normalized answers in the frozen, answer-only and support-supervised arms.

[HotpotQA scores](hotpot/summary.json) · [MuSiQue scores](musique/summary.json) · [Comparison scores](comparison/summary.json) · [Checkpoint record](adapters/training.json) · [Title sensitivity](hotpot/title_overlap.json)

## Related work

[Pinned MARC paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/16049e0c3f47899091ac46f8b3afb178-Abstract-Conference.html) links efficient access to relevant video segments with training a small reader to make use of the evidence. [O-MARC](https://arxiv.org/abs/2605.26584) extends this question to temporally grounded audio and visual evidence. This experiment uses text and a training-only sentence objective. The original [HotpotQA study](https://aclanthology.org/D18-1259/) and [Select, Answer and Explain](https://arxiv.org/abs/1911.00484) already use joint supporting-sentence supervision. Here the contribution is the controlled comparison of that signal in a compact causal reader at fixed decoding budgets. The MLX adapters follow the general quantized adaptation approach of [QLoRA](https://arxiv.org/abs/2305.14314) using the existing MLX checkpoint format.

## Reproduce

The repository includes the two small trained adapter tensors and the numerical case ledgers. The base model and prepared text and answers are separate inputs. Use the matching Qwen3-1.7B checkpoint listed in the [base lock](../cross-corpus-qa/LOCK.json), Python 3.12, NumPy, fast Transformers tokenization, and MLX with mlx-lm 0.32.0 on Apple Silicon. Set numerical library threads to one. The saved checkpoints let the answer evaluation replay independently of training. The disposable workspace is needed only for token reconstruction and model replay.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
base=/path/to/Qwen3-1.7B-4bit
tmp=$(mktemp -d)
python cross-corpus-qa/run.py fetch "$tmp/hotpot.json"
python qa-grounding/prepare_data.py "$tmp/hotpot.json" "$base" "$tmp/hotpot-inputs"
python qa-grounding/audit_prepared.py "$tmp/hotpot.json" "$base" "$tmp/hotpot-inputs"
python qa-grounding/study.py verify --adapters qa-grounding/adapters --output qa-grounding/hotpot
python qa-grounding/study.py train --model "$base" --data "$tmp/hotpot-inputs" --output "$tmp/adapters"
python qa-grounding/study.py score --model "$base" --data "$tmp/hotpot-inputs" --adapters "$tmp/adapters" --output "$tmp/hotpot"
python qa-grounding/audit_results.py "$tmp/hotpot-inputs" "$tmp/hotpot" "$tmp/adapters/training.json"
python qa-grounding/score_transfer.py verify --output qa-grounding/musique
python study.py download
python qa-grounding/prepare_transfer.py "$tmp/hotpot.json" data/musique_ans_v1.0_dev.jsonl "$base" "$tmp/musique-inputs"
python qa-grounding/score_transfer.py score --model "$base" --data "$tmp/musique-inputs" --adapters qa-grounding/adapters --output "$tmp/musique"
python qa-grounding/audit_results.py "$tmp/musique-inputs" "$tmp/musique" qa-grounding/adapters/training.json
```

The training lock includes the split file digests and the preparation manifest from the same run. Prepared example JSONL files are deterministic; the preparation manifest also records elapsed time and RSS. For an exact scoring replay with the published adapters, reproduce the prepared example files, then use the archived preparation manifest to satisfy that administrative fingerprint. See [the numeric manifest](hotpot/preparation.json).

```sh
cp qa-grounding/hotpot/preparation.json "$tmp/hotpot-inputs/manifest.json"
python qa-grounding/study.py score --model "$base" --data "$tmp/hotpot-inputs" --adapters qa-grounding/adapters --output "$tmp/hotpot-saved"
python qa-grounding/audit_results.py "$tmp/hotpot-inputs" "$tmp/hotpot-saved" qa-grounding/adapters/training.json
```

## Reproduce the comparison panel

The comparison panel uses the identical input IDs for both adapters. For the frozen question-last control, move the whole question block after the evidence while preserving the exact prompt token multiset. The frozen question-first control uses the unchanged adapter prompt. Shared frozen prefix entries are reused only after an equality check against standard generation on development inputs.

```sh
python qa-grounding/prepare_comparison.py "$tmp/hotpot.json" "$base" "$tmp/comparison-inputs"
python qa-grounding/score_comparison.py verify --data "$tmp/comparison-inputs" --output qa-grounding/comparison
python qa-grounding/score_comparison.py score --model "$base" --hotpot-data "$tmp/hotpot-inputs" --data "$tmp/comparison-inputs" --adapters qa-grounding/adapters --output "$tmp/comparison"
```
