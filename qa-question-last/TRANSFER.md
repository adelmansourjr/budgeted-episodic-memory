# Question-last Qwen transfer to MuSiQue

The HotpotQA-trained question-last adapter was evaluated on 169 new MuSiQue reader-evaluation groups at its published weights. The frozen comparator used the same Qwen3-1.7B base checkpoint. Both received identical prompt IDs on every case and the same 24-token generation cap. The mean logical input was **834.31 tokens**.

| Reader | Answer F1 | Exact match | Mean generated tokens |
|---|---:|---:|---:|
| Frozen question-last Qwen | 17.81 | 10.06% | 4.30 |
| Saved question-last adapter | 24.50 | 13.61% | 5.73 |

The paired F1 difference was **+6.69 points** with a 95% component-bootstrap interval of **+2.65 to +11.06**. The panel contained 26 wins, 134 ties and 9 losses for the adapter.

The point estimate cleared the prespecified three-point threshold, and the confidence interval excluded zero.

This panel uses the remaining eligible support components from the earlier MuSiQue retrieval-test split. It is disjoint from the prior 160-group MuSiQue reader evaluation. The shared retriever was selected before this transfer test on a separate MuSiQue retrieval development split. These scores weight support components equally and use answer F1 and exact match with the released aliases.

Peak MLX allocation was **1.74 GiB**, with one loaded model and one active generation at a time. Shared-prefix reuse matched the exact generated strings and token counts on the prior development replay.

[Frozen protocol](TRANSFER_PROTOCOL.md) · [Input preparation](TRANSFER_DATA.md) · [Verified summary](transfer/summary.json) · [Numeric ledger](transfer/cases.jsonl) · [Preparation fingerprint](transfer/preparation.json)

A follow-up tests the same readers with exactly matched generated and scored token counts. [Exact-length comparison](EQUAL_DECODE.md)

## Reproduce

Use the pinned checkpoint and the original datasets. Keep tokenized prompts, labels and output replays in scratch storage. The commands below use the archived adapter directly.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
model=/path/to/Qwen3-1.7B-4bit
scratch=$(mktemp -d)
python cross-corpus-qa/run.py fetch "$scratch/hotpot.json"
python study.py download
python qa-question-last/prepare.py "$scratch/hotpot.json" "$model" "$scratch/hotpot"
python qa-question-last/prepare_transfer.py "$scratch/hotpot.json" data/musique_ans_v1.0_dev.jsonl "$model" "$scratch/musique"
python qa-question-last/score_transfer.py verify --data "$scratch/musique" --output qa-question-last/transfer
python qa-question-last/score_transfer.py score --data "$scratch/musique" --hotpot-data "$scratch/hotpot" --model "$model" --output "$scratch/transfer-replay"
```
