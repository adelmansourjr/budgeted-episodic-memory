# MuSiQue answer quality at equal generated lengths

The published question-last adapter and frozen Qwen each emitted **12 ordinary tokens** on every case. The primary comparison scored the entire first **6 tokens** from each response using MuSiQue alias-aware answer F1. Prompt token IDs, vocabulary mask, greedy decoding and base model checkpoint were shared within every pair. The adapter used its original HotpotQA-trained weights.

| Panel | Groups | Frozen F1 | Adapter F1 | Paired gain | 95% paired interval |
|---|---:|---:|---:|---:|---:|
| Prior transfer panel | 169 | 7.70 | 16.25 | +8.56 | +5.04 to +12.08 |
| Disjoint replication | 160 | 7.40 | 17.20 | +9.80 | +6.29 to +13.49 |
| Pooled | 329 | 7.55 | 16.71 | +9.16 | +6.64 to +11.72 |

The transfer panel met the prespecified three-point estimate and positive interval criterion. On that panel the adapter had 40 paired wins, 121 ties and 8 losses.

The replication panel met the prespecified three-point estimate and positive interval criterion. On that panel the adapter had 45 paired wins, 108 ties and 7 losses.

Both panels cleared the criterion, supporting a replicated gain at six tokens.

## Sensitivity to exact budget

| Scored tokens per answer | Frozen F1 | Adapter F1 | Paired gain | Simultaneous 95% interval |
|---:|---:|---:|---:|---:|
| 2 | 10.04 | 13.44 | +3.40 | +0.27 to +6.53 |
| 4 | 13.60 | 21.17 | +7.57 | +4.44 to +10.70 |
| 6 | 7.55 | 16.71 | +9.16 | +6.03 to +12.29 |
| 8 | 7.81 | 14.79 | +6.98 | +3.85 to +10.10 |
| 12 | 7.41 | 11.76 | +4.35 | +1.22 to +7.48 |

At six tokens, pooled exact match was 0.61% for frozen Qwen and 2.13% for the adapter. The paired difference was +1.52 points [+0.00 to +3.04].

By token six, frozen Qwen would have chosen a blocked token at least once in 300 cases. The corresponding adapter count was 273. The separate pre-intervention diagnostic matched an average of 2.58 scored tokens, with a difference of +3.87 points [+1.35 to +6.44]. That diagnostic conditions length on the earlier greedy boundary from either reader. The six-token test fixes the scored length independently of both models.

The earlier shared-cap evaluation used a first-line answer parser and allowed natural stopping. The present evaluation scores full token prefixes under a single-line vocabulary constraint. For short gold answers the constraint can extend the generation past its preferred endpoint, so the estimates describe this controlled decode policy.

The verified run used one loaded local model, one active sequence and **1.74 GiB** peak MLX allocation (1.25 GiB peak process RSS). The shared-prefix path matched standard decoding exactly on the engineering prompts. Standalone six-token decoding matched the six-token prefixes in the engineering check.

[Frozen protocol](EQUAL_DECODE_PROTOCOL.md) · [Summary](equal-decode/summary.json) · [Numerical ledger](equal-decode/cases.jsonl) · [Original panel manifest](equal-decode/transfer_preparation.json) · [Replication manifest](equal-decode/replication_preparation.json)

## Reproduce

Use the released local checkpoint and prior reproducible panels. The numeric ledger includes emitted token counts and sequence digests. The evaluator keeps raw generated token arrays in scratch, separate from the public artifacts.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
model=/path/to/Qwen3-1.7B-4bit
work=$(mktemp -d)
python cross-corpus-qa/run.py fetch "$work/hotpot.json"
python study.py download
musique=data/musique_ans_v1.0_dev.jsonl
python qa-question-last/prepare.py "$work/hotpot.json" "$model" "$work/hotpot"
python qa-grounding/prepare_transfer.py "$work/hotpot.json" "$musique" "$model" "$work/earlier"
python qa-question-last/prepare_transfer.py "$work/hotpot.json" "$musique" "$model" "$work/remaining"
python qa-question-last/prepare_equal_replication.py "$work/earlier" "$work/remaining" "$model" "$work/replication"
python qa-question-last/score_equal_decode.py score --transfer "$work/remaining" --replication "$work/replication" --hotpot-data "$work/hotpot" --model "$model" --output "$work/run"
python qa-question-last/score_equal_decode.py verify --transfer "$work/remaining" --replication "$work/replication" --output "$work/run"
```
