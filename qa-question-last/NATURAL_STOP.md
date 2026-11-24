# MuSiQue at matched prefixes before natural stopping

Frozen Qwen and the original HotpotQA-trained adapter answered with stock greedy decoding and their configured EOS tokens. For every question, both answers were scored at the shorter observed pre-EOS length. Both received identical prompt token IDs and a 24-token generation cap. This is a new stopping analysis on the same two disjoint MuSiQue development panels used for the exact-length experiment.

| Panel | Groups | Frozen F1 | Adapter F1 | Paired gain [95% interval] |
|---|---:|---:|---:|---:|
| Original transfer panel | 169 | 16.97 | 22.16 | +5.20 [+1.86 to +8.77] |
| Disjoint replication | 160 | 17.03 | 20.23 | +3.21 [-0.80 to +7.15] |
| Pooled | 329 | 17.00 | 21.23 | +4.23 [+1.65 to +6.85] |

The matched prefix averaged **2.65 answer tokens** across all 329 questions. Zero-token pairs numbered 0. All pairs remain in the result. The adapter recorded 38 wins, 278 ties and 13 losses on paired F1.

The original panel cleared the prespecified criterion. The disjoint panel was directionally consistent and its interval includes zero. The pooled estimate supports a gain on the combined cases. Evidence across panels is weaker than in the forced-length comparison.

## Did the result depend on the stopping rule?

| Pooled readout | Frozen F1 | Adapter F1 | Paired gain [95% interval] |
|---|---:|---:|---:|
| Equal pre-EOS prefix | 17.00 | 21.23 | +4.23 [+1.65 to +6.85] |
| Equal first-answer-line prefix | 17.03 | 21.20 | +4.17 [+1.60 to +6.80] |
| Entire native response | 17.06 | 23.97 | +6.92 [+3.96 to +9.99] |

The both-EOS sensitivity included 324 pairs and yielded +4.29 [+1.65 to +7.01] F1 points. The primary matched-prefix exact-match difference was +0.00 [-2.13 to +2.13] points.

| Reader | Mean emitted answer tokens | Mean decoder steps including EOS | Capped responses |
|---|---:|---:|---:|
| Frozen Qwen | 3.44 | 4.44 | 1 |
| Saved adapter | 4.48 | 5.47 | 4 |

The two native outputs can contain different token counts. The scoring horizon is shared within each pair and depends on those two stopping decisions. The earlier fixed-length experiment assigned the output and score lengths independently of either reader. [Exact-length comparison](EQUAL_DECODE.md)

One local model and one active sequence used **1.74 GiB** peak MLX allocation and **0.89 GiB** peak process RSS on macOS. Development replays matched the standard and shared-cache token sequences and stop events exactly.

A separate uncached run matched every token and stop event for both readers on 16 MuSiQue pairs selected by group hash, eight per panel. [Replay audit](natural-stop/standard-replay-audit.json)

[Prespecified protocol](NATURAL_STOP_PROTOCOL.md) · [Summary](natural-stop/summary.json) · [Numeric ledger](natural-stop/cases.jsonl)

## Reproduce

Build the same pinned HotpotQA development input and the two MuSiQue panels following the earlier experiment’s preparation commands. Use the identical saved local model. Private token traces are written to the scratch output and are separate from the public numerical ledger. [Input preparation](EQUAL_DECODE.md#reproduce)

```sh
python qa-question-last/score_natural_stop.py score --transfer "$work/remaining" --replication "$work/replication" --hotpot-data "$work/hotpot" --model "$model" --output "$work/natural"
python qa-question-last/score_natural_stop.py verify --transfer "$work/remaining" --replication "$work/replication" --model "$model" --private-tokens "$work/natural/private_tokens.jsonl" --output "$work/natural"
```
