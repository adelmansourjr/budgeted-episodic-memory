# Exact reader inputs and sentence supervision

The source is the checksum-pinned HotpotQA distractor file used in the earlier [cross-corpus experiment](../cross-corpus-qa/). Grouping connects questions that share a normalized supporting title. Use a deterministic representative from each component, then exclude the 3,268 components used by the previous experiments. Half-open positions in the ordered component list define the new splits.

| Split | Group positions | Questions | Mean prompt tokens | Full supporting sentence set retained | Prompts truncated |
|---|---|---:|---:|---:|---:|
| Adaptation | 3268–3908 | 640 | 728.18 | 557/640 | 3 |
| Development | 3908–3972 | 64 | 746.42 | 56/64 | 1 |
| Held out | 3972–4292 | 320 | 729.29 | 283/320 | 3 |

Acquisition is the original frozen reveal-bridge policy. It receives the question and candidate titles, starts from the same 24-word preview per candidate and spends at most 320 acquisition units. It uses no labels for its choices. After acquisition, the reader input contains the question first and the selected full articles in source order. A 1,024-token cap drops whole trailing sentences when required. Both adaptations and frozen inference receive the same per-question Qwen chat-token IDs. Answer-side targets are independently tokenized without a chat wrapper.

Moving the question before the articles makes each causal sentence endpoint query-conditioned. The fast tokenizer gives offsets on the complete serialized chat. Both the separately tokenized chat IDs and the canonical template IDs must agree exactly. Each nonempty shown sentence has a single strictly increasing last-token position. Only the support-supervised training loss uses the binary sentence labels. The model sees the same original sentence text and order in all arms.

No two splits or earlier component groups share supporting titles. That split does not guarantee complete separation of article text. Three training prompts and one development prompt included a later test supporting article as an unlabelled distractor. Evaluation still observes disjoint supporting components. The public repository does not include the prepared text, answer labels, or token files.

```sh
python cross-corpus-qa/run.py fetch /tmp/hotpot-dev-original.json
python qa-grounding/prepare_data.py /tmp/hotpot-dev-original.json /path/to/Qwen3-1.7B-4bit /tmp/grounding-inputs
python qa-grounding/audit_prepared.py /tmp/hotpot-dev-original.json /path/to/Qwen3-1.7B-4bit /tmp/grounding-inputs
```

The CPU audit regenerated all 1,024 example records, including exact token sequences and support endpoints. The builder used one CPU worker and peaked at 429 MB RSS on the recorded Apple Silicon machine. The [HotpotQA paper](https://aclanthology.org/D18-1259/) already describes joint answer and supporting-fact supervision. This experiment studies the training-only objective in a compact causal reader under matched inference.
