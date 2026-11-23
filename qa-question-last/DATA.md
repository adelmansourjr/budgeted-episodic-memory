# Exact question-last preparation

The preparation uses the frozen reveal-bridge retriever and matching Qwen tokenizer from the earlier [HotpotQA experiment](../qa-grounding/DATA.md). The selector sees the question and the same passage previews with the existing 320-unit acquisition budget. The 1,024-token reader cap drops complete trailing sentences where needed. The prepared input is shared exactly by the frozen and adapted readers.

The prompt follows the [previous question-last control](../qa-grounding/COMPARISON_DATA.md). The formatter builds the verified original chat input, then moves its contiguous question block after the passages block. Both token count and token multiset match the original input. Decoding verifies that the instruction, evidence, question and answer cue preserve their content. The evidence-position metadata follows the moved passages, which now precede the question. This split is intended for answer-only training.

| Split | Component indices | Groups | Mean prompt tokens | Full supporting sentences retained |
|---|---:|---:|---:|---:|
| Training | 3268–3907 | 640 | 728.18 | 557 |
| Development | 3908–3971 | 64 | 746.42 | 56 |
| Fresh test | 5304–5626 | 320 | 729.86 | 278 |

The test selects the first 320 eligible representatives in the existing salted support-component order starting at 5304. That start follows the last selected component of the earlier comparison panel. The title screen removes three candidate groups whose normalized supporting titles occurred in the fitting groups’ gold or acquired article titles. The test has 320 unique components and a maximum input of 1,018 tokens. Separation by support components and the exact-title screen address split overlap. The corpus and the pretrained model may share Wikipedia text.

## Reproduce

Use the local checkpoint whose fingerprints appear in the [base lock](../cross-corpus-qa/LOCK.json). The builder verifies the source and checkpoint hashes and records the generated split digests in its manifest. Prepared text, token sequences and answer labels belong in temporary storage. The initial preparation peaked at 444 MB process RSS.

```sh
python cross-corpus-qa/run.py fetch /tmp/hotpot-dev-original.json
python qa-question-last/prepare.py /tmp/hotpot-dev-original.json /path/to/Qwen3-1.7B-4bit /tmp/question-last
python qa-question-last/prepare.py /tmp/hotpot-dev-original.json /path/to/Qwen3-1.7B-4bit /tmp/question-last-replay
diff -qr /tmp/question-last /tmp/question-last-replay
```
