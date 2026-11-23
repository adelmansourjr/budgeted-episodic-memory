# MuSiQue reader-transfer panel

This panel tests the frozen HotpotQA reader adapters on a different question dataset. It uses the repository's checksum-pinned [MuSiQue-Ans](https://aclanthology.org/2022.tacl-1.31/) development file and the original retrieval split. Retrieval development records and the separately designated largest stress component are excluded. The acquisition policy was previously selected on the retrieval development split. No new transfer scores are used to choose adapters, thresholds or prompts.

Select one MuSiQue question per each of 330 retrieval-test components by salted hash. Exclude the representative of any component where a normalized MuSiQue support title matches a HotpotQA original training or checkpoint-selection gold title or acquired article title. Then keep the first 160 eligible representatives by independent salted component order. One representative was disqualified. This exact-title screen reduces obvious overlap and allows both corpora to draw from Wikipedia, with possible pretraining or unrecognized textual overlap. The panel measures question performance with equal component weights.

The same pinned Qwen base checkpoint, question-first chat formatter and frozen reveal-bridge policy prepare the reader input. The acquisition cap stays at 320 units with a 24-word preview. Each arm uses exactly the same prompt IDs per question with a maximum of 1,024 input tokens. MuSiQue supplies paragraphs as evidence units. The cap drops whole trailing paragraphs on overflow. Support flags and token positions refer to these paragraphs and remain evaluation metadata. Primary answers and valid aliases stay outside the prompt. To score free-text answers, the [official MuSiQue answer metric](https://github.com/StonyBrookNLP/musique/blob/main/metrics/answer.py) maximizes token F1 and normalized exact match independently over the canonical answer and its aliases.

The panel has 146 two-hop questions, 10 three-hop questions and four four-hop questions. There are 53 questions with aliases. Prompts average 833.16 tokens and reach 1,024 at most. All annotated supporting paragraphs remain in 111 cases and 25 prompts had to drop at least one trailing paragraph. These are acquisition diagnostics rather than answer scores.

## Reproduce

Use the existing local model and the archives described in [the data setup](DATA.md) and [the repository's MuSiQue source record](../data/README.md). The prepared JSONL contains answers and tokenized text. Keep it outside the Git repository.

```sh
python study.py download
python cross-corpus-qa/run.py fetch /tmp/hotpot-dev-original.json
python qa-grounding/prepare_transfer.py /tmp/hotpot-dev-original.json data/musique_ans_v1.0_dev.jsonl /path/to/Qwen3-1.7B-4bit /tmp/musique-transfer
shasum -a 256 /tmp/musique-transfer/test.jsonl
```

The expected SHA-256 of the panel is `6b907b8fd08ce914ffcc7aed106d3dc8afdec74099d582520e377a376f61d7c6`. The builder validates the prior retrieval, tokenizer and base-checkpoint fingerprints, distinguishes exact token IDs from tokenizer metadata, checks paragraph offsets and runs on CPU only. Peak process RSS was 392 MB on the recorded machine.
