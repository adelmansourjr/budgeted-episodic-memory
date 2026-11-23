# Fresh MuSiQue question-last transfer panel

This panel uses the checksum-pinned MuSiQue-Ans corpus and the frozen reveal-bridge retriever from the [earlier reader transfer](../qa-grounding/TRANSFER_DATA.md). Reconstruct the original salted order using one representative per retrieval-test support component and the same HotpotQA fitting-title screen. Exclude the first 160 eligible components because they were already used for reader transfer. The remaining 169 components comprise this panel. One of the 330 retrieval-test components is excluded by the fitting-title screen. The selection uses no answer scores.

The preparation reuses the prior full-text acquisition and whole-paragraph truncation policy. The 320-unit acquisition cap and 24-word previews stay fixed. The exact [question-last permutation](DATA.md) moves the question after the selected evidence. Length and token multiset are verified, along with an independent reconstruction of the decoded instruction, question, passages, and answer cue. The two inference arms share the resulting token IDs. Canonical answers and valid aliases are separate from the prompt.

The 169 prompts average 834.31 tokens and reach at most 1,019, within the 1,024-token cap. There are 163 two-hop questions, five three-hop questions, and one four-hop question. All supporting paragraphs remain in 117 prompts. The preparation drops at least one trailing paragraph in 26 prompts, and 49 questions include valid alternative answers. The support-title screen reduces exact known HotpotQA fitting overlap. Both data sources use Wikipedia.

## Reproduce

Use the local checkpoint and pinned archives from [the HotpotQA setup](../qa-grounding/DATA.md) and [the MuSiQue source record](../data/README.md). Keep prepared JSONL in temporary storage because it contains exact prompt tokens and answer labels.

```sh
python study.py download
python cross-corpus-qa/run.py fetch /tmp/hotpot-dev-original.json
python qa-question-last/prepare_transfer.py /tmp/hotpot-dev-original.json data/musique_ans_v1.0_dev.jsonl /path/to/Qwen3-1.7B-4bit /tmp/question-last-transfer
shasum -a 256 /tmp/question-last-transfer/test.jsonl
```

Expected panel SHA-256 is `1dd9b421940902c49f4f990c4dc0d74f72140a0521a7507eb6a8169743d56ee4`. Preparation runs entirely on CPU, peaked at roughly 392–407 MB process RSS across the checked builds, and reproduced the exact panel bytes on a second run. Earlier group IDs were independently matched to the original 160-question panel. All 329 group IDs and representative question IDs are unique across the two panels.
