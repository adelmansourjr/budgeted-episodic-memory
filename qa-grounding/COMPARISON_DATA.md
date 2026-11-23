# Held-out HotpotQA comparison panel

The main HotpotQA evaluation included an exploratory comparison-question subgroup. This follow-up fixes a new comparison-only panel before inference. Starting at zero-based position 4,292 in the already salted order of support-title components, consider each component's fixed representative if its annotated type is comparison. Exclude representatives with a normalized supporting title seen among the acquired or gold articles of either HotpotQA training or checkpoint selection. Then take the first 192 eligible representatives. The last included component is at position 5,303.

The standard prompt is identical to the [training study's data builder](DATA.md). It places the question before the acquired passages. The comparison also prepares a question-last prompt for the frozen model as a prompt-order control. Only the contiguous question and passages token blocks exchange positions. For each question the two prompts have exactly the same token multiset and length. A separate decoded-text check verifies that the instruction, question, acquired text and answer suffix remain identical, with the question and passages reordered. The tokenized prompt itself preserves both substantive content and exact cost of processing the input. The model comparisons still use the same generation limit. The cap does not force outputs that terminate early to use more tokens.

The panel has 170 questions where every annotated supporting sentence is retained, one prompt shortened by the cap, and a mean of 759.59 input tokens. The maximum is 1,015. It tests the effect on one focused question family using previously unused components and the same checkpoint. Disjoint normalized titles do not rule out shared Wikipedia text or pretraining exposure.

## Reproduce the input panel

```sh
python cross-corpus-qa/run.py fetch /tmp/hotpot-dev-original.json
python qa-grounding/prepare_comparison.py /tmp/hotpot-dev-original.json /path/to/Qwen3-1.7B-4bit /tmp/comparison-panel
shasum -a 256 /tmp/comparison-panel/test.jsonl
```

The expected SHA-256 of the temporary input is `edd83110ab17671008fbb95ef479e983fcd5ecae0721f56a1ba3cdc26063d0e2`. That file contains prepared input and answers and stays outside Git. The CPU builder validates the pinned corpus, retrieval code, model and tokenizer fingerprints. It used 437 MB peak process RSS on the recorded machine.
