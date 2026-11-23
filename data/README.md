# Source data

Trivedi, Balasubramanian, Khot and Sabharwal, [MuSiQue, TACL 2022](https://aclanthology.org/2022.tacl-1.31/). The [authors' repository](https://github.com/StonyBrookNLP/musique) distributes MuSiQue under CC BY 4.0. This work uses the public MuSiQue-Ans development file as a stand-alone benchmark pool, with an internal development and grouped held-out test split.

Run `python study.py download` to retrieve the pinned 29 MB JSONL from the [voidful/MuSiQue mirror](https://huggingface.co/datasets/voidful/MuSiQue/tree/a7d9f9adf6191604fc67cde318ee1a86fcf7babc). The downloader verifies SHA-256 `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`. The public repository contains no article text, answers or dataset copy. It retains IDs, selections, split assignments and derived measurements for replay.
