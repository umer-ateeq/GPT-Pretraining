# ZeroToGPT

**We trained a GPT from scratch on 7 billion tokens, as undergraduates, without a research lab or
massive compute.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/umer-ateeq/GPT-Pretraining/blob/main/gpt_style_transformer_134m__training_on_7b_tokens_using_colab.ipynb)

134M parameters, every component written out in PyTorch, pretrained on a 7B-token FineWeb-Edu corpus
using a single free-tier GPU. No `nn.Transformer`, no HuggingFace model class. Open the notebook and
continue the run yourself.

## Overview

| | |
|---|---|
| Parameters | **134,077,440** (95,283,456 non-embedding) |
| Model | 8-block decoder-only transformer, 768 wide, 12 heads |
| Dataset | **7B tokens**, FineWeb-Edu `CC-MAIN-2024-10` |
| Hardware | one Tesla P100 16 GB, free Kaggle session |
| Held-out perplexity | **38.89** on FineWeb-Edu, context 128 |
| Throughput | **10,200 tok/s**, **31.7% MFU**, 12.24 GB peak of 16 |
| Weights | [umerateeq/zerotogpt-134m](https://huggingface.co/umerateeq/zerotogpt-134m), 538 MB |
| Write-up | [ZeroToGPT on Level Up Coding](https://levelup.gitconnected.com/zerotogpt-a-comprehensive-guide-to-train-custom-gpt-from-scratch-on-7b-tokens-for-free-7bcd8aef07c3) |
| Post | [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7376159272583311360/) |

---

## Architecture

| | | | |
|---|---|---|---|
| Type | decoder-only transformer | Blocks | 8 |
| Residual width | 768 | Attention heads | 12, head dim 64 |
| Attention | multi-head causal self-attention | Masking | causal, upper triangular |
| Feed-forward | 768 to 3072 to 768, ReLU | Normalization | pre-norm LayerNorm, `eps = 1e-5` |
| Token embedding | `nn.Embedding(50257, 768)` | Positional encoding | learned absolute |
| Tokenizer | GPT-2 BPE (`tiktoken`) | Vocabulary | 50,257 |
| Context | 128 | Dropout | 0.1 |
| Output head | **untied**, `nn.Linear(768, 50257, bias=False)` | Sampling | greedy, temperature with top-k |

Every module is implemented directly: multi-head causal self-attention, the causal mask, LayerNorm,
the feed-forward block, the residual wiring, the sampler and the training loop.

---

## Training

| | | | |
|---|---|---|---|
| Optimizer | AdamW | Learning rate | 4e-4 |
| Weight decay | 0.1 | Grad clip | global norm 1.0 |
| Precision | fp16 + `GradScaler` | Sequence length | 128 |
| Batch | 32 x 128 = 4,096 tok/step | Dropout | 0.1 |
| Throughput | **10,200 tok/s** | Peak GPU memory | **12.24 GB** of 16 |
| Achieved | **5.93 TFLOP/s** | MFU | **31.7%** of the P100's 18.7 TFLOP/s fp16 peak |

### Data pipeline

| Stage | Implementation |
|---|---|
| Source | FineWeb-Edu `CC-MAIN-2024-10`, streamed from Hugging Face |
| Tokenizer | GPT-2 BPE via `tiktoken`, `encode_ordinary` |
| Storage | pre-tokenized into a flat `.bin` file, written through `np.memmap` |
| Dtype | **`uint16`**, since GPT-2's highest token id is 50256 and fits in two bytes |
| Batching | memmap reopened per batch, pinned, copied to GPU asynchronously |

---

## Results

| Dataset | Perplexity | Context |
|---|---|---|
| **Held-out FineWeb-Edu** | **38.89** | 128 |
| TinyStories | 35.41 | 128 |
| WikiText-2 | 184.96 | 128 |

Scored on full raw test sets with non-overlapping windows.

---

## Continue the pretraining run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/umer-ateeq/GPT-Pretraining/blob/main/gpt_style_transformer_134m__training_on_7b_tokens_using_colab.ipynb)

Open the notebook in Colab and hit **Run all**. Nothing to install. It pulls the pre-tokenized
`train.bin` and `validation.bin` and the released `weights.pth`, so the run picks up pretraining
from the published checkpoint and plots the loss curve as it goes. Skip the "Load my weights" cell
to train from scratch instead.

This is a base model: it completes text.

---

## Credits

With thanks to:

- **Umar Jamil**, [Attention is all you need (Transformer): model explanation including math, inference and training](https://www.youtube.com/watch?v=bCz4OMemCcA)
- **Sebastian Raschka**, [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)
- **Andrej Karpathy**, [nanoGPT](https://github.com/karpathy/nanoGPT)

Data: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
