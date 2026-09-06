"""Perplexity for the ZeroToGPT 134M checkpoint, with GPT-2-small as a baseline.

Perplexity is exp(average cross-entropy per token). Read it as "on average the model
was as uncertain as if it were choosing uniformly among this many tokens". Lower is
better and the floor is 1.

A perplexity number without its protocol is not comparable to anything, so this script
is explicit about which one it used. Three modes:

  --mode bin        sequential non-overlapping windows over a tokenized uint16 .bin
                    file. Every token scored exactly once. Strict and fast.
  --mode wikitext   the strided sliding window from the GPT-2 paper, over the
                    WikiText-2 raw test set. Each token gets up to max_length - 1
                    tokens of left context, the window advances by --stride, and
                    overlap tokens that exist only to provide context are masked out
                    of the loss so nothing is counted twice.
  --mode hf         the same strided protocol over any HuggingFace text dataset.

--model gpt2 runs HuggingFace's GPT-2-small through the identical scoring function, so
a comparison isolates the model rather than the evaluation code. That baseline is what
makes any number here checkable: GPT-2-small's published WikiText-2 perplexity is about
29.4 at its full context, so a sane figure at a shorter window means the harness is
sound.

The model classes below mirror the ones in the notebooks in this repository. They are
inlined so this script runs standalone, with no imports beyond pip.

Usage:
    python evaluate.py --ckpt weights.pth --all --data-bin validation.bin
    python evaluate.py --ckpt weights.pth --mode bin --data-bin validation.bin
    python evaluate.py --ckpt weights.pth --mode wikitext
    python evaluate.py --ckpt weights.pth --mode hf --hf-dataset roneneldan/TinyStories
    python evaluate.py --model gpt2 --mode wikitext
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GPT_CONFIG_134M = {
    "vocab_size": 50257,      # GPT-2 BPE vocabulary (tiktoken "gpt2")
    "context_length": 256,    # positions the model allocates embeddings for
    "emb_dim": 768,           # model width, d_model
    "n_heads": 12,            # 768 / 12 = 64 dimensions per head
    "n_layers": 8,            # transformer blocks
    "drop_rate": 0.1,         # inactive here, the model is in eval mode
    "qkv_bias": False,
}


# ----------------------------------------------------------------------------
# Model. Mirrors the notebooks in this repository.
# ----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, _ = x.shape

        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)          # (b, heads, tokens, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # scale by sqrt(head_dim), which is keys.shape[-1] after the view above
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.ReLU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"], num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))    # pre-norm
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        x = self.tok_emb(in_idx) + self.pos_emb(
            torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def trained_positions(state):
    """Recover how many positional rows actually received gradient.

    Untrained rows decay toward zero under AdamW weight decay while trained rows keep
    a healthy norm, so the boundary shows up as a cliff in the row norms.
    """
    norms = state["pos_emb.weight"].float().norm(dim=1)
    threshold = 0.1 * norms.max().item()
    n = 0
    for i in range(norms.numel()):
        if norms[i] < threshold:
            break
        n = i + 1
    return n


def load_scratch_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = GPTModel(GPT_CONFIG_134M)
    model.load_state_dict(state)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return (lambda x: model(x)), GPT_CONFIG_134M["context_length"], n_params, state


def load_gpt2_baseline(device):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return (lambda x: model(x).logits), model.config.n_positions, n_params, None


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

@torch.no_grad()
def strided_perplexity(forward, token_ids, max_length, stride, device):
    """Sliding-window perplexity with the overlap masked out of the loss."""
    n = token_ids.size(0)
    nll_sum, n_scored, prev_end = 0.0, 0, 0

    for begin in range(0, n, stride):
        end = min(begin + max_length, n)
        target_len = end - prev_end          # tokens in this window not yet scored
        if target_len <= 0:
            continue
        ids = token_ids[begin:end].unsqueeze(0).to(device)
        targets = ids.clone()
        targets[:, :-target_len] = -100      # -100 is ignored by cross_entropy

        logits = forward(ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
            ignore_index=-100, reduction="sum")

        nll_sum += loss.item()
        n_scored += int((targets[:, 1:] != -100).sum().item())
        prev_end = end
        if end == n:
            break

    avg_nll = nll_sum / n_scored
    return math.exp(avg_nll), avg_nll, n_scored


@torch.no_grad()
def bin_perplexity(forward, path, context, batch_size, device):
    """Sequential non-overlapping windows over a uint16 token file."""
    data = np.fromfile(path, dtype=np.uint16)
    n_batches = ((len(data) - 1) // context) // batch_size

    total_loss, total_tokens = 0.0, 0
    for b in range(n_batches):
        starts = [(b * batch_size + i) * context for i in range(batch_size)]
        x = torch.stack([torch.from_numpy(data[s:s + context].astype(np.int64))
                         for s in starts]).to(device)
        y = torch.stack([torch.from_numpy(data[s + 1:s + 1 + context].astype(np.int64))
                         for s in starts]).to(device)
        logits = forward(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg = total_loss / total_tokens
    return math.exp(avg), avg, total_tokens


def load_text_tokens(dataset, config, split, column, max_tokens):
    """Tokenize a HuggingFace text dataset with the GPT-2 BPE."""
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(dataset, config, split=split) if config \
        else load_dataset(dataset, split=split)
    text = "\n\n".join(t for t in ds[column] if t)
    ids = enc.encode_ordinary(text)
    if max_tokens and max_tokens < len(ids):
        ids = ids[:max_tokens]
    return torch.tensor(ids, dtype=torch.long)


# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="scratch", choices=["scratch", "gpt2"],
                   help="'scratch' loads --ckpt, 'gpt2' loads GPT-2-small as a baseline")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--mode", default="wikitext", choices=["bin", "wikitext", "hf"])
    p.add_argument("--all", action="store_true",
                   help="run every dataset and print a markdown table")
    p.add_argument("--data-bin", default="validation.bin", help="--mode bin only")
    p.add_argument("--window", type=int, default=128,
                   help="scoring window. The run trained at 128")
    p.add_argument("--stride", type=int, default=None,
                   help="window advance, defaults to --window (non-overlapping)")
    p.add_argument("--batch-size", type=int, default=8, help="--mode bin only")
    p.add_argument("--hf-dataset", default="roneneldan/TinyStories")
    p.add_argument("--hf-config", default=None)
    p.add_argument("--hf-split", default="validation")
    p.add_argument("--hf-column", default="text")
    p.add_argument("--max-tokens", type=int, default=0, help="0 = the whole set")
    return p.parse_args()


def warn_if_past_trained(state, window):
    if state is None:
        return
    trained = trained_positions(state)
    if window > trained:
        print(f"[WARN]  only positions 0-{trained - 1} of this checkpoint were trained, "
              f"but the window is {window}.")
        print(f"[WARN]  positions {trained}-{window - 1} never received gradient. "
              f"Re-run with --window {trained} for the honest number.")


def run_one(label, forward, state, args, mode, dataset_label, **kw):
    window = args.window
    warn_if_past_trained(state, window)
    t0 = time.time()

    if mode == "bin":
        print(f"[data]  {kw['path']}")
        print(f"[eval]  sequential non-overlapping windows | context {window}", flush=True)
        ppl, nll, scored = bin_perplexity(
            forward, kw["path"], window, args.batch_size, kw["device"])
    else:
        tokens = load_text_tokens(kw["dataset"], kw["config"], kw["split"],
                                  kw["column"], args.max_tokens)
        stride = args.stride or window
        print(f"[data]  {dataset_label}: {tokens.size(0):,} tokens")
        print(f"[eval]  max_length {window} | stride {stride}", flush=True)
        ppl, nll, scored = strided_perplexity(
            forward, tokens, window, stride, kw["device"])

    print(f"[result] perplexity {ppl:.2f} | avg NLL {nll:.4f} | "
          f"{scored:,} tokens scored | {time.time() - t0:.0f}s")
    return ppl


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)

    if args.model == "gpt2":
        forward, model_ctx, n_params, state = load_gpt2_baseline(device)
        label = "GPT-2-small (baseline)"
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt is required unless --model gpt2")
        forward, model_ctx, n_params, state = load_scratch_model(args.ckpt, device)
        label = f"ZeroToGPT 134M ({os.path.basename(args.ckpt)})"

    if args.window > model_ctx:
        raise SystemExit(f"window {args.window} exceeds the model context {model_ctx}")

    print(f"[model] {label} | {n_params / 1e6:.2f}M params | {device}")

    common = dict(device=device)
    if not args.all:
        if args.mode == "bin":
            run_one(label, forward, state, args, "bin",
                    args.data_bin, path=args.data_bin, **common)
        elif args.mode == "wikitext":
            run_one(label, forward, state, args, "text", "WikiText-2 raw test",
                    dataset="Salesforce/wikitext", config="wikitext-2-raw-v1",
                    split="test", column="text", **common)
        else:
            run_one(label, forward, state, args, "text",
                    f"{args.hf_dataset} [{args.hf_split}]",
                    dataset=args.hf_dataset, config=args.hf_config,
                    split=args.hf_split, column=args.hf_column, **common)
        return

    results = {}
    if os.path.exists(args.data_bin):
        print(f"\n=== Held-out FineWeb-Edu ===")
        results["Held-out FineWeb-Edu"] = run_one(
            label, forward, state, args, "bin", args.data_bin,
            path=args.data_bin, **common)
    else:
        print(f"\n[skip]  {args.data_bin} not found, skipping the FineWeb-Edu row")

    print(f"\n=== TinyStories ===")
    results["TinyStories"] = run_one(
        label, forward, state, args, "text", "TinyStories [validation]",
        dataset="roneneldan/TinyStories", config=None,
        split="validation", column="text", **common)

    print(f"\n=== WikiText-2 ===")
    results["WikiText-2"] = run_one(
        label, forward, state, args, "text", "WikiText-2 raw test",
        dataset="Salesforce/wikitext", config="wikitext-2-raw-v1",
        split="test", column="text", **common)

    print("\n\nPaste into the README:\n")
    print("| Dataset | Perplexity | Context |")
    print("|---|---|---|")
    for name, ppl in results.items():
        bold = "**" if name.startswith("Held-out") else ""
        print(f"| {bold}{name}{bold} | {bold}{ppl:.2f}{bold} | {args.window} |")


if __name__ == "__main__":
    main()
