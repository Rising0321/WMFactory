# From https://github.com/karpathy/nanoGPT/blob/master/model.py - Thanks Andrej Karpathy

# MIT License
# Copyright (c) 2022 Andrej Karpathy
#               2023 Microsoft Research

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.


"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from dataclasses import dataclass
import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

NEGATIVE_INFINITE_FLOAT = -float("inf")
CROSS_ENTROPY_INVALID_CLASS_TARGET = -1

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def limit_logits_to_valid_range(logits, valid_token_range):
    """
    MODIFIES logits INPLACE.
    Mask out invalid positions in the logits tensor with -inf so they are not considered by the softmax.

    Args:
        logits: logits tensor of shape (batch_size, vocab_size)
        valid_token_range: tuple of (start, end) indices of valid positions in the logits tensor (inclusive).
                           Everything outside is masked out with -inf.
    """
    logits[:, : valid_token_range[0]] = NEGATIVE_INFINITE_FLOAT
    logits[:, valid_token_range[1] + 1 :] = NEGATIVE_INFINITE_FLOAT


def default_sample_token(logits, valid_token_range=None, temperature=1.0, deterministic=False, top_k=None, top_p=None, min_tokens_to_keep=1):
    """
    Given a vector of logits, sample and return an index according to settings.

    logits: tensor of shape (batch_size, vocab_size)

    valid_token_range should be a tuple, specifying start and end indices we'd like to sample from (inclusive).
    If None, we'll sample from the full vocab.

    If deterministic is True, we'll take the argmax of the logits which implies top-k sampling with top_k = 1, therefore user inputted values of top_p and top_k will be ignored.

    Otherwise, either top-p (float) value can be specified or top-k (int) value can be specified.
    Top-p (float top_p) : only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    Top-k (int top_k) : selects top_k tokens for generation.
    min_tokens_to_keep: Used with both top_p and top_k sampling.
    """
    assert top_k is None or top_p is None, "Can only specify one of top-k or top-p sampling."
    if temperature < 0.1:
        # Avoid too low a temp, especially 0
        temperature = 0.1
    logits = logits / temperature
    if valid_token_range is not None:
        limit_logits_to_valid_range(logits, valid_token_range)
    if deterministic:
        selected_logits = select_logits(logits, top_k=1)
    else:
        selected_logits = select_logits(logits, top_p=top_p, top_k=top_k, min_tokens_to_keep=min_tokens_to_keep)
    probs = F.softmax(selected_logits, dim=-1)
    # More robustly handle errors in the sampling here
    sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return sampled_idx


def select_logits(logits, top_k=None, top_p=None, min_tokens_to_keep=1):
    """
    Select from original logits using top-k or top-p sampling.

    Args:
        logits (torch.Tensor): Logits to sample from.
        k (int, optional): Number of top elements to consider in top-k sampling.
        p (float, optional): Threshold probability for top-p sampling.
        min_tokens_to_keep (int, optional): Minimum number of tokens to keep in the output.

    Returns:
        logits: Selected logits after top-k or top-p sampling. Sets all logits outside the selected ones to NEGATIVE_INFINITE_FLOAT.
    """
    assert top_k is None or top_p is None, "Can only specify one of top-k or top-p sampling."
    min_tokens_to_keep = min(min_tokens_to_keep, logits.size(-1))
    if top_k is not None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        # Top-k sampling
        top_k = max(top_k, min_tokens_to_keep)
        top_k = min(top_k, logits.size(-1))
        top_k_logits, _ = torch.topk(logits, top_k)
        indices_to_remove = logits < top_k_logits[..., -1:]
        logits = torch.where(indices_to_remove, NEGATIVE_INFINITE_FLOAT, logits)

    elif top_p is not None:
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        # Top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = torch.where(indices_to_remove, NEGATIVE_INFINITE_FLOAT, logits)

    else:
        # Return logits as is
        pass

    return logits


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class LayerNormMinimal(nn.Module):
    """LayerNorm like above, but without learnable parameters"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.ndim = (ndim,)

    def forward(self, input):
        return F.layer_norm(input, self.ndim, eps=1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and self.dropout == 0.0
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size), persistent=False)

        self.cached_k = None
        self.cached_v = None
        self.current_cache_size = 0

    def _manual_causal_attention(self, q, k, v, mask):
        # q, k and v should be of shape (B, nh, T, hs)
        token_len = q.size(-2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask[:, :, :token_len, :token_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        return y

    def forward(self, x, cache=False):
        batch_size, token_len, n_embd = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, token_len, self.n_head, n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, token_len, self.n_head, n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, token_len, self.n_head, n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and not cache:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        elif cache:
            # manual implemention of attention (as below), but cache arrays we can reuse
            assert token_len == 1, "Cache only works for single step"
            assert self.cached_k is not None, "Must call reset_cache() before using cache"
            assert self.current_cache_size < self.cached_k.size(2), "Trying to generate more steps than provided in reset_cache() `num_steps_to_come`"
            assert self.dropout == 0.0, "Dropout not supported with caching"
            this_step_q = q
            self.cached_k[:, :, self.current_cache_size, :] = k[:, :, 0, :]
            self.cached_v[:, :, self.current_cache_size, :] = v[:, :, 0, :]
            # Remove the zero parts
            k = self.cached_k[:, :, : self.current_cache_size + 1, :]
            # compute last row of the attention mask
            this_step_att_row = (this_step_q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            this_step_att_row = F.softmax(this_step_att_row, dim=-1)
            # We only need output for the current step
            y = this_step_att_row @ self.cached_v[:, :, : self.current_cache_size + 1, :]
            # Update cache
            self.current_cache_size += 1
        else:
            y = self._manual_causal_attention(q, k, v, self.bias)
        y = y.transpose(1, 2).contiguous().view(batch_size, token_len, n_embd)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def reset_cache(self, x, num_steps_to_come):
        """
        Reset caches by doing initial pass with x data (returning same output as forward).
        Also set the number of steps to come, which is used to initialize the buffers
        """
        batch_size, token_len, n_embd = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, token_len, self.n_head, n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, token_len, self.n_head, n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, token_len, self.n_head, n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Use SDPA instead of a manual implementation
        # y = self._manual_causal_attention(q, k, v, self.bias)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(batch_size, token_len, n_embd)
        # output projection
        y = self.resid_dropout(self.c_proj(y))

        # Create full k,q,v for predicting all future steps.
        # Just null-out the last num_steps_to_come-1 steps
        pad_size = num_steps_to_come
        self.current_cache_size = token_len
        self.cached_k = torch.cat([k, torch.zeros(batch_size, self.n_head, pad_size, n_embd // self.n_head, device=k.device)], dim=2)
        self.cached_v = torch.cat([v, torch.zeros(batch_size, self.n_head, pad_size, n_embd // self.n_head, device=v.device)], dim=2)

        return y

class SelfAttention(nn.Module):
    """
    Non-causal self-attention layer, the same as CausalSelfAttention but without the causal mask.
    Duplicating the code to keep this separate for clarity.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and self.dropout == 0.0
        assert self.flash, "SelfAttention only supports flash attention for now."

        self.register_buffer("attn_mask", torch.ones((config.block_size, config.block_size)).bool().unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        batch_size, token_len, n_embd = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, token_len, self.n_head, n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, token_len, self.n_head, n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, token_len, self.n_head, n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask, dropout_p=self.dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(batch_size, token_len, n_embd)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GELU_MLP(nn.Module):
    """MLP Block using PyTorch's native GELU activation function"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cache=False, reset_cache_with_num_steps_to_come=None):
        """
        Args:
            cache: If True, use the cache to predict the next token (assumes model was initialized with `reset_cache`).
            reset_cache_with_num_steps_to_come:
                If not None, reset and prepare the cache for cached prediction of the next `reset_cache_with_num_steps_to_come` tokens.
                This is same as calling `reset_cache` with the same argument, but we include option here in `forward` to support torch hook functions (used to get embeddings from this module output).

        Caching example:
            ```
            # Initialize model with reset_cache_with_num_steps_to_come=10
            outputs[0] = model(inputs, reset_cache_with_num_steps_to_come=10)
            # Predict next 10 tokens using cache
            for i in range(10):
                outputs[i+1] = model(inputs, cache=True)
            ```
        """
        if reset_cache_with_num_steps_to_come:
            return self.reset_cache(x, num_steps_to_come=reset_cache_with_num_steps_to_come)
        x = x + self.attn(self.ln_1(x), cache=cache)
        x = x + self.mlp(self.ln_2(x))
        return x

    def reset_cache(self, x, num_steps_to_come):
        x = x + self.attn.reset_cache(self.ln_1(x), num_steps_to_come=num_steps_to_come)
        x = x + self.mlp(self.ln_2(x))
        return x

class BlockV2(nn.Module):
    """
    Compared to the Block in the original implementation, this one uses non-parametric LayerNorm and Pytorch's GELU.
    These two changes save significant vram but are incompatible with previously trained models.
    Hence the separate class.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNormMinimal(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNormMinimal(config.n_embd, bias=config.bias)
        self.mlp = GELU_MLP(config)

    def forward(self, x, cache=False, reset_cache_with_num_steps_to_come=None):
        if reset_cache_with_num_steps_to_come:
            return self.reset_cache(x, num_steps_to_come=reset_cache_with_num_steps_to_come)
        x = x + self.attn(self.ln_1(x), cache=cache)
        x = x + self.mlp(self.ln_2(x))
        return x

    def reset_cache(self, x, num_steps_to_come):
        x = x + self.attn.reset_cache(self.ln_1(x), num_steps_to_come=num_steps_to_come)
        x = x + self.mlp(self.ln_2(x))
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    version: int = 1 # Version 1 is the original GPT, Version 2 is the one with non-parametric LayerNorm and Pytorch's GELU


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.version = config.version

        print(f"[nanoGPT] creating model with version {self.version}")

        if self.version == 1:
            transformer_dict = dict(
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        elif self.version == 2:
            transformer_dict = dict(
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([BlockV2(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias), # This one is still parametric due to user error
            )

        transformer_dict["wte"] = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless.
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_pos_encoding(self, x):
        device = x.device
        token_len = x.size(1)
        pos = torch.arange(0, token_len, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        x = x + pos_emb
        return x

    def original_forward(self, idx, targets=None, loss_mask=None, loss_reduction="mean"):
        batch_size, seq_len = idx.shape[:2]
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(self._apply_pos_encoding(tok_emb))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            if loss_mask is not None:
                # Feeding target = CROSS_ENTROPY_INVALID_CLASS_TARGET to cross_entropy will ignore the loss
                # for that position. This is useful for padding tokens.
                targets[loss_mask == 0] = CROSS_ENTROPY_INVALID_CLASS_TARGET
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, self.config.vocab_size), targets.view(-1), ignore_index=CROSS_ENTROPY_INVALID_CLASS_TARGET, reduction=loss_reduction
            )
            if loss_reduction == "none":
                # Reshape back into batch_size and seq_len
                loss = loss.view(batch_size, seq_len)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def forward(self, x, targets=None, loss_mask=None, loss_reduction="mean"):
        token_len = x.size(1)
        assert token_len <= self.config.block_size, f"Cannot forward sequence of length {token_len}, block size is only {self.config.block_size}"
        return self.original_forward(x, targets, loss_mask, loss_reduction)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, valid_token_range=None, temperature=1.0, top_k=None, raise_cropping=False, deterministic=False):
        """
        valid_token_range should be a tuple, specifying start and end indices we'd like to sample from (inclusive).
        if None, we'll sample from the full vocab.

        If raise_cropping is True, we'll raise an error if we need to crop the sequence context.
        """
        if valid_token_range is None:
            valid_token_range = (0, self.config.vocab_size - 1)
        assert len(valid_token_range) == 2
        assert valid_token_range[0] < valid_token_range[1]
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx
            if idx.size(1) > self.config.block_size:
                if raise_cropping:
                    raise ValueError("Tried to crop idxs but flag told to raise this")
                else:
                    idx_cond = idx[:, -self.config.block_size :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature  # logits is B T Vocabsize -> B Vocabsize
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = NEGATIVE_INFINITE_FLOAT

            # Crop out the logits we don't want to sample from
            if valid_token_range is not None:
                limit_logits_to_valid_range(logits, valid_token_range)

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                # Take max of the results
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def optimized_generate(
        self,
        idx,
        num_new_tokens,
        valid_token_ranges=None,
        temperature=1.0,
        deterministic=False,
        raise_cropping=False,
        top_k=None,
        top_p=None,
        min_tokens_to_keep=1,
    ):
        """
        Generate function but optimized by caching the results in transformer blocks (think this is referred to as "attention caching").
        The higher the num_new_tokens, the more the speedup compared to original generate.

        Caveat: the context length + num_new_tokens must be less than the block size. This means that the first
                generated tokens do not have full context length.

        valid_token_ranges should be None or list of length num_new_tokens, specifying valid range for tokens for every step
        """
        # Properly compile the modules used and/or quantize for improved speed.
        logit_layer = self.lm_head
        embedder_fn = self.transformer.wte

        if valid_token_ranges is None:
            valid_token_ranges = [[0, self.config.vocab_size] for _ in range(num_new_tokens)]
        assert len(valid_token_ranges) == num_new_tokens, "valid_token_ranges should be list of length num_new_tokens or None"

        _, token_len = idx.size()
        if token_len + num_new_tokens > self.config.block_size:
            raise ValueError("Can't use optimized generation with num_new_tokens + context_length > block_size")
        new_idxs = torch.zeros(idx.size(0), num_new_tokens, dtype=torch.long, device=idx.device)
        # First, we need to cull the sequence to the block size
        # and remove first max_new_tokens so we can reuse same position embeddings
        # and not have to recompute them
        num_original_tokens = idx.size(1)
        original_idx = idx
        if (num_original_tokens + num_new_tokens) > self.config.block_size:
            if raise_cropping:
                raise ValueError("Tried to crop idxs but flag told to raise this")
            original_idx = idx[:, -self.config.block_size + num_new_tokens :]
        original_pos = torch.arange(0, original_idx.size(1), dtype=torch.long, device=idx.device).unsqueeze(0)
        # Now cache results with the original context
        original_tok_emb = embedder_fn(original_idx)
        original_pos_emb = self.transformer.wpe(original_pos)
        original_x = original_tok_emb + original_pos_emb
        for block in self.transformer.h:
            # Reset the cache for each block, and cache new result
            original_x = block(original_x, reset_cache_with_num_steps_to_come=num_new_tokens)

        # Sample the first token
        original_x = self.transformer.ln_f(original_x)
        last_logit = logit_layer(original_x[:, [-1], :])
        new_idxs[:, 0] = default_sample_token(
            last_logit[:, -1, :], valid_token_ranges[0], temperature, deterministic, top_k=top_k, top_p=top_p, min_tokens_to_keep=min_tokens_to_keep
        )

        # Generate rest of the steps
        for generation_idx in range(1, num_new_tokens):
            # forward the model to get the logits for the index in the sequence
            # This is the position of the latest generated token, not the currently going-to-be-generated token
            latest_token_pos = num_original_tokens + generation_idx - 1
            # We only need to pass in the latest token
            newest_idx = new_idxs[:, generation_idx - 1].unsqueeze(-1)
            newest_tok_emb = embedder_fn(newest_idx)
            newest_pos_emb = self.transformer.wpe(torch.tensor(latest_token_pos, dtype=torch.long, device=idx.device).unsqueeze(0))
            newest_x = newest_tok_emb + newest_pos_emb
            for block in self.transformer.h:
                newest_x = block(newest_x, cache=True)

            newest_x = self.transformer.ln_f(newest_x)
            newest_logit = logit_layer(newest_x)
            # Check this function isn't slowing things down noticeably
            new_idxs[:, generation_idx] = default_sample_token(
                newest_logit[:, -1, :], valid_token_ranges[generation_idx], temperature, deterministic, top_k=top_k, top_p=top_p, min_tokens_to_keep=min_tokens_to_keep
            )

        # Combine indices
        new_idxs = torch.cat((idx, new_idxs), dim=1)
        return new_idxs
