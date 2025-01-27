import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass
from torch.cuda.amp import autocast, GradScaler  # For mixed precision

# Causal Self Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# Define MLP
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

# Define Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Define GPTConfig with the new parameters
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12    # Set to 12 layers
    n_head: int = 16     # Set to 16 attention heads
    n_embd: int = 1024   # Set embedding size to 1024      # Adjusted embedding dimension

# Define GPT
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

# DataLoaderLite with tiktoken
class DataLoaderLite:
    def __init__(self, B, T, data_path, tokenizer_name="gpt2"):
        self.B = B
        self.T = T
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        end = self.current_position + B * T + 1
        if end > len(self.tokens):
            self.current_position = 0  # Reset to the beginning if we exceed data length
            end = self.current_position + B * T + 1

        buf = self.tokens[self.current_position:end]
        self.current_position += B * T

        # Ensure the buffer length matches the batch size requirements
        if len(buf) < B * T + 1:  # If insufficient data, pad with zeros
            pad_size = (B * T + 1) - len(buf)
            buf = torch.cat([buf, torch.zeros(pad_size, dtype=buf.dtype)], dim=0)

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        return x, y
