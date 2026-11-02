"""The released SimpleTransformer (model/transformer.py), reimplemented with an optional causal mask.

Faithful details: fixed sinusoidal positions; per layer, ONE LayerNorm before attention only, the MLP
(d -> 2d -> d, ReLU, dropout after the ReLU) reads the un-normalised residual stream; no final norm; separate
output projection; linear layers initialised N(0, 0.02^2) with zero bias (their init_weights), embeddings with
PyTorch defaults. causal=False reproduces the released code, where position i can attend to position i + 1,
which is its own next-token label.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal(length, width):
    position = torch.arange(length).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, width, 2) * -(math.log(10000.0) / width))
    encoding = torch.zeros(1, length, width)
    encoding[0, :, 0::2] = torch.sin(position * frequency)
    encoding[0, :, 1::2] = torch.cos(position * frequency[: width // 2])
    return encoding


class Layer(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))
        self.norm = nn.LayerNorm(width)
        self.attention_dropout, self.output_dropout = nn.Dropout(dropout), nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(2 * width, width))

    def forward(self, x, causal):
        batch, length, width = x.shape
        residual = x
        x = self.norm(x)
        q, k, v = (proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for proj in (self.q, self.k, self.v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if causal:
            future = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(future, -1e9)
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        out = residual + self.output_dropout(self.o(out))
        return out + self.mlp(out)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, width=30, layers=4, heads=6, max_length=20, dropout=0.0, causal=True):
        super().__init__()
        self.causal = causal
        self.embedding = nn.Embedding(vocab_size, width)
        self.register_buffer("positions", sinusoidal(2 * max_length, width), persistent=False)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(Layer(width, heads, dropout) for _ in range(layers))
        self.head = nn.Linear(width, vocab_size)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        x = self.embedding_dropout(self.embedding(ids) + self.positions[:, : ids.shape[1]])
        for layer in self.layers:
            x = layer(x, self.causal)
        return self.head(x)  # [batch, length, vocab]
