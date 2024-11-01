# _*_ coding: utf-8 _*_

"""
    @Time : 2023/4/19 11:53 
    @Author : smile 笑
    @File : real_former.py
    @desc :
"""


import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        d_head, remainder = divmod(d_model, num_heads)
        assert remainder == 0, "`d_model` should be divisible by `num_heads`"
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(d_head)

        self.kqv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev):
        batch_size, seq_len, _ = x.shape

        kqv = self.kqv_proj(x)
        key, query, value = torch.chunk(kqv, 3, dim=-1)
        # shape == (batch_size, seq_len, d_model)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # key.shape == (batch_size, num_heads, d_head, seq_len)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # qv.shape == (batch_size, num_heads, seq_len, d_head)

        energy = self.scale * torch.matmul(query, key)
        # energy.shape == (batch_size, num_heads, seq_len, seq_len)
        if prev is not None:
            energy = energy + prev

        attn = F.softmax(energy, -1)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        context = torch.matmul(attn, value).transpose(1, 2)
        # context.shape == (batch_size, seq_len, num_heads, d_head)
        context = context.reshape(batch_size, seq_len, -1)
        out = self.dropout(self.out_proj(context))

        return out, energy


class FeedForward(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        d_hidden = d_model * expansion_factor
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        out = self.dropout(self.fc2(x))
        return out


class RealFormerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, expansion_factor, dropout):
        super().__init__()
        self.attn = ResidualMultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, prev=None):
        residual = x
        x, prev = self.attn(x, prev)
        x = self.norm1(x + residual)
        residual = x
        x = self.ff(x)
        out = self.norm2(x + residual)
        return out, prev


class RealFormerEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        num_heads=8,
        expansion_factor=2,
        dropout=0.5,
        num_layers=6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RealFormerEncoderLayer(d_model, num_heads, expansion_factor, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        prev = None
        for layer in self.layers:
            x, prev = layer(x, prev)
        return x


if __name__ == '__main__':
    a = torch.randn([2, 196, 768])
    model = RealFormerEncoder(768, 12, 4, 0.2, 12)
    print(model(a).shape)
    print(sum(x.numel() for x in model.parameters()))  # 85054464




