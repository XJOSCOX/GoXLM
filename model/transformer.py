from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GoXLMConfig:
    vocab_size: int = 512
    context_length: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GoXLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd
        )

        self.position_embedding = nn.Embedding(
            config.context_length, config.n_embd
        )

        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )

        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > self.config.context_length:
            raise ValueError("Sequence exceeds context length")

        positions = torch.arange(T, device=idx.device)

        x = self.token_embedding(idx)
        x = x + self.position_embedding(positions)

        x = self.blocks(x)
        x = self.ln_final(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss
