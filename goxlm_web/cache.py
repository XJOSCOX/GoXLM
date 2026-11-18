"""Request-local inference replay of the existing GoXLM blocks with a KV cache."""
import torch
import torch.nn.functional as F


class CachedDecoder:
    def __init__(self, model, device):
        if model.training:
            raise ValueError("KV caching requires evaluation mode")
        self.model = model
        self.device = device
        self.context = model.config.context_length
        self.layers = []
        self.length = 0
        self.seen = 0
        self.rebuilds = 0

    @torch.inference_mode()
    def logits(self, tokens):
        if not tokens:
            raise ValueError("An empty prefix cannot be decoded")
        append = (self.length > 0 and len(tokens) == self.seen + 1 and self.length < self.context)
        if append:
            start = self.length
            inputs = tokens[-1:]
        else:
            if self.seen:
                self.rebuilds += 1
            start = 0
            inputs = tokens[-self.context:]
        end = start + len(inputs)
        ids = torch.tensor([inputs], dtype=torch.long, device=self.device)
        positions = torch.arange(start, end, device=self.device)
        x = self.model.token_embedding(ids) + self.model.position_embedding(positions)

        for i, block in enumerate(self.model.blocks):
            attention = block.attn
            batch, width, embedding = x.shape
            q, k, v = attention.qkv(block.ln1(x)).chunk(3, dim=-1)
            q, k, v = (t.view(batch, width, attention.n_heads, attention.head_dim).transpose(1, 2)
                       for t in (q, k, v))
            if i == len(self.layers):
                shape = (batch, attention.n_heads, self.context, attention.head_dim)
                self.layers.append((k.new_empty(shape), v.new_empty(shape)))
            keys, values = self.layers[i]
            keys[:, :, start:end].copy_(k)
            values[:, :, start:end].copy_(v)
            if append:
                # The sole query is at the end; all provided keys are causally valid.
                y = F.scaled_dot_product_attention(q, keys[:, :, :end], values[:, :, :end],
                                                   dropout_p=0.0, is_causal=False)
            else:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            x = x + attention.proj(y.transpose(1, 2).contiguous().view(batch, width, embedding))
            x = x + block.mlp(block.ln2(x))
        self.seen, self.length = len(tokens), end
        return self.model.lm_head(self.model.ln_final(x[:, -1]))[0]

    def close(self):
        self.layers.clear()
        self.length = self.seen = 0
