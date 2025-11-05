import torch
import torch.nn as nn
import math
from typing import Optional

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: (batch, seq_len, embed_dim)
        b, t, _ = x.size()
        qkv = self.qkv_proj(x)  # (b, t, 3*embed)
        qkv = qkv.reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, t, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (b, heads, t, head_dim)

        # attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b, heads, t, t)
        if mask is not None:
            # mask expected to be (b, t) or (b, 1, 1, t) or (t, t)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (b, heads, t, head_dim)
        out = out.transpose(1, 2).reshape(b, t, self.embed_dim)  # (b, t, embed_dim)
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_multiplier=4, dropout=0.0, activation='gelu'):
        super().__init__()
        hidden = int(embed_dim * ff_multiplier)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int = 256,
                 n_heads: int = 4,
                 ff_multiplier: float = 4.0,
                 dropout: float = 0.0,
                 layer_norm_eps: float = 1e-5,
                 pre_norm: bool = True,
                 activation: str = 'gelu'):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout=dropout)
        self.ff = FeedForward(embed_dim, ff_multiplier, dropout=dropout, activation=activation)
        self.pre_norm = pre_norm
        self.ln1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Pre-norm or Post-norm option
        if self.pre_norm:
            y = self.ln1(x)
            y = self.attn(y, mask=mask)
            x = x + self.dropout(y)
            y = self.ln2(x)
            y = self.ff(y)
            return x + self.dropout(y)
        else:
            y = self.attn(x, mask=mask)
            x = self.ln1(x + self.dropout(y))
            y = self.ff(x)
            return self.ln2(x + self.dropout(y))
        

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_multiplier=4.0, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.cross_attn = CrossAttention(embed_dim, n_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_multiplier, dropout, activation)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        # self-attention
        y = self.self_attn(self.ln1(x), mask=self_mask)
        x = x + self.dropout(y)
        # cross-attention
        y = self.cross_attn(self.ln2(x), memory, mask=cross_mask)
        x = x + self.dropout(y)
        # feed-forward
        y = self.ff(self.ln3(x))
        return x + self.dropout(y)



class CrossAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, mask=None):
        # x: decoder hidden (b, t_dec, d)
        # memory: encoder outputs (b, t_enc, d)
        b, t_dec, d = x.size()
        t_enc = memory.size(1)

        q = self.q_proj(x).reshape(b, t_dec, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(memory).reshape(b, t_enc, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).re
