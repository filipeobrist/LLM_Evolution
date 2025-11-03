import torch
import torch.nn as nn
import math
from .block import TransformerBlock, DecoderBlock

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, n_layers_enc=4, n_layers_dec=4,
                 n_heads=4, ff_multiplier=4.0, dropout=0.1, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len)
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ff_multiplier, dropout)
            for _ in range(n_layers_enc)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(embed_dim, n_heads, ff_multiplier, dropout)
            for _ in range(n_layers_dec)
        ])
        self.ln_enc = nn.LayerNorm(embed_dim)
        self.ln_dec = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def encode(self, src_ids, src_mask=None):
        x = self.pos_enc(self.embed(src_ids))
        for layer in self.encoder:
            x = layer(x, mask=src_mask)
        return self.ln_enc(x)

    def decode(self, tgt_ids, memory, tgt_mask=None, cross_mask=None):
        y = self.pos_enc(self.embed(tgt_ids))
        for layer in self.decoder:
            y = layer(y, memory, self_mask=tgt_mask, cross_mask=cross_mask)
        y = self.ln_dec(y)
        return self.output_head(y)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None, cross_mask=None):
        memory = self.encode(src_ids, src_mask)
        logits = self.decode(tgt_ids, memory, tgt_mask, cross_mask)
        return logits
