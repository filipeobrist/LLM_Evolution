import torch
import torch.nn as nn
import math
from .transformer_block import TransformerBlock, DecoderBlock

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
    def __init__(self, vocab_size, embed_dim, n_heads, ff_multiplier, n_layers_enc, n_layers_dec, dropout=0.1, activation="gelu"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * ff_multiplier,
            dropout=dropout,
            activation=activation
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * ff_multiplier,
            dropout=dropout,
            activation=activation
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers_enc)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers_dec)
        self.out = nn.Linear(embed_dim, vocab_size)

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
