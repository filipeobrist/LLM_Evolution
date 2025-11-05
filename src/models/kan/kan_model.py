"""
KAN summarizer model (encoder-decoder lite).
This module provides a simple encoder-decoder style summarizer using KAN blocks.
It's intentionally light for NAS evaluation: small sizes for quick runs.
If you prefer a full seq2seq, we can later add an explicit decoder stack.
"""

import torch
import torch.nn as nn
from .kan_block import KANBlock

class KANEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, n_layers=4, mlp_hidden=1024, n_heads=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            KANBlock(embed_dim=embed_dim, mlp_hidden=mlp_hidden, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, attn_mask=None):
        x = self.embed(input_ids)  # (batch, seq_len, embed_dim)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.ln(x)
        return x  # return sequence representations


class KANDecoder(nn.Module):
    """
    Minimal autoregressive decoder using KAN blocks.
    For prototyping abstractive summarization we include a small decoder that reuses KANBlock.
    """
    def __init__(self, vocab_size, embed_dim=256, n_layers=2, mlp_hidden=1024, n_heads=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            KANBlock(embed_dim=embed_dim, mlp_hidden=mlp_hidden, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt_ids, encoder_outputs=None, attn_mask=None):
        # Note: simple decoder that does not include cross-attention to encoder outputs.
        # You can later extend it to include cross-attention for better abstractive summaries.
        x = self.embed(tgt_ids)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.ln(x)
        logits = self.head(x)
        return logits


class KANSummarizer(nn.Module):
    """
    Combined encoder-decoder summarizer using KAN building blocks.
    For faster prototyping, cross-attention is omitted; later we can add it.
    """
    def __init__(self, vocab_size, embed_dim=256, enc_layers=4, dec_layers=2,
                 mlp_hidden=1024, n_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = KANEncoder(vocab_size, embed_dim, enc_layers, mlp_hidden, n_heads, dropout)
        self.decoder = KANDecoder(vocab_size, embed_dim, dec_layers, mlp_hidden, n_heads, dropout)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        enc = self.encoder(src_ids, src_mask)
        # For now, decoder does not attend to encoder outputs; a future improvement is to add cross-attention.
        logits = self.decoder(tgt_ids, enc, tgt_mask)
        return logits


def build_kan_from_gene(gene: dict, vocab_size: int):
    """
    Helper to instantiate a KAN summarizer from a gene dict:
    gene example keys: embed_dim, enc_layers, dec_layers, mlp_hidden, n_heads, dropout
    """
    return KANSummarizer(
        vocab_size=vocab_size,
        embed_dim=gene.get("embed_dim", 256),
        enc_layers=gene.get("enc_layers", 4),
        dec_layers=gene.get("dec_layers", 2),
        mlp_hidden=gene.get("mlp_hidden", 1024),
        n_heads=gene.get("n_heads", 4),
        dropout=gene.get("dropout", 0.1),
    )
