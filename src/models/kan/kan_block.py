"""
KAN Block based on the KAT/KAN implementation.

This file attempts to import a true KAN linear operator (KANLinear) from one of the
popular KAN repositories. If not found, it falls back to a Linear-based shim with a
warning so you can prototype before installing the true KAN package.

To use real KAN:
- Clone the repository (example): git clone https://github.com/Adamdad/kat.git
- Add the repo path to PYTHONPATH or install as editable:
    pip install -e /path/to/kat
- Then this module will import the proper KANLinear operator.

Author: Assistant (adapted for EvoMix-NAS)
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing known KAN implementations (common module names)
KANLinear = None
_try_names = [
    "kat_rational.kat_1dgroup_torch",  # rational kat cu style
    "kat.kat_1dgroup_torch",           # alternate layout
    "kat.kat_1dgroup",                 # generic
    "kat_1dgroup_torch",               # possible direct file
]

for modname in _try_names:
    try:
        module = __import__(modname, fromlist=["KANLinear"])
        if hasattr(module, "KANLinear"):
            KANLinear = getattr(module, "KANLinear")
            break
        # Some repos use different symbol names; try to discover possible candidates
        for cand in ["KANLinear", "KanLinear", "KatLinear", "kat_linear"]:
            if hasattr(module, cand):
                KANLinear = getattr(module, cand)
                break
        if KANLinear is not None:
            break
    except Exception:
        continue

if KANLinear is None:
    warnings.warn(
        "KANLinear not found. The KANBlock will use a Linear fallback. "
        "To enable real KAN, clone a KAN implementation (e.g. https://github.com/Adamdad/kat) "
        "and install it (pip install -e /path/to/kat) or add it to PYTHONPATH."
    )

    class KANLinearFallback(nn.Module):
        """Fallback that mimics the KAN API but uses a Linear layer."""
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        def forward(self, x):
            # x: (batch, seq_len, in_features)
            # behave like an elementwise mapping similar to KAN
            return self.linear(x)

    KANLinear = KANLinearFallback  # assign the fallback


class KANBlock(nn.Module):
    """
    A KAN block that uses KANLinear for function-based transforms.
    Structure:
      - LayerNorm
      - (Optional) Self-attention (we keep a simple MHSA; KAT papers often integrate KAN inside MLP)
      - Residual + KANLinear MLP mapping
    """

    def __init__(
        self,
        embed_dim: int = 256,
        mlp_hidden: int = 1024,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_attn: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_attn = use_attn

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        if self.use_attn:
            # keep a standard MultiheadAttention for token mixing (batch_first)
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        else:
            self.attn = None

        # KANLinear is expected to be a module mapping (batch, seq_len, embed_dim) -> same shape
        # Many KAN implementations take shape (in_dim, out_dim), but some expect different args.
        # We assume KANLinear(in, out) signature; fallback uses nn.Linear.
        self.kan_fc1 = KANLinear(embed_dim, mlp_hidden)
        self.activation = nn.SiLU()
        self.kan_fc2 = KANLinear(mlp_hidden, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: (batch, seq_len, embed_dim)
        """
        # self-attention / mixing
        residual = x
        if self.attn is not None:
            x_ln = self.ln1(x)
            # nn.MultiheadAttention expects (batch, seq_len, embed_dim) with batch_first=True
            attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask)
            x = residual + self.dropout(attn_out)
        else:
            x = residual

        # KAN MLP
        residual = x
        x_ln = self.ln2(x)
        # apply first KAN-based linear mapping
        x_mid = self.kan_fc1(x_ln)
        x_mid = self.activation(x_mid)
        x_mid = self.dropout(x_mid)
        x_mid = self.kan_fc2(x_mid)
        x = residual + self.dropout(x_mid)
        return x
