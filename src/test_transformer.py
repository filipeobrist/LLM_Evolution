import torch
from models.transformer.transformer_model import TransformerModel

# Define architecture gene
gene = {
    "embed_dim": 256,
    "n_heads": 4,
    "ff_multiplier": 4,
    "n_layers_enc": 2,
    "n_layers_dec": 2,
    "dropout": 0.1,
    "activation": "gelu"
}

# Create model
vocab_size = 2000
model = TransformerModel(
    vocab_size=vocab_size,
    embed_dim=gene["embed_dim"],
    n_heads=gene["n_heads"],
    ff_multiplier=gene["ff_multiplier"],
    n_layers_enc=gene["n_layers_enc"],
    n_layers_dec=gene["n_layers_dec"],
    dropout=gene["dropout"],
)

# Dummy data
src = torch.randint(0, vocab_size, (2, 64))   # 2 samples, 64 input tokens
tgt = torch.randint(0, vocab_size, (2, 16))   # 16 target tokens

# Forward pass
out = model(src, tgt)

print("Output shape:", out.shape)
print("Transformer smoke test passed ✅")
