import torch
from models.kan.kan_model import build_kan_from_gene

# Define a small architecture gene to test
gene = {
    "embed_dim": 128,
    "enc_layers": 2,
    "dec_layers": 1,
    "mlp_hidden": 256,
    "n_heads": 2,
    "dropout": 0.1
}

# Create model
model = build_kan_from_gene(gene, vocab_size=2000)

# Dummy data (batch of 2 samples, 64 input tokens, 16 target tokens)
src = torch.randint(0, 2000, (2, 64))
tgt = torch.randint(0, 2000, (2, 16))

# Forward pass
out = model(src, tgt)

print("Output shape:", out.shape)
print("Test completed successfully")
