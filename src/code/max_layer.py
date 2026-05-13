import torch
import gc
import random
import numpy as np

# ------------------------------------------------------------
# 1.  Same imports and config as run_evolution.py (AG News)
# ------------------------------------------------------------
from jamba_model_evolve import (
    JambaLM, JambaLMConfig, JambaClassifier, 
)

NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# 2.  Experiment settings – keep them identical to your evolution
# ------------------------------------------------------------
BATCH_SIZE = 32            # AG News batch size
MAX_LENGTH = 128           # AG News token length
MAX_LAYERS = 16            # maximum possible layers
NUM_CLASSES = 4            # AG News classes

# ------------------------------------------------------------
# 3.  Build a config that exactly matches Mini‑Jamba
#     (this is what base_model.config gives during evolution)
# ------------------------------------------------------------
config = JambaLMConfig(
    d_model=256,
    n_layers=MAX_LAYERS,
    mlp_size=512,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    d_state=16,
    expand_factor=2,
    d_conv=4,
    dt_rank=16,
    dt_min=0.001,
    dt_max=0.1,
    dt_init="random",
    dt_scale=1.0,
    bias=False,
    conv_bias=True,
    inner_layernorms=True,
    use_cuda=False,
    pscan=True,
    num_attention_heads=32,
    num_key_value_heads=8,
    attention_dropout=0.0,
    num_experts=16,
    num_experts_per_tok=2,
    attn_layer_offset=4,
    attn_layer_period=8,
    expert_layer_offset=1,
    expert_layer_period=2,
    vocab_size=65536,
    pad_token_id=0,
    tie_lm_weights=True
)

# ------------------------------------------------------------
# 4.  Build the heaviest possible model (20 layers, random genotype)
# ------------------------------------------------------------
genotype = [0 for _ in range(MAX_LAYERS)]
print(f"Test genotype: {genotype}")

base_lm = JambaLM(config, genotype).to(DEVICE)
model = JambaClassifier(base_lm, NUM_CLASSES).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ------------------------------------------------------------
# 5.  Simulate a single real batch (same dimensions as AG News)
# ------------------------------------------------------------
dummy_ids = torch.randint(0, 20000, (BATCH_SIZE, MAX_LENGTH), device=DEVICE)
dummy_labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)

# ------------------------------------------------------------
# 6.  Clean caches and reset memory stats
# ------------------------------------------------------------
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# ------------------------------------------------------------
# 7.  One training step (forward + backward + optimizer)
# ------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
model.train()
out = model(dummy_ids)
loss = torch.nn.functional.cross_entropy(out, dummy_labels)
loss.backward()
optimizer.step()

# ------------------------------------------------------------
# 8.  Report peak GPU memory usage
# ------------------------------------------------------------
peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
peak_gb = peak_mb / 1024
print(f"Peak memory used: {peak_mb:.1f} MiB  ({peak_gb:.2f} GiB)")