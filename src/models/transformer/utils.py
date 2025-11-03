import torch
import torch.nn as nn
import math

# small helpers you might need

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dummy_inputs(batch=2, seq_len=16, device='cpu'):
    import torch
    return torch.randint(0, 2000, (batch, seq_len), dtype=torch.long, device=device)
