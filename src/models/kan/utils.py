"""
Training / evaluation helpers for KAN summarizer.
These are simple utilities similar to transformer utils, designed for small prototyping runs.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, batch: Tuple[torch.Tensor, torch.Tensor], device='cpu'):
    """
    batch: (src_ids, tgt_ids) where tgt_ids are the target token ids for teacher forcing
    """
    model.train()
    src_ids, tgt_ids = batch
    src_ids = src_ids.to(device)
    tgt_ids = tgt_ids.to(device)
    optimizer.zero_grad()
    # Shifted targets for teacher forcing
    input_t = tgt_ids[:, :-1]
    target_t = tgt_ids[:, 1:]
    logits = model(src_ids, input_t)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, logits.size(-1)), target_t.contiguous().view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device='cpu'):
    model.eval()
    total_loss = 0.0
    count = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    for src_ids, tgt_ids in dataloader:
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)
        input_t = tgt_ids[:, :-1]
        target_t = tgt_ids[:, 1:]
        logits = model(src_ids, input_t)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_t.contiguous().view(-1))
        total_loss += float(loss.item()) * src_ids.size(0)
        count += src_ids.size(0)
    return total_loss / max(1, count)
