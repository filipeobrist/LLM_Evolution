import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset
from typing import Union
import random
import copy
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import gc
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# ------------------------------------------------------------
# 1.  Parallel scan
# ------------------------------------------------------------
def npo2(len):
    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))
        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return
        Aa = A[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa = X[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])
        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))
        Aa = A
        Xa = X
        for _ in range(num_steps-2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])
            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return
        Aa = A[:, :, 0:L:2**(num_steps-2)]
        Xa = X[:, :, 0:L:2**(num_steps-2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])
        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]
            T = Xa.size(2)
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        L = X_in.size(1)
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in)
            X = pad_npo2(X_in)
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)
        PScan.pscan(A, X)
        ctx.save_for_backward(A_in, X)
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, X = ctx.saved_tensors
        L = grad_output_in.size(1)
        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in)
            A_in = pad_npo2(A_in)
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)
        A = F.pad(A_in[:, :, 1:], (0, 0, 0, 1))
        PScan.pscan_rev(A, grad_output)
        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])
        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]

pscan = PScan.apply

# ------------------------------------------------------------
# 2.  Mamba components
# ------------------------------------------------------------
@dataclass
class MambaConfig:
    d_model: int
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    rms_norm_eps: float = 1e-5
    base_std: float = 0.02
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False
    mup: bool = False
    mup_base_width: float = 128
    pscan: bool = True
    use_cuda: bool = False

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x

    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm(x), cache)
        return output + x, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner, padding=config.d_conv - 1)
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        if config.inner_layernorms:
            self.dt_layernorm = RMSNorm(config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(config.d_state, config.rms_norm_eps, config.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None
        if config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        y = self.ssm(x, z)
        if self.config.use_cuda:
            return self.out_proj(y)
        z = F.silu(z)
        output = y * z
        return self.out_proj(output)

    def ssm(self, x, z):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)
        if self.config.use_cuda:
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
            return y.transpose(1, 2)
        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)
            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)
            return y

    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * (x.unsqueeze(-1))
        hs = pscan(deltaA, BX)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x
        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * (x.unsqueeze(-1))
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        hs = []
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x
        return y

    def step(self, x, cache):
        h, inputs = cache
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1]
        x = F.silu(x)
        y, h = self.ssm_step(x, h)
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        return output, (h, inputs)

    def ssm_step(self, x, h):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)
        BX = deltaB * (x.unsqueeze(-1))
        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        h = deltaA * h + BX
        y = (h @ C.unsqueeze(-1)).squeeze(2)
        y = y + D * x
        return y, h

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()
        self.use_mup = use_mup
        self.eps = eps
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if not self.use_mup:
            return output * self.weight
        else:
            return output

# ------------------------------------------------------------
# 3.  Jamba model 
# ------------------------------------------------------------
@dataclass
class JambaLMConfig:
    d_model: int
    n_layers: int
    mlp_size: int
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4
    dt_rank: Union[int, str] = 'auto'
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = True
    use_cuda: bool = False
    pscan: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    attention_dropout: float = 0.
    num_experts: int = 16
    num_experts_per_tok: int = 2
    attn_layer_offset: int = 4
    attn_layer_period: int = 8
    expert_layer_offset: int = 1
    expert_layer_period: int = 2
    vocab_size: int = 65536
    pad_token_id: int = 0
    tie_lm_weights: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        self.mamba_config = MambaConfig(d_model=self.d_model, n_layers=0, dt_rank=self.dt_rank,
                                        d_state=self.d_state, expand_factor=self.expand_factor,
                                        d_conv=self.d_conv, dt_min=self.dt_min, dt_max=self.dt_max,
                                        dt_init=self.dt_init, dt_scale=self.dt_scale,
                                        rms_norm_eps=self.rms_norm_eps, bias=self.bias,
                                        conv_bias=self.conv_bias, inner_layernorms=self.inner_layernorms,
                                        pscan=self.pscan, use_cuda=self.use_cuda)

def from_pretrained(name: str):
    """Load a model with pretrained weights, but only to extract the config we need."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("transformers needed")
        return
    model_hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32,
                                                    use_mamba_kernels=False, device_map="auto",
                                                    trust_remote_code=True)
    config = JambaLMConfig(
        vocab_size=model_hf.config.vocab_size, d_model=model_hf.config.hidden_size,
        n_layers=model_hf.config.num_hidden_layers,
        rms_norm_eps=model_hf.config.rms_norm_eps, mlp_size=model_hf.config.intermediate_size,
        inner_layernorms=model_hf.config.mamba_inner_layernorms,
        expand_factor=model_hf.config.mamba_expand, dt_rank=model_hf.config.mamba_dt_rank,
        d_state=model_hf.config.mamba_d_state, d_conv=model_hf.config.mamba_d_conv,
        conv_bias=model_hf.config.mamba_conv_bias, initializer_range=model_hf.config.initializer_range,
        num_experts=model_hf.config.num_experts, num_experts_per_tok=model_hf.config.num_experts_per_tok,
        attn_layer_offset=model_hf.config.attn_layer_offset,
        attn_layer_period=model_hf.config.attn_layer_period,
        expert_layer_offset=model_hf.config.expert_layer_offset,
        expert_layer_period=model_hf.config.expert_layer_period,
        num_key_value_heads=model_hf.config.num_key_value_heads,
        num_attention_heads=model_hf.config.num_attention_heads,
        pad_token_id=model_hf.config.pad_token_id, bias=model_hf.config.mamba_proj_bias,
        attention_dropout=model_hf.config.attention_dropout,
        tie_lm_weights=model_hf.config.tie_word_embeddings
    )
    del model_hf
    return config  # return config only – we’ll not use pretrained weights

class JambaLM(nn.Module):
    def __init__(self, config: JambaLMConfig, genotype: list = None):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.jamba = Jamba(config, genotype)                    # ← genotype passed
        self.final_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_lm_weights:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def forward(self, tokens):
        x = self.embedding(tokens)
        x, router_logits = self.jamba(x)
        x = self.final_layernorm(x)
        logits = self.lm_head(x)
        if self.config.num_experts == 1:
            return logits
        else:
            return logits, router_logits

    def step(self, tokens, caches):
        x = self.embedding(tokens)
        x, caches = self.jamba.step(x, caches)
        x = self.final_layernorm(x)
        logits = self.lm_head(x)
        return logits, caches

    def generate(self, tokenizer, prompt: str, max_tokens: int = 50, batch_size: int = 1,
                 sample: bool = True, top_k: int = 40, temperature: float = 1.0):
        self.eval()
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device)
        input_ids = input_ids.repeat(batch_size, 1)
        caches = [self.jamba.layers[i].get_empty_cache(batch_size, input_ids.device)
                  for i in range(len(self.jamba.layers))]
        for i in range(input_ids.size(1) + max_tokens - 1):
            with torch.no_grad():
                next_token_logits, caches = self.step(input_ids[:, [i]], caches)
                next_token_logits = next_token_logits.squeeze(1)
            if i+1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k)
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)
                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(probs, dim=-1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        outputs = [tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in input_ids[:, 1:]]
        self.train()
        return outputs[0] if batch_size==1 else outputs

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class Jamba(nn.Module):
    def __init__(self, config: JambaLMConfig, genotype: list = None):
        super().__init__()
        self.config = config
        decoder_layers = []
        if genotype is None:
            # original Mini‑Jamba pattern
            for i in range(config.n_layers):
                is_attn = (i - config.attn_layer_offset) % config.attn_layer_period == 0
                is_expert = (i - config.expert_layer_offset) % config.expert_layer_period == 0
                num_experts = config.num_experts if is_expert else 1
                if is_attn:
                    decoder_layers.append(AttentionLayer(config, num_experts=num_experts))
                else:
                    decoder_layers.append(MambaLayer(config, num_experts=num_experts))
        else:
            # genotype‑driven: 0 → Mamba, 1 → Attention
            for i, gene in enumerate(genotype):
                is_expert = (i - config.expert_layer_offset) % config.expert_layer_period == 0
                num_experts = config.num_experts if is_expert else 1
                if gene == 0:
                    decoder_layers.append(MambaLayer(config, num_experts=num_experts))
                elif gene == 1:
                    decoder_layers.append(AttentionLayer(config, num_experts=num_experts))
                else:
                    raise ValueError(f"Invalid gene: {gene}")
        self.layers = nn.ModuleList(decoder_layers)

    def forward(self, x):
        router_logits = []
        for layer in self.layers:
            layer_output, _ = layer(x)
            x = layer_output[0]
            router_logits.append(layer_output[1])
        return x, router_logits

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            layer_output, caches[i] = layer(x, caches[i])
            x = layer_output[0]
        return x, caches

class AttentionLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()
        self.self_attn = AttentionSDPA(config)
        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache=None):
        # no active flag
        residual = x
        x = self.input_layernorm(x)
        x, cache = self.self_attn(x, cache)
        x = residual + x
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x
        return (x, router_logits), cache

    def get_empty_cache(self, batch_size, device):
        return (None, None)

class AttentionSDPA(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, cache=None):
        B, L, _ = x.size()
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            past_keys, past_values = cache
            if past_keys is not None:
                keys = torch.cat([past_keys, keys], dim=2)
                values = torch.cat([past_values, values], dim=2)
            cache = (keys, values)
        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(queries, keys, values,
                                                     dropout_p=self.attention_dropout if self.training else 0.0,
                                                     is_causal=(cache is None))
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, cache

class MambaLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()
        self.config = config
        self.mamba = MambaBlock(config=config.mamba_config)
        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache=None):
        # no active flag
        residual = x
        x = self.input_layernorm(x)
        if cache is None:
            x = self.mamba(x)
        else:
            x, cache = self.mamba.step(x.squeeze(1), cache)
            x = x.unsqueeze(1)
        x = residual + x
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x
        return (x, router_logits), cache

    def get_empty_cache(self, batch_size, device):
        return (None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv-1, device=device))

class SparseMoEBlock(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        if num_experts > 1:
            self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        else:
            self.router = None
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

    def forward(self, x):
        batch_size, sequence_length, hidden_dim = x.shape
        if self.num_experts == 1:
            final_hidden_states = self.experts[0](x)
            router_logits = torch.ones((batch_size * sequence_length, 1), device=x.device,
                                       dtype=x.dtype, requires_grad=x.requires_grad)
            return final_hidden_states, router_logits
        x = x.view(-1, hidden_dim)
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(x.dtype)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

class MLP(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()
        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size
        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

def load_balancing_loss(router_logits, num_experts, num_experts_per_tok):
    router_logits = torch.cat([r for r in router_logits if r.shape[1] > 1], dim=0)
    routing_weights = F.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_experts)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    return torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# ------------------------------------------------------------
# 4.  Classifier wrapper
# ------------------------------------------------------------
class JambaClassifier(nn.Module):
    def __init__(self, base_lm, num_classes):
        super().__init__()
        self.lm = base_lm
        d_model = int(base_lm.config.d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids):
        x = self.lm.embedding(input_ids)
        outputs = self.lm.jamba(x)                # returns (hidden, router_logits)
        hidden_states = self.lm.final_layernorm(outputs[0])
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)