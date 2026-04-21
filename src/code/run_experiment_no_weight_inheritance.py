import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset
from typing import Union
import random
import copy
from transformers import AutoTokenizer
import gc
from torch.utils.data import DataLoader
# from tqdm import tqdm
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt


# Count the execution time
import time


"""

An implementation of the parallel scan operation in PyTorch (Blelloch version).
Please see docs/pscan.ipynb for a detailed explanation of what happens here.

"""

def npo2(len):
    """
    Returns the next power of 2 above len
    """

    return 2 ** math.ceil(math.log2(len))

def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        # only supports L that is a power of two (mainly for a clearer code)
        
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
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

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
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
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
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

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
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
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        L = X_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            A = pad_npo2(A_in) # (B, npo2(L), D, N)
            X = pad_npo2(X_in) # (B, npo2(L), D, N)
        
        # prepare tensors
        A = A.transpose(2, 1) # (B, D, npo2(L), N)
        X = X.transpose(2, 1) # (B, D, npo2(L), N)

        # parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)
        
        # slice [:, :L] (cut if there was padding)
        return X.transpose(2, 1)[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in) # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in) # (B, npo2(L), D, N)

        # prepare tensors
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1) # (B, D, npo2(L), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1)) # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)

        # reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]
    
pscan = PScan.apply

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)


        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y) # (B, L, D)
            return output # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output
    
    def ssm(self, x, z):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)
        
        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2) # (B, L, ED)
        
        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output

# from mamba import MambaConfig, MambaBlock, RMSNorm

"""

This file implements the Jamba architecture, as proposed by AI21labs (altough others have proposed blending Mamba & attention in the past).
A Jamba model combines Mamba and attention layers, as well as MoE for the MLP blocks.

This file closely follows the official Jamba implementation (https://huggingface.co/ai21labs/Jamba-v0.1).
But it is quite shorter (800 lines vs 2100 lines), because it has been stripped of all optional features that come with transformers.
It is thus easier to read, while keeping the same performances.
It supports training (using official CUDA mamba backend or mamba.py) & inference.
You can also load pretrained Jamba models from HF using the from_pretrained function.

Architecture of the torch modules found in this file :
- JambaLM: the final object, used for language modeling. has an embedding layer, an lm head...
- Jamba: core model. inputs (B, L, D), outputs (B, L, D). (B=batch size, L=seq len, D=d_model).
  It is composed of two types of layers : MambaLayer and AttentionLayer.
- AttentionLayer: standard GQA attention layer + MoE MLP (the attn computations are located in the AttentionSDPA module)
- MambaLayer : standard Mamba layer + MoE MLP. (the Mamba computations are located in the mamba.py file)
- SparseMoEBlock and MLP : Moe MLP

Notes :
-if using use_cuda, you must train in float32. If not, the following error is triggered : 
"Expected B.scalar_type() == (!is_variable_B ? weight_type : input_type) to be true, but got false."
when calling the selective_scan_fn function. Not clear why this error shows up when in (b)float16. TODO: investigate.

"""

@dataclass
class JambaLMConfig:
    
    d_model: int
    n_layers: int
    
    mlp_size: int
    
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5

    # mamba related
    d_state: int = 16 # N in paper
    expand_factor: int = 2 # N in paper
    d_conv: int = 4
    dt_rank: Union[int, str] = 'auto'

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = True
    use_cuda: bool = False
    pscan: bool = True # use parallel scan mode or sequential mode when training

    # attention related
    num_attention_heads: int = 32
    num_key_value_heads: int = 8 # GQA
    attention_dropout: float = 0.

    # MoE related
    num_experts: int = 16
    num_experts_per_tok: int = 2

    # structure
    attn_layer_offset: int = 4
    attn_layer_period: int = 8
    expert_layer_offset: int = 1
    expert_layer_period: int = 2

    # language modeling
    vocab_size: int = 65536
    pad_token_id: int = 0
    tie_lm_weights: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        self.mamba_config = MambaConfig(d_model=self.d_model, n_layers=0, dt_rank=self.dt_rank, d_state=self.d_state,
                                        expand_factor=self.expand_factor, d_conv=self.d_conv, dt_min=self.dt_min, dt_max=self.dt_max,
                                        dt_init=self.dt_init, dt_scale=self.dt_scale, rms_norm_eps=self.rms_norm_eps,
                                        bias=self.bias, conv_bias=self.conv_bias, inner_layernorms=self.inner_layernorms,
                                        pscan=self.pscan, use_cuda=self.use_cuda)

def from_pretrained(name: str):
    """
    Returns a model loaded with pretrained weights pulled from HuggingFace.
    The model has to follow the same structure as the original Jamba model on HF (ai21labs/Jamba-v0.1).
    You can easily adapt this function.

    Args:
        name: for example:
            * 'TechxGenus/Mini-Jamba'
            * 'ai21labs/Jamba-v0.1'

    Returns:
        model: a Jamba model configured with the proper parameters and initialized with the proper weights
    """

    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("The from_pretrained function pulls weights from HuggingFace and thus needs transformers to be installed (pip install transformers)")
        return

    model_hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, use_mamba_kernels=False, device_map="auto", trust_remote_code=True)
        
    # copy config data
    config = JambaLMConfig(vocab_size=model_hf.config.vocab_size, d_model=model_hf.config.hidden_size, n_layers=model_hf.config.num_hidden_layers, 
                                rms_norm_eps=model_hf.config.rms_norm_eps, mlp_size=model_hf.config.intermediate_size, inner_layernorms=model_hf.config.mamba_inner_layernorms,
                                expand_factor=model_hf.config.mamba_expand, dt_rank=model_hf.config.mamba_dt_rank, d_state=model_hf.config.mamba_d_state,
                                d_conv=model_hf.config.mamba_d_conv, conv_bias=model_hf.config.mamba_conv_bias, initializer_range=model_hf.config.initializer_range,
                                num_experts=model_hf.config.num_experts, num_experts_per_tok=model_hf.config.num_experts_per_tok, 
                                attn_layer_offset=model_hf.config.attn_layer_offset, attn_layer_period=model_hf.config.attn_layer_period, 
                                expert_layer_offset=model_hf.config.expert_layer_offset, expert_layer_period=model_hf.config.expert_layer_period,
                                num_key_value_heads=model_hf.config.num_key_value_heads, num_attention_heads=model_hf.config.num_attention_heads,
                                pad_token_id=model_hf.config.pad_token_id, bias=model_hf.config.mamba_proj_bias, attention_dropout=model_hf.config.attention_dropout,
                                tie_lm_weights=model_hf.config.tie_word_embeddings)

    model = JambaLM(config)

    # copy weights
    for name, param in model_hf.named_parameters():
        name = name.replace("model.", "jamba.")
        
        if "embed_tokens" in name:
            name = "embedding.weight"
        
        if "final_layernorm" in name:
            name = name.replace("jamba.", "")

        counterpart_param = model.get_parameter(name)
        if counterpart_param is not None:
            counterpart_param.data.copy_(param.data)

    del model_hf

    return model

class JambaLM(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        # Added
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.jamba = Jamba(config)
        self.final_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if self.config.tie_lm_weights:
            self.lm_head.weight = self.embedding.weight 

        self.apply(self._init_weights)

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)
        # router_logits : (B*L, n_experts) if n_experts>1

        x = self.embedding(tokens)

        x, router_logits = self.jamba(x)
        x = self.final_layernorm(x)

        logits = self.lm_head(x)

        if self.config.num_experts == 1:
            return logits
        else:
            return logits, router_logits
    
    def step(self, tokens, caches):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)

        x, caches = self.jamba.step(x, caches)
        x = self.final_layernorm(x)

        logits = self.lm_head(x)

        return logits, caches

    # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    def generate(self, tokenizer, prompt: str, max_tokens: int = 50, batch_size: int = 1, sample: bool = True, top_k: int = 40, temperature: float = 1.0):
        self.eval()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device) # (1, num_tokens)
        input_ids = input_ids.repeat(batch_size, 1)

        # caches is a list of cache, one per layer
        # cache is composed of : - if Mamba layer : the hidden state, and the last d_conv-1 inputs (see more in mamba_lm.py)
        #                        - if Attention layer : the KV cache, ie 2 tensors of shape (B, num_kv_heads, L, head_dim)
        caches = [self.jamba.layers[i].get_empty_cache(batch_size, input_ids.device) for i in range(self.config.n_layers)]

        for i in range(input_ids.size(1) + max_tokens - 1):
            with torch.no_grad():
                # forward the new output, get new cache
                next_token_logits, caches = self.step(input_ids[:, [i]], caches) # (batch_size, 1, vocab_size), caches
                next_token_logits = next_token_logits.squeeze(1)

            # sample (no sampling when the prompt is being processed)
            if i+1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1) # (batch_size, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k) # (batch_size, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size)
                else:
                    next_token = torch.argmax(probs, dim=-1) # (batch_size)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        outputs = [tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in input_ids[:, 1:]]

        self.train()

        if batch_size==1:
            return outputs[0]
        else:
            return outputs
    
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
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.config = config

        # init each model layer, decide if it's mamba/attention and has experts or not
        decoder_layers = []
        for i in range(config.n_layers):
            is_attn = True if (i - self.config.attn_layer_offset) % self.config.attn_layer_period == 0 else False
            is_expert = True if (i - self.config.expert_layer_offset) % self.config.expert_layer_period == 0 else False

            num_experts = self.config.num_experts if is_expert else 1

            if is_attn:
                decoder_layers.append(AttentionLayer(config, num_experts=num_experts))
            else:
                decoder_layers.append(MambaLayer(config, num_experts=num_experts))

        self.layers = nn.ModuleList(decoder_layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x):
        # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)

        router_logits = []

        for decoder_layer in self.layers:
            layer_output, _ = decoder_layer(x)
            x = layer_output[0]
            router_logits.append(layer_output[1])

        return x, router_logits
    
    def step(self, x, caches):
        # x: (B, L, D)

        # logits: (B, L, D)
        # caches

        for i, decoder_layer in enumerate(self.layers):
            layer_output, caches[i] = decoder_layer(x, caches[i])
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

        ## ADDED
        self.layer_type = "attention"
        self.active = True   # ← THIS 

    def forward(self, x, cache = None):
        if not self.active:
            # Identity layer
            return (x, None), cache
        # x: (B, L, D)

        # outputs: (B, L, D)
        
        # attention
        residual = x
        x = self.input_layernorm(x)
        x, cache = self.self_attn(x, cache)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)
        return outputs, cache

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

    def forward(self, x, cache = None):
        # x: (B, L, D)

        # attn_output: (B, L, D)

        B, L, _ = x.size()

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv cache implementation
        if cache is not None:
            past_keys, past_values = cache
            
            # not first in the sequence
            if past_keys is not None:
                keys = torch.cat([past_keys, keys], dim=2)
                values = torch.cat([past_values, values], dim=2)
            
            cache = (keys, values) # prepare cache for next token

        # GQA related
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

        ## ADDED
        self.layer_type = "mamba"
        self.active = True   # ← THIS

    def forward(self, x, cache = None):
        if not self.active:
            return (x, None), cache
        # x: (B, L, D)

        # outputs: (B, L, D)

        # mamba
        residual = x
        x = self.input_layernorm(x)
        if cache is None:
            x = self.mamba(x)
        else:
            x, cache = self.mamba.step(x.squeeze(1), cache)
            x = x.unsqueeze(1)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)

        return outputs, cache
    
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
        # x: (B, L, D)

        # final_hidden_states: (B, L, D)
        # router_logits: (B*L, n_experts)

        #note : it is not clear why we work with shape (B*L, D) here.
        #I copied this code from the official jamba imple, and did not have time to think it through.
        
        batch_size, sequence_length, hidden_dim = x.shape

        # no routing
        if self.num_experts == 1:
            final_hidden_states = self.experts[0](x)
            router_logits = torch.ones(
                (batch_size * sequence_length, 1),
                device=x.device,
                dtype=x.dtype,
                requires_grad=x.requires_grad,
            )
            return final_hidden_states, router_logits

        # routing
        x = x.view(-1, hidden_dim) # (B*L, D)

        router_logits = self.router(x) # (B*L, n_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
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
        # x : (B, L, D)

        # y : (B, L, D)

        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    
def load_balancing_loss(router_logits, num_experts, num_experts_per_tok):
    # router_logits: list of router_logit, one per layer, each (B*D, n_experts)

    # moe_aux_loss : scalar

    router_logits = torch.cat([r for r in router_logits if r.shape[1] > 1], dim=0)

    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    moe_aux_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return moe_aux_loss * num_experts

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Configurations start here
# --- CONSTANTS & CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAMBA, ATTN = 0, 1
MIN_LAYERS, MAX_LAYERS = 4, 20
NUM_CLASSES = 4
MUTATION_RATE = 0.10
CROSSOVER_RATE = 0.8 
POP_SIZE = 30
GENERATIONS = 100
ELITISM = 1
LEARNING_RATE = 4e-4
BATCH_SIZE = 32
STEPS_1, STEPS_2 = 100, 200
FINE_TUNE = True

BATCH_SIZE = 32
STEPS_1 = 100

# Benchmark Mini-Jamba original 
# BASELINE_LATENCY = 0.35 
# BASELINE_PARAMS = 70_000_000

BASELINE_LATENCY = 0.01
BASELINE_PARAMS = 50_000_000


class JambaClassifier(nn.Module):
    def __init__(self, base_lm, num_classes):
        super().__init__()
        self.lm = base_lm
        d_model = int(base_lm.config.d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids):
        x = self.lm.embedding(input_ids) 
        outputs = self.lm.jamba(x)
        hidden_states = self.lm.final_layernorm(outputs[0])
        pooled = hidden_states.mean(dim=1) 
        return self.classifier(pooled)

def load_agnews(tokenizer, n_train=15000, n_val=1000):
    """Loads the AG News dataset and pre-tokenizes it using the provided tokenizer. It returns tokenized train and validation datasets ready for PyTorch."""
    print("Pre-tokenizing dataset...")
    ds = load_dataset("ag_news")
    
    def tokenize_function(examples):
        # Pre-Tokenization
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    # Split
    train = ds["train"].shuffle(seed=42).select(range(n_train))
    val = ds["test"].shuffle(seed=42).select(range(n_val))
    
    tokenized_train = train.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val.map(tokenize_function, batched=True, remove_columns=["text"])
    
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return tokenized_train, tokenized_val


def train_model(model, train_ds, steps, val_ds=None, patience=3, gen0=False, ind_id="0"):
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    best_f1 = 0
    best_weights = copy.deepcopy(model.state_dict())
    no_improve_count = 0
    val_check_interval = 100 
    
    # Listas para guardar métricas apenas se gen0 for True para não gastar RAM à toa
    train_losses = [] if gen0 else None
    val_f1s = [] if gen0 else None
    
    start_time = time.time()
    step_count = 0
    
    while step_count < steps:
        for batch in loader:
            if step_count >= steps: break
            
            input_ids, labels = batch["input_ids"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Só guarda a loss se estivermos na Geração 0
            if gen0:
                train_losses.append(loss.item())
            
            step_count += 1
            
            # --- Verificação de Validação & Early Stopping ---
            if step_count % val_check_interval == 0 and val_ds is not None:
                current_f1, _ = evaluate_model(model, val_ds)
                model.train()
                
                if gen0:
                    val_f1s.append((step_count, current_f1))
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_weights = copy.deepcopy(model.state_dict())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # O Early Stopping corre sempre, mas o gráfico só é gerado na Gen 0
                if no_improve_count >= patience:
                    model.load_state_dict(best_weights)
                    if gen0:
                        save_training_plot(train_losses, val_f1s, ind_id)
                    return time.time() - start_time, train_losses
    
    # Se chegarmos ao fim dos steps sem disparar o Early Stopping
    if gen0:
        save_training_plot(train_losses, val_f1s, ind_id)
        
    return time.time() - start_time, train_losses

def save_training_plot(losses, f1s, ind_id):
    """Gera um gráfico da evolução do treino."""
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(losses, color='tab:red', alpha=0.5, label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Val F1', color='tab:blue')
    steps, f1_values = zip(*f1s) if f1s else ([], [])
    ax2.plot(steps, f1_values, color='tab:blue', marker='o', label='F1')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f'Training Progress - Indivíduo {ind_id}')
    plt.savefig(f'../plots/training_plot_{ind_id}.png')
    plt.close()


# Utils
@torch.no_grad()
def evaluate_model(model, val_ds):
    """Returns F1 score and average latency per sample on the validation set."""
    model.eval()
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    all_preds, all_labels, latencies = [], [], []
    
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"] # Mantém no CPU para facilitar
        
        torch.cuda.synchronize() 
        start_lat = time.perf_counter()
        logits = model(input_ids)
        torch.cuda.synchronize() 
        latencies.append(time.perf_counter() - start_lat)
        
        preds = torch.argmax(logits, dim=-1).cpu() 
        
   
        all_preds.extend(preds.numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
            
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # Latência média por amostra (dividimos pelo batch size para ser real)
    avg_lat = (sum(latencies) / len(latencies)) / BATCH_SIZE
    
    return f1, avg_lat


def get_trainable_state_dict(model):
    """Returns a state_dict containing only parameters that require gradients."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items() if v.requires_grad}


def setup_model_trainability(model, full_fine_tune=False):
    """
    If full_fine_tune is True: Unfreezes everything for max accuracy.
    If full_fine_tune is False: Only unfreezes Norms, MoE, and Classifier.
    """
    if full_fine_tune:
        # print("High-Performance Mode: Unfreezing all parameters...")
        for p in model.parameters():
            p.requires_grad = True
    else:
        # strategy for low VRAM
        for name, p in model.named_parameters():
            if (
                "classifier" in name
                or "layernorm" in name.lower()
                or "moe" in name.lower()
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False


# --- GENETIC OPERATORS ---

def generate_random_genotype():
    """Generates a random genotype with a random length between MIN_LAYERS and MAX_LAYERS. Each gene is either 0 (Mamba) or 1 (Attention)."""
    length = random.randint(MIN_LAYERS, MAX_LAYERS)
    return [random.randint(0, 1) for _ in range(length)]

def crossover(g1, g2):
    """Crossover de ponto único para genótipos de tamanhos variáveis"""
    point = random.randint(1, min(len(g1), len(g2)) - 1)
    # Combina a primeira parte de um com a segunda do outro
    child_g = g1[:point] + g2[point:]
    
    # Garante que o filho respeita os limites de profundidade da tese
    if len(child_g) > MAX_LAYERS:
        child_g = child_g[:MAX_LAYERS]
    elif len(child_g) < MIN_LAYERS:
        # Se for muito pequeno, adiciona camadas aleatórias até ao mínimo
        while len(child_g) < MIN_LAYERS:
            child_g.append(random.randint(0, 1))
            
    return child_g

def mutate(genotype, mutation_rate=0.10):
    """Mutação Bit-flip e Estrutural (agora segura pois treinamos do zero)"""
    new_genotype = list(genotype)
    
    # 1. Bit Flips (Muda o tipo de camada: Jamba vs Mamba)
    for i in range(len(new_genotype)):
        if random.random() < mutation_rate:
            new_genotype[i] = 1 - new_genotype[i]
            
    # 2. Mutação Estrutural (Muda a profundidade)
    # 15% de chance de alterar o número de camadas
    if random.random() < 0.15: 
        if random.random() > 0.5 and len(new_genotype) < MAX_LAYERS:
            # Insere em qualquer posição (agora é seguro!)
            new_genotype.insert(random.randint(0, len(new_genotype)), random.randint(0, 1))
        elif len(new_genotype) > MIN_LAYERS:
            new_genotype.pop(random.randint(0, len(new_genotype) - 1))
            
    return new_genotype

# --- SELECTION ---

def tournament_selection(population, k=3):
    # 1. Pick k random individuals from the whole population
    selection_pool = random.sample(population, k)
    
    # 2. The winner is the one with the best fitness
    winner = max(selection_pool, key=lambda x: x['fitness'])
    
    return winner

# --- GENOTYPE ---

def apply_genotype(model, genotype):
    """Applies the genotype to the model by activating/deactivating layers and setting them to Mamba or Attention based on the gene value."""
    for i, layer in enumerate(model.lm.jamba.layers):
        if i < len(genotype):
            layer.active = True
            layer.use_mamba = (genotype[i] == MAMBA)
            layer.use_attention = (genotype[i] == ATTN)
        else:
            layer.active = False

def evaluate_individual(base_model, genotype, train_ds, val_ds, steps, inherited_weights=None, gen0=False, ind_id="0"):
    """Evaluates an individual by applying its genotype to a fresh model, optionally loading inherited weights, 
    training it, and then evaluating its F1 score and latency. It returns the trainable weights for inheritance
    and a stats dictionary for fitness calculation."""

    model = JambaClassifier(copy.deepcopy(base_model), NUM_CLASSES).to(DEVICE)
    apply_genotype(model, genotype)
    
    if inherited_weights:
        model.load_state_dict(inherited_weights, strict=False)
    
    setup_model_trainability(model, full_fine_tune=FINE_TUNE)

    
    train_time, losses = train_model(model, train_ds, steps, val_ds=val_ds, patience=3, gen0=gen0, ind_id=ind_id)
    f1, latency = evaluate_model(model, val_ds)

    print(f"Evaluated Genotype: {genotype} | F1: {f1:.4f} | Latency: {latency:.4f}s | Train Time: {train_time:.2f}s")

    total_params = 0
    for name, p in model.named_parameters():
        # Só conta se não pertencer a uma camada inativa
        parts = name.split('.')
        if "layers" in parts:
            layer_idx = int(parts[parts.index("layers") + 1])
            if layer_idx < len(genotype):
                total_params += p.numel()
        else:
            total_params += p.numel()
    
    stats = {
        'f1': f1, 'latency': latency, 
        'params': total_params,
        'depth': len(genotype), 'train_time': train_time
    }
    
    # Return only trainable weights for inheritance
    return {k: v.cpu().clone() for k, v in model.state_dict().items() if v.requires_grad}, stats, losses

def fitness(population_list):
    if not population_list: return

    for ind in population_list:
        stats = ind['stats']
        f1 = stats['f1']
        
        base_score = f1

        # lat_ratio =  BASELINE_LATENCY / stats['latency']
        # param_ratio = BASELINE_PARAMS / stats['params']  
        
        # penalty = 1.0 * lat_ratio + 0.5 * param_ratio # Ver o porque de eles usarem o exp 

        ind['fitness'] = base_score # * penalty



def smart_weight_inheritance(child_model, parent_weights, child_genotype, parent_genotype):
    child_dict = child_model.state_dict()
    new_weights = {}

    for name, param in child_dict.items():
        # Se o peso não for de uma camada evoluível (ex: embeddings, final_norm, classifier)
        if "layers" not in name:
            if name in parent_weights:
                new_weights[name] = parent_weights[name]
            continue
        
        # Extrair o índice da camada: "lm.jamba.layers.0.mamba.weights" -> 0
        parts = name.split('.')
        layer_idx = int(parts[parts.index("layers") + 1])
        
        # Só herdar se a camada existir no pai E for do mesmo tipo
        if layer_idx < len(parent_genotype):
            if child_genotype[layer_idx] == parent_genotype[layer_idx]:
                if name in parent_weights:
                    new_weights[name] = parent_weights[name]
    
    # Carregar apenas o que foi filtrado
    child_model.load_state_dict(new_weights, strict=False)

def plot_population_vs_best(data):
    import numpy as np
    plt.figure(figsize=(12, 6))
    
    # 1. Extrair todas as curvas de loss
    all_curves = [d['losses'] for d in data]
    # Garantir que todas têm o mesmo tamanho para a média (padding)
    max_len = STEPS_1
    padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in all_curves])
    
    # 2. Calcular Média
    mean_loss = np.mean(padded, axis=0)
    
    # 3. Identificar o Melhor Indivíduo (por F1)
    best_idx = np.argmax([d['f1'] for d in data])
    best_curve = padded[best_idx]
    
    # 4. Plot
    steps = np.arange(max_len)
    plt.plot(steps, mean_loss, label='Population Average', color='black', linewidth=2, linestyle='--')
    plt.plot(steps, best_curve, label=f'Best model (F1: {data[best_idx]["f1"]:.4f})', color='blue', linewidth=2)
    
    # Estética
    plt.fill_between(steps, np.min(padded, axis=0), np.max(padded, axis=0), color='gray', alpha=0.1, label='Range da População')
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Convergência na Geração 0: Média vs Melhor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../plots/gen0_population_analysis.png')
    plt.close()

# --- EVOLUTION LOOP ---
def evolve(base_model, train_ds, val_ds, pop_size=30, generations=100, elitism=1):
    history_logs = []
    gen0_data = []
    
    # --- Geração 0 ---
    gen_start_time = time.time()
    print(f"--- Generation 0: Initializing {pop_size} individuals ---")
    population = []
    for i in range(pop_size):
        g = generate_random_genotype()
        # Treino do zero para a base inicial
        w, s, losses = evaluate_individual(base_model, g, train_ds, val_ds, STEPS_1, inherited_weights=None, gen0=True, ind_id=str(i))
        population.append({'genotype': g, 'weights': w, 'stats': s})

    fitness(population)
    population = sorted(population, key=lambda x: x['fitness'], reverse=True)
    gen0_data = [{'losses': ind['losses'], 'f1': ind['stats']['f1']} for ind in population]

    gen_duration = (time.time() - gen_start_time) / 60

    for i, ind in enumerate(population):
        history_logs.append({
            'generation': 0, 'rank': i, 'fitness': ind['fitness'],
            'f1': ind['stats']['f1'], 'latency': ind['stats']['latency'],
            'params': ind['stats']['params'], 'depth': ind['stats']['depth'],
            'gen_time_min': gen_duration, 'genotype': str(ind['genotype'])
        })

    pd.DataFrame(history_logs).to_csv("agnews_evolution_results_no_weight_inheritance.csv", index=False)
    print(f"Gen 0 Best: F1 {population[0]['stats']['f1']:.4f}")

    plot_population_vs_best(gen0_data)
    

    # 3. LOOP DE EVOLUÇÃO
    for gen in range(1, generations):
        gen_start_time = time.time()
        new_candidates = population[:elitism]

        print(f"--- Generation {gen}: Evolving population ---")
        
        while len(new_candidates) < pop_size:
            # 1. SELEÇÃO E CROSSOVER
            if random.random() < CROSSOVER_RATE:
                p1, p2 = tournament_selection(population), tournament_selection(population)
                child_g = crossover(p1['genotype'], p2['genotype'])
            else:
                # Se não há crossover, clonamos um progenitor
                parent = tournament_selection(population)
                child_g = list(parent['genotype'])
            
            # 2. MUTAÇÃO (Sempre aplicada ao genótipo resultante)
            child_g = mutate(child_g, MUTATION_RATE)

            # 3. AVALIAÇÃO DO ZERO
            # Passamos inherited_weights=None para forçar o modelo a inicializar do zero
            # O indivíduo é avaliado apenas pelo potencial da sua arquitetura
            w, s, _ = evaluate_individual(base_model, child_g, train_ds, val_ds, STEPS_1, inherited_weights=None)
            
            new_candidates.append({'genotype': child_g, 'weights': w, 'stats': s})

        population = new_candidates
        fitness(population)
        population = sorted(population, key=lambda x: x['fitness'], reverse=True)

        # Logs e Monitorização
        gen_duration = (time.time() - gen_start_time) / 60
        for i, ind in enumerate(population):
            history_logs.append({
                'generation': gen, 'rank': i, 'fitness': ind['fitness'],
                'f1': ind['stats']['f1'], 'latency': ind['stats']['latency'],
                'params': ind['stats']['params'], 'depth': ind['stats']['depth'],
                'gen_time_min': gen_duration, 'genotype': str(ind['genotype'])
            })

        pd.DataFrame(history_logs).to_csv("agnews_evolution_results_no_weight_inheritance.csv", index=False)
        print(f"Gen {gen} Best: F1 {population[0]['stats']['f1']:.4f}")
        
        gc.collect()
        torch.cuda.empty_cache()

    return population[0]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")



tokenizer = AutoTokenizer.from_pretrained("TechxGenus/Mini-Jamba")
train_ds, val_ds = load_agnews(tokenizer)

# We load the base model once and move it to the device
# It stays "frozen" as a template; deepcopy will be used for individuals
base_model = from_pretrained("TechxGenus/Mini-Jamba").to(device)

# Run
def run_thesis_experiment():
    print("Starting Evolution")

    best_individual = evolve(
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        elitism=ELITISM
    )
    
    print(f"\nEvolution Complete!")
    print(f"Best Genotype: {best_individual['genotype']}")
    print(f"Stats: F1 {best_individual['stats']['f1']:.4f}, Latency {best_individual['stats']['latency']:.4f}s, Params {best_individual['stats']['params']}, Depth {best_individual['stats']['depth']}")
    
    return best_individual

best_model_data = run_thesis_experiment()