import torch
from dataclasses import dataclass
from typing import Optional, Tuple

import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self
import time

MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]


class Timer:
    def __init__(self, device: torch.device) -> None:
        self.start_time = None
        self.stop_time = None
        self.device = device

    def start(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()
            self.start_time = torch.cuda.Event(enable_timing=True)
            self.stop_time = torch.cuda.Event(enable_timing=True)
            self.start_time.record()
            return
        elif self.device == "cpu":
            # self.start_time = timeit.default_timer()
            self.start_time = time.time()
            return
        else:
            print("unknown device type!")

    def stop(self) -> None:
        if self.device == "cuda":
            self.stop_time.record()
            torch.cuda.synchronize()
            return
        elif self.device == "cpu":
            # self.stop_time = timeit.default_timer()
            self.stop_time = time.time()
            return
        else:
            print("unknown device type!")

    def get_latency_ms(self) -> float:
        if self.device == "cuda":
            return self.start_time.elapsed_time(self.stop_time)
        elif self.device == "cpu":
            return (self.stop_time - self.start_time) * 10**3
        else:
            print("unknown device type!")


@dataclass
class LLaMAConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    tensor_parallelism: int = 1
    device: torch.device = "cuda"
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


# RoPE cache is: (B, n_head/2 ,2), Attn_Mask=(1,1,B,B)
class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd // config.tensor_parallelism,
            dtype=config.dtype,
            bias=False,
        )
        # output projection
        self.c_proj = nn.Linear(
            config.n_embd // config.tensor_parallelism,
            config.n_embd,
            dtype=config.dtype,
            bias=False,
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.tensor_parallelism = config.tensor_parallelism

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd // self.tensor_parallelism, dim=2)

        head_size = C // (self.n_head)
        k = k.view(B, T, self.n_head // self.tensor_parallelism, head_size)
        q = q.view(B, T, self.n_head // self.tensor_parallelism, head_size)
        v = v.view(B, T, self.n_head // self.tensor_parallelism, head_size)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        # torch.cuda.synchronize()
        # start_qkv = torch.cuda.Event(enable_timing=True)
        # stop_qkv = torch.cuda.Event(enable_timing=True)
        # start_qkv.record()
        # att = F.softmax(att, dim=-1)
        # stop_qkv.record()
        # torch.cuda.synchronize()
        # qkv_time_in_msec=start_qkv.elapsed_time(stop_qkv)
        # print("Time in msec for softmax output of block is: "+str(qkv_time_in_msec))
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        # print("q: "+str(q.size()))
        # print("k: "+str(k.size()))
        # print("v: "+str(v.size()))
        # print("mask: "+str(mask.size()))

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C // self.tensor_parallelism)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y, kv_cache


class LLamaMLP(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256) // config.tensor_parallelism

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, dtype=config.dtype, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, dtype=config.dtype, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, dtype=config.dtype, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(
        self, size: int, dtype: torch.dtype, dim: int = -1, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        result = self.scale * x_normed
        return result.to(self.dtype)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def build_mask_cache(seq_len, device) -> MaskCache:
    ones = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (
        base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem)
    )

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: RoPECache) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class Block(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd, config.dtype)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd, config.dtype)
        self.mlp = LLamaMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        h, new_kv_cache = self.attn(
            self.rms_1(x), rope, mask, max_seq_length, input_pos, kv_cache
        )
        x = x + h
        x = x + self.mlp(self.rms_2(x))
        return x, new_kv_cache
