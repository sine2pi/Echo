
import base64, os, evaluate, random, gzip, math, torch, numpy as np, json, warnings
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from datasets import load_dataset, IterableDatasetDict, Audio
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor,WhisperFeatureExtractor,
WhisperTokenizerFast)
import torch.nn.functional as F
import transformers
from itertools import chain
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Tuple
from torch import Tensor, nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List, Any
from torch.nn.functional import scaled_dot_product_attention

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
transformers.utils.logging.set_verbosity_error()
device = torch.device(device="cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)


@dataclass
class Dimensions:
    mels: int
    audio_ctx: int
    audio_state: int
    audio_head: int
    audio_layerA: int
    audio_layerB: int
    vocab: int
    text_ctx: int
    text_state: int
    text_head: int
    text_layerA: int
    text_layerB: int
    dropout: float
    activation: str
    checkpoint: bool


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(input=x.float()).type(dtype=x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            input=x,
            weight=self.weight.to(dtype=x.dtype),
            bias=None if self.bias is None else self.bias.to(dtype=x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            input=x,
            weight=weight.to(dtype=x.dtype),
            bias=None if bias is None else bias.to(dtype=x.dtype),
        )


# @torch.jit.script
# def _apply_qrotation(x: torch.Tensor, theta: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#     u = u / torch.norm(u)
#     v = v / torch.norm(v)

#     half_theta = theta / 2
#     cos_ht = torch.cos(half_theta)
#     sin_ht = torch.sin(half_theta)

#     q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
    
#     x_shape = x.shape
#     x = x.view(-1, 3)

#     uv_cross = torch.cross(u.unsqueeze(0), x)
#     uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
#     x_rot = x + 2 * (q[0] * uv_cross + uuv_cross)

#     x_rot = x_rot.view(*x_shape)
#     return x_rot

# @torch.jit.script
# def _create_rotation_matrix(dims: int, i: int, j: int, theta: torch.Tensor, device: torch.device) -> torch.Tensor:
#     G = torch.eye(dims, device=device)
#     c, s = torch.cos(theta), torch.sin(theta)
#     G[i, i], G[j, j] = c, c
#     G[i, j], G[j, i] = -s, s
    
#     if dims == 3:
#         u = torch.eye(dims, device=device)[i]
#         v = torch.eye(dims, device=device)[j]
#         x = torch.eye(dims, device=device)
        
#         Q = _apply_qrotation(x, theta=theta, u=u, v=v)
#         G = (G + Q) / 2
#     return G

# @torch.jit.script
# def _apply_rope_transform(
#     x: torch.Tensor, 
#     sin: torch.Tensor, 
#     cos: torch.Tensor
# ) -> torch.Tensor:
#     x1, x2 = x[..., ::2], x[..., 1::2]
#     return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# class rotary2(nn.Module):
#     def __init__(self, ctx, dims, heads, base=10000, theta_learnable=False,
#         rot_learnable=False, matrix_learnable=False, freq_learnable=False,
#     ):
#         super().__init__()
#         self.ctx = ctx
#         self.dims = dims
#         self.heads = heads
#         self.base = base

#         self.head_dim = self.dims // self.heads
#         self.rot = self.head_dim // 2

#         self.thetas = nn.Parameter(torch.zeros(self.rot))
#         self.r_pairs = nn.Parameter(torch.rand(self.rot, 2) * self.head_dim)
#         self.theta_scale = nn.Parameter(torch.ones(1), requires_grad=theta_learnable)
#         self.rot_scale = nn.Parameter(torch.ones(1), requires_grad=rot_learnable)
#         self.r_matrix = nn.Parameter(
#             torch.eye(self.head_dim), requires_grad=matrix_learnable
#         )

#         freq_data = 1.0 / (
#             self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
#         )

#         self.inv_freq = nn.Parameter(freq_data, requires_grad=freq_learnable)

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.orthogonal_(self.r_matrix)
#         nn.init.zeros_(self.thetas)

#     def q_rotation(self, x, theta, u, v):
#         return _apply_qrotation(x, theta, u, v)

#     def rotation_matrix(self, dims, i, j, theta):
#         return _create_rotation_matrix(dims, i, j, theta, theta.device)

#     @torch.jit.script_method # type: ignore
#     def apply_rotations(self, x: torch.Tensor) -> torch.Tensor:
#         adjusted_rot = int(self.rot_scale.item() * self.rot)
#         for k in range(adjusted_rot):
#             i, j = int(self.r_pairs[k, 0].item()), int(self.r_pairs[k, 1].item())
#             theta = self.thetas[k] * self.theta_scale
#             G = _create_rotation_matrix(self.head_dim, i, j, theta, x.device)
#             x = x @ G
#         return x

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len = x.shape[0], x.shape[1]
        
#         if x.dim() == 3:
#             if x.shape[2] != self.dims:
#                 raise ValueError(f"Expected dim {self.dims}, got {x.shape[2]}")
#             x = x.view(batch_size, seq_len, self.heads, self.head_dim)
#         elif x.dim() == 4:
#             if x.shape[2] != self.heads or x.shape[3] != self.head_dim:
#                 raise ValueError(f"Expected {self.heads} heads and {self.head_dim} head_dim")
#         else:
#             raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

#         x_flat = x.reshape(-1, self.head_dim)
#         x_rotated = self.apply_rotations(x_flat)
#         x_rotated = x_rotated @ self.r_matrix
        
#         x = x_rotated.view(batch_size, seq_len, self.heads, self.head_dim)
        
#         position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
#         div_term = self.inv_freq.unsqueeze(0)
#         sinusoid_inp = position * div_term
        
#         sin = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(2)
#         cos = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(2)
        
#         x = _apply_rope_transform(x, sin, cos)
        
#         x = x.view(batch_size, seq_len, self.dims)
#         x = x * math.sqrt(self.dims)
#         return x


class rotary(nn.Module):
    def __init__(self, ctx, dims, heads, base=10000, theta_learnable=False,
        rot_learnable=False, matrix_learnable=False, freq_learnable=False,
    ):
        super().__init__()
        self.ctx = ctx
        self.dims = dims
        self.heads = heads
        self.base = base

        self.head_dim = self.dims // self.heads
        self.rot = self.head_dim // 2

        self.thetas = nn.Parameter(torch.zeros(self.rot))
        self.r_pairs = nn.Parameter(torch.rand(self.rot, 2) * self.head_dim)
        self.theta_scale = nn.Parameter(torch.ones(1), requires_grad=theta_learnable)
        self.rot_scale = nn.Parameter(torch.ones(1), requires_grad=rot_learnable)
        self.r_matrix = nn.Parameter(
            torch.eye(self.head_dim), requires_grad=matrix_learnable
        )

        freq_data = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )

        self.inv_freq = nn.Parameter(freq_data, requires_grad=freq_learnable)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.r_matrix)
        nn.init.zeros_(self.thetas)

    def q_rotation(self, x, theta, u, v):
        u = u / torch.norm(u)
        v = v / torch.norm(v)

        half_theta = theta / 2
        cos_ht = torch.cos(half_theta)
        sin_ht = torch.sin(half_theta)

        q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
        q_conj = torch.cat([cos_ht.unsqueeze(0), -sin_ht * u])

        x_shape = x.shape
        x = x.view(-1, 3)

        uv_cross = torch.cross(u.unsqueeze(0), x)
        uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        x_rot = x + 2 * (q[0] * uv_cross + uuv_cross)

        x_rot = x_rot.view(*x_shape)
        return x_rot

    def rotation_matrix(self, dims, i, j, theta):
        G = torch.eye(dims, device=theta.device)
        c, s = torch.cos(theta), torch.sin(theta)
        G[i, i], G[j, j] = c, c
        G[i, j], G[j, i] = -s, s

        if dims == 3:
            u = torch.eye(dims, device=theta.device)[i]
            v = torch.eye(dims, device=theta.device)[j]
            Q = self.q_rotation(
                torch.eye(dims, device=theta.device), theta=theta, u=u, v=v
            )
            G = (G + Q) / 2
        return G

    def apply_rotations(self, x):
        adjusted_rot = int(torch.round(self.rot_scale * self.rot))
        for k in range(adjusted_rot):
            i, j = self.r_pairs[k].long()
            theta = self.thetas[k] * self.theta_scale
            G = self.rotation_matrix(self.head_dim, i.item(), j.item(), theta)
            x = x @ G
        return x

    def forward(self, x):
        batch_size, seq_len, *rest = x.size()

        if len(rest) == 1:
            dims = rest[0]
            if dims != self.heads * self.head_dim:
                raise ValueError(
                    f"Needed {self.heads * self.head_dim}, but got too many {dims}"
                )
        elif len(rest) == 2:
            heads, head_dim = rest
            if heads != self.heads or head_dim != self.head_dim:
                raise ValueError(
                    f"This many heads {self.heads} and head_dims {self.head_dim} we need, got this many heads {heads} and head_dims {head_dim} we did."
                )
        else:
            raise ValueError(f"Expected the thingy to be 3D or 4D, but got {x.dim()}D")

        x = x.view(batch_size, seq_len, self.heads, self.head_dim)
        x = x.reshape(-1, self.head_dim)

        x = self.apply_rotations(x)
        x = x @ self.r_matrix

        x = x.view(batch_size, seq_len, self.heads, self.head_dim)

        position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = self.inv_freq.unsqueeze(0)
        sinusoid_inp = position * div_term

        sin = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(2)
        cos = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(2)

        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x = x.view(batch_size, seq_len, self.dims)
        x = x * math.sqrt(self.dims)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dims, ctx):
        super(PositionalEncoding, self).__init__()
        self.dims = dims
        self.ctx = ctx
        self.pe = self.get_positional_encoding(max_seq_len=ctx)

    def get_positional_encoding(self, max_seq_len):
        pe = torch.zeros(max_seq_len, self.dims)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dims, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.dims)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe.to(device)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        x = x * math.sqrt(self.dims)
        x = x + pe

        return x


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# @torch.jit.script
# def _manual_attention(q: Tensor, k: Tensor, v: Tensor,  scale: float,
#     mask: Optional[Tensor] = None,  ctx: int = 0, k_ctx: int = 0
# ) -> Tuple[Tensor, Tensor]:
#     qk = (q * scale) @ ((k * scale).transpose(-1, -2))
#     if mask is not None:
#         qk = qk + mask[:ctx, :k_ctx]
#     qk_float = qk.float()
#     w = F.softmax(qk_float, dim=-1).to(q.dtype)
#     out = (w @ v)
#     return out, qk_float

class MultiheadA(nn.Module):
    use_sdpa: bool = True

    def __init__(self, dims: int, heads: int):
        super().__init__()

        assert dims % heads == 0, f"dims ({dims}) must be divisible by heads ({heads})"
        assert isinstance(dims, int) and isinstance(
            heads, int
        ), "dims and heads must be integers"

        self.heads = heads
        self.dims = dims
        self.head_dim = dims // heads
        self.scale = (self.head_dim) ** -0.25

        self.query = nn.Linear(in_features=dims, out_features=dims)
        self.key = nn.Linear(in_features=dims, out_features=dims, bias=False)
        self.value = nn.Linear(in_features=dims, out_features=dims)
        self.out = nn.Linear(in_features=dims, out_features=dims)

        self._init_weights()

        self.register_buffer("_has_cuda", torch.tensor(torch.cuda.is_available()))

    def _init_weights(self):

        std = 0.02
        nn.init.normal_(self.query.weight, std=std)
        nn.init.normal_(self.key.weight, std=std)
        nn.init.normal_(self.value.weight, std=std)
        nn.init.normal_(self.out.weight, std=std)
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Dict] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if __debug__:
            assert x.dim() == 3, f"Expected 3D input tensor, got {x.dim()}D"
            if xa is not None:
                assert (
                    xa.dim() == 3
                ), f"Expected 3D cross-attention tensor, got {xa.dim()}D"

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            kv_input = xa if xa is not None else x

            k = self.key(kv_input)
            v = self.value(kv_input)

            if kv_cache is not None and xa is not None:
                kv_cache[self.key] = k
                kv_cache[self.value] = v
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self._attention(q=q, k=k, v=v, mask=mask)

        return self.out(wv), qk

    def _attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):

        batch, ctx, _ = q.shape
        k_ctx = k.size(1)

        head_dim = self.dims // self.heads
        reshape_dim = (batch, -1, self.heads, head_dim)

        q = q.view(*reshape_dim).transpose(1, 2)
        k = k.view(batch, k_ctx, self.heads, head_dim).transpose(1, 2)
        v = v.view(batch, k_ctx, self.heads, head_dim).transpose(1, 2)

        if MultiheadA.use_sdpa:
            with torch.autocast(device_type="cuda", enabled=True):
                out = F.scaled_dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    attn_mask=None,
                    is_causal=mask is not None and ctx > 1,
                )
            out = out.transpose(1, 2).flatten(2)
            return out, None
        else:
            qk = (q * self.scale) @ ((k * self.scale).transpose(-1, -2))
            if mask is not None:
                qk = qk + mask[:ctx, :k_ctx]
            qk_float = qk.float()
            w = F.softmax(qk_float, dim=-1).to(q.dtype)
            out = (w @ v).transpose(1, 2).flatten(2)
            print("mulita",out.shape)
            return out, qk_float.detach()


class MultiHeadB(nn.Module):

    use_sdpa: bool = True

    def __init__(self, dims: int, heads: int):
        super().__init__()

        if dims % heads != 0:
            raise ValueError(f"dims ({dims}) must be divisible by heads ({heads})")
        if not isinstance(dims, int) or not isinstance(heads, int):
            raise TypeError("dims and heads must be integers")

        self.heads = heads
        self.dims = dims
        self.head_dim = dims // heads

        self.query = Linear(in_features=dims, out_features=dims)
        self.key = Linear(in_features=dims, out_features=dims, bias=False)
        self.value = Linear(in_features=dims, out_features=dims)
        self.out = Linear(in_features=dims, out_features=dims)

    def init_weights(self):
        nn.init.normal_(self.query.weight, std=0.02)
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.value.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        if xa is not None and xa.dim() != 3:
            raise ValueError(f"Expected 3D cross-attention tensor, got {xa.dim()}D")

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q=q, k=k, v=v, mask=mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        batch, ctx, dims = q.shape
        scale = (dims // self.heads) ** -0.25

        q = q.view(batch, ctx, self.heads, -1).permute(0, 2, 1, 3)
        k = k.view(batch, k.size(1), self.heads, -1).permute(0, 2, 1, 3)
        v = v.view(batch, v.size(1), self.heads, -1).permute(0, 2, 1, 3)

        if self.use_sdpa and torch.cuda.is_available():
            with torch.autocast("cuda"):
                a = scaled_dot_product_attention(
                    query=q, key=k, value=v, is_causal=mask is not None and ctx > 1
                )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:

            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:ctx, :ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk

class MultiheadC(nn.Module):
    use_sdpa: bool = True

    def __init__(self, dims: int, heads: int, max_dist: int):
        super().__init__()
        if dims % heads != 0:
            raise ValueError(f"dims ({dims}) must be divisible by heads ({heads})")
        if dims % 2 != 0:
            raise ValueError(f"dims ({dims}) must be even for rotary embeddings")
        self.heads = heads
        self.head_dim = dims // heads
        self.dims = dims
        self.max_dist = max_dist

        scale = 1 / math.sqrt(self.head_dim)
        self.query = nn.Linear(in_features=dims, out_features=dims)
        self.key = nn.Linear(in_features=dims, out_features=dims, bias=False)
        self.value = nn.Linear(in_features=dims, out_features=dims)
        self.out = nn.Linear(in_features=dims, out_features=dims)

        nn.init.normal_(tensor=self.query.weight, std=scale)
        nn.init.normal_(tensor=self.key.weight, std=scale)
        nn.init.normal_(tensor=self.value.weight, std=scale)
        nn.init.zeros_(tensor=self.out.bias)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Dict] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q=q, k=k, v=v, mask=mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:

        batch, ctx, dims = q.shape
        scale = (dims // self.heads) ** -0.25
        q = q.view(batch, ctx, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch, ctx, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, ctx, self.heads, self.head_dim).permute(0, 2, 1, 3)

        with torch.autocast(device_type="cuda"):
            a = scaled_dot_product_attention(
                query=q, key=k, value=v, is_causal=mask is not None and ctx > 1
            )
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        qk = None

        return out, qk

class miniAttention(nn.Module):
    def __init__(self, dims, max_dist, heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.0,
        proj_drop=0.0):
        super().__init__()
        if dims % heads != 0:
            raise ValueError(f"dims ({dims}) must be divisible by heads ({heads})")
        if dims % 2 != 0:
            raise ValueError(f"dims ({dims}) must be even for rotary embeddings")
        self.heads = heads
        self.head_dim = dims // heads
        self.dims = dims
        self.max_dist = max_dist
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dims, dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dims, dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        B, N, C = x.shape
        qkv = (self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Refiner:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.R = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default_value = 0.0

    def get_value(self, state, action):
        return self.R.get((state, action), self.default_value)

    def set_value(self, state, action, value):
        self.R[(state, action)] = value

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actions)
        else:
            action_values = [self.get_value(state, a) for a in range(self.actions)]
            return np.argmax(action_values)

    def update(self, state, action, reward, next_state):
        next_values = [self.get_value(next_state, a) for a in range(self.actions)]
        best_next_value = max(next_values)

        old_value = self.get_value(state, action)
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - old_value
        new_value = old_value + self.alpha * td_error
        self.set_value(state, action, new_value)

class Predictor(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.linear = nn.Linear(in_features=dims, out_features=1)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, global_out):
        if global_out.dim() > 2:
            global_out = global_out.mean(dim=1)
        scale = torch.sigmoid(self.linear(global_out))
        
        return scale

class AdaptiveSpan(nn.Module):
    def __init__(self, dims, heads, max_dist, sharpen=True, temp_scale=0.01):
        super().__init__()
        self.heads = heads
        self.max_dist = max_dist
        self.dims = dims
        self.temp_scale = temp_scale
        self.sharpen = sharpen
        self.span_scale = nn.Parameter(torch.tensor(1.0))

        self.head_dim = dims // heads
        self.register_buffer("scale", torch.tensor(self.head_dim**-0.25))

    def forward(self, query, key, value, max_dist=None, max_span=None, span_scale=None):
        if max_dist is None:
            max_dist = self.max_dist
        if max_span is None:
            max_span = query.shape[1]  # Default to sequence length
        if span_scale is None:
            span_scale = self.span_scale
            
        span_mean = span_scale.mean().item()
        span_len = min(int(max_span * span_mean), query.shape[1], key.shape[1], value.shape[1])
        eff_span = min(span_len, max_dist)
        
        if eff_span == 0:
            batch_size = query.shape[0]
            return (torch.zeros(batch_size, eff_span, self.dims, device=query.device), None)
            
        q_span = query[:, :eff_span, :]
        k_span = key[:, :eff_span, :]
        v_span = value[:, :eff_span, :]

        batch_size = q_span.shape[0]

        reshape_dims = (batch_size, -1, self.heads, self.head_dim)
        q = q_span.view(*reshape_dims).permute(0, 2, 1, 3)
        k = k_span.view(*reshape_dims).permute(0, 2, 1, 3)
        v = v_span.view(*reshape_dims).permute(0, 2, 1, 3)

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            temperature = (
                1.0 + self.temp_scale * (1.0 - span_mean)
                if self.sharpen
                else 0.5 + self.temp_scale * span_mean
            )
            scores = torch.matmul(q, k.transpose(-2, -1))
            weights = torch.softmax((scores / temperature) * self.scale, dim=-1)
            out = torch.matmul(weights, v)
            out = out.permute(0, 2, 1, 3).reshape(batch_size, eff_span, self.dims)

        return out, weights

class FocusA(nn.Module):
    def __init__(self, dims, heads, max_dist, sharpen=True, win_size=256, max_span=512):
        super().__init__()
        self.heads = heads
        self.max_dist = max_dist
        self.dims = dims
        self.max_span = max_span
        self.sliding_window = win_size
        self.temp_scale = 0.01
        self.sharpen = sharpen
        self.head_dim = dims // heads
        self.batch_size = None  # Will be set during forward pass

        self.refiner = Refiner(
            states=10000, actions=10, alpha=0.1, gamma=0.9, epsilon=0.1
        )
        self.span_pred = Predictor(dims=dims)
        self.attn_local = AdaptiveSpan(
            dims=dims, heads=heads, max_dist=max_dist, sharpen=True, temp_scale=0.01
        )
        self.attn_global = MultiheadC(dims=dims, heads=heads, max_dist=max_dist)

        self.projection = nn.Linear(in_features=2 * dims, out_features=dims)

        self.ln_a = nn.LayerNorm(normalized_shape=dims)
        self.ln_b = nn.LayerNorm(normalized_shape=dims)

        mask = torch.empty(max_span, max_span).fill_(float("-inf")).triu_(diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

        self.register_buffer("window_mask", None, persistent=False)
        self.register_buffer("threshold", torch.tensor(1e-4), persistent=False)
        self.register_buffer("s_factor", torch.tensor(0.1), persistent=False)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        if mask is None:
            mask = self.mask
            
        local = self.ln_a(x)
        globe = self.ln_b(x)

        globe_out, _ = self.attn_global(globe, globe, globe)
        base_scale = self.span_pred(globe_out)
        state = self.extract(local)

        action = self.refiner.choose_action(state=state)
        refine = self.action_scale(action=action)

        span_scale = torch.clamp(base_scale * refine, min=0.0, max=1.0)
        span_mean = span_scale.mean().item()

        with torch.no_grad():
            current_win_size = max(1, int(self.sliding_window * span_mean))
            current_span_len = max(1, int(self.max_span * span_mean))

            effective_max = min(self.max_dist, local.size(1))
            local_max = min(self.max_dist, current_span_len, current_win_size)
            globe_max = effective_max

        self.attn_local.max_dist = local_max
        self.attn_global.max_dist = globe_max

        local_out = self.slide_win(
            x=local,
            win_size=current_win_size,
            span_len=current_span_len,
            span_scale=span_scale,
            mask=mask,
        )
        with torch.no_grad():
            quality = self.quality(output=local_out)
            next_state = self.extract(local_out)
            self.refiner.update(
                state=state, action=action, reward=quality, next_state=next_state)
        combined = torch.cat([local_out, globe_out], dim=-1)
        x = self.projection(combined)
        return x

    def quality(self, output):
        with torch.no_grad():
            safe_output = output.clamp(min=1e-10)
            entropy = -(safe_output * torch.log(safe_output)).sum(-1).mean()
            coverage = (output > 0.01).float().mean()
            return float(coverage - 0.1 * entropy)

    def extract(self, x):
        with torch.no_grad():
            mean_state = x.mean(dim=(0, 1))
            var_state = x.var(dim=(0, 1), unbiased=False)
            state = torch.cat([mean_state, var_state])
            state_id = self.discretize(state.cpu().numpy())
        return state_id

    def discretize(self, state):
        bins = np.linspace(-1, 1, num=10)
        state_discrete = np.digitize(state, bins)
        state_hash = hash(tuple(state_discrete))
        state_id = state_hash % (self.refiner.states - 1)
        return state_id

    def action_scale(self, action):
        span_value = action / (self.refiner.actions - 1)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        span_scale = torch.tensor([span_value], device=device, dtype=dtype)
        return span_scale

    def _focus(self, query, key, value, span_scale, mask):
        max_iterations = 10
        iteration = 0
        prev_attn = torch.zeros_like(input=query)
        attn_out = torch.zeros_like(input=query)
        attn_weights = None

        threshold = self.threshold.item()
        s_factor = self.s_factor.item()

        while iteration < max_iterations:
            span_len = int(self.max_span * span_scale.mean().item())
            span_len = min(span_len, query.size(1), key.size(1), value.size(1))
            eff_span = min(span_len, self.max_dist)

            if eff_span == 0:
                break

            q_span = query[:, :eff_span, :]
            k_span = key[:, :eff_span, :]
            v_span = value[:, :eff_span, :]

            batch_size, seq_len, dims = q_span.size()
            d_k = dims // self.heads
            scale_factor = 1 / math.sqrt(d_k)

            q = q_span.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
            k = k_span.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
            v = v_span.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)

            if self.sharpen:
                temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
            else:
                temperature = 0.5 + self.temp_scale * span_scale.mean().item()
            attn_scores = (
                torch.matmul(q, k.transpose(-2, -1)) * scale_factor / temperature
            )
            if mask.size(-2) != attn_scores.size(-2) or mask.size(
                -1
            ) != attn_scores.size(-1):

                mask_q_len = min(mask.size(-2), attn_scores.size(-2))
                mask_k_len = min(mask.size(-1), attn_scores.size(-1))
                resized_mask = torch.ones(
                    (
                        batch_size,
                        self.heads,
                        attn_scores.size(-2),
                        attn_scores.size(-1),
                    ),
                    device=mask.device,
                    dtype=mask.dtype,
                )
                resized_mask[:, :, :mask_q_len, :mask_k_len] = mask[
                    :, :, :mask_q_len, :mask_k_len
                ]
                mask = resized_mask

            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)
            attn_out = (
                attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            )

            diff = torch.abs(attn_out - prev_attn).mean()
            dynamic_threshold = threshold + s_factor * diff

            if diff < dynamic_threshold:
                break

            prev_attn = attn_out
            query = query + attn_out
            iteration += 1
        return attn_out, attn_weights

    def slide_win(self, x, win_size, span_len, span_scale, mask):
        batch_size, seq_len, dims = x.size()
        self.batch_size = batch_size
        num_windows = (seq_len + win_size - 1) // win_size
        output = torch.zeros_like(x)
        device = x.device
        default_mask = None

        for i in range(num_windows):
            start_idx = i * win_size
            end_idx = min((i + 1) * win_size, seq_len)
            window_size = end_idx - start_idx

            key_start = max(0, start_idx - span_len + win_size)
            key_end = min(start_idx + span_len, seq_len)
            span_size = key_end - key_start

            query = x[:, start_idx:end_idx, :]
            key = x[:, key_start:key_end, :]
            value = key

            if mask is not None:
                if mask.dim() == 4:
                    window_mask = mask[:, :, start_idx:end_idx, key_start:key_end]
                    if window_mask.size(1) == 1:
                        window_mask = window_mask.expand(-1, self.heads, -1, -1)
                else:
                    if (
                        default_mask is None
                        or default_mask.size(-2) != window_size
                        or default_mask.size(-1) != span_size
                    ):
                        default_mask = torch.ones(
                            (batch_size, self.heads, window_size, span_size),
                            device=device,
                            dtype=torch.bool,
                        )
                    window_mask = default_mask
            else:
                if (
                    default_mask is None
                    or default_mask.size(-2) != window_size
                    or default_mask.size(-1) != span_size
                ):
                    default_mask = torch.ones(
                        (batch_size, self.heads, window_size, span_size),
                        device=device,
                        dtype=torch.bool,
                    )
                window_mask = default_mask

            attn_out, _ = self._focus(
                query=query,
                key=key,
                value=value,
                span_scale=span_scale,
                mask=window_mask,
            )

            output[:, start_idx:end_idx, :] = attn_out

        return output


class Residual(nn.Module):
    def __init__(
        self, param: Dimensions, dims: int, heads: int, dropout: float, activation: str
    ):
        super().__init__()
        self.param = param
        self.dims = dims

        activation_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
        }
        act_fn = activation_map.get(activation, nn.ReLU())

        self.attn = MultiheadA(dims=dims, heads=heads)
        self.cross = MultiHeadB(dims=dims, heads=heads)

        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=dims, out_features=dims * 4, bias=True),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(in_features=dims * 4, out_features=dims, bias=True),
        )

        self.ln_a = nn.LayerNorm(normalized_shape=dims)
        self.ln_b = nn.LayerNorm(normalized_shape=dims)
        self.ln_c = nn.LayerNorm(normalized_shape=dims)

        self._init_weights()

    def _init_weights(self):

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        
        y = x
        z = self.ln_a(x)
        x = x + self.attn(z, mask=mask, kv_cache=kv_cache)[0]
        if xa is not None:
            z = self.ln_b(x)
            x = x + self.cross(z, xa, mask=mask, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.ln_c(x))

        return x + y
    
class AudioEncoder(nn.Module):
    def __init__(self, param: Dimensions, mels: int, ctx: int, dims: int, heads: int, 
                checkpoint: bool, dropout: float, activation: str, layerA: int, layerB: int):
        super().__init__()
        
        self.checkpoint = checkpoint

        act_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
        }
        act = act_map.get(activation, nn.ReLU())

        self.rotation = rotary(ctx=ctx, dims=dims, heads=heads, base=10000)
        self.position = sinusoids(length=ctx, channels=dims)
        self.register_buffer("positions", self.position, persistent=False)

        self.convx = nn.Sequential(
            nn.Conv1d(mels, dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(dims),
            act,
            nn.Dropout(p=dropout),
            nn.Conv1d(dims, dims, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(dims),
            act,
            nn.Dropout(p=dropout),
        )

        for m in self.convx:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

        self.blockA = nn.ModuleList([
            Residual(param, dims, heads, dropout, activation) 
            for _ in range(layerA)]) if layerA > 0 else None

        self.blockB = nn.ModuleList([
            FocusA(dims=dims, heads=heads, max_dist=ctx) 
            for _ in range(layerB)]) if layerB > 0 else None

        self.ln_post = nn.LayerNorm(dims)

    def forward(self, x) -> Tensor:
        x = checkpoint(self._forward, x, use_reentrant=True) if self.checkpoint else self._forward(x)
        for block in chain(self.blockB or [], self.blockA or []):
            x = checkpoint(block, x, use_reentrant=True) if self.checkpoint else block(x)  
        return self.ln_post(x)

    def _forward(self, x) -> Tensor:
        x = F.gelu(self.convx(x))
        x = x.permute(0, 2, 1)  
        x = (x + self.positions).to(x.dtype)  # type: ignore
        x = self.rotation(x)
        return x

class TextDecoder(nn.Module):
    def __init__(self, param: Dimensions, vocab: int, ctx: int, dims: int, heads: int, 
                checkpoint: bool, dropout: float, activation: str, layerA: int, layerB: int):
        super().__init__()
        
        self.checkpoint = checkpoint
        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims))
        nn.init.normal_(tensor=self.positional_embedding, mean=0.0, std=0.02)
        
        self.positional_encoding = PositionalEncoding(ctx=ctx, dims=dims)
        self.ln = LayerNorm(normalized_shape=dims)
        
        self.blockA = nn.ModuleList(modules=[Residual(param=param, dims=dims, heads=heads, 
                dropout=dropout, activation=activation) 
                for _ in range(layerA)]) if layerA > 0 else None

        self.blockB = nn.ModuleList(modules=[FocusA(dims=dims, heads=heads, max_dist=ctx) 
                for _ in range(layerB)]) if layerB > 0 else None

        mask = torch.empty(ctx, ctx).fill_(value=-np.inf).triu_(diagonal=1)
        self.register_buffer(name="mask", tensor=mask, persistent=False)
        self.mask = mask
        
    def forward(self, x, xa, kv_cache = None):
        x = checkpoint(function=self._forward, x=x, xa=xa, kv_cache=kv_cache) if self.checkpoint else self._forward(x=x, xa=xa, kv_cache=kv_cache)
        for block in chain(self.blockA or [], self.blockB or []):        
            x = checkpoint(function=block, x=x, xa=xa, mask=self.mask, kv_cache=kv_cache) if self.checkpoint else block(x=x, xa=xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)
        x = (x @ torch.transpose(self.token_embedding.weight.to(dtype=x.dtype), dim0=0, dim1=1)).float()
        return x
        
    def _forward(self, x, xa, kv_cache):       
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]])
        x = self.positional_encoding(x)
        x = x.to(dtype=xa.dtype)
        return x


@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_state: int
    text_head: int
    text_layerA: int
    text_layerB: int
    audio_ctx: int
    audio_state: int
    audio_head: int
    audio_layerA: int
    audio_layerB: int
    mels: int
    checkpoint: bool = False
    dropout: float = 0.1
    activation: str = "gelu"

class Echo(nn.Module):

    PAD_TOKEN_ID = 50257
    START_TOKEN_ID = 50258

    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param

        self._build_model()

        self.to(self.device)

    def _build_model(self):

        self.encoder = AudioEncoder(
            param=self.param,
            mels=self.param.mels,
            ctx=self.param.audio_ctx,
            dims=self.param.audio_state,
            heads=self.param.audio_head,
            layerA=self.param.audio_layerA,
            layerB=self.param.audio_layerB,
            checkpoint=self.param.checkpoint,
            dropout=self.param.dropout,
            activation=self.param.activation,
        )

        self.decoder = TextDecoder(
            param=self.param,
            vocab=self.param.vocab,
            ctx=self.param.text_ctx,
            dims=self.param.text_state,
            heads=self.param.text_head,
            layerA=self.param.text_layerA,
            layerB=self.param.text_layerB,
            checkpoint=self.param.checkpoint,
            dropout=self.param.dropout,
            activation=self.param.activation,
        )

    @property
    def device(self) -> torch.device:

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def shift_tokens_right(
        input_ids: torch.Tensor,
        pad_token_id: int = PAD_TOKEN_ID,
        decoder_start_token_id: int = START_TOKEN_ID,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        shifted_input_ids = torch.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        dec_input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:

        if labels is not None and dec_input_ids is None:
            dec_input_ids = self.shift_tokens_right(
                input_ids=labels,
                pad_token_id=self.PAD_TOKEN_ID,
                decoder_start_token_id=self.START_TOKEN_ID,
            )

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            encoded_features = self.encoder(input_features)

            logits = self.decoder(dec_input_ids, encoded_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(logits.device).long()

            flattened_logits = logits.view(-1, self.param.vocab)
            flattened_labels = labels.view(-1)

            loss = loss_fct(flattened_logits, flattened_labels)

        return {"loss": loss, "logits": logits}

    def _init_weights(self, module):
        std = 0.02

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, AudioEncoder):
            module.convx.apply(self._init_weights)

        elif isinstance(module, TextDecoder):
            nn.init.normal_(module.positional_embedding, mean=0.0, std=std)
            nn.init.normal_(module.token_embedding.weight, mean=0.0, std=std)

        elif isinstance(module, Residual):
            for layer in module.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=std)
                    nn.init.zeros_(layer.bias)

            for ln_name in ["ln_a", "ln_b", "ln_c"]:
                if hasattr(module, ln_name):
                    ln = getattr(module, ln_name)
                    nn.init.normal_(ln.weight, mean=1.0, std=std)
                    nn.init.zeros_(ln.bias)

            if hasattr(module, "attn") and hasattr(module.attn, "init_weights"):
                module.attn.init_weights()
            if hasattr(module, "cross") and hasattr(module.cross, "init_weights"):
                module.cross.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    @torch.no_grad()
    def generate(
        self,
        audio_features: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        encoded_features = self.encoder(audio_features).to(self.device)

        batch_size = audio_features.size(0)
        generated = torch.full(
            (batch_size, 1), self.START_TOKEN_ID, dtype=torch.long, device=self.device)

        kv_cache = {}

        for _ in range(max_length - 1):
            logits = self.decoder(generated, encoded_features, kv_cache=kv_cache)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-7)
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tokens], dim=-1)
            if (next_tokens == self.PAD_TOKEN_ID).all():
                break

        return generated


from datetime import datetime

log_dir = os.path.join("./output/Whisper", datetime.now().strftime(format="%m-%d_%H"))
os.makedirs(name=log_dir, exist_ok=True)

param = Dimensions(
    mels=128,
    audio_ctx=1500,
    audio_head=8,
    audio_layerA=8,
    audio_layerB=2,
    audio_state=1024,
    vocab=51865,
    text_ctx=448,
    text_head=8,
    text_layerA=8,
    text_layerB=0,
    text_state=1024,
    checkpoint=False,
    dropout=0.001,
    activation="gelu",
)

model = Echo(param=param).to(device=device)
model.init_weights()

class MaxFactor(torch.optim.Optimizer):

    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(1e-12, 1e-8), d=1.0, 
                 weight_decay=0.01, gamma=0.99, max=False, full_matrix=False, clip=0, 
                 lookahead=False, lookahead_k=5):
        
        eps1, eps2 = eps
        if eps1 is None:
            eps1 = torch.finfo(torch.float32).eps
            
        defaults = dict(
            lr=lr, beta2_decay=beta2_decay, eps=(eps1, eps2), d=d, weight_decay=weight_decay, 
            gamma=gamma, max=max, full_matrix=full_matrix, clip=clip, 
            lookahead=lookahead, lookahead_k=lookahead_k)
        
        super().__init__(params=params, defaults=defaults)
          
    
    def _get_lr(self, param_group, param_state):
        step = param_state["step"]
        min_step = (1e-6 * step + 1e-12)
        rel_step = min(min_step, 1.0 / step.sqrt())
        param_scale = max(param_group["eps"][1], param_state["RMS"])
        return min(param_group["lr"], param_scale * rel_step)

    @staticmethod
    def _rms(tensor):
        if tensor.numel() == 0:
            return torch.tensor(0.0, device=tensor.device)
        return tensor.norm() / (tensor.numel() ** 0.5 + 1e-12)

    def _adaptive_clip(self, grad, norm, clip_threshold):
        return grad * torch.clamp(clip_threshold / (norm + 1e-12), max=1.0)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            params_with_grad = []
            grads = []
            row_vars = []
            col_vars = []
            v = []
            state_steps = []
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    
                    if p.dim() > 1 and not group["full_matrix"]:
                        row_shape = list(p.shape)
                        row_shape[-1] = 1
                        state["row_var"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
                        
                        col_shape = list(p.shape)
                        col_shape[-2] = 1
                        state["col_var"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
                    
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    state["RMS"] = self._rms(p).item()

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for j, param in enumerate(params_with_grad):
                grad = grads[j]
                state = self.state[param]
                                
                if group["max"]:
                    grad = -grad
                    
                step_t = state_steps[j]
                row_var = row_vars[j]
                col_var = col_vars[j]
                vi = v[j]
                
                step_t += 1
                step_float = step_t.item()
                
                one_minus_beta2_t = min(0.999, step_float ** group["beta2_decay"])

                state = self.state[param]
                state["RMS"] = self._rms(param).item()
                adaptive_lr = self._get_lr(param_group=group, param_state=state)
                                
                if group["weight_decay"] != 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"] + eps1)
                    
                norm = grad.norm(2)
                if norm > group["clip"] > 0:
                    grad = self._adaptive_clip(grad=grad, norm=norm, clip_threshold=group["clip"])

                if param.dim() > 1 and not group["full_matrix"]:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_()
                    row_mean.div_(grad.size(-1) + eps1)
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_()
                    col_mean.div_(grad.size(-2) + eps1)
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))
                else:
                    vi.mul_(group["gamma"]).add_(grad.square_(), alpha=1 - group["gamma"])
                    var_estimate = vi
                    
                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                  
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                param.add_(update.sign() * update.abs().max(dim=-1, keepdim=True)[0], alpha=-adaptive_lr / denom)
            
                state["step"] = step_t

        return loss

token=""

extractor = WhisperFeatureExtractor.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small", token=token,
    feature_size=128, sampling_rate=16000, return_tensors="pt", do_normalize=True)

tokenizer = WhisperTokenizerFast.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small", 
    language="en", task="transcribe", token=token)

processor = WhisperProcessor.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small", token=token)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    extractor: Any
    tokenizer: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], Tensor]]]) -> Dict[str, Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
    
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor, extractor=extractor,
    tokenizer=tokenizer, decoder_start_token_id=50258)

dataset = IterableDatasetDict()

dataset["train"] = load_dataset(
    path="mozilla-foundation/common_voice_17_0", split="train",
    name="en", streaming=True, token=token, 
    trust_remote_code=True, save_infos=True)#.shuffle()#.take(10000)

dataset["test"] = load_dataset(
    path="mozilla-foundation/common_voice_17_0",
    name="en", split="test", streaming=True, 
    token=token, trust_remote_code=True, save_infos=True).take(500)

dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))

dataset = dataset.map(function=prepare_dataset, 
    remove_columns=list(next(iter(dataset.values()))
                        .features)).with_format(type="torch")

metric = evaluate.load(path="wer")

def compute_metrics(eval_pred):
    pred_logits = eval_pred.predictions
    label_ids = eval_pred.label_ids

    if isinstance(pred_logits, tuple):
        pred_ids = pred_logits[0]
    else:
        pred_ids = pred_logits
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str) # type: ignore
    pred_flat = pred_ids.flatten()
    labels_flat = label_ids.flatten()
    mask = labels_flat != tokenizer.pad_token_id
  
    if len(pred_str) > 0:
        sample_idx = random.randint(0, len(pred_str) - 1)
        print("-" * 10)
        print(f"Prediction: {pred_str[sample_idx]}")
        print(f"Label: {label_str[sample_idx]}")
        print("-" * 10)

    acc = accuracy_score(y_true=labels_flat[mask], y_pred=pred_flat[mask])
    pre = precision_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], 
    average='weighted', zero_division=0)
    rec = recall_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], 
    average='weighted', zero_division=0)
    f1 = f1_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], 
    average='weighted', zero_division=0)
    
    return {
        "wer": wer,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1}
    
log_dir = os.path.join(os.getcwd(), "whisper_training_logs")
os.makedirs(log_dir, exist_ok=True)

args = Seq2SeqTrainingArguments(
    output_dir=log_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    tf32=True,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",
    max_steps=100000,
    save_steps=1000,
    eval_steps=1000,
    warmup_steps=300,
    num_train_epochs=1,
    logging_steps=1,
    logging_dir=os.path.join(log_dir, "logs_hf"),
    report_to=["tensorboard"],
    push_to_hub=False,
    disable_tqdm=False,
    save_total_limit=1,
    remove_unused_columns=False,
    label_names=["labels"],
    eval_on_start=False,
    # optim="adafactor",
)

optimizer = MaxFactor(
    model.parameters(), 
    lr=0.01,  
    beta2_decay=-0.8,
    eps=(1e-10, 1e-4),  
    d=1.0,
    weight_decay=0.01,  
    gamma=0.99,         
    eps_rms=1e-8,
    maximize=False,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=args.max_steps,
    eta_min=0.0,
    last_epoch=-1  
)

trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=extractor,
    optimizers=(optimizer, scheduler),
)


trainer.train(resume_from_checkpoint=False)


