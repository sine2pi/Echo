
import base64
import gzip
import math
import os
import functools
import warnings
import numpy as np
import torch
import transformers
import aiohttp
import torch.nn.functional as F
from torch import Tensor, amp, optim, nn
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from threading import Thread
from typing import Dict, Optional, Tuple, Union, List, Any
from transformers.modeling_utils import PreTrainedModel 
from dataclasses import dataclass
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments, PretrainedConfig, TrainerCallback,
    WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizerFast
)

import evaluate
from evaluate import module
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from datasets import  load_dataset
from torch.nn.functional import scaled_dot_product_attention

transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings(action="ignore")
warnings.warn = lambda *args, **kwargs: None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch_dtype = torch.float32
torch.set_default_dtype(dtype)



class CustomEmbedding(nn.Module):
    def __init__(self, initial_value, learnable=True):
        super(CustomEmbedding, self).__init__()
        if learnable:
            self.value = nn.Parameter(torch.tensor(initial_value))
        else:
            self.register_buffer('value', torch.tensor(initial_value))
    def forward(self):
        return self.value

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )
    
class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class CombinedRotaryEmbedding(nn.Module):
    def __init__(self, n_state, n_head, num_rotations, base=10000, checkpointing=False):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.h_dim = n_state // n_head
        self.num_rotations = num_rotations
        self.base = base
        self.checkpointing = checkpointing
        
        self.thetas = nn.Parameter(torch.zeros(num_rotations))
        self.rotation_pairs = nn.Parameter(data=torch.rand(num_rotations, 2) * self.h_dim)
        self.theta_scale = nn.Parameter(data=torch.ones(1))  
        self.rotation_matrix = nn.Parameter(data=torch.eye(n=self.h_dim))
        self.inv_freq = nn.Parameter(data=1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)))

        self.num_rotations_scale = nn.Parameter(data=torch.ones(1))
    
    def givens_rotation_matrix(self, n_state, i, j, theta):
        G = torch.eye(n_state, device=theta.device)
        G[i, i] = math.cos(theta)
        G[i, j] = -math.sin(theta)
        G[j, i] = math.sin(theta)
        G[j, j] = math.cos(theta)
        return G
    
    def update_base(self, new_base):
        self.base = float(new_base)
        self.base = new_base
        self.inv_freq = nn.Parameter(data=1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)))
    
    def reset_parameters(self):
        nn.init.orthogonal_(tensor=self.rotation_matrix)
        nn.init.zeros_(tensor=self.thetas)
    
    def forward(self, x):
        if self.checkpointing:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")
        
        if x.dim() == 3:
            batch_size, seq_len, n_state = x.size()
            x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
        else:
            batch_size, seq_len, n_head, h_dim = x.size()
            if n_head != self.n_head or h_dim != self.h_dim:
                raise ValueError(f"Expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")
        
        x = x.reshape(-1, self.h_dim)
        
        adjusted_num_rotations = int(self.num_rotations * self.num_rotations_scale.item())
        
        for k in range(adjusted_num_rotations):
            i, j = self.rotation_pairs[k].long()
            theta = self.thetas[k] * self.theta_scale  
            G = self.givens_rotation_matrix(n_state=self.h_dim, i=i, j=j, theta=theta)
            x = torch.matmul(input=x, other=G)
        
        x = torch.matmul(input=x, other=self.rotation_matrix)
        x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
        
        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(end=seq_len, device=x.device), self.inv_freq.to(device=x.device))
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]
        
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat(tensors=[x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x = x.view(batch_size, seq_len, self.n_state)
        return x

class LearnedSinusoidalEmbeddings(nn.Module):
    def __init__(self, n_ctx, n_state, checkpointing=False):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.checkpointing = checkpointing

        position = torch.arange(start=0, end=n_ctx, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(input=torch.arange(start=0, end=n_state, step=2).float() * -(math.log(10000.0) / n_state))
        features = torch.zeros(n_ctx, n_state)
        features[:, 0::2] = torch.sin(input=position * div_term)
        features[:, 1::2] = torch.cos(input=position * div_term)
        self.register_buffer('my_big_toe', tensor=features)
        self.positional_embeddings = nn.Parameter(self.my_big_toe.clone())

    def forward(self, positions):
        if self.checkpointing:
            position_embeddings = checkpoint(lambda x: self.positional_embeddings[x], positions)
        else:
            position_embeddings = self.positional_embeddings[positions]
        position_embeddings = torch.nn.functional.normalize(input=position_embeddings, p=2, dim=-1)
        return position_embeddings

class MultiheadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int, max_rel_dist, base):
        super().__init__()
        assert n_state % n_head == 0, "n_state must be divisible by n_head"
        self.n_head = n_head
        self.h_dim = n_state // n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

        self.positional_scaling = nn.Parameter(torch.ones(1))      

        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        self.kv_cache = {}

        self.max_rel_dist = max_rel_dist
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
        self.register_buffer(name='inv_freq', tensor=inv_freq)

        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * self.max_rel_dist - 1, self.n_head)))

        self.combined_rotary = CombinedRotaryEmbedding(
            n_state,
            n_head,
            num_rotations=self.h_dim // 2, 
            base=base,
            checkpointing=False  
        )

    def update_base(self, new_base):
        self.base = float(new_base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim)) 
        self.register_buffer('inv_freq', inv_freq) 
        self.combined_rotary.update_base(self.base)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:

            k = kv_cache[self.key]
            v = kv_cache[self.value]

        q = self.combined_rotary(q) * self.positional_scaling
        k = self.combined_rotary(k) * self.positional_scaling

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape

        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if MultiheadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
            qk = qk if qk is not None else 0
        else:
            L, S = q.size(-2), k.size(-2)
            scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=q.dtype)
            w = q @ k.transpose(-2, -1) * scale_factor
            w += attn_bias.to(q.dtype).to(device)
            w = torch.softmax(w, dim=-1).to(q.dtype)

            qk = (q * scale) @ (k * scale).transpose(-1, -2)

            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]

            qk = qk.float()
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        positions = torch.arange(end=seq_len_q, device=q.device).unsqueeze(dim=1) - torch.arange(end=seq_len_k, device=q.device).unsqueeze(dim=0)
        positions = positions.clamp(min=-self.max_rel_dist + 1, max=self.max_rel_dist - 1) + self.max_rel_dist - 1
        rel_bias = self.rel_pos_bias[positions]
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)
        qk = qk + rel_bias
        return out, qk

class CombinedSparseAdaptiveAttention(nn.Module):
    def __init__(self, n_state, n_head, max_rel_dist, base, sparsity_factor, max_span):
        super().__init__()
        self.n_head = n_head
        self.multihead_attn = MultiheadAttention(n_state, n_head, max_rel_dist, base)
        self.sparsity_factor = sparsity_factor
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value):
        assert query.dim() == 2 or query.dim() == 3, "query should be unbatched 2D or batched 3D tensor but received {}-D tensor".format(query.dim())
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3]) 

        batch_size, seq_len, n_state = query.size()

        k = max(1, int(seq_len * self.sparsity_factor)) 
        indices = torch.topk(query.norm(dim=-1), k, dim=1).indices
        query_sparse = query.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        key_sparse = key.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        value_sparse = value.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))

        if query_sparse.shape[1] > 0 and key_sparse.shape[1] > 0 and value_sparse.shape[1] > 0:
            query_sparse = query_sparse.view(query_sparse.shape[0], query_sparse.shape[1], self.n_head, -1)
            key_sparse = key_sparse.view(key_sparse.shape[0], key_sparse.shape[1], self.n_head, -1)
            value_sparse = value_sparse.view(value_sparse.shape[0], value_sparse.shape[1], self.n_head, -1)

        span_length = int(self.max_span * self.span_scale.item())
        span_length = min(span_length, query.shape[1])
        query_span = query_sparse[:, :span_length, :]
        key_span = key_sparse[:, :span_length, :]
        value_span = value_sparse[:, :span_length, :]

        attn_output, attn_weights = self.multihead_attn(query_span, key_span, value_span)
        return attn_output, attn_weights

class SparseAttention(nn.Module):
    def __init__(self, n_state, n_head, max_rel_dist, base, sparsity_factor):
        super().__init__()
        self.n_head = n_head
        self.multihead_attn = MultiheadAttention(n_state, n_head, max_rel_dist, base)
        self.sparsity_factor = sparsity_factor
            
    def forward(self, query, key, value):
        assert query.dim() == 2 or query.dim() == 3, "query should be unbatched 2D or batched 3D tensor but received {}-D tensor".format(query.dim())
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3]) 

        batch_size, seq_len, n_state = query.size()
        k = max(1, int(seq_len * self.sparsity_factor))  

        indices = torch.topk(query.norm(dim=-1), k, dim=1).indices
        query_sparse = query.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        key_sparse = key.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        value_sparse = value.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))

        if query_sparse.shape[1] > 0 and key_sparse.shape[1] > 0 and value_sparse.shape[1] > 0:
            query_sparse = query_sparse.view(query_sparse.shape[0], query_sparse.shape[1], self.n_head, -1)
            key_sparse = key_sparse.view(key_sparse.shape[0], key_sparse.shape[1], self.n_head, -1)
            value_sparse = value_sparse.view(value_sparse.shape[0], value_sparse.shape[1], self.n_head, -1)

        attn_output, attn_weights = self.multihead_attn(query_sparse, key_sparse, value_sparse)
        return attn_output, attn_weights

class AdaptiveSpanAttention(nn.Module):
    def __init__(self, n_state, n_head, max_span, max_rel_dist, base):
        super().__init__()
        self.multihead_attn = MultiheadAttention(n_state, n_head, max_rel_dist, base) 
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value):
        span_length = int(self.max_span * self.span_scale.item())
        span_length = min(span_length, query.shape[1])
        query_span = query[:, :span_length, :]
        key_span = key[:, :span_length, :]
        value_span = value[:, :span_length, :]
        attn_output, attn_weights = self.multihead_attn(query_span, key_span, value_span)
        return attn_output, attn_weights

class RecurrentAttention(nn.Module):
    def __init__(self, n_state, n_head, max_rel_dist, base, chunk_size):
        super().__init__()
        self.multihead_attn = MultiheadAttention(n_state, n_head, max_rel_dist, base)
        self.chunk_size = chunk_size

    def forward(self, query, key, value, kv_cache=None):
        batch_size, seq_len, n_state = query.size()  
        output = torch.zeros_like(query).to(query.device)

        if kv_cache is None:
            kv_cache = {}
        key_global = key
        value_global = value

        for i in range(0, seq_len, self.chunk_size):
            end = min(seq_len, i + self.chunk_size)
            query_chunk = query[:, i:end, :]  

            if 'k' not in kv_cache:
                kv_cache['k'] = key_global.clone().detach().to(query.device)
                kv_cache['v'] = value_global.clone().detach().to(query.device)

            key_chunk = kv_cache['k'][:, :end, :]
            value_chunk = kv_cache['v'][:, :end, :]

            attn_output, _ = self.multihead_attn(query_chunk, key_chunk, value_chunk)
            output[:, i:end, :] = attn_output
        return output, kv_cache

class CombinedAdaptiveSpanRecurrentAttention(nn.Module):
    def __init__(self, n_state, n_head, max_rel_dist, base, max_span, chunk_size):
        super().__init__()
        self.n_head = n_head
        self.multihead_attn = MultiheadAttention(n_state, n_head, max_rel_dist, base)
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))
        self.chunk_size = chunk_size

    def forward(self, query, key, value, kv_cache=None):
        assert query.dim() == 2 or query.dim() == 3, "query should be unbatched 2D or batched 3D tensor but received {}-D tensor".format(query.dim())
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])  

        batch_size, seq_len, n_state = query.size()
        output = torch.zeros_like(query).to(query.device)

        if kv_cache is None:
            kv_cache = {}
        key_global = key
        value_global = value

        for i in range(0, seq_len, self.chunk_size):
            end = min(seq_len, i + self.chunk_size)
            query_chunk = query[:, i:end, :]  

            if 'k' not in kv_cache:
                kv_cache['k'] = key_global.clone().detach().to(query.device)
                kv_cache['v'] = value_global.clone().detach().to(query.device)

            key_chunk = kv_cache['k'][:, :end, :]
            value_chunk = kv_cache['v'][:, :end, :]

            span_length = int(self.max_span * self.span_scale.item())
            span_length = min(span_length, query_chunk.shape[1])
            query_span = query_chunk[:, :span_length, :]
            key_span = key_chunk[:, :span_length, :]
            value_span = value_chunk[:, :span_length, :]

            attn_output, _ = self.multihead_attn(query_span, key_span, value_span)
            output[:, i:end, :] = attn_output

        return output, kv_cache

class HybridAttention(nn.Module):
    def __init__(self, n_state, n_head, n_layer, window_size=40, alpha=0.001, sparsity_factor=0.333, 
                 max_span=50, chunk_size=50, max_rel_dist=1, base=10000, loss=None):
        super().__init__()
        self.local_attn = AdaptiveSpanAttention(n_state, n_head, max_rel_dist, base, max_span)
        self.global_attn = MultiheadAttention(n_state, n_head, max_rel_dist, base)# RecurrentAttention(n_state, n_head, chunk_size)
        self.ln_local = nn.LayerNorm(n_state)
        self.ln_global = nn.LayerNorm(n_state)
        self.window_scale = nn.Parameter(torch.tensor(float(window_size)))  
        self.best_loss = float('inf')
        self.projection = nn.Linear(2 * n_state, n_state) 

        self.window_size = window_size
        self.loss = loss

    def update_window(self, new_window):
        new_window = max(1, int(new_window + 0.5))
        self.window_size = new_window
        self.local_attn.max_span = new_window
        self.global_attn.chunk_size = new_window

    def forward(self, x, loss=None):
        if loss is not None:
            self.update_window(loss)
        window_size = self.window_size 

        x_local = self.ln_local(x)
        x_global = self.ln_global(x)
        x_local = x_local.permute(1, 0, 2)
        x_global = x_global.permute(1, 0, 2)

        local_out = self.sliding_window_attention(x_local, window_size)
        kv_cache = {}
        global_out, _ = self.global_attn(x_global, x_global, x_global, kv_cache=kv_cache)

        if local_out.shape[1] != global_out.shape[1]:
            seq_len_diff = local_out.shape[1] - global_out.shape[1]
            if seq_len_diff > 0: 
                local_out = local_out[:, :global_out.shape[1], :]
            elif seq_len_diff < 0:
                pad = (0, 0, 0, -seq_len_diff, 0, 0) 
                local_out = F.pad(local_out, pad).to(local_out.device)

        combined = torch.cat([local_out, global_out], dim=-1)
        combined_out = self.projection(combined)
        combined_out = combined_out.permute(1, 0, 2)
        return combined_out

    def sliding_window_attention(self, x, window_size):
        batch_size, seq_len, n_state = x.size()
        output = torch.zeros_like(x)

        for i in range(0, seq_len, window_size):
            end = min(i + window_size, seq_len)
            query = x[i:end, :, :]
            start = max(0, i - window_size)
            key = x[start:end, :, :]
            value = x[start:end, :, :]
            attn_output, _ = self.local_attn(query, key, value)
            output[i:end, :, :] = attn_output[:end - i, :, :]
        return output

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, max_rel_dist, window_size, base, cross_attention, checkpointing=False):
        super().__init__()
        self.attn = HybridAttention(n_state, n_head, window_size, max_rel_dist, base)
        self.cross_attn = MultiheadAttention(n_state, n_head, max_rel_dist, base) if cross_attention else None
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        self.cross_attention = cross_attention
        self.checkpointing = checkpointing
        self.max_rel_dist = max_rel_dist

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x, xa=None, mask=None, loss=None, kv_cache=None):
        if self.checkpointing:
            x = checkpoint(self._attn_forward, x, mask, loss, kv_cache)
        else:
            x = self._attn_forward(x, mask, loss, kv_cache)
        if self.cross_attention:
            if self.checkpointing:
                x = checkpoint(self._cross_attn_forward, x, xa, kv_cache)
            else:
                x = self._cross_attn_forward(x, xa, kv_cache)
        if self.checkpointing:
            x = checkpoint(self._mlp_forward, x)
        else:
            x = self._mlp_forward(x)
        return x

    def _attn_forward(self, x, mask=None, loss=None, kv_cache=None):
        residual = x
        x = self.attn_ln(x)
        if isinstance(self.attn, HybridAttention):
            x = residual + self.attn(x, loss)[0]
        else:
            x = residual + self.attn(x, mask=mask, kv_cache=kv_cache)[0]
        return x

    def _cross_attn_forward(self, x, xa, kv_cache=None):
        residual = x
        x = self.cross_attn_ln(x)
        x = residual + self.cross_attn(x, xa, kv_cache=kv_cache)[0]
        return x

    def _mlp_forward(self, x):
        residual = x
        x = self.mlp_ln(x)
        x = residual + self.mlp(x)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer, window_size, max_rel_dist, checkpointing=False, cross_attention=True, base=10000):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state, checkpointing=checkpointing)
        self.checkpointing = checkpointing
        self.h_dim = n_state // n_head

        self.combined_rotary = CombinedRotaryEmbedding(
            n_state,
            n_head,
            num_rotations=self.h_dim // 2,
            base=base,
            checkpointing=False)

        self.blocks = nn.ModuleList(
            modules=[ResidualAttentionBlock(n_state, 
                                            n_head, 
                                            cross_attention=cross_attention, 
                                            max_rel_dist=max_rel_dist, 
                                            checkpointing=checkpointing,
                                            window_size=window_size,
                                            base=base,
                                            ) for _ in range(n_layer)])
        self.ln_post = LayerNorm(n_state)

    def update_base(self, new_base):
        self.base = float(new_base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim)) 
        self.register_buffer('inv_freq', inv_freq) 
        self.combined_rotary.update_base(self.base)

    def forward(self, x):
        if self.checkpointing:
            x = checkpoint(self._conv_forward, x)
        else:
            x = self._conv_forward(x=x)
        for block in self.blocks:
            if self.checkpointing:
                x = checkpoint(block, x)
            else:
                x = block(x)
        x = self.ln_post(x)
        return x

    def _conv_forward(self, x):
        x = F.gelu(input=self.conv1(x))
        x = F.gelu(input=self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.combined_rotary(x)
        pos_emb = self.positional_embedding(torch.arange(end=x.size(1), device=x.device)).unsqueeze(0)
        x = x + pos_emb
        return x

class TextDecoder(nn.Module):
    def __init__(self, vocab_size, n_ctx, n_state, n_head, n_layer, window_size, max_rel_dist, checkpointing=False, 
                 cross_attention=False, base=10000):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_state)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state, checkpointing=checkpointing)
        self.checkpointing = checkpointing
        self.n_head = n_head
        self.h_dim = n_state // n_head
        
        self.combined_rotary = CombinedRotaryEmbedding(
            n_state,
            n_head,
            num_rotations=self.h_dim // 2, 
            base=base,
            checkpointing=False)

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, 
                                    n_head, 
                                    cross_attention=cross_attention, 
                                    max_rel_dist=max_rel_dist, 
                                    checkpointing=checkpointing,
                                    window_size=window_size,
                                    base=base,
                                    )
            for _ in range(n_layer)])
        
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(value=-np.inf).triu_(diagonal=1)
        self.register_buffer(name="mask", tensor=mask, persistent=False)

    def update_base(self, new_base):
        self.base = float(new_base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim)) 
        self.register_buffer('inv_freq', inv_freq) 
        self.combined_rotary.update_base(self.base)

    def forward(self, x, xa, loss=None, kv_cache=None): 
        if self.checkpointing:
            x = checkpoint(self._embedding_forward, x, xa, kv_cache)
        else:
            x = self._embedding_forward(x, xa, kv_cache)

        for block in self.blocks:
            if self.checkpointing:
                x = checkpoint(block, x, xa, self.mask, loss, kv_cache)  
            else:
                x = block(x, xa, self.mask, loss, kv_cache) 

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(dtype=x.dtype), 0, 1)).float()
        return logits

    def _embedding_forward(self, x, xa, kv_cache):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        positions = torch.arange(end=x.shape[1], device=x.device) + offset
        pos_emb = self.positional_embedding(positions).unsqueeze(0)
        x = self.token_embedding(x) + pos_emb
        x = x.to(xa.dtype)
        batch_size, seq_length, embedding_dim = x.shape
        num_heads = self.n_head
        head_dim = embedding_dim // num_heads
        x = x.view(batch_size, seq_length, num_heads, head_dim)
        x = self.combined_rotary(x)
        x = x.view(batch_size, seq_length, embedding_dim)
        return x


class EchoConfig(PretrainedConfig):
    model_type = "Echo"
    def __init__(
        self,
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=1024,
        n_audio_head=16,
        n_audio_layer=24,
        vocab_size=51865,
        n_text_ctx=448,
        n_text_state=1024,
        n_text_head=16,
        n_text_layer=16,
        max_rel_dist=256,
        base=10000,
        window_size=128,
        init_std=0.02,
        pad_token_id=50257,
        unk_token_id=50257,
        bos_token_id=50257,
        eos_token_id=50257,
        decoder_start_token_id=50258,
        cross_attention=False,
        checkpointing=False,
        **kwargs,
    ):
        super(EchoConfig, self).__init__(**kwargs)
        self.base = base
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.checkpointing = checkpointing
        self.cross_attention = cross_attention
        self.init_std = init_std
        self.max_rel_dist = max_rel_dist
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_audio_state = n_audio_state
        self.n_mels = n_mels
        self.n_text_ctx = n_text_ctx
        self.n_text_head = n_text_head
        self.n_text_layer = n_text_layer
        self.n_text_state = n_text_state 
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.decoder_start_token_id = decoder_start_token_id

class Echo(PreTrainedModel):
    config_class = EchoConfig
    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.config = config  
        self.encoder = AudioEncoder(
            n_mels=self.config.n_mels,
            n_ctx=self.config.n_audio_ctx,
            n_state=self.config.n_audio_state,
            n_head=self.config.n_audio_head,
            n_layer=self.config.n_audio_layer,
            max_rel_dist=self.config.max_rel_dist,
            cross_attention=self.config.cross_attention,
            checkpointing=self.config.checkpointing,
            base=self.config.base,
            window_size=self.config.window_size,
        )
        self.decoder = TextDecoder(
            vocab_size=self.config.vocab_size,
            n_ctx=self.config.n_text_ctx,
            n_state=self.config.n_text_state,
            n_head=self.config.n_text_head,
            n_layer=self.config.n_text_layer,
            max_rel_dist=self.config.max_rel_dist,
            cross_attention=self.config.cross_attention,
            checkpointing=self.config.checkpointing,
            base=self.config.base,
            window_size=self.config.window_size,
        )

        all_heads = torch.zeros(self.config.n_text_layer, self.config.n_text_head, dtype=torch.bool)
        all_heads[self.config.n_text_layer // 2:] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

        self.base = self.config.base
        self.window_size = self.config.window_size
        self.adjust_counter = 0
        self.best_loss = float('inf')
        self.kv_cache = {}

    def update_base(self, new_base):
        self.base = new_base
        for name, module in self.encoder.named_modules():
            if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding, AudioEncoder)):
                module.update_base(self.base)
        for name, module in self.decoder.named_modules():
            if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding, TextDecoder)):
                module.update_base(self.base)

    def update_window(self, new_window):
        self.window_size = new_window
        for name, module in self.encoder.named_modules():
            if isinstance(module, HybridAttention):
                module.update_window(self.window_size)
        for name, module in self.decoder.named_modules():
            if isinstance(module, HybridAttention):
                module.update_window(self.window_size)

    def adjust_base(self, loss, factor=1.00005):
        if self.adjust_counter % 2 == 0:  
            if loss < self.best_loss:
                new_base = self.base * factor
            else:
                new_base = self.base / factor
            self.update_base(new_base=new_base)
            self.best_loss = loss
            return new_base
        return self.base

    def adjust_window(self, loss, factor=1.00005):
        if self.adjust_counter % 2 == 1: 
            if loss < self.best_loss:
                new_window = self.window_size * factor
            else:
                new_window = self.window_size / factor
            self.update_window(new_window=new_window)
            self.best_loss = loss
            return new_window
        return self.window_size

    def print_update(self, x):
        self.adjust_counter += 1
        if self.adjust_counter % 100 == 0:
            print(f"Update: {self.adjust_counter}: Loss: {self.best_loss}  Base: {self.base}, Window size: {self.window_size}")
        return x

    @staticmethod
    def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id) -> torch.Tensor:
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward(self, input_features, labels=None, dec_input_ids=None):
        if labels is not None:
            if dec_input_ids is None:
                dec_input_ids = self.shift_tokens_right(
                    input_ids=labels, pad_token_id=self.config.pad_token_id, 
                    decoder_start_token_id=self.config.decoder_start_token_id
                )

        encoded_features = self.encoder(input_features).to(device)
        logits = self.decoder(dec_input_ids, encoded_features)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(logits.device).long()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

            self.adjust_window(loss.item())
            self.adjust_base(loss.item())
            self.print_update(None)

        return {
            "loss": loss,
            "logits": logits,
        }

    def _initialize_weights(self, module):
            nn.init.normal_(self.decoder.token_embedding.weight, mean=0.0, std=self.config.init_std)
            if hasattr(self.decoder.positional_embedding, 'weight'):
                nn.init.normal_(self.decoder.positional_embedding.weight, mean=0.0, std=self.config.init_std)
            for block in self.decoder.blocks:
                for layer in block.children():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

            nn.init.constant_(self.decoder.ln.weight, 1)
            if self.decoder.ln.bias is not None:
                nn.init.constant_(self.decoder.ln.bias, 0)

            nn.init.xavier_normal_(self.encoder.conv1.weight)
            if self.encoder.conv1.bias is not None:
                nn.init.zeros_(self.encoder.conv1.bias)

            nn.init.kaiming_normal_(self.encoder.conv2.weight, mode='fan_out', nonlinearity='relu')
            if self.encoder.conv2.bias is not None:
                nn.init.zeros_(self.encoder.conv2.bias)

            nn.init.constant_(self.encoder.ln_post.weight, 1)
            if self.encoder.ln_post.bias is not None:
                nn.init.constant_(self.encoder.ln_post.bias, 0)
                
    def apply_initialization(self, module):
        self._initialize_weights( module )  

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.config.n_text_layer, self.config.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, labels, input_features):
        return self.decoder(labels, input_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.config.vocab_size >= 51865

    @property
    def num_languages(self):
        return self.config.vocab_size - 51765 - int(self.is_multilingual)
    
    @property
    def supports_gradient_checkpointing(self):
        return True

    def install_kv_cache_hooks(self, cache = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.config.n_text_ctx:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiheadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_features': input_ids}

    def _prepare_decoder_input_ids_for_generation(self, batch_size, decoder_start_token_id=None, bos_token_id=None):
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * decoder_start_token_id

    def can_generate(self):
        return True
    
    def generate(self, inputs, **kwargs):
        encoder_outputs = self.encoder(inputs)
        decoder_input_ids = torch.zeros((inputs.size(0), 1), dtype=torch.long, device=inputs.device)
        outputs = self.decoder(decoder_input_ids, encoder_outputs)
        return outputs.argmax(dim=-1)
    
    def generate_beam_search(self, inputs, **kwargs):
        encoder_outputs = self.encoder(inputs)
        decoder_input_ids = torch.zeros((inputs.size(0), 1), dtype=torch.long, device=inputs.device)
        outputs = self.decoder(decoder_input_ids, encoder_outputs)
        return outputs.argmax(dim=-1)

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=checkpoint):
        self.checkpointing = enable
        self.gradient_checkpointing_func = gradient_checkpointing_func
        for module in self.modules():
            if hasattr(module, 'checkpointing'):
                module.checkpointing = enable
                module.gradient_checkpointing_func = gradient_checkpointing_func

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}
        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)


from datetime import datetime
log_dir = os.path.join('./output/', datetime.now().strftime('%Y-%m-%d_%H'))
os.makedirs(log_dir, exist_ok=True)

name="/echo_test/"
config = EchoConfig(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=1024,
    n_audio_head=16,
    n_audio_layer=24,
    vocab_size=51865,
    n_text_ctx=448,
    n_text_state=1024,
    n_text_head=16,
    n_text_layer=16,
    max_rel_dist=256,
    base=10000,
    window_size=128,
    init_std=0.02,
    pad_token_id=50257,
    unk_token_id=50257,
    bos_token_id=50257,
    eos_token_id=50257,
    decoder_start_token_id=50258,
    cross_attention=False,
    checkpointing=False,
)
config.save_pretrained(log_dir+name)
model = Echo(config=config).to('cuda')
model.apply_initialization(module)
# model.save_pretrained(log_dir+name, safe_serialization=False)
# torch.save(model.state_dict(), log_dir+name+"state_dict.pt")
# model = Echo.from_pretrained(pretrained_model_name_or_path=(log_dir+name)).to('cuda')



feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path="openai/whisper-small")
tokenizer = WhisperTokenizerFast.from_pretrained(pretrained_model_name_or_path="openai/whisper-small", language="en", task="transcribe")
processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path="openai/whisper-small")


class GradientClippingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.nn.utils.clip_grad_norm_(parameters=kwargs["model"].parameters(), max_norm=0.98)

class MetricsCallback(TrainerCallback):
    def __init__(self, tb_writer, tokenizer, metric, log_every_n_steps=1):
        super().__init__()
        self.tb_writer = tb_writer
        self.tokenizer = tokenizer
        self.metric = metric
        self.log_every_n_steps = log_every_n_steps
        self.predictions = None
        self.label_ids = None

    def compute_wer(self, pred_str, label_str):
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return wer

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        if metrics is not None:
            eval_loss = metrics.get('eval_loss')
            if eval_loss is not None:

                adjusted_base = model.adjust_base(eval_loss)
                
                adjusted_window_size = model.adjust_window(eval_loss)

            if state.global_step % self.log_every_n_steps == 0:
                for key, value in metrics.items():
                    if key.startswith("eval_"):
                        self.tb_writer.add_scalar(key, value, state.global_step)

        if self.predictions is not None and self.label_ids is not None:
            pred_str = self.tokenizer.batch_decode(self.predictions, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(self.label_ids, skip_special_tokens=True)

            for sample_index in range(1):
                if sample_index < len(pred_str):
                    self.tb_writer.add_text(f"Prediction_{sample_index}", pred_str[sample_index], state.global_step)
                    self.tb_writer.add_text(f"Label_{sample_index}", label_str[sample_index], state.global_step)

                    print(f"Sample {sample_index + 0} - Step {state.global_step} - Eval Loss: {eval_loss:.4f} - Adjusted Base: {adjusted_base:.4f} - Adjusted Window: {adjusted_window_size:.4f}")
                    print(f"Prediction: {pred_str[sample_index]}")
                    print(f"Label: {label_str[sample_index]}")
                    
                    print("-" * 30)

        self.predictions = None
        self.label_ids = None

def create_compute_metrics(callback_instance):
    def compute_metrics(eval_pred):
        pred_logits = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(pred_logits, tuple):
            pred_ids = pred_logits[0]
        else:
            pred_ids = pred_logits
        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        label_ids[label_ids == -100] = callback_instance.tokenizer.pad_token_id
        callback_instance.predictions = pred_ids
        callback_instance.label_ids = label_ids

        pred_str = callback_instance.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = callback_instance.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * callback_instance.metric.compute(predictions=pred_str, references=label_str)

        pred_flat = pred_ids.flatten()
        labels_flat = label_ids.flatten()
        mask = labels_flat != callback_instance.tokenizer.pad_token_id

        accuracy = accuracy_score(y_true=labels_flat[mask], y_pred=pred_flat[mask])
        precision = precision_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], average='weighted', zero_division=0)
        recall = recall_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], average='weighted', zero_division=0)
        f1 = f1_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], average='weighted', zero_division=0)

        return {"wer": wer, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    return compute_metrics

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    tokenizer: Any
    feature_extractor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def get_length_of_dataset(dataset):
    length = 0
    for item in dataset:
        length += len(item["audio"]["array"]) / item["audio"]["sampling_rate"]
    return length / 3600  


from datasets import load_dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    transcription = batch["text"]
    batch["labels"] = tokenizer(transcription).input_ids
    return batch

train = load_dataset("fixie-ai/librispeech_asr", "clean", 
        split="train.100"
        streaming=False, 
        trust_remote_code=True, cache_dir="d:/hf").take(10) \
        .to_iterable_dataset(num_shards=1) \
        .map(prepare_dataset).select_columns(["input_features", "labels"])

test = load_dataset("fixie-ai/librispeech_asr", "clean", 
        split="test", 
        streaming=False, 
        trust_remote_code=True, cache_dir="d:/hf").take(10) \
        .to_iterable_dataset(num_shards=1) \
        .map(prepare_dataset).select_columns(["input_features", "labels"])


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer, feature_extractor=feature_extractor)

metric = evaluate.load(path="wer")
tb_writer = SummaryWriter(log_dir=log_dir)

metrics_callback = MetricsCallback(tb_writer=tb_writer, tokenizer=tokenizer, metric=metric, log_every_n_steps=100)
compute_metrics = create_compute_metrics(callback_instance=metrics_callback)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()
torch.cuda.set_device(device=0)

training_args = Seq2SeqTrainingArguments(
    output_dir=log_dir,  
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    tf32=True,
    bf16=True,
    # warmup_steps=10,
    evaluation_strategy="steps",
    max_steps=2000,
    save_steps=200,
    eval_steps=5,
    logging_steps=2,
    logging_dir= log_dir+"/logs_hf",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    optim="adafactor",
    weight_decay=0.0025,
    disable_tqdm=False,
    save_total_limit=2,
    save_strategy="steps",
    remove_unused_columns=False,
    label_names=["labels"],
    gradient_checkpointing=True,
    eval_on_start=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train,
    eval_dataset=test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
    callbacks=[metrics_callback]
)    

trainer.train(resume_from_checkpoint=False)

model.save_pretrained(log_dir+"/models/echo4_trained/")
import tensorboard


