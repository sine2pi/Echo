

import base64, gzip, math, os, functools, warnings, numpy as np, torch, transformers, aiohttp, torch.nn.functional as F, evaluate, json, random
from torch import Tensor, amp, optim, nn
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard.writer import SummaryWriter
from threading import Thread
from typing import Dict, Optional, Tuple, Union, List, Any
from transformers.modeling_utils import PreTrainedModel 
from dataclasses import dataclass
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, PretrainedConfig, TrainerCallback, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizerFast)

import evaluate
from evaluate import module
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from datasets import load_dataset, Dataset, concatenate_datasets, IterableDatasetDict, Audio
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
        super().__init__() 
        value = torch.tensor(initial_value)
        if learnable:
            self.value = nn.Parameter(value)
        else:
            self.register_buffer('value', value)

    def forward(self):
        return self.value

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:# type: ignore
        return F.linear(x, self.weight.to(x.dtype), 
                         None if self.bias is None else self.bias.to(x.dtype))

class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:# type: ignore
        return super()._conv_forward(x, weight.to(x.dtype), 
                                     None if bias is None else bias.to(x.dtype))

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return super().forward(x.float()).type(x.dtype)



class CombinedRotaryEmbedding(nn.Module):
    def __init__(self, base, dims: int, head: int, theta_scale_learnable: bool = True,
                 n_rots_scale_learnable: bool = True, r_matrix_learnable: bool = False, inv_freq_learnable: bool = True):
        super().__init__()
        self.dims = dims
        self.head = head
        self.base = base
        
        assert self.dims % self.head == 0, "dims must be divisible by head"
        self.h_dim = self.dims // self.head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"
        self.n_rots = ((dims // head) // 2)

        self.thetas = nn.Parameter(torch.zeros(self.n_rots))
        self.r_pairs = nn.Parameter(data=torch.rand(self.n_rots, 2) * self.h_dim)

        self.theta_scale = nn.Parameter(torch.ones(1), requires_grad=theta_scale_learnable)
        self.n_rots_scale = nn.Parameter(torch.ones(1), requires_grad=n_rots_scale_learnable)

        # --- R Matrix --- loss += embedding_layer.orthogonal_regularization_term()
        self.r_matrix = nn.Parameter(torch.eye(n=self.h_dim), requires_grad=r_matrix_learnable)

        inv_freq_data = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
        self.inv_freq = nn.Parameter(inv_freq_data, requires_grad=inv_freq_learnable)

        self.orthogonal_reg_weight = 0.01  
        
    def givens_r_matrix(self, dims, i, j, theta):
        G = torch.eye(dims).to(theta.device)
        G[i, i] = math.cos(theta)
        G[i, j] = -math.sin(theta)
        G[j, i] = math.sin(theta)
        G[j, j] = math.cos(theta)
        return G

    def update_base(self, new_base):
        if new_base is not None and new_base != self.base:
            self.base = new_base
            inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
            self.inv_freq.data.copy_(inv_freq)
            self.update_pairs()
            # print("Pairs updated")

    def reset_parameters(self):
        nn.init.orthogonal_(tensor=self.r_matrix)
        nn.init.zeros_(tensor=self.thetas)

    def orthogonal_regularization_term(self):
        loss = torch.tensor(0.0, device=self.r_matrix.device) 
        if self.r_matrix.requires_grad: 
            product = torch.matmul(self.r_matrix, self.r_matrix.t())
            identity = torch.eye(self.r_matrix.size(0)).to(self.r_matrix.device)
            loss = ((product - identity) ** 2).sum()
        return self.orthogonal_reg_weight * loss
   
    def update_pairs(self):   
        pairs = []
        while len(pairs) < self.n_rots:
            i, j = random.randint(0, self.h_dim - 1), random.randint(0, self.h_dim - 1)
            if i != j and (i, j) not in pairs and (j, i) not in pairs:
                pairs.append((i, j))
        self.r_pairs.data.copy_(torch.tensor(pairs, dtype=torch.float32))
        
    def forward(self, x, global_step=None):
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")

        batch_size, seq_len, *rest = x.size()

        if x.dim() == 3:
            dims = rest[0]
            if dims != self.head * self.h_dim:
                raise ValueError(
                    f"Expected dims ({dims}) to be compatible with head ({self.head}) * h_dim ({self.h_dim}={self.head * self.h_dim})")
        else:
            head, h_dim = rest
            if head != self.head or h_dim != self.h_dim:
                raise ValueError(
                    f"For 4D input, expected head {self.head} and h_dim {self.h_dim}, but got head {head} and h_dim {h_dim}")

        x = x.view(batch_size, seq_len, self.head, self.h_dim)
        x = x.reshape(-1, self.h_dim)
        adjusted_n_rots = int(torch.round(self.n_rots_scale * self.n_rots))

        for k in range(adjusted_n_rots):
            i, j = self.r_pairs[k].long()
            theta = self.thetas[k] * self.theta_scale
            G = self.givens_r_matrix(dims=self.h_dim, i=i, j=j, theta=theta)
            x = torch.matmul(input=x, other=G)

        x = torch.matmul(input=x, other=self.r_matrix)
        x = x.view(batch_size, seq_len, self.head, self.h_dim)

        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(end=seq_len, device=x.device),
                                     self.inv_freq.to(device=x.device))
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]

        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat(tensors=[x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x = x.view(batch_size, seq_len, self.dims)

        return x

        



class LearnedSinusoidalEmbeddings(nn.Module):
    def __init__(self, n_ctx, dims, checkpoint=False):
        super().__init__()
        self.n_ctx = n_ctx
        self.dims = dims
        self.checkpoint = checkpoint

        position = torch.arange(0, n_ctx, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dims, 2).float() * -(math.log(10000.0) / dims))
        features = torch.zeros(n_ctx, dims)
        features[:, 0::2] = torch.sin(position * div_term)
        features[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('my_big_toe', features) 
        self.positional_embeddings = nn.Parameter(self.my_big_toe.clone())

    def forward(self, positions):
        if self.checkpoint:
            position_embeddings = checkpoint(lambda x: self.positional_embeddings[x], positions)
        else:
            position_embeddings = self.positional_embeddings[positions]
        return F.normalize(position_embeddings, p=2, dim=-1) # type: ignore

class MultiheadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, base, dims, head, max_dist): 
        super().__init__()
        assert dims % head == 0, "dims must be divisible by head"
        self.head = head
        self.h_dim = dims // head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"
        self.max_dist = max_dist
        
        self.query = nn.Linear(dims, dims)
        self.key = nn.Linear(dims, dims, bias=False)
        self.value = nn.Linear(dims, dims)
        self.out = nn.Linear(dims, dims)

        self.combined_rotary = CombinedRotaryEmbedding(base, dims, head)

    def forward(self, x, xa = None, mask = None, kv_cache = None):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        q = self.combined_rotary(q)
        k = self.combined_rotary(k) 

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, 
                      mask: Optional[Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, dims = q.shape

        scale = (dims // self.head) ** -0.25
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        if MultiheadAttention.use_sdpa:
            a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and n_ctx > 1)
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None 
        else:
            L, S = q.size(-2), k.size(-2)
            scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=q.dtype)  
            w = q @ k.transpose(-2, -1) * scale_factor
            w += attn_bias.to(q.dtype).to(q.device)  
            w = torch.softmax(w, dim=-1).to(q.dtype)

            qk = (q * scale) @ (k * scale).transpose(-1, -2)

            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]

            qk = qk.float()
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()
 
        return out, qk
    
### wip

# class AdaptiveSpanAttention(nn.Module):
#     def __init__(self, base, dims, head, max_dist, win_size, max_span, temp_scale=0.01):
#         super().__init__()

#         self.max_dist = max_dist
#         self.win_size = win_size
#         self.max_span = max_span
#         self.temp_scale = temp_scale
#         self.multihead_attn = nn.MultiheadAttention(dims, head)
#         self.span_scale = nn.Parameter(torch.tensor(1.0))

#     def forward(self, query, key, value, span_scale):
#         span_len = int(self.max_span * span_scale.mean().item())
#         span_len = min(span_len, query.shape[1], key.shape[1], value.shape[1])

#         eff_span = min(span_len, self.max_dist)
#         q_span = query[:, :eff_span, :]
#         k_span = key[:, :eff_span, :]
#         v_span = value[:, :eff_span, :]

#         attn_out, attn_weights = self.multi_attn(q_span, k_span, v_span)
#         temperature = 1.0 - self.temp_scale * span_scale  

#         n_batch, n_ctx, dims = q_span.shape
#         scale = (dims // self.multi_attn.head) ** -0.25

#         q = q_span.view(*q_span.shape[:2], self.multi_attn.head, -1).permute(0, 2, 1, 3)
#         k = k_span.view(*k_span.shape[:2], self.multi_attn.head, -1).permute(0, 2, 1, 3)
#         v = v_span.view(*v_span.shape[:2], self.multi_attn.head, -1).permute(0, 2, 1, 3)

#         attn_scores = torch.matmul(q, k.transpose(-2, -1))
#         attn_weights = torch.softmax((attn_scores / temperature) * scale, dim=-1)
#         attn_out = torch.matmul(attn_weights, v)
#         attn_out = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
#         attn_weights = attn_weights * (1.0 / span_scale)     
       
#         attn_out = torch.bmm(attn_weights.view(-1, *attn_weights.shape[2:]), v.view(-1, *v.shape[2:]))
#         attn_out = attn_out.view(query.size(0), query.size(1), -1)
#         attn_out = attn_out.permute(0, 2, 1).contiguous().view(query.size(0), -1, query.size(2))    
#         # print(f"Adaptive {attn_out.shape}")
#         return attn_out, attn_weights

# class SpanPredictor(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         self.linear = nn.Linear(dims, 1)

#     def forward(self, global_out):
#         scale = torch.sigmoid(self.linear(global_out))
#         return scale
    
# class HybridAttention(nn.Module):
#     def __init__(self, base, dims, head, max_dist, win_size = 32, max_span = 32, slid_win = 32):
#         super().__init__()
#         self.max_dist = max_dist
#         self.win_size = win_size
#         self.max_span = max_span
#         self.slid_win = slid_win

#         self.span_pred = SpanPredictor(dims)
#         self.dist_local = max_dist  
#         self.dist_global = max_dist
#         self.attn_local = AdaptiveSpanAttention(base, dims, head, self.dist_local, win_size, max_span)
#         self.attn_global = MultiheadAttention(base, dims, head, self.dist_global)
#         self.ln_local = LayerNorm(dims)
#         self.ln_global = LayerNorm(dims)
#         self.projection = Linear(2 * dims, dims)

#     def forward(self, x, new_dist=None, new_base=None, xa = None, mask = None, kv_cache = None):
 
#         local = self.ln_local(x)
#         globe= self.ln_global(x)

#         globe_out, _ = self.attn_global(globe, globe, globe)
#         span_scale = self.span_pred(globe_out.mean(dim=1)) 

#         win_size = max(1, int(self.slid_win * span_scale.mean().item()))
#         span_len = max(1, int(self.max_span * span_scale.mean().item()))

#         effective_max_dist = min(self.max_dist, local.size(1))
#         local_max_dist = min(self.dist_local, span_len, win_size)
#         globe_max_dist = effective_max_dist
#         self.attn_local.max_dist = local_max_dist
#         self.attn_global.max_dist = globe_max_dist

#         local_out = self.slid_wiattention(local, win_size, span_len, span_scale)
#         combined = torch.cat([local_out.permute(1, 0, 2), globe_out.permute(1, 0, 2)], dim=-1)
#         x = self.projection(combined)
#         return x
    
#     def slid_wiattention(self, x, win_size, span_len, span_scale):
#         batch_size, seq_len, dims = x.size()
#         out = torch.zeros_like(x, device=x.device)

#         for i in range(0, seq_len, win_size):
#             end = min(i + win_size, seq_len)
#             query = x[:, i:end, :]
#             start = max(0, i - span_len + win_size)
#             key = x[:, start:i + span_len, :]
#             value = x[:, start:i + span_len, :]
 
#             attn_out, _ = self.attn_local(query, key, value, span_scale)
#             x[:, i:end, :] = attn_out
#         # print(f"Hybrid {x.shape}")
#         return x


class CombinedSparseAdaptiveAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, max_span, sparsity_factor):
        super().__init__()
        self.head = head
        self.multihead_attn = nn.MultiheadAttention(dims, head)
        self.sparsity_factor = sparsity_factor
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value): 
        assert query.dim() in [2, 3], ("query should be unbatched 2D or batched 3D tensor "
                                       f"but received {query.dim()}-D tensor")
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])

        batch_size, seq_len, dims = query.size()
        k = max(1, int(seq_len * self.sparsity_factor))
        indices = torch.topk(query.norm(dim=-1), k, dim=1).indices
        query_sparse = query.gather(1, indices.unsqueeze(-1).expand(-1, -1, dims))
        key_sparse = key.gather(1, indices.unsqueeze(-1).expand(-1, -1, dims))
        value_sparse = value.gather(1, indices.unsqueeze(-1).expand(-1, -1, dims))

        if query_sparse.shape[1] > 0 and key_sparse.shape[1] > 0 and value_sparse.shape[1] > 0: 
            query_sparse = query_sparse.view(query_sparse.shape[0], query_sparse.shape[1], self.head, -1)
            key_sparse = key_sparse.view(key_sparse.shape[0], key_sparse.shape[1], self.head, -1)
            value_sparse = value_sparse.view(value_sparse.shape[0], value_sparse.shape[1], self.head, -1)

        span_length = int(self.max_span * self.span_scale.item())
        span_length = min(span_length, query.shape[1])
        query_span = query_sparse[:, :span_length, :]
        key_span = key_sparse[:, :span_length, :]
        value_span = value_sparse[:, :span_length, :]

        attn_output, attn_weights = self.multihead_attn(query_span, key_span, value_span) 
        return attn_output, attn_weights

class SparseAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, sparsity_factor):
        super().__init__()
        self.head = head
        self.multihead_attn = nn.MultiheadAttention(dims, head)
        self.sparsity_factor = sparsity_factor

    def forward(self, query, key, value):
        assert query.dim() in [2, 3], ("query should be unbatched 2D or batched 3D tensor "
                                       f"but received {query.dim()}-D tensor")
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])

        batch_size, seq_len, dims = query.size()
        k = max(1, int(seq_len * self.sparsity_factor))
        indices = torch.topk(query.norm(dim=-1), k, dim=1).indices
        query_sparse = query.gather(1, indices.unsqueeze(-1).expand(-1, -1, dims))
        key_sparse = key.gather(1, indices.unsqueeze(-1).expand(-1, -1, dims))
        value_sparse = value.gather(1, indices.unsqueeze(-1).expand(-1, -1, dims))

        if query_sparse.shape[1] > 0 and key_sparse.shape[1] > 0 and value_sparse.shape[1] > 0:
            query_sparse = query_sparse.view(query_sparse.shape[0], query_sparse.shape[1], self.head, -1)
            key_sparse = key_sparse.view(key_sparse.shape[0], key_sparse.shape[1], self.head, -1)
            value_sparse = value_sparse.view(value_sparse.shape[0], value_sparse.shape[1], self.head, -1)

        attn_output, attn_weights = self.multihead_attn(query_sparse, key_sparse, value_sparse)
        return attn_output, attn_weights

class AdaptiveSpanAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, max_span):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dims, head)
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
    def __init__(self, base, dims, head, max_dist, chunk_size):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dims, head)
        self.chunk_size = chunk_size

    def forward(self, query, key, value, kv_cache=None):
        batch_size, seq_len, dims = query.size()
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
    def __init__(self, base, dims, head, max_dist, max_span, chunk_size):
        super().__init__()
        self.head = head
        self.multihead_attn = nn.MultiheadAttention(dims, head)
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))
        self.chunk_size = chunk_size

    def forward(self, query, key, value, kv_cache=None):
        assert query.dim() in [2, 3], ("query should be unbatched 2D or batched 3D tensor "
                                       f"but received {query.dim()}-D tensor")
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])

        batch_size, seq_len, dims = query.size()
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
    def __init__(self, base, dims, head, max_dist, win_size, max_span=64, sparsity_factor=0.1): 
        super().__init__()

        # self.local_attn = AdaptiveSpanAttention(base, dims, head, max_dist, max_span)
        self.local_attn = SparseAttention(base, dims, head, max_dist, sparsity_factor)
        self.global_attn = MultiheadAttention(base, dims, head, max_dist)
        self.ln_local = LayerNorm(dims)
        self.ln_global = LayerNorm(dims)

        self.projection = Linear(2 * dims, dims)

    def update_window(self, new_window):
        new_window = max(1, int(new_window + 0.5))
        self.win_size = new_window
        # self.local_attn.max_span = new_window

    def forward(self, x): 
        win_size = self.win_size
        x_local = self.ln_local(x)
        x_global = self.ln_global(x)
        x_local = x_local.permute(1, 0, 2)  
        x_global = x_global.permute(1, 0, 2) 

        local_out = self.sliding_window_attention(x_local, win_size)
        global_out, _ = self.global_attn(x_global, x_global, x_global)

        if local_out.shape[1] != global_out.shape[1]:
            seq_len_diff = local_out.shape[1] - global_out.shape[1]
            if seq_len_diff > 0:
                local_out = local_out[:, :global_out.shape[1], :]
            elif seq_len_diff < 0:
                pad = (0, 0, 0, -seq_len_diff, 0, 0)
                local_out = F.pad(local_out, pad).to(local_out.device)

        combined = torch.cat([local_out, global_out], dim=-1)
        combined_out = self.projection(combined)
        return combined_out.permute(1, 0, 2) 

    def sliding_window_attention(self, x, win_size):
        batch_size, seq_len, dims = x.size()
        output = torch.zeros_like(x)

        for i in range(0, seq_len, win_size):
            end = min(i + win_size, seq_len)
            query = x[i:end, :, :]
            start = max(0, i - win_size)
            key = x[start:end, :, :]
            value = x[start:end, :, :]
            attn_output, _ = self.local_attn(query, key, value)
            output[i:end, :, :] = attn_output[:end - i, :, :]
        return output




class ResidualAttentionBlock(nn.Module):
    def __init__(self, base, dims, head, max_dist, win_size, max_span, hybrid):
        super().__init__()

        if hybrid:
            print("HybridDrive ON")
            self.attn = HybridAttention(base, dims, head, max_dist, win_size, max_span)
            self.attn_ln = LayerNorm(dims)
        else:
            self.attn = MultiheadAttention(base, dims, head, max_dist)
            self.attn_ln = LayerNorm(dims)

        n_mlp = dims * 4
        self.mlp = nn.Sequential(nn.Linear(dims, n_mlp), nn.GELU(), nn.Linear(n_mlp, dims))
        self.mlp_ln = LayerNorm(dims)

    def forward(self, x, mask=None, kv_cache=None): 
        x = self._attn_forward(x, mask, kv_cache)
        x = self._mlp_forward(x)
        return x

    def _attn_forward(self, x, mask=None, kv_cache=None):
        residual = x
        x = self.attn_ln(x)
        if isinstance(self.attn, HybridAttention):
            x = residual + self.attn(x)[0]  
        else:
            x = residual + self.attn(x, mask=mask, kv_cache=kv_cache)[0]  
        return x

    def _mlp_forward(self, x):
        residual = x
        x = self.mlp_ln(x)
        return residual + self.mlp(x)

class AudioEncoder(nn.Module):
    def __init__(self, base, mels, dims, head, n_layer, n_ctx, max_dist, 
                 win_size, max_span, hybrid, checkpoint, cross):  
        super().__init__()
        self.conv1 = Conv1d(mels, dims, kernel_size=3, padding=1)
        self.conv2 = Conv1d(dims, dims, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, dims)
        self.checkpoint = checkpoint 

        self.combined_rotary = CombinedRotaryEmbedding(base, dims, head)

        self.blocks = nn.ModuleList([ResidualAttentionBlock(base, dims, head, max_dist, win_size, max_span, hybrid) for _ in range(n_layer)])
        
        self.ln_post = LayerNorm(dims)

    def forward(self, x):
        if self.checkpoint:
            x = checkpoint(self._conv_forward, x)
        else:
            x = self._conv_forward(x)

        for block in self.blocks:
            if self.checkpoint:
                x = checkpoint(block, x) 
            else:
                x = block(x) 
        return self.ln_post(x)

    def _conv_forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.combined_rotary(x)
        p = self.positional_embedding(torch.arange(x.size(1), device=x.device)).unsqueeze(0)
        x = x + p
        return x

class TextDecoder(nn.Module):
    def __init__(self, base, vocab, dims, head, n_layer, n_ctx, max_dist, 
                 win_size, max_span, hybrid, checkpoint, cross):  
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, dims)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, dims)
        self.checkpoint = checkpoint

        self.combined_rotary = CombinedRotaryEmbedding(base, dims, head)

        self.blocks = nn.ModuleList([ResidualAttentionBlock(base, dims, head, max_dist, win_size, max_span, hybrid) for _ in range(n_layer)])
        
        self.ln_post = LayerNorm(dims)
        self.ln = LayerNorm(dims)

        mask = torch.empty(n_ctx, n_ctx).fill_(value=-np.inf).triu_(diagonal=1)
        self.register_buffer("mask", mask, persistent=False)
        self.mask=mask

    def forward(self, x, xa, kv_cache=None):  
            
        if self.checkpoint:
            x = checkpoint(self._embedding_forward, x, xa, kv_cache)
        else:
            x = self._embedding_forward(x, xa, kv_cache)

        for block in self.blocks:
            if self.checkpoint:
                x = checkpoint(block, x, self.mask, kv_cache)  
            else:
                x = block(x, self.mask, kv_cache)  

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(dtype=x.dtype), 0, 1)).float()
        return logits

    def _embedding_forward(self, x, xa, kv_cache):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        positions = torch.arange(x.shape[1], device=x.device) + offset
        pos_emb = self.positional_embedding(positions).unsqueeze(0)
        x = self.token_embedding(x) + pos_emb
        x = x.to(xa.dtype)

        # x = self.combined_rotary(x)
        return x

class EchoConfig(PretrainedConfig):
    model_type = "Echo"
    def __init__(
        self,
        checkpoint=False,
        cross=False,
        hybrid=True,

        a_ctx=1500,
        a_head=8,
        a_layer=8,
        a_dims=1024,
        mels=128,
        t_ctx=448,
        t_head=8,
        t_layer=4,
        t_dims=1024,
        win_size=64,
        max_span=64,
        max_dist=128,
        base=10000,
        
        pad_token_id=50257,
        unk_token_id=50257,
        vocab=51865,
        eos_token_id=50257,
        bos_token_id=50257,
        decoder_start_token_id=50258,
        **kwargs,
    ):
        
        super().__init__(**kwargs) 
        self.base = base
        self.bos_token_id = bos_token_id
        self.checkpoint = checkpoint
        self.cross = cross
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id
        self.hybrid = hybrid
        self.max_dist = max_dist
        self.max_span = max_span
        self.a_ctx = a_ctx
        self.a_head = a_head
        self.a_layer = a_layer
        self.a_dims = a_dims
        self.mels = mels
        self.t_ctx = t_ctx
        self.t_head = t_head
        self.t_layer = t_layer
        self.t_dims = t_dims
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.vocab = vocab
        self.win_size = win_size


class Echo(PreTrainedModel):
    config_class = EchoConfig
    
    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.config = config
            
        self.encoder = AudioEncoder(
            base=self.config.base,
            mels=self.config.mels,
            dims=self.config.a_dims, 
            head=self.config.a_head,
            n_layer=self.config.a_layer,
            n_ctx=self.config.a_ctx,
            max_dist=self.config.max_dist,
            win_size=self.config.win_size,  
            max_span=self.config.max_span,
            hybrid=self.config.hybrid,
            checkpoint=self.config.checkpoint,
            cross=self.config.cross,
        )

        self.decoder = TextDecoder(
            base=self.config.base,
            vocab=self.config.vocab,
            dims=self.config.t_dims, 
            head=self.config.t_head,
            n_layer=self.config.t_layer,
            n_ctx=self.config.t_ctx,
            max_dist=self.config.max_dist,
            win_size=self.config.win_size,  
            max_span=self.config.max_span,
            hybrid=self.config.hybrid,
            checkpoint=self.config.checkpoint,
            cross=self.config.cross,
        )


        all_heads = torch.zeros(self.config.t_layer, self.config.t_head, dtype=torch.bool) 
        all_heads[self.config.t_layer // 2:] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

        self.base = self.config.base
        self.win_size = self.config.win_size
        self.adjust_counter = 0
        self.best_loss = float('inf')
        self.kv_cache = {}

    def update_window(self, new_window):
        self.win_size = new_window
        for module in self.modules(): 
            if isinstance(module, HybridAttention):
                module.update_window(self.win_size)

    def adjust_window(self, loss, factor=1.00005):
        if self.adjust_counter % 10 == 0:
            if loss < self.best_loss:
                new_window = self.win_size * factor
            else:
                new_window = self.win_size / factor
            self.update_window(new_window)
            self.best_loss = loss
            self.adjust_counter += 1
            return new_window
        return self.win_size

    def update_dist(self, new_dist):
        self.new_dist=new_dist
        for name, module in self.encoder.named_modules():
            if isinstance(module, (MultiheadAttention)):
                module.update_dist(self.new_dist)

    def adjust_n_dist(self, loss, step_size=1, threshold=0.0005) -> int:
        if self.adjust_counter % 25 == 0:
            if loss < self.best_loss:
                potential_new_dist=self.n_dist + step_size
            else:
                potential_new_dist=max(1, self.n_dist - step_size)            
            if abs(potential_new_dist - self.n_dist) >= threshold:
                new_dist=potential_new_dist
                self.update_dist(new_dist)
                self.n_dist=new_dist
                self.best_loss=loss
        self.adjust_counter += 1
        return self.n_dist

    def adjust_base(self, loss, factor=1.0025):
                if self.adjust_counter % 25 == 0:
                    if loss < self.best_loss:
                        new_base=self.base*factor
                    else:
                        new_base=self.base/factor
                    self.update_base(new_base)
                    self.base=new_base
                    self.best_loss=loss
                self.adjust_counter += 1
                return self.base
            
    def update_base(self, new_base):
        self.new_base=new_base
        for name, module in self.encoder.named_modules():
            if isinstance(module, (CombinedRotaryEmbedding)):
                module.update_base(self.new_base)

    def print_update(self, x=None):
        self.adjust_counter += 1
        if self.adjust_counter % 10 == 0:
            print(f"Update: {self.adjust_counter}: Loss: {self.best_loss}  "
                  f"Base: {self.base}, Window size: {self.win_size}")
        return x  

    @staticmethod
    def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone() 
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward(self, input_features, labels=None, dec_input_ids=None):
        if labels is not None:
            if dec_input_ids is None:
                dec_input_ids = self.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        encoded_features = self.encoder(input_features).to(self.device)  
        logits = self.decoder(dec_input_ids, encoded_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(logits.device).long()
            loss = loss_fct(logits.view(-1, self.config.vocab), labels.view(-1))

            # self.adjust_window(loss.item())
            self.adjust_base(loss.item())
            # self.print_update()  

        return {"loss": loss, "logits": logits}


    def reset_parameters(self):
        for name, module in self.encoder.named_modules():
            if isinstance(module, CombinedRotaryEmbedding):
                module.reset_parameters()
        self.encoder.apply(self._init_weights)
        
    def _initialize_weights(self, module):
            nn.init.normal_(self.decoder.token_embedding.weight, mean=0.0, std=0.02)

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
        self._initialize_weights(module)

    def set_alignment_heads(self, dump: bytes):
        array=np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask=torch.from_numpy(array).reshape(
            self.config.t_layer, self.config.t_head
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
    def supports_gradient_checkpointing(self):
        return True

    def install_kv_cache_hooks(self, cache=None):
        cache={**cache} if cache is not None else {}
        hooks=[]

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.config.t_ctx:
                cache[module]=output
            else:
                cache[module]=torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiheadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    def prepare_inputs_for_generation(self, input_features, **kwargs):
        return {'input_features': input_features}

    def _prepare_decoder_input_ids_for_generation(self, batch_size, decoder_start_token_id=None, bos_token_id=None):
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * self.config.decoder_start_token_id

    def can_generate(self):
        return True
        
    def generate(self, input_features, max_length=32, num_samples=None, **kwargs):
        if num_samples is not None:
            input_features = input_features[:num_samples]

        encoder_outputs = self.encoder(input_features)
        decoder_input_ids = torch.ones((input_features.size(0), 1), dtype=torch.long, 
                                    device=input_features.device) * self.config.decoder_start_token_id   
        generated_sequences = decoder_input_ids
        
        for step in range(max_length - 1):
            outputs = self.decoder(generated_sequences, encoder_outputs)
            next_token_logits = outputs[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            
            generated_sequences = torch.cat([generated_sequences, next_token_id], dim=-1)
        return generated_sequences

    def generate_beam_search(self, input_features, num_beams=1, max_length=32, **kwargs):
        encoder_outputs = self.encoder(input_features)
        batch_size = input_features.size(0)
        
        decoder_input_ids = torch.ones((batch_size * num_beams, 1), dtype=torch.long, device=input_features.device) * self.config.decoder_start_token_id
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_features.device)
        
        for step in range(max_length - 1):
            outputs = self.decoder(decoder_input_ids, encoder_outputs.repeat_interleave(num_beams, dim=0))
            next_token_logits = outputs[:, -1, :]
            next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
            
            beam_scores = beam_scores.view(-1, 1) + next_token_scores
            beam_scores, beam_tokens = beam_scores.view(batch_size, -1).topk(num_beams, dim=-1)
            
            beam_indices = beam_tokens // self.config.vocab
            next_token_id = beam_tokens % self.config.vocab
            decoder_input_ids = torch.cat([decoder_input_ids[beam_indices], next_token_id.unsqueeze(-1)], dim=-1)
            
            if torch.all(next_token_id == self.config.eos_token_id):
                break
        
        return decoder_input_ids

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
        


class GradientClippingCallback(TrainerCallback):
    def on_step_end(self, args, dims, control, **kwargs):
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

    def on_evaluate(self, args, dims, control, model, metrics=None, **kwargs):
        if metrics is not None:
            self.eval_loss = metrics.get('eval_loss')

            if dims.global_step % self.log_every_n_steps == 0:
                for key, value in metrics.items():
                    if key.startswith("eval_"):
                        self.tb_writer.add_scalar(key, value, dims.global_step)

        if self.predictions is not None and self.label_ids is not None:
            pred_str = self.tokenizer.batch_decode(self.predictions, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(self.label_ids, skip_special_tokens=True)

            if dims.global_step % self.log_every_n_steps == 0:
                sample_index = 0  
                self.tb_writer.add_text(f"Prediction", pred_str[sample_index], dims.global_step) 
                self.tb_writer.add_text(f"Label", label_str[sample_index], dims.global_step)
                print(f"Evaluation: - Step {dims.global_step} - Loss: {self.eval_loss:.2f}")
                print(f"Prediction: {pred_str[sample_index]}")
                print(f"Label: {label_str[sample_index]}")
                print("-" * 10)

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

def prepare_dataset(batch):
    batch["input_features"] = feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch



if __name__ == '__main__':
    
    from datetime import datetime
    log_dir = os.path.join('./output/', datetime.now().strftime('%Y-%m-%d_%H'))
    os.makedirs(log_dir, exist_ok=True)
    
    name="/echo_test/"
    config = EchoConfig(
        checkpoint=False,
        cross=False,
        hybrid=False,
        a_ctx=1500,
        a_head=8,
        a_layer=8,
        a_dims=1024,
        mels=128,
        t_ctx=448,
        t_head=8,
        t_layer=4,
        t_dims=1024,
        win_size=64,
        max_span=64,
        max_dist=128,
        base=10000,
        pad_token_id=50257,
        unk_token_id=50257,
        vocab=51865,
        eos_token_id=50257,
        bos_token_id=50257,
        decoder_start_token_id=50258
        )
    
    
    config.save_pretrained(log_dir+name)
    model = Echo(config).to(device)
    model.apply_initialization(module)

    # model.save_pretrained(log_dir+name, safe_serialization=False)
    # torch.save(model.dims_dict(), log_dir+name+"dims_dict.pt")
    # model = Echo.from_pretrained(pretrained_model_name_or_path=(log_dir+name)).to(device)

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", feature_size=128)
    tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-small", language="en", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    train=load_dataset("fixie-ai/librispeech_asr", "clean", split="train.100", streaming=True, trust_remote_code=True).map(prepare_dataset).select_columns(["input_features", "labels"])
    train=train.shuffle()

    test=load_dataset("fixie-ai/librispeech_asr", "clean", split="test", streaming=True, trust_remote_code=True).map(prepare_dataset).select_columns(["input_features", "labels"]).take(50)
    test=test.shuffle()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer, feature_extractor=feature_extractor)

    metric = evaluate.load(path="wer")
    tb_writer = SummaryWriter(log_dir=log_dir)
    metrics_callback = MetricsCallback(tb_writer=tb_writer, tokenizer=tokenizer, metric=metric, log_every_n_steps=5)
    compute_metrics = create_compute_metrics(callback_instance=metrics_callback)

    training_args = Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        tf32=True,
        bf16=True,
        evaluation_strategy="steps",
        max_steps=20000,
        save_steps=5000,
        eval_steps=100,
        logging_steps=2,
        logging_dir=log_dir + "/logs_hf",
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
    eval_results = trainer.evaluate()



