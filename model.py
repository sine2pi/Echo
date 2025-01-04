import base64, gzip, evaluate, math, os, sys, time
import gzip, neologdn
from transformers.modeling_utils import PreTrainedModel 

import functools
from evaluate import module
from functools import partial, wraps
from threading import Thread 

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from torch import amp, Tensor, optim
from torch.utils.checkpoint import checkpoint
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from contextlib import contextmanager
from dataclasses import dataclass
from transformers.models.whisper.modeling_whisper import WhisperPreTrainedModel
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from transformers.optimization import Adafactor, AdafactorSchedule
from datasets import IterableDatasetDict, Audio, load_dataset, load_from_disk
import numpy as np
import torch, transformers, warnings
from typing import Dict, Iterable, Optional, Tuple, Union, List, Any, Type
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Seq2SeqTrainer, PretrainedConfig, TrainerCallback, Seq2SeqTrainingArguments, WhisperTokenizer, WhisperForConditionalGeneration, WhisperConfig, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperTokenizerFast
from whisper.decoding import decode as decode_function
from whisper.decoding import detect_language as detect_language_function
from whisper.transcribe import transcribe as transcribe_function
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings(action="ignore")
warnings.warn = lambda *args,**kwargs: None
torch.autograd.set_detect_anomaly(True)
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

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.gamma * x + self.beta

class BiasedCrossAttention(nn.Module):
    def __init__(self, n_state, n_head, dropout_rate=0.001):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head

        self.query = nn.Linear(in_features=n_state, out_features=n_state)
        self.key = nn.Linear(in_features=n_state, out_features=n_state, bias=False)
        self.value = nn.Linear(in_features=n_state, out_features=n_state)
        self.out = nn.Linear(in_features=n_state, out_features=n_state)

        self.bias = nn.Parameter(data=torch.zeros(n_head, 1, self.head_dim))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = LayerNorm(num_features=n_state)
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_length, _ = q.size()

        q = self.query(q).view(batch_size, seq_length, self.n_head, self.head_dim)
        k = self.key(k).view(batch_size, seq_length, self.n_head, self.head_dim)
        v = self.value(v).view(batch_size, seq_length, self.n_head, self.head_dim)

        qk = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5) + self.bias
        if mask is not None:
            qk = qk.masked_fill(mask == 0, float('-inf'))

        w = F.softmax(qk, dim=-1)
        w = self.dropout(w)

        out = (w @ v).transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        out = self.norm(self.out(out) + q.view(batch_size, seq_length, -1))
        return out

class DynamicConvAttention(nn.Module):
    def __init__(self, n_state, n_head, kernel_size=3, base=10000):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.h_dim = n_state // n_head
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(in_channels=n_state, out_channels=n_state, kernel_size=kernel_size, padding=kernel_size // 2, groups=n_head)
        self.query = nn.Linear(in_features=n_state, out_features=n_state)
        self.key = nn.Linear(in_features=n_state, out_features=n_state, bias=False)
        self.value = nn.Linear(in_features=n_state, out_features=n_state)
        self.out_proj = nn.Linear(in_features=n_state, out_features=n_state)
        self.norm = LayerNorm(num_features=n_state)

        self.combined_rotary = CombinedRotaryEmbedding(
            n_state=n_state,
            n_head=n_head,
            num_rotations=self.h_dim // 2,
            base=base,
            checkpointing=False 
        )

    def update_base(self, new_base):
        self.base = new_base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim)) 
        self.register_buffer('inv_freq', inv_freq) 
        self.combined_rotary.update_base(self.base)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        x = x.permute(0, 2, 1)
        conv_out = self.conv(x)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.norm(conv_out)

        attention_out = F.softmax(input=torch.matmul(input=q, other=k.transpose(-2, -1)) / (self.n_state ** 0.5), dim=-1)
        attention_out = torch.matmul(input=attention_out, other=v)
        
        combined_out = conv_out + attention_out
        combined_out = self.norm(combined_out)
        return self.out_proj(combined_out) + x.permute(0, 2, 1)

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
        
        for k in range(self.num_rotations):
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
    
class AdaptiveSpanAttention(nn.Module):
    def __init__(self, n_state, n_head, max_span=50):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(n_state, n_head)  # Custom version
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value):
        span_length = int(self.max_span * self.span_scale.item())
        query_span = query[:, :span_length, :]
        key_span = key[:, :span_length, :]
        value_span = value[:, :span_length, :]
        attn_output, attn_weights = self.multihead_attn(query_span, key_span, value_span)
        return attn_output, attn_weights

class RecurrentAttention(nn.Module):
    def __init__(self, n_state, n_head, chunk_size):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(n_state, n_head)
        self.chunk_size = chunk_size

    def forward(self, query, key, value):
        batch_size, seq_len, n_state = query.size()
        output = torch.zeros_like(query)
        
        for i in range(0, seq_len, self.chunk_size):
            query_chunk = query[:, i:i + self.chunk_size, :]
            key_chunk = key[:, i:i + self.chunk_size, :]
            value_chunk = value[:, i:i + self.chunk_size, :]
            
            if query_chunk.size(1) > 0 and key_chunk.size(1) > 0 and value_chunk.size(1) > 0:
                attn_output, attn_weights = self.multihead_attn(query_chunk, key_chunk, value_chunk)
                
                output[:, i:i + self.chunk_size, :] = attn_output

        return output


class SparseAttention(nn.Module):
    def __init__(self, n_state, n_head, sparsity_factor):
        super().__init__()
        self.n_head = n_head
        self.multihead_attn = nn.MultiheadAttention(n_state, n_head)
        self.sparsity_factor = sparsity_factor
            
    def forward(self, query, key, value):
        assert query.dim() == 2 or query.dim() == 3, "query should be unbatched 2D or batched 3D tensor but received {}-D tensor".format(query.dim())
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])  # Adjust this based on your requirements

        batch_size, seq_len, n_state = query.size()
        k = max(1, int(seq_len * self.sparsity_factor))  # Ensure k is at least 1

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

class MultiheadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state, n_head, max_rel_dist = 1, base = 10000):
        super().__init__()
        assert n_state % n_head == 0, "n_state must be divisible by n_head"
        self.n_head = n_head
        self.h_dim = n_state // n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

        self.positional_scaling = nn.Parameter(torch.ones(1))

        self.query = nn.Linear(in_features=n_state, out_features=n_state)
        self.key = nn.Linear(in_features=n_state, out_features=n_state, bias=False)
        self.value = nn.Linear(in_features=n_state, out_features=n_state)
        self.out = nn.Linear(in_features=n_state, out_features=n_state)

        self.max_rel_dist = max_rel_dist
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
        self.register_buffer(name='inv_freq', tensor=inv_freq)

        self.rel_pos_bias = nn.Embedding(num_embeddings=2 * self.max_rel_dist - 1, embedding_dim=self.n_head)
        self.rel_pos_bias.weight.data.fill_(value=0)

        self.combined_rotary = CombinedRotaryEmbedding(
            n_state=n_state,
            n_head=n_head,
            num_rotations=self.h_dim // 2,
            base=base,
            checkpointing=False 
        )

        if device:
            self.to(device=device)
            
    def update_base(self, new_base): 
        self.base = float(new_base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim)) 
        self.register_buffer('inv_freq', inv_freq) 
        self.combined_rotary.update_base(self.base)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        q = self.query(x)

        if kv_cache is None or xa is None or 'k' not in kv_cache:
            k_input = x if xa is None else xa
            k = self.key(k_input)
            v = self.value(k_input)
            if kv_cache is not None:
                kv_cache['k'] = k
                kv_cache['v'] = v
        else:
            k = kv_cache['k']
            v = kv_cache['v']

        q = q.view(q.shape[0], q.shape[1], self.n_head, -1)
        k = k.view(k.shape[0], k.shape[1], self.n_head, -1)
        v = v.view(v.shape[0], v.shape[1], self.n_head, -1)
        
        q = self.combined_rotary(q) 
        k = self.combined_rotary(k)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)

        wv, qk = self.qkv_attention(q=q, k=k, v=v, mask=mask)
        return self.out(wv), qk

        
    def qkv_attention(self, q, k, v, mask=None):
        n_batch, n_ctx, n_state = q.shape

        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        positions = torch.arange(end=seq_len_q, device=q.device).unsqueeze(dim=1) - torch.arange(end=seq_len_k, device=q.device).unsqueeze(dim=0)
        positions = positions.clamp(min=-self.max_rel_dist + 1, max=self.max_rel_dist - 1) + self.max_rel_dist - 1
        rel_bias = self.rel_pos_bias(positions)
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)
        qk = qk + rel_bias

        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(input=qk, dim=-1).to(dtype=q.dtype)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        qk = qk.detach()
        return out, qk


class HybridAttention(nn.Module):
    def __init__(self, n_state, n_head, window_size=40, alpha=0.001, sparsity_factor=0.333, 
                 max_span=50, chunk_size=50, max_rel_dist=1, base=10000, loss=None):
        super().__init__()
        self.local_attn = AdaptiveSpanAttention(n_state, n_head,  max_span)
        self.global_attn = RecurrentAttention(n_state, n_head, chunk_size)
        self.ln_local = nn.LayerNorm(n_state)
        self.ln_global = nn.LayerNorm(n_state)
        self.window_size = window_size
        self.loss = loss
        self.window_scale = nn.Parameter(torch.tensor(float(window_size)))  
        self.best_loss = float('inf')

    def update_window(self, new_window):
        self.window_size = int(new_window)

    def forward(self, x, loss=None):
        if loss is not None:
            window_size = self.update_window(loss)
        else:
            window_size = self.window_size

        x_local = self.ln_local(x)
        x_global = self.ln_global(x)
        x_local = x_local.permute(1, 0, 2)
        x_global = x_global.permute(1, 0, 2)

        local_out = self.sliding_window_attention(x_local, window_size)
        global_out = self.global_attn(x_global, x_global, x_global)
        combined_out = local_out + global_out
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
    def __init__(self, n_state, n_head, cross_attention=True, max_rel_dist=1, checkpointing=False, 
                 use_hybrid_attention=False, window_size=40):
        super().__init__()

        self.attn = MultiheadAttention(n_state=n_state, n_head=n_head)
        self.attn_ln = LayerNorm(num_features=n_state)
        self.checkpointing = checkpointing
        self.max_rel_dist = max_rel_dist
        self.cross_attention = cross_attention
        self.use_hybrid_attention = use_hybrid_attention

        if self.use_hybrid_attention:
            self.attn = HybridAttention(n_state=n_state, n_head=n_head, window_size=window_size)
            self.cross_attn = MultiheadAttention(n_state=n_state, n_head=n_head)

        self.cross_attn = MultiheadAttention(n_state=n_state, n_head=n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(num_features=n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(in_features=n_state, out_features=n_mlp), nn.GELU(), Linear(in_features=n_mlp, out_features=n_state)
        )
        self.mlp_ln = LayerNorm(num_features=n_state)

    def forward(self, x, xa=None, mask=None, loss=None, kv_cache=None):
        if self.checkpointing:
            x = checkpoint(self._attn_forward, x, mask, loss, kv_cache)
        else:
            x = self._attn_forward(x, mask, loss, kv_cache)
        if self.cross_attn:
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

class EnhancedResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, cross_attention=True, max_rel_dist=1, checkpointing=False,
                 use_hybrid_attention=False, window_size=40):
        super().__init__()

        self.attn = MultiheadAttention(n_state=n_state, n_head=n_head, max_rel_dist=max_rel_dist)
        self.attn_ln = LayerNorm(num_features=n_state)
        self.checkpointing = checkpointing
        self.max_rel_dist = max_rel_dist
        self.cross_attention = cross_attention
        self.use_hybrid_attention = use_hybrid_attention

        if self.use_hybrid_attention:
            self.attn = HybridAttention(n_state=n_state, n_head=n_head, window_size=window_size)
            self.cross_attn = MultiheadAttention(n_state=n_state, n_head=n_head, max_rel_dist=max_rel_dist)

        self.cross_attn = MultiheadAttention(n_state=n_state, n_head=n_head, max_rel_dist=max_rel_dist) if cross_attention else None
        self.cross_attn_ln = LayerNorm(num_features=n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(in_features=n_state, out_features=n_mlp), nn.GELU(), Linear(in_features=n_mlp, out_features=n_state)
        )
        self.mlp_ln = LayerNorm(num_features=n_state)

    def forward(self, x, xa=None, mask=None, loss=None, kv_cache=None):
        if self.checkpointing:
            x = checkpoint(self._attn_forward, x, mask, loss, kv_cache)
        else:
            x = self._attn_forward(x, mask, loss, kv_cache)
        if self.cross_attn:
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
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer, window_size = 40, max_rel_dist = 1, checkpointing=False,
                 use_hybrid_attention = False, cross_attention=True, base=10000):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_mels, out_channels=n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=n_state, out_channels=n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx=n_ctx, n_state=n_state, checkpointing=checkpointing)
        self.checkpointing = checkpointing
        self.h_dim = n_state // n_head

        self.combined_rotary = CombinedRotaryEmbedding(
            n_state=n_state,
            n_head=n_head,
            num_rotations=self.h_dim // 2,
            base=base,
            checkpointing=False 
        )

        self.blocks = nn.ModuleList(
            modules=[ResidualAttentionBlock(n_state=n_state, 
                                            n_head=n_head, 
                                            use_hybrid_attention=use_hybrid_attention, 
                                            cross_attention=cross_attention, 
                                            max_rel_dist=max_rel_dist, 
                                            checkpointing=checkpointing,
                                            window_size=window_size,
                                            ) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(num_features=n_state)

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
    def __init__(self, vocab_size, n_ctx, n_state, n_head, n_layer, window_size=40, max_rel_dist=1, checkpointing=False, 
                 use_hybrid_attention=False, cross_attention=True, base=10000):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_state)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx=n_ctx, n_state=n_state, checkpointing=checkpointing)
        self.checkpointing = checkpointing
        self.n_head = n_head
        self.h_dim = n_state // n_head
        
        self.combined_rotary = CombinedRotaryEmbedding(
            n_state=n_state,
            n_head=n_head,
            num_rotations=self.h_dim // 2, 
            base=base,
            checkpointing=False  
        )

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state=n_state, 
                                    n_head=n_head, 
                                    use_hybrid_attention=use_hybrid_attention, 
                                    cross_attention=cross_attention, 
                                    max_rel_dist=max_rel_dist, 
                                    checkpointing=checkpointing,
                                    window_size=window_size,
                                    )
            for _ in range(n_layer)
        ])
        
        self.ln = LayerNorm(num_features=n_state)
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
    


class AutonomicLayer(nn.Module):
    def __init__(self, encoder, decoder, alpha=0.0001, beta=0.9, factor=1.005, threshold=0.01, window_size=40, base=10000):
        super(AutonomicLayer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.best_loss = float('inf')
        self.base = base
        self.window_size = window_size
        self.adjust_counter = 0
        self.alpha = alpha
        self.beta = beta
        self.factor = factor
        self.threshold = threshold
        self.running_loss = None


    def update_base(self, new_base):
        self.base = new_base
        for name, module in self.encoder.named_modules():
            if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding, AudioEncoder)):
                module.update_base(self.base)
        for name, module in self.decoder.named_modules():
            if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding, TextDecoder)):
                module.update_base(self.base)

    def adjust_base(self, loss, factor=1.0005):
        self.adjust_counter += 1 
        if loss < self.best_loss:
            new_base = self.base * factor
        else:
            new_base = self.base / factor
        self.update_base(new_base=new_base)
        self.best_loss = loss
        return new_base

    def update_window(self, new_window):
        self.window_size = new_window
        for name, module in self.encoder.named_modules():
            if isinstance(module, HybridAttention):
                module.update_window(self.window_size)
        for name, module in self.decoder.named_modules():
            if isinstance(module, HybridAttention):
                module.update_window(self.window_size)
    
    def adjust_window(self, loss, factor=1.0005):
        self.adjust_counter += 1 
        if loss < self.best_loss:
            new_window = self.window_size * factor
        else:
            new_window = self.window_size / factor
        self.update_window(new_window=new_window)
        self.best_loss = loss
        return new_window

    def forward(self, x):
        self.adjust_counter += 1
        if self.adjust_counter % 10 == 0:
            print(f"Iteration {self.adjust_counter}: Current base: {self.base}, Current window size: {self.window_size}")
        return x



class EchoConfig(PretrainedConfig):
    model_type = "Echo"
    def __init__(
        self,
        n_mels=128,
        n_audio_ctx=1500,
        n_audio_state=1024,
        n_audio_head=16,
        n_audio_layer=24,
        vocab_size=15141,
        n_text_ctx=448,
        n_text_state=1024,
        n_text_head=16,
        n_text_layer=16,
        max_rel_dist=1000,
        base=10000,
        window_size = 100,
        init_std=0.02,
        pad_token_id = 112,
        unk_token_id = 113,
        bos_token_id = 114,
        eos_token_id = 115,
        decoder_start_token_id = 120,
        cross_attention = False,
        checkpointing = False,
        use_hybrid_attention = True,
        is_encoder_decoder = False,
        **kwargs,
    ):
        self.base = base
        self.use_hybrid_attention = use_hybrid_attention
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.checkpointing = checkpointing
        self.cross_attention = cross_attention
        self.init_std = init_std
        self.is_encoder_decoder = is_encoder_decoder
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
        super().__init__(**kwargs)
        
config = EchoConfig()

class Echo(PreTrainedModel):
    config_class = EchoConfig
    def __init__(self, config: EchoConfig):
        super().__init__(config)
        config = EchoConfig

        self.encoder = AudioEncoder(
            n_mels=self.config.n_mels,
            n_ctx=self.config.n_audio_ctx,
            n_state=self.config.n_audio_state,
            n_head=self.config.n_audio_head,
            n_layer=self.config.n_audio_layer,
            max_rel_dist=self.config.max_rel_dist,
            use_hybrid_attention=self.config.use_hybrid_attention,
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
            use_hybrid_attention=self.config.use_hybrid_attention,
            cross_attention=self.config.cross_attention,
            checkpointing=self.config.checkpointing,
            base=self.config.base,
            window_size=self.config.window_size,
        )

        all_heads = torch.zeros(self.config.n_text_layer, self.config.n_text_head, dtype=torch.bool)
        all_heads[self.config.n_text_layer // 2:] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    base=config.base
    window_size=config.window_size
    adjust_counter = 0
    best_loss = float('inf')

    def update_base(self, new_base):
        self.base = new_base
        for name, module in self.encoder.named_modules():
            if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding, AudioEncoder)):
                module.update_base(self.base)
        for name, module in self.decoder.named_modules():
            if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding, TextDecoder)):
                module.update_base(self.base)

    def adjust_base(self, loss, factor=1.0005):
        self.adjust_counter += 1 
        if loss < self.best_loss:
            new_base = self.base * factor
        else:
            new_base = self.base / factor
        self.update_base(new_base=new_base)
        self.best_loss = loss
        return new_base

    def update_window(self, new_window):
        self.window_size = new_window
        for name, module in self.encoder.named_modules():
            if isinstance(module, HybridAttention):
                module.update_window(self.window_size)
        for name, module in self.decoder.named_modules():
            if isinstance(module, HybridAttention):
                module.update_window(self.window_size)
    
    def adjust_window(self, loss, factor=1.0005):
        self.adjust_counter += 1 
        if loss < self.best_loss:
            new_window = self.window_size * factor
        else:
            new_window = self.window_size / factor
        self.update_window(new_window=new_window)
        self.best_loss = loss
        return new_window

    def print_update(self, x):
        self.adjust_counter += 1
        if self.adjust_counter % 100 == 0:
            if config.use_hybrid_attention:
                print(f"Update: {self.adjust_counter}: Loss: {self.best_loss}  Base: {self.base}, Window size: {self.window_size}")
            else:
                print(f"Update: {self.adjust_counter}: Loss: {self.best_loss}  Base: {self.base}")
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
                    input_ids=labels, pad_token_id = 112, decoder_start_token_id = 120)

        encoded_features = self.encoder(input_features).to(device)
        logits = self.decoder(dec_input_ids, encoded_features)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) 
            labels = labels.to(logits.device).long()
            loss = loss_fct(logits.view(-1, config.vocab_size), labels.view(-1))
            
        if config.use_hybrid_attention:
            self.adjust_window(loss.item())
            self.adjust_base(loss.item())
        else: 
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

            nn.init.constant_(self.decoder.ln.gamma, 1)
            if self.decoder.ln.beta is not None:
                nn.init.constant_(self.decoder.ln.beta, 0)

            nn.init.xavier_normal_(self.encoder.conv1.weight)
            if self.encoder.conv1.bias is not None:
                nn.init.zeros_(self.encoder.conv1.bias)

            nn.init.kaiming_normal_(self.encoder.conv2.weight, mode='fan_out', nonlinearity='relu')
            if self.encoder.conv2.bias is not None:
                nn.init.zeros_(self.encoder.conv2.bias)

            nn.init.constant_(self.encoder.ln_post.gamma, 1)
            if self.encoder.ln_post.beta is not None:
                nn.init.constant_(self.encoder.ln_post.beta, 0)
                
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
        return self.config.vocab_size >= 15141

    @property
    def num_languages(self):
        return self.config.vocab_size - 15041 - int(self.is_multilingual)
    
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

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

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

name="/echo15/"
config = EchoConfig()
config.save_pretrained(log_dir+name)
model = Echo(config=config).to(device)
model.apply_initialization(module)
model.save_pretrained(log_dir+name, safe_serialization=False)
torch.save(model.state_dict(), log_dir+name+"state_dict.pt")
model = Echo.from_pretrained(pretrained_model_name_or_path=(log_dir+name)).to(device)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path="D:/newproject/my_extractor", feature_size=128, sampling_rate=16000)
tokenizer = WhisperTokenizerFast.from_pretrained(pretrained_model_name_or_path="D:/newproject/my_tokenizer")
processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path="D:/newproject/my_processor")



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

    def compute_cer(self, pred_str, label_str):
        cer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return cer

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        if metrics is not None:
            eval_loss = metrics.get('eval_loss')
            if eval_loss is not None:

                adjusted_base = model.adjust_base(eval_loss)
                if config.use_hybrid_attention:
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

                    print(f"Sample {sample_index + 0} - Step {state.global_step} - Eval Loss: {eval_loss:.4f} - Adjusted Base: {adjusted_base:.4f}")
                    print(f"Prediction: {pred_str[sample_index]}")
                    print(f"Label: {label_str[sample_index]}")
                    print("-" * 30)

            # eval_loss = eval_loss if eval_loss is not None else 0.0
            # adjusted_base = adjusted_base if adjusted_base is not None else 0.0
            # if config.use_hybrid_attention:
            #     adjusted_window_size = adjusted_window_size if adjusted_window_size is not None else 0.0

            #print(f"Step {state.global_step} - Eval Loss: {eval_loss:.4f} - Sample Prediction: {pred_str[sample_index]} - Sample Label: {label_str[sample_index]} - Adjusted Base: {adjusted_base:.4f}")# - Adjusted Window Size: {adjusted_window_size:.4f}")

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
        cer = 100 * callback_instance.metric.compute(predictions=pred_str, references=label_str)

        pred_flat = pred_ids.flatten()
        labels_flat = label_ids.flatten()
        mask = labels_flat != callback_instance.tokenizer.pad_token_id

        accuracy = accuracy_score(y_true=labels_flat[mask], y_pred=pred_flat[mask])
        precision = precision_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], average='weighted', zero_division=0)
        recall = recall_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], average='weighted', zero_division=0)
        f1 = f1_score(y_true=labels_flat[mask], y_pred=pred_flat[mask], average='weighted', zero_division=0)

        return {"cer": cer, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
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
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    transcription = batch["sentence"]
    transcription = neologdn.normalize(transcription, repeat=1)
    batch["labels"] = tokenizer(transcription).input_ids
    return batch

####  -- Sanity Check

test = load_dataset(path="audiofolder", data_dir="D:/projold/datasets/gv_test")["train"] \
        .map(prepare_dataset).select_columns(["input_features", "labels"]).take(1)

train = load_dataset(path="audiofolder", data_dir="D:/projold/datasets/gv_test")["train"] \
        .map(prepare_dataset).select_columns(["input_features", "labels"]).take(1)

# train = load_dataset(path="audiofolder", data_dir="D:/proj/datasets/gv")["train"] \
#         .take(1000) \
#         .to_iterable_dataset(num_shards=200) \
#         .map(prepare_dataset).select_columns(["input_features", "labels"])



data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer, feature_extractor=feature_extractor)
optimizer = transformers.Adafactor(params=model.parameters(), 
                                clip_threshold=1.0, 
                                weight_decay=0.0025, 
                                scale_parameter=False, 
                                relative_step=False, 
                                warmup_init=False, 
                                lr=2.25e-5)

scheduler = transformers.optimization.AdafactorSchedule(optimizer=optimizer)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

metric = evaluate.load(path="cer")
tb_writer = SummaryWriter(log_dir=log_dir)

metrics_callback = MetricsCallback(tb_writer=tb_writer, tokenizer=tokenizer, metric=metric, log_every_n_steps=5)
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
    warmup_steps=100,
    evaluation_strategy="steps",
    max_steps=1000,
    save_steps=1000,
    eval_steps=5,
    logging_steps=5,
    logging_dir= log_dir+"/logs_hf",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
    optim="adafactor",
    weight_decay=0.0025,
    disable_tqdm=False,
    save_total_limit=2,
    torch_empty_cache_steps=10,
    save_strategy="steps",
    remove_unused_columns=False,
    label_names=["labels"],
    gradient_checkpointing=True,
    eval_on_start=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train,
    eval_dataset=test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
    callbacks=[metrics_callback]#, GradientClippingCallback]
)    

trainer.train(resume_from_checkpoint=False)

model.save_pretrained(log_dir+"/models/echo4_trained/")
import tensorboard


# import torch
# import numpy as np
# import random

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)
#     torch.cuda.manual_seed_all(42)

# trainer.evaluate()


