


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
from torch.utils.tensorboard.writer import SummaryWriter
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
from sklearn.model_selection import KFold, train_test_split
from datasets import load_dataset, Dataset, concatenate_datasets, IterableDatasetDict, Audio
from torch.nn.functional import scaled_dot_product_attention

import warnings
transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings(action="ignore")
warnings.warn = lambda *args, **kwargs: None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
dd = (dtype, device)
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
    def forward(self, x: Tensor) -> Tensor: # type: ignore
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class Conv1d(nn.Conv1d):
    def _conv_forward( # type: ignore
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor: # type: ignore
        return super().forward(x.float()).type(x.dtype)

class CombinedRotaryEmbedding(nn.Module):
    def __init__(self, n_state: int, n_head: int, base: float):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.base = base
        assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"
        self.h_dim = self.n_state // self.n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"
        self.num_rotations = ((n_state // n_head) // 2)
        
        self.thetas = nn.Parameter(torch.zeros(self.num_rotations)) 
        self.rotation_pairs = nn.Parameter(data=torch.rand(self.num_rotations, 2) * self.h_dim)
        self.theta_scale = nn.Parameter(data=torch.ones(1))
        self.rotation_matrix = nn.Parameter(data=torch.eye(n=self.h_dim))
        self.inv_freq = nn.Parameter(data=1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)))
        self.num_rotations_scale = nn.Parameter(data=torch.ones(1))

    def givens_rotation_matrix(self, n_state, i, j, theta):
        G = torch.eye(n_state).to(device)
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

    def reset_parameters(self):
        nn.init.orthogonal_(tensor=self.rotation_matrix)
        nn.init.zeros_(tensor=self.thetas)

    def forward(self, x, new_base=None):


        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")

        batch_size, seq_len, *rest = x.size()

        if x.dim() == 3:
            n_state = rest[0]
            if n_state != self.n_head * self.h_dim:
                raise ValueError(f"Expected n_state ({n_state}) to be compatible with n_head ({self.n_head}) * h_dim ({self.h_dim} = {self.n_head * self.h_dim})") 
        else: 
            n_head, h_dim = rest
            if n_head != self.n_head or h_dim != self.h_dim:
                raise ValueError(f"For 4D input, expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")

        x = x.view(batch_size, seq_len, self.n_head, self.h_dim) 
        x = x.reshape(-1, self.h_dim)
        adjusted_num_rotations = int(torch.round(self.num_rotations * self.num_rotations_scale))

        for k in range(adjusted_num_rotations):
            i, j = self.rotation_pairs[k].long()
            theta = self.thetas[k] * self.theta_scale
            G = self.givens_rotation_matrix(n_state=self.h_dim, i=i, j=j, theta=theta).to(device)
            x = torch.matmul(input=x, other=G).to(device)

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
    def __init__(self, n_ctx: int, n_state: int):
        super().__init__()

        position = torch.arange(start=0, end=n_ctx, dtype=dtype).unsqueeze(dim=1)
        div_term = torch.exp(input=torch.arange(start=0, end=n_state, step=2, dtype=dtype) * -(math.log(10000.0) / n_state))
        features = torch.zeros(n_ctx, n_state)
        features[:, 0::2] = torch.sin(position * div_term)
        features[:, 1::2] = torch.cos(position* div_term)
        self.register_buffer('my_big_toe', tensor=features)
        self.positional_embeddings = nn.Parameter(self.my_big_toe.clone())

    def forward(self, positions):
        position_embeddings = self.positional_embeddings[positions]
        # position_embeddings = torch.nn.functional.normalize(position_embeddings, p=2, dim=-1)
        return position_embeddings

class MultiheadAttention(nn.Module):
    use_sdpa = True
    def __init__(self, n_state: int, n_head: int, max_dist: int):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.max_dist = max_dist

        assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"
        self.h_dim = self.n_state // self.n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

        self.query = nn.Linear(self.n_state, self.n_state)
        self.key = nn.Linear(self.n_state, self.n_state, bias=False)
        self.value = nn.Linear(self.n_state, self.n_state)
        self.out = nn.Linear(self.n_state, self.n_state)
        
        self.kv_cache = {}
        
        self.positional_scaling = nn.Parameter(torch.ones(1))
        self.pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_dist) - 1, self.n_head))).to(device)
                   
    def update_dist(self, new_dist):
        if new_dist is not None and new_dist != self.max_dist:
            self.max_dist = new_dist
            # print("pos_bias")
            new_pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_dist) - 1, self.n_head))).to(self.pos_bias.device)  
            self.pos_bias = new_pos_bias          
                   
    def forward(self, x: Tensor, xa: Optional[Tensor] = None, 
                mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None, new_dist=None) -> tuple[Any, Tensor | None]:

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:

            k = kv_cache[self.key]
            v = kv_cache[self.value]

        # q = self.givens_rotary(q)# * self.positional_scaling
        # k = self.givens_rotary(k)# * self.positional_scaling

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk
   

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,  relative_positional_embeddings=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    
        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) 
        
        if relative_positional_embeddings is not None:
            rel_emb = torch.einsum("b h q d, q k d -> b h q k", q, relative_positional_embeddings)
        else:
            rel_emb = torch.zeros_like(attn_scores)
        
        positions = torch.arange(end=seq_len_q, device=q.device).unsqueeze(dim=1) - torch.arange(end=seq_len_k, device=q.device).unsqueeze(dim=0)
        positions = positions.clamp(min=-self.max_dist + 1, max=self.max_dist - 1) + self.max_dist - 1
        rel_bias = self.pos_bias[positions]
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)
        
        attn_scores = (attn_scores + rel_bias + rel_emb) * scale

        if MultiheadAttention.use_sdpa:
            a = scaled_dot_product_attention(q, k, v, attn_mask = attn_scores ,is_causal=mask is not None and n_ctx > 1)
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()
            
        return out, qk

class ResidualAttentionBlock(nn.Module):

    def __init__(self, n_ctx: int, n_state: int, n_head: int, n_layer: int, max_dist: int, base: float):
        super().__init__()

        self.attn = MultiheadAttention(n_state, n_head, max_dist)
        self.attn_ln = nn.LayerNorm(n_state)
        
        # self.cross_attn = (MultiheadAttention(n_state, n_head) if cross_attention else None)
        # self.cross_attn_ln = (LayerNorm(n_state) if cross_attention else None)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor, xa: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, kv_cache=None) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        # print(f"Residual out = {x.shape}")
        return x



# class AdaptiveSpanAttention(nn.Module):
#   def __init__(self, n_state, n_head, max_dist, win_size, max_span, temp_scale=0.01):
#       super().__init__()

#       self.max_dist = max_dist
#       self.win_size = win_size
#       self.max_span = max_span
#       self.temp_scale = temp_scale
#       self.multi_attn = MultiheadAttention(n_state, n_head, max_dist)
#       self.span_scale = nn.Parameter(torch.tensor(1.0))

#   def forward(self, query, key, value, span_scale):
#       span_len = int(self.max_span * span_scale.mean().item())
#       span_len = min(span_len, query.shape[1], key.shape[1], value.shape[1])

#       eff_span = min(span_len, self.max_dist)
#       q_span = query[:, :eff_span, :]
#       k_span = key[:, :eff_span, :]
#       v_span = value[:, :eff_span, :]

#       attn_out, attn_weights = self.multi_attn(q_span, k_span, v_span)
#       temperature = 1.0 - self.temp_scale * span_scale  

#       n_batch, n_ctx, n_state = q_span.shape
#       scale = (n_state // self.multi_attn.n_head) ** -0.25

#       q = q_span.view(*q_span.shape[:2], self.multi_attn.n_head, -1).permute(0, 2, 1, 3)
#       k = k_span.view(*k_span.shape[:2], self.multi_attn.n_head, -1).permute(0, 2, 1, 3)
#       v = v_span.view(*v_span.shape[:2], self.multi_attn.n_head, -1).permute(0, 2, 1, 3)

#       attn_scores = torch.matmul(q, k.transpose(-2, -1))
#       attn_weights = torch.softmax((attn_scores / temperature) * scale, dim=-1)
#       attn_out = torch.matmul(attn_weights, v)
#       attn_out = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
#       attn_weights = attn_weights * (1.0 / span_scale)     
     
#       attn_out = torch.bmm(attn_weights.view(-1, *attn_weights.shape[2:]), v.view(-1, *v.shape[2:]))
#       attn_out = attn_out.view(query.size(0), query.size(1), -1)
#       attn_out = attn_out.permute(0, 2, 1).contiguous().view(query.size(0), -1, query.size(2))    

#       return attn_out, attn_weights

# class SpanPredictor(nn.Module):
#   def __init__(self, n_state):
#       super().__init__()
#       self.linear = nn.Linear(n_state, 1)

#   def forward(self, global_out):
#       scale = torch.sigmoid(self.linear(global_out))
#       return scale
  
# class HybridAttention(nn.Module):
#   def __init__(self, n_state, n_head, max_dist, win_size = 32, max_span = 32, slid_win = 32):
#       super().__init__()
#       self.max_dist = max_dist
#       self.win_size = win_size
#       self.max_span = max_span
#       self.slid_win = slid_win

#       self.span_pred = SpanPredictor(n_state)
#       self.dist_local = max_dist  
#       self.dist_global = max_dist
#       self.attn_local = AdaptiveSpanAttention(n_state, n_head, self.dist_local, win_size, max_span)
#       self.attn_global = MultiheadAttention(n_state, n_head, self.dist_global)
#       self.ln_local = LayerNorm(n_state)
#       self.ln_global = LayerNorm(n_state)
#       self.projection = Linear(2 * n_state, n_state)

#   def forward(self, x, new_dist=None, new_base=None, xa = None, mask = None, kv_cache = None):

#       local = self.ln_local(x)
#       globe= self.ln_global(x)

#       globe_out, _ = self.attn_global(globe, globe, globe)
#       span_scale = self.span_pred(globe_out.mean(dim=1)) 

#       win_size = max(1, int(self.slid_win * span_scale.mean().item()))
#       span_len = max(1, int(self.max_span * span_scale.mean().item()))

#       effective_max_dist = min(self.max_dist, local.size(1))
#       local_max_dist = min(self.dist_local, span_len, win_size)
#       globe_max_dist = effective_max_dist
#       self.attn_local.max_dist = local_max_dist
#       self.attn_global.max_dist = globe_max_dist

#       local_out = self.slid_win_attention(local, win_size, span_len, span_scale)
#       combined = torch.cat([local_out.permute(1, 0, 2), globe_out.permute(1, 0, 2)], dim=-1)
#       x = self.projection(combined)
#       return x
  
#   def slid_win_attention(self, x, win_size, span_len, span_scale):
#       batch_size, seq_len, n_state = x.size()
#       out = torch.zeros_like(x, device=x.device)

#       for i in range(0, seq_len, win_size):
#           end = min(i + win_size, seq_len)
#           query = x[:, i:end, :]
#           start = max(0, i - span_len + win_size)
#           key = x[:, start:i + span_len, :]
#           value = x[:, start:i + span_len, :]

#           attn_out, _ = self.attn_local(query, key, value, span_scale)
#           x[:, i:end, :] = attn_out

#       return x
  
# class ResidualAttentionBlock(nn.Module):

#   def __init__(self, n_ctx: int, n_state: int, n_head: int, n_layer: int, max_dist: int, base: float, hybrid: bool = False, cross_att: bool = False):
#       super().__init__()
#       self.cross_att = cross_att
#       self.hybrid_ = hybrid

#       if hybrid:
#           self.attn = HybridAttention(n_state, n_head, max_dist)
#       else:
#           self.attn = MultiheadAttention(n_state, n_head, max_dist)
#       self.attn_ln = nn.LayerNorm(n_state)

#       if cross_att:
#           if hybrid:
#                self.cross_attn = HybridAttention(n_state, n_head, max_dist)
#           else:
#               self.cross_attn = MultiheadAttention(n_state, n_head, max_dist)
#           self.cross_attn_ln = nn.LayerNorm(n_state)


#       n_mlp = n_state * 4
#       self.mlp = nn.Sequential(
#           nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
#       )
#       self.mlp_ln = nn.LayerNorm(n_state)

#   def forward(self, x: torch.Tensor, xa: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, kv_cache=None) -> torch.Tensor:
#       x = x + self.attn(self.attn_ln(x), mask=mask)[0]

#       if self.cross_att:
#           x = x + self.cross_attn(self.cross_attn_ln(x), xa, mask=mask)[0]

#       x = x + self.mlp(self.mlp_ln(x))
#       return x




class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer, max_dist, base) -> None:
        super().__init__()
        self.h_dim = n_state // n_head
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        
        
        self.sinusoidal_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state)
        self.givens_rotary = CombinedRotaryEmbedding(n_state, n_head, base)

        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_ctx, n_state, n_head, n_layer, max_dist, base) for _ in range(n_layer)])

        self.ln_post = LayerNorm(n_state)

    def forward(self, x: torch.Tensor):
        x = F.gelu(input=self.conv1(x))
        x = F.gelu(input=self.conv2(x))
        x = x.permute(0, 2, 1)
        rot = self.givens_rotary(x)
        pos = self.sinusoidal_embedding(torch.arange(end=x.size(1))).unsqueeze(0)
        x = rot + pos
        x = x.to(x.dtype)       
        for block in self.blocks:
                  x = block(x)
        x = self.ln_post(x)      
        # print(f"Encoder : {x.shape}")
        return x




class RelativePositionalEmbedding(nn.Module):
    def __init__(self, max_dist, emb_dim):
        super().__init__()
        self.max_distance = max_dist
        self.embeddings = nn.Embedding(2 * max_dist + 1, emb_dim)

    def forward(self, seq_len_q, seq_len_k):
        relative_positions = (torch.arange(seq_len_q)[:, None] - torch.arange(seq_len_k)[None, :]).to(self.embeddings.weight.device)
        clipped_positions = torch.clamp(relative_positions, -self.max_distance, self.max_distance)
        final_positions = clipped_positions + self.max_distance
        return self.embeddings(final_positions.long())




class TextDecoder(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer, max_dist, base):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        
        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_ctx, n_state, n_head, n_layer, max_dist, base) for _ in range(n_layer)])
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):     
        
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
       
        x = (self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]])
        x = x.to(xa.dtype)
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
    
        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        # print(f"Decoder = {logits.shape}")  
        return logits




# class TextDecoder(nn.Module):
#     def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, max_dist: int, base: float, max_relative_distance, embedding_dim):
#         super().__init__()
        
#         # self.relative_positional_embeddings = RelativePositionalEmbedding(max_relative_distance, embedding_dim)
#         self.token_embedding = nn.Embedding(n_vocab, n_state)
#         self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        
#         self.blocks = nn.ModuleList([ResidualAttentionBlock(n_ctx, n_state, n_head, n_layer, max_dist, base) for _ in range(n_layer)])
#         self.ln = LayerNorm(n_state)

#         mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
#         self.register_buffer("mask", mask, persistent=False)

#     def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):     
#         offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
#         # x = self.token_embedding(x)
#         # for block in self.blocks:
#         #     relative_embeddings = self.relative_positional_embeddings(x.shape[1], xa.shape[1])
#         #     x = block(x, xa, mask=self.mask, kv_cache=kv_cache, relative_positional_embeddings=relative_embeddings) 
        
        
#         x = (self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]])
#         x = x.to(xa.dtype)
#         for block in self.blocks:
#             x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
       

#         x = self.ln(x)
#         logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
#         print(f"Decoder = {logits.shape}")  
#         return logits
    
#         # attn_out = self.qkv_attention(q, k, v, relative_positional_embeddings=relative_positional_embeddings)



class EchoConfig(PretrainedConfig):
    model_type = "Echo"
    def __init__(
        self,
        base=10000,
        bos_token_id=50257,
        decoder_start_token_id=50258,
        eos_token_id=50257,
        init_std=0.03,
        max_dist=128,
        n_audio_ctx=1500,
        n_audio_head=16,
        n_audio_layer=24,
        n_audio_state=1024,
        n_mels=128,
        n_text_ctx=448,
        n_text_head=16,
        n_text_layer=16,
        n_text_state=1024,
        pad_token_id=50257,
        unk_token_id=50257,
        n_vocab=51865,
        hybrid = False, 
        cross_att = False,
        
        **kwargs,
    ):
        super(EchoConfig, self).__init__(**kwargs)
        self.base = base
        self.bos_token_id = bos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id
        self.init_std = init_std
        self.max_dist = max_dist
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_audio_state = n_audio_state
        self.n_mels = n_mels
        self.n_text_ctx = n_text_ctx
        self.n_text_head = n_text_head
        self.n_text_layer = n_text_layer
        self.n_text_state = n_text_state
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.n_vocab = n_vocab
        self.hybrid = hybrid, 
        self.cross_att = cross_att,

class Echo(PreTrainedModel):
    config_class = EchoConfig

    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.config = config

        self.encoder = AudioEncoder(
            n_mels=self.config.n_mels,
            n_state=self.config.n_audio_state, 
            n_head=self.config.n_audio_head,
            n_layer=self.config.n_audio_layer,
            n_ctx=self.config.n_audio_ctx,
            max_dist=self.config.max_dist,
            base = self.config.base,
            # hybrid = self.config.hybrid, 
            # cross_att = self.config.cross_att,
        )

        self.decoder = TextDecoder(
            n_vocab=self.config.n_vocab,
            n_state=self.config.n_text_state, 
            n_head=self.config.n_text_head,
            n_layer=self.config.n_text_layer,
            n_ctx=self.config.n_text_ctx,
            max_dist=self.config.max_dist,
            base = self.config.base,
            # hybrid = self.config.hybrid, 
            # cross_att = self.config.cross_att,

        )

        self.reset_parameters()
        
        all_heads = torch.zeros(self.config.n_text_layer, self.config.n_text_head, dtype=torch.bool) 
        all_heads[self.config.n_text_layer // 2:] = True 
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

        self.base = self.config.base
        self.max_dist= self.config.max_dist
        self.adjust_counter = 0
        self.best_loss = float('inf')
        self.kv_cache = {}

    def update_dist(self, new_dist):
        self.new_dist = new_dist
        for name, module in self.encoder.named_modules():
            if isinstance(module, MultiheadAttention):
                module.update_dist(self.new_dist)

    def adjust_max_dist(self, loss, step_size=1, threshold=0.99975):
        if self.adjust_counter % 25 == 0:
            if loss < self.best_loss:
                potential_new_dist = self.max_dist + step_size
            else:
                potential_new_dist = max(1, self.max_dist - step_size)
            if abs(potential_new_dist - self.max_dist) >= threshold:
                new_dist = potential_new_dist
                self.update_dist(new_dist)
                self.max_dist = new_dist
                self.best_loss = loss

        self.adjust_counter += 1
        return self.max_dist

    def adjust_base(self, loss, factor=1.0025):
        if self.adjust_counter % 25 == 0:
            if loss < self.best_loss:
                new_base = self.base * factor
            else:
                new_base = self.base / factor
            self.update_base(new_base)
            self.base = new_base
            self.best_loss = loss
   
        self.adjust_counter += 1
        return self.base

    def update_base(self, new_base):
        self.new_base=new_base
        for name, module in self.encoder.named_modules():
            if isinstance(module, (CombinedRotaryEmbedding)):
                module.update_base(self.new_base)
            
    def print_update(self):
        self.adjust_counter += 1
        if self.adjust_counter % 25 == 0:
            print(f"{self.adjust_counter}: Loss: {self.best_loss:.4f}  Base: {self.base:.4f}, Distance: {self.max_dist}")
            
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
            loss = loss_fct(logits.view(-1, self.config.n_vocab), labels.view(-1))

            self.adjust_base(loss.item())
            self.adjust_max_dist(loss.item())
            self.print_update()  

        return {"loss": loss, "logits": logits}

    def reset_parameters(self):
        for name, module in self.encoder.named_modules():
            if isinstance(module, CombinedRotaryEmbedding):
                module.reset_parameters()
        self.encoder.apply(self._init_weights)
        
    def _initialize_weights(self, module):
            nn.init.normal_(self.decoder.token_embedding.weight, mean=0.0, std=self.config.init_std)
            if hasattr(self.decoder.positional_embedding, 'weight'):
                nn.init.normal_(self.decoder.positional_embedding.weight, mean=0.0, std=self.config.init_std) # type: ignore
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

#

config = EchoConfig(
    base=10000,
    bos_token_id=50257,
    decoder_start_token_id=50258,
    eos_token_id=50257,
    init_std=0.02,
    max_dist=128,
    n_audio_ctx=1500,
    n_audio_head=16,
    n_audio_layer=24,
    n_audio_state=1024,
    n_mels=80,
    n_text_ctx=448,
    n_text_head=16,
    n_text_layer=16,
    n_text_state=1024,
    pad_token_id=50257,
    n_vocab=51865,
    # hybrid = False, 
    # cross_att = False,

)

model = Echo(config).to(device)
model.apply_initialization(module)

from datetime import datetime
log_dir = os.path.join('./output/', datetime.now().strftime('%Y-%m-%d_%H'))
os.makedirs(log_dir, exist_ok=True)

name="/echo_test/"
config.save_pretrained(log_dir+name)
model.save_pretrained(log_dir+name, safe_serialization=False)
torch.save(model.state_dict(), log_dir+name+"state_dict.pt")
model = Echo.from_pretrained(pretrained_model_name_or_path=(log_dir+name)).to(device)

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
            self.eval_loss = metrics.get('eval_loss')

            if state.global_step % self.log_every_n_steps == 0:
                for key, value in metrics.items():
                    if key.startswith("eval_"):
                        self.tb_writer.add_scalar(key, value, state.global_step)

        if self.predictions is not None and self.label_ids is not None:
            pred_str = self.tokenizer.batch_decode(self.predictions, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(self.label_ids, skip_special_tokens=True)

            if state.global_step % self.log_every_n_steps == 0:
                sample_index = 0
                self.tb_writer.add_text(f"Prediction", pred_str[sample_index], state.global_step)
                self.tb_writer.add_text(f"Label", label_str[sample_index], state.global_step)
                print(f"Evaluation: - Step {state.global_step} - Loss: {self.eval_loss:.4f}")
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

train = load_dataset("fixie-ai/librispeech_asr", "clean", split="train.100", streaming=True, trust_remote_code=True).map(prepare_dataset).select_columns(["input_features", "labels"]).take(1000)

test = load_dataset("fixie-ai/librispeech_asr", "clean", split="test", streaming=True, trust_remote_code=True).map(prepare_dataset).select_columns(["input_features", "labels"]).take(100)


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer, feature_extractor=feature_extractor)

metric = evaluate.load(path="wer")
tb_writer = SummaryWriter(log_dir=log_dir)
metrics_callback = MetricsCallback(tb_writer=tb_writer, tokenizer=tokenizer, metric=metric, log_every_n_steps=1)
compute_metrics = create_compute_metrics(callback_instance=metrics_callback)

### 
training_args = Seq2SeqTrainingArguments(
    output_dir=log_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    tf32=True,
    bf16=True,
    evaluation_strategy="steps",
    # warmup_steps = 100,
    max_steps=10,
    # save_steps=500,
    eval_steps=2,
    logging_steps=1,
    logging_dir=log_dir + "/logs_hf",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    optim="adafactor",
    weight_decay=0.0025,
    disable_tqdm=False,
    # save_total_limit=2,
    # save_strategy="steps",
    remove_unused_columns=False,
    label_names=["labels"],
    gradient_checkpointing=False,
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
    callbacks=[metrics_callback]
)





trainer.train(resume_from_checkpoint=False)
eval_results = trainer.evaluate()
eval_results
# model.save_pretrained(name+"_a", safe_serialization=False)




