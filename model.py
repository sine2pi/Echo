

import base64, json
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
from typing import Dict, Optional, Tuple, Union, List, Any, Iterable
from transformers.modeling_utils import PreTrainedModel
from dataclasses import dataclass
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments, PretrainedConfig, TrainerCallback,
    WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizerFast, WhisperTokenizer
)
from torch.nn.functional import scaled_dot_product_attention
import evaluate
from evaluate import module
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from datasets import load_dataset, Dataset, concatenate_datasets, IterableDatasetDict, Audio
# from torch.nn.functional import scaled_dot_product_attention

import warnings
transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings(action="ignore")
warnings.warn=lambda *args, **kwargs: None
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype=torch.float32
dd=(dtype, device)
torch.set_default_dtype(dtype)



class CustomEmbedding(nn.Module):
    def __init__(self, initial_value, learnable=True):
        super(CustomEmbedding, self).__init__()
        if learnable:
            self.value=nn.Parameter(torch.tensor(initial_value))
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
    def __init__(self, n_state: int, n_head: int, n_freq: float,
                 theta_scale_learnable: bool = True,
                 n_rots_scale_learnable: bool = True,
                 r_matrix_learnable: bool = False,
                 inv_freq_learnable: bool = True):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.n_freq = n_freq
        assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"
        self.h_dim = self.n_state // self.n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"
        self.n_rots = ((n_state // n_head) // 2)

        # --- Learnable Parameters ---
        self.thetas = nn.Parameter(torch.zeros(self.n_rots))
        self.r_pairs = nn.Parameter(data=torch.rand(self.n_rots, 2) * self.h_dim)

        # --- Scaling Parameters ---
        self.theta_scale = nn.Parameter(torch.ones(1), requires_grad=theta_scale_learnable)
        self.n_rots_scale = nn.Parameter(torch.ones(1), requires_grad=n_rots_scale_learnable)

        # --- R Matrix ---
        self.r_matrix = nn.Parameter(torch.eye(n=self.h_dim), requires_grad=r_matrix_learnable)

        # --- Frequency Parameters for RoPE ---
        inv_freq_data = 1.0 / (self.n_freq ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
        self.inv_freq = nn.Parameter(inv_freq_data, requires_grad=inv_freq_learnable)

        # --- Regularization ---
        self.orthogonal_reg_weight = 0.01  

    def givens_r_matrix(self, n_state, i, j, theta):
        G = torch.eye(n_state).to(theta.device)
        G[i, i] = math.cos(theta)
        G[i, j] = -math.sin(theta)
        G[j, i] = math.sin(theta)
        G[j, j] = math.cos(theta)
        return G

    def update_base(self, new_base):
        if new_base is not None and new_base != self.n_freq:
            self.n_freq = new_base
            inv_freq = 1.0 / (self.n_freq ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
            self.inv_freq.data.copy_(inv_freq)

    def reset_parameters(self):
        nn.init.orthogonal_(tensor=self.r_matrix)
        nn.init.zeros_(tensor=self.thetas)

    def orthogonal_regularization_term(self):
        """Calculates the orthogonal regularization term for r_matrix."""
        loss = torch.tensor(0.0, device=self.r_matrix.device) 
        if self.r_matrix.requires_grad: 
            product = torch.matmul(self.r_matrix, self.r_matrix.t())
            identity = torch.eye(self.r_matrix.size(0)).to(self.r_matrix.device)
            loss = ((product - identity) ** 2).sum()
        return self.orthogonal_reg_weight * loss

    def forward(self, x):
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")

        batch_size, seq_len, *rest = x.size()

        if x.dim() == 3:
            n_state = rest[0]
            if n_state != self.n_head * self.h_dim:
                raise ValueError(
                    f"Expected n_state ({n_state}) to be compatible with n_head ({self.n_head}) * h_dim ({self.h_dim}={self.n_head * self.h_dim})")
        else:
            n_head, h_dim = rest
            if n_head != self.n_head or h_dim != self.h_dim:
                raise ValueError(
                    f"For 4D input, expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")

        x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
        x = x.reshape(-1, self.h_dim)
        adjusted_n_rots = int(torch.round(self.n_rots_scale * self.n_rots))

        for k in range(adjusted_n_rots):
            i, j = self.r_pairs[k].long()
            theta = self.thetas[k] * self.theta_scale
            G = self.givens_r_matrix(n_state=self.h_dim, i=i, j=j, theta=theta)
            x = torch.matmul(input=x, other=G)

        x = torch.matmul(input=x, other=self.r_matrix)
        x = x.view(batch_size, seq_len, self.n_head, self.h_dim)

        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(end=seq_len, device=x.device),
                                     self.inv_freq.to(device=x.device))
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]

        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat(tensors=[x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x = x.view(batch_size, seq_len, self.n_state)

        return x
#     loss += embedding_layer.orthogonal_regularization_term()

class LearnedSinusoidalEmbeddings(nn.Module):
    def __init__(self, n_ctx: int, n_state: int):
        super().__init__()

        position=torch.arange(start=0, end=n_ctx, dtype=dtype).unsqueeze(dim=1)
        div_term=torch.exp(input=torch.arange(start=0, end=n_state, step=2, dtype=dtype) * -(math.log(10000.0) / n_state))
        features=torch.zeros(n_ctx, n_state)
        features[:, 0::2]=torch.sin(position * div_term)
        features[:, 1::2]=torch.cos(position* div_term)
        self.register_buffer('my_big_toe', features)
        self.positional_embeddings=nn.Parameter(self.my_big_toe.clone())

    def forward(self, positions):
        position_embeddings=self.positional_embeddings[positions]
        position_embeddings=torch.nn.functional.normalize(position_embeddings, p=2, dim=-1)
        return position_embeddings


class MultiheadAttention(nn.Module):
    use_sdpa = True
    def __init__(self, n_state, n_head, n_dist, n_freq):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.n_dist = n_dist
        assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"
        self.h_dim = self.n_state // self.n_head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"
        self.givens_rotary=CombinedRotaryEmbedding(n_state, n_head, n_freq)
        self.query = nn.Linear(self.n_state, self.n_state)
        self.key = nn.Linear(self.n_state, self.n_state, bias=False)
        self.value = nn.Linear(self.n_state, self.n_state)
        self.out = nn.Linear(self.n_state, self.n_state)
        
        self.kv_cache = {}
                           
    def forward(self, x, xa = None,  mask = None, kv_cache = None, r_bias=None):

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:

            k = kv_cache[self.key]
            v = kv_cache[self.value]
            
        q = self.givens_rotary(q)# * self.positional_scaling
        k = self.givens_rotary(k)# * self.positional_scaling

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk
   
    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask):
        
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)


        if MultiheadAttention.use_sdpa:
            a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and n_ctx > 1 )

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
    

# class RelativePositionalBias(nn.Module):
#     def __init__(self, max_dist: int, n_head: int):
#         super().__init__()
#         self.max_dist = max_dist
#         self.n_head = n_head
#         self.pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_dist) - 1, self.n_head)))

#     def forward(self, seq_len_q, seq_len_k, device):
#         positions = torch.arange(end=seq_len_q, device=device).unsqueeze(dim=1) - torch.arange(end=seq_len_k, device=device).unsqueeze(dim=0)
#         positions = positions.clamp(min=-self.max_dist + 1, max=self.max_dist - 1) + self.max_dist - 1
#         rel_bias = self.pos_bias[positions]
#         rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_head, seq_len_q, seq_len_k)
#         return rel_bias

#     def update_dist(self, new_dist, device):
#         if new_dist is not None and new_dist != self.max_dist:
#             self.max_dist = new_dist
#             self.pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_dist) - 1, self.n_head))).to(device)

# class MultiheadAttention(nn.Module):
#     use_sdpa = True

#     def __init__(self, n_state: int, n_head: int, n_freq: int):
#         super().__init__()
#         self.n_state = n_state
#         self.n_head = n_head

#         assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"
#         self.h_dim = self.n_state // self.n_head
#         assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

#         self.query = nn.Linear(self.n_state, self.n_state)
#         self.key = nn.Linear(self.n_state, self.n_state, bias=False)
#         self.value = nn.Linear(self.n_state, self.n_state)
#         self.out = nn.Linear(self.n_state, self.n_state)
#         self.givens_rotary = CombinedRotaryEmbedding(n_state, n_head, n_freq)
#         self.kv_cache = {}

#         self.positional_scaling = nn.Parameter(torch.ones(1))

#     def forward(self, x: Tensor, xa: Optional[Tensor] = None,
#                 mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None,
#                 rel_pos_bias: Optional[Tensor] = None) -> tuple[Any, Tensor | None]:

#         q = self.query(x)

#         if kv_cache is None or xa is None or self.key not in kv_cache:
#             k = self.key(x if xa is None else xa)
#             v = self.value(x if xa is None else xa)
#         else:
#             k = kv_cache[self.key]
#             v = kv_cache[self.value]

#         q = self.givens_rotary(q)
#         k = self.givens_rotary(k)
        
#         wv, qk = self.qkv_attention(q, k, v, mask, rel_pos_bias)
#         return self.out(wv), qk

#     def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
#                       rel_pos_bias: Optional[Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         n_batch, n_ctx, n_state = q.shape
#         scale = (n_state // self.n_head) ** -0.25
#         q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
#         k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
#         v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

#         attn_scores = torch.matmul(q, k.transpose(-2, -1))

#         if rel_pos_bias is not None:
#             attn_scores = attn_scores + rel_pos_bias

#         attn_scores = attn_scores * scale

#         if MultiheadAttention.use_sdpa:
#             a = scaled_dot_product_attention(q, k, v, attn_mask=attn_scores, is_causal=mask is not None and n_ctx > 1)
#             out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
#             qk = None
#         else:
#             qk = (q * scale) @ (k * scale).transpose(-1, -2) # moved the scaling here
#             if mask is not None:
#                 qk = qk + mask[:n_ctx, :n_ctx]
#             qk = qk.float()

#             w = F.softmax(qk, dim=-1).to(q.dtype)
#             out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
#             qk = qk.detach()

#         return out, qk


class ResidualAttentionBlock(nn.Module):

    def __init__(self, n_state, n_head, n_dist, n_freq):
        super().__init__()

        self.attn=MultiheadAttention(n_state, n_head, n_dist, n_freq)
        self.attn_ln=nn.LayerNorm(n_state)
        
        n_mlp=n_state * 4
        self.mlp=nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln=nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,

    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]

        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer, n_dist, n_freq, hybrid_attn, cross_attn) -> None:
        super().__init__()
        self.n_head = n_head
        self.h_dim = n_state // n_head
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.n_state = n_state

        self.givens_rotary = CombinedRotaryEmbedding(n_state, n_head, n_freq)

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, n_dist, n_freq)
            for _ in range(n_layer)
        ])
        self.ln_post = nn.LayerNorm(n_state)


    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.givens_rotary(x)
        x = x.to(x.dtype)

        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)

        return x

class TextDecoder(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer, n_dist, n_freq, hybrid_attn, cross_attn):
        super().__init__()

        self.sin_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state)
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, n_dist, n_freq)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]])
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class EchoConfig(PretrainedConfig):
       
    def __init__(
        self,
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_head=16,
        n_audio_layer=24,
        n_audio_state=1024,
        n_vocab=51865,
        n_text_ctx=448,
        n_text_head=16,
        n_text_layer=16,
        n_text_state=1024,
        n_dist=128,
        n_freq=10000,
        hybrid_attn=False,
        cross_attn=False,
        bos_token_id=50257,
        eos_token_id=50257,
        pad_token_id=50257,
        decoder_start_token_id=50258,       
        **kwargs,
        
    ):
        super(EchoConfig, self).__init__(**kwargs)
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_audio_state = n_audio_state
        self.n_vocab = n_vocab
        self.n_text_ctx = n_text_ctx
        self.n_text_head = n_text_head
        self.n_text_layer = n_text_layer
        self.n_text_state = n_text_state
        self.n_dist = n_dist
        self.n_freq = n_freq
        self.cross_attn = cross_attn
        self.hybrid_attn = hybrid_attn
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id


class Echo(PreTrainedModel):
    config_class = EchoConfig
    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.config = config
        
        self.encoder=AudioEncoder(
            self.config.n_mels,
            self.config.n_audio_ctx,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            self.config.n_dist,
            self.config.n_freq,
            self.config.hybrid_attn, 
            self.config.cross_attn,
        )

        self.decoder=TextDecoder(
            self.config.n_vocab,
            self.config.n_text_ctx,
            self.config.n_text_state, 
            self.config.n_text_head,
            self.config.n_text_layer,
            self.config.n_dist,
            self.config.n_freq,
            self.config.hybrid_attn, 
            self.config.cross_attn,
        )
        
        self.encoder.givens_rotary.reset_parameters()
        
        self.init_std=0.02
        all_heads=torch.zeros(self.config.n_text_layer, self.config.n_text_head, dtype=torch.bool) 
        all_heads[self.config.n_text_layer // 2:]=True 
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

        self.n_freq=self.config.n_freq
        self.n_dist=self.config.n_dist
        self.adjust_counter=0
        self.best_loss=float('inf')
        self.kv_cache={}
   
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
                        new_base=self.n_freq   * factor
                    else:
                        new_base=self.n_freq   / factor
                    self.update_base(new_base)
                    self.n_freq=new_base
                    self.best_loss=loss
                self.adjust_counter += 1
                return self.n_freq
            
    def update_base(self, new_base):
        self.new_base=new_base
        for name, module in self.encoder.named_modules():
            if isinstance(module, (CombinedRotaryEmbedding)):
                module.update_base(self.new_base)
            
    def print_update(self):
        self.adjust_counter += 1
        if self.adjust_counter % 25 == 0:
            print(f"{self.adjust_counter}: Loss: {self.best_loss:.4f}  Base: {self.n_freq:.4f}, Distance: {self.n_dist}")
            
    @staticmethod
    def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
        shifted_input_ids=input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:]=input_ids[:, :-1].clone() 
        shifted_input_ids[:, 0]=decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward(self, input_features, labels=None, dec_input_ids=None):
        if labels is not None:
            if dec_input_ids is None:
                dec_input_ids=self.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        encoded_features=self.encoder(input_features).to(device)  
        logits=self.decoder(dec_input_ids, encoded_features)

        loss=None
        if labels is not None:
            loss_fct=nn.CrossEntropyLoss(ignore_index=-100)
            labels=labels.to(logits.device).long()
            loss=loss_fct(logits.view(-1, self.config.n_vocab), labels.view(-1))

            self.adjust_base(loss.item())
            # self.print_update()
            # self.adjust_n_dist(loss.item())

        return {"loss": loss, "logits": logits}

    def reset_parameters(self):
        for name, module in self.encoder.named_modules():
            if isinstance(module, CombinedRotaryEmbedding):
                module.reset_parameters()
        self.encoder.apply(self._init_weights)
        
    def _initialize_weights(self, module):
            nn.init.normal_(self.decoder.token_embedding.weight, mean=0.0, std=0.02)
            if hasattr(self.decoder.positional_embedding, 'weight'):
                nn.init.normal_(self.decoder.positional_embedding, mean=0.0, std=0.02)
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

    def install_kv_cache_hooks(self, cache=None):
        cache={**cache} if cache is not None else {}
        hooks=[]

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.config.n_text_ctx:
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
            # if step % 1 == 0:
            #     print(f"Step {step+1}: Generated {generated_sequences.size(1)} tokens so far.")
        #     if torch.all(next_token_id == self.config.eos_token_id):
        #         break
        # print("Generation complete!")
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
            
            beam_indices = beam_tokens // self.config.n_vocab
            next_token_id = beam_tokens % self.config.n_vocab
            decoder_input_ids = torch.cat([decoder_input_ids[beam_indices], next_token_id.unsqueeze(-1)], dim=-1)
            
            if torch.all(next_token_id == self.config.eos_token_id):
                break
        
        return decoder_input_ids

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=checkpoint):
        self.checkpointing=enable
        self.gradient_checkpointing_func=gradient_checkpointing_func
        for module in self.modules():
            if hasattr(module, 'checkpointing'):
                module.checkpointing
                module.gradient_checkpointing_func=gradient_checkpointing_func

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs={"use_reentrant": True}
        gradient_checkpointing_func=functools.partial(checkpoint, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        
    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f)
    
    @classmethod
    def from_pretrained(cls, load_directory):
        config_path = os.path.join(load_directory, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = EchoConfig(**config_dict) 
        model = cls(config)

        model_path = os.path.join(load_directory, 'pytorch_model.bin')
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        
        return model

    def to_dict(self):
        return {
            "architectures": ["Echo"],
            "model_type": "Echo",
            "n_mels": self.n_mels,
            "n_audio_ctx": self.n_audio_ctx,
            "n_audio_head": self.n_audio_head,
            "n_audio_layer": self.n_audio_layer,
            "n_audio_state": self.n_audio_state,
            "n_vocab": self.n_vocab,
            "n_text_ctx": self.n_text_ctx,
            "n_text_head": self.n_text_head,
            "n_text_layer": self.n_text_layer,
            "n_text_state": self.n_text_state,
            "n_dist": self.n_dist,
            "n_freq": self.n_freq,
            "cross_attn": self.cross_attn,
            "hybrid_attn": self.hybrid_attn,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
            "torch_dtype": self.float32,
            "use_bfloat16": True,
            "is_encoder_decoder": True,
        }


config = EchoConfig(
        model_type = "Echo",
        architectures = "Transformer",
        torch_dtype = torch.float32,
        n_mels = 80,
        n_audio_ctx = 1500,
        n_audio_head = 16,
        n_audio_layer = 20,
        n_audio_state = 1024,
        n_vocab = 51865,
        n_text_ctx = 448,
        n_text_head = 16,
        n_text_layer = 16,
        n_text_state = 1024,
        n_dist = 128,
        n_freq = 10000,
        bos_token_id = 50257,
        eos_token_id = 50257,
        pad_token_id = 50257,
        decoder_start_token_id = 50258,
        hybrid_attn = False, 
        cross_attn = False,
        ###
        use_bfloat16 = True,
        is_encoder_decoder = True,
        tie_word_embeddings = False,
        do_sample = True,
        tokenizer_class = "WhisperTokenizer"
        )

model=Echo(config=config).to(device)
model.apply_initialization(module)


from datetime import datetime
log_dir=os.path.join('./output/', datetime.now().strftime('%Y-%m-%d_%H'))
os.makedirs(log_dir, exist_ok=True)

name="/echo_test/"
config.save_pretrained(log_dir+name)
model.save_pretrained(log_dir+name)
model = Echo.from_pretrained((log_dir+name)).to(device)



feature_extractor=WhisperFeatureExtractor.from_pretrained("openai/whisper-small", do_normalize=True, sampling_rate=16000, return_tensors="pt")
tokenizer=WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")
processor=WhisperProcessor.from_pretrained("openai/whisper-small")

class GradientClippingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.nn.utils.clip_grad_norm_(parameters=kwargs["model"].parameters(), max_norm=0.98)

metric=evaluate.load(path="wer")

def compute_metrics(eval_pred, tokenizer=tokenizer, metric=metric):

    pred_logits = eval_pred.predictions
    label_ids = eval_pred.label_ids
    pred_ids = pred_logits[0] if isinstance(pred_logits, tuple) else pred_logits
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    sample_index = 0
    print(f"Prediction: {pred_str[sample_index]}")
    print(f"Label: {label_str[sample_index]}")
    print("-" * 10)

    pred_flat = pred_ids.flatten()
    labels_flat = label_ids.flatten()
    mask = labels_flat != tokenizer.pad_token_id

    return {"wer": wer}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    tokenizer: Any
    feature_extractor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features=[{"input_features": feature["input_features"]} for feature in features]
        batch=feature_extractor.pad(input_features, return_tensors="pt")
        label_features=[{"input_ids": feature["labels"]} for feature in features]
        labels_batch=tokenizer.pad(label_features, return_tensors="pt")
        labels=labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == tokenizer.bos_token_id).all().cpu().item():
            labels=labels[:, 1:]
        batch["labels"]=labels
        return batch

def get_length_of_dataset(dataset):
    length=0
    for item in dataset:
        length += len(item["audio"]["array"]) / item["audio"]["sampling_rate"]
    return length / 3600

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


train=load_dataset("fixie-ai/librispeech_asr", "clean", split="train.100", streaming=True, trust_remote_code=True).map(prepare_dataset).select_columns(["input_features", "labels"])
train = train.shuffle(seed=32)

test=load_dataset("fixie-ai/librispeech_asr", "clean", split="test", streaming=True, trust_remote_code=True).map(prepare_dataset).select_columns(["input_features", "labels"]).take(100)


data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer, feature_extractor=feature_extractor)

metric=evaluate.load(path="wer")
tb_writer=SummaryWriter(log_dir=log_dir)


### 
training_args=Seq2SeqTrainingArguments(
    output_dir=log_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    tf32=True,
    bf16=True,
    learning_rate=3e-6,
    eval_strategy="steps",
    warmup_steps=100,
    max_steps=10000,
    save_steps=500,
    eval_steps=100,
    logging_steps=5,
    logging_dir=log_dir + "/logs_hf",
    report_to=["tensorboard"],
    push_to_hub=False,
    optim="adafactor",
    weight_decay=0.0025,
    disable_tqdm=False,
    save_total_limit=1,
    save_strategy="steps",
    remove_unused_columns=True,
    label_names=["labels"],
    # gradient_checkpointing=False,
    eval_on_start=True,
    max_grad_norm = 0.98,
    predict_with_generate=False,
    generation_max_length = 32,
    generation_num_beams = 2,
    # bf16_full_eval = True,
    # torch_empty_cache_steps = 10,
)

trainer=Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train,
    eval_dataset=test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    # callbacks=[metrics_callback]
)





trainer.train(resume_from_checkpoint=False)
eval_results=trainer.evaluate()
print(eval_results)
import tensorboard




# print(torch.cuda.memory_summary(device=None, abbreviated=False))


