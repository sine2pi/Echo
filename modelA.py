
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
    def __init__(self, base, n_state, n_head):
        super().__init__()
        self.base = base
        self.n_state = n_state
        self.n_head = n_head
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
        self.update_base(new_base) 

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
    def __init__(self, n_ctx, n_state, checkpointing=False):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.checkpointing = checkpointing

        position = torch.arange(start=0, end=self.n_ctx, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(input=torch.arange(start=0, end=self.n_state, step=2).float() * -(math.log(10000.0) / self.n_state))
        features = torch.zeros(self.n_ctx, self.n_state)
        features[:, 0::2] = torch.sin(input=position * div_term)
        features[:, 1::2] = torch.cos(input=position * div_term)
        self.register_buffer('my_big_toe', tensor=features)
        self.positional_embeddings = nn.Parameter(self.my_big_toe.clone())

    def forward(self, positions):
        position_embeddings = self.positional_embeddings[positions]
        return position_embeddings

class MultiheadAttention(nn.Module):
    def __init__(self, base, n_state, n_head, max_dist):
        super().__init__()
        self.base = base
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

        self.combined_rotary = CombinedRotaryEmbedding(base, n_state, n_head)
        self.kv_cache = {}
        
        self.positional_scaling = nn.Parameter(torch.ones(1))
        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_dist) - 1, self.n_head))).to(device)
        self.inv_freq = nn.Parameter(data=1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)))

    def update_base(self, new_base):
        if new_base is not None and new_base != self.base:
            self.base = new_base
            inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
            self.inv_freq.data.copy_(inv_freq)
            self.combined_rotary.update_base(self.base)

    def update_dist(self, new_dist):
        if new_dist is not None and new_dist != new_dist:
            self.max_dist = new_dist
            rel_pos_bias = nn.Parameter(torch.zeros((2 * int(self.max_dist) - 1, self.n_head)))
            self.rel_pos_bias.data.copy_(rel_pos_bias)
        
    def forward(self, x: Tensor, xa: Optional[Tensor] = None, 
                mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None, new_dist=None, new_base=None):
        
        self.update_base(new_base) 
        self.update_dist(new_dist)

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
   
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple     [torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        seq_len_q = q.size(2)
        seq_len_k = k.size(2)
        positions = torch.arange(end=seq_len_q, device=q.device).unsqueeze(dim=1) - torch.arange(end=seq_len_k, device=q.device).unsqueeze(dim=0)
        positions = positions.clamp(min=-self.max_dist + 1, max=self.max_dist - 1) + self.max_dist - 1
        rel_bias = self.rel_pos_bias[positions]
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  

        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale 
        attn_scores = attn_scores + rel_bias

        a = scaled_dot_product_attention(q, k, v, attn_mask = attn_scores , is_causal=mask is not None and n_ctx > 1) 
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        attn_scores = attn_scores.reshape(n_batch, n_ctx, -1)[:,:,:self.n_state]

        return out, attn_scores

class ResidualAttentionBlock(nn.Module):
    """Residual attention block with only standard multi-head attention."""

    def __init__(self, base: int, n_state: int, n_head: int, max_dist: int):
        super().__init__()

        self.attn = MultiheadAttention(base, n_state, n_head, max_dist)
        self.attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

class AudioEncoder(nn.Module):
    def __init__(self, base, n_mels: int, n_state: int, n_head: int, n_layer: int, n_ctx: int, max_dist: int):
        super().__init__()
        self.h_dim = n_state // n_head
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        self.combined_rotary = CombinedRotaryEmbedding(base, n_state,  n_head)

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(base, n_state, n_head, max_dist) for _ in range(n_layer)
        ])

        self.ln_post = LayerNorm(n_state)

    def update_base(self, new_base):
        if new_base is not None:
            self.combined_rotary.update_base(new_base)

    def forward(self, x, new_base=None):
        self.update_base(new_base)
        x = F.gelu(input=self.conv1(x))
        x = F.gelu(input=self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.combined_rotary(x) 
        for block in self.blocks:
                  x = block(x)
        x = self.ln_post(x)
        return x

class TextDecoder(nn.Module):
    def __init__(self, base, n_vocab, n_state, n_head, n_layer, n_ctx, max_dist):  
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state)

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(base, n_state, n_head, max_dist) for _ in range(n_layer)])
        
        self.ln_post = LayerNorm(n_state)
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(value=-np.inf).triu_(diagonal=1)
        self.register_buffer(name="mask", tensor=mask, persistent=False)

    def forward(self, x, xa, kv_cache=None, new_base=None):

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        positions = torch.arange(end=x.shape[1], device=x.device) + offset
        pos = self.positional_embedding(positions).unsqueeze(0)
        tok = self.token_embedding(x)
        # rot = self.combined_rotary(x)
        x = pos + tok
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, kv_cache)
        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(dtype=x.dtype), 0, 1)).float()
        return logits

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

class Echo(PreTrainedModel):
    config_class = EchoConfig

    def __init__(self, config: EchoConfig):
        super().__init__(config)
        self.config = config

        self.encoder = AudioEncoder(
            base=self.config.base,
            n_state=self.config.n_audio_state, 
            n_head=self.config.n_audio_head,
            n_mels=self.config.n_mels,
            n_layer=self.config.n_audio_layer,
            n_ctx=self.config.n_audio_ctx,
            max_dist=self.config.max_dist,
        )

        self.decoder = TextDecoder(
            base=self.config.base,
            n_state=self.config.n_text_state, 
            n_head=self.config.n_text_head,
            n_vocab=self.config.n_vocab,
            n_layer=self.config.n_text_layer,
            n_ctx=self.config.n_text_ctx,
            max_dist=self.config.max_dist,

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
        for name, module in self.decoder.named_modules():
            if isinstance(module, MultiheadAttention):
                module.update_dist(self.new_dist)

    def adjust_max_dist(self, loss, step_size=1, threshold=0.0005):
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
            if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding, AudioEncoder)):
                module.update_base(self.new_base)
        for name, module in self.decoder.named_modules():
            if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding)):
                module.update_base(self.new_base)

    def print_update(self):
        self.adjust_counter += 1
        if self.adjust_counter % 20 == 0:
            print(f"{self.adjust_counter}: Loss: {self.best_loss}  Base: {self.base}, Distance: {self.max_dist}")
            
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
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id)

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

config = EchoConfig(
    base=10000,
    bos_token_id=50257,
    decoder_start_token_id=50258,
    eos_token_id=50257,
    init_std=0.02,
    max_dist=128,
    n_audio_ctx=1500,
    n_audio_head=16,
    n_audio_layer=20,
    n_audio_state=1024,
    n_mels=128,
    n_text_ctx=448,
    n_text_head=16,
    n_text_layer=16,
    n_text_state=1024,
    pad_token_id=50257,
    unk_token_id=50257,
    n_vocab=51865,
    )

model = Echo(config=config).to('cuda')
model.apply_initialization(module)

name="./echo/"
config.save_pretrained(name)
model.save_pretrained(name)
torch.save(model.state_dict(), name+"state_dict.pt")
model = Echo.from_pretrained(name).to('cuda')
