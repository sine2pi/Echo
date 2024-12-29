
import base64, gzip, evaluate, math, os, sys, time
import gzip, neologdn
from transformers.modeling_utils import PreTrainedModel 
import collections
import copy
import functools
from evaluate import module
from functools import partial, wraps
from threading import Thread
import gc
import importlib.metadata
import inspect
import itertools
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
from huggingface_hub import PyTorchModelHubMixin
from datasets import IterableDatasetDict, Audio, load_dataset, load_from_disk
import numpy as np
import torch, transformers, warnings
from typing import Dict, Iterable, Optional, Tuple, Union, List, Any, Type
import torch.nn.functional as F
from torch import Tensor, nn
import torchaudio, torchaudio.transforms as T
from transformers import Seq2SeqTrainer, TrainerCallback, Seq2SeqTrainingArguments, WhisperTokenizer, WhisperForConditionalGeneration, WhisperConfig, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperTokenizerFast
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
torch.set_default_dtype(dtype)

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
        self.register_buffer('sinusoidal_features', tensor=features)

        self.positional_embeddings = nn.Parameter(data=self.sinusoidal_features.clone())

    def forward(self, positions):
        if self.checkpointing:
            position_embeddings = checkpoint(lambda x: self.positional_embeddings[x], positions)
        else:
            position_embeddings = self.positional_embeddings[positions]

        position_embeddings = torch.nn.functional.normalize(input=position_embeddings, p=2, dim=-1)
        return position_embeddings

class MultiHeadAttention(nn.Module):
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
        self.base = new_base 
        inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)) 
        self.register_buffer(name='inv_freq', tensor=inv_freq) 
        self.combined_rotary.update_base(new_base=new_base)

    def forward(self, x, xa = None, mask = None, kv_cache = None):
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
    
    def qkv_attention(self, q, k, v, mask = None):
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
    def __init__(self, n_state, n_head, window_size=40, alpha=0.1):
        super().__init__()
        self.local_attn = nn.MultiheadAttention(n_state, n_head)
        self.global_attn = nn.MultiheadAttention(n_state, n_head)
        self.ln_local = LayerNorm(num_features=n_state)
        self.ln_global = LayerNorm(num_features=n_state)
        self.window_size = window_size
        self.window_scale = nn.Parameter(torch.tensor(1.0))
        self.running_loss = None
        self.alpha = alpha

    def forward(self, x, loss=None):
        x_local = self.ln_local(x)
        x_global = self.ln_global(x)
        x_local = x_local.permute(1, 0, 2)
        x_global = x_global.permute(1, 0, 2)

        effective_window_size = self.adjust_window_size(loss)

        local_out = self.sliding_window_attention(x_local, effective_window_size)
        global_out, _ = self.global_attn(x_global, x_global, x_global)
        combined_out = local_out + global_out
        combined_out = combined_out.permute(1, 0, 2)
        return combined_out

    def adjust_window_size(self, loss, factor=1.05):
        
        if loss is not None:  
            if self.running_loss is None:
                self.running_loss = loss
            else:
                self.running_loss = self.alpha * loss + (1 - self.alpha) * self.running_loss

        threshold = self.running_loss
        if loss is not None and loss < threshold:
            adjusted_window_size = int(self.window_size * self.window_scale.item() * factor)
        else:
            adjusted_window_size = max(1, int(self.window_size * self.window_scale.item() / factor))
        return adjusted_window_size

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
    def __init__(self, n_state, n_head, cross_attention=True, max_rel_dist=1, checkpointing=False, use_hybrid_attention=False, window_size=1):
        super().__init__()

        self.attn = MultiHeadAttention(n_state=n_state, n_head=n_head)
        self.attn_ln = LayerNorm(num_features=n_state)
        self.checkpointing = checkpointing
        self.max_rel_dist = max_rel_dist
        self.cross_attention = cross_attention
        self.use_hybrid_attention = use_hybrid_attention

        if self.use_hybrid_attention:
            self.attn = HybridAttention(n_state=n_state, n_head=n_head, window_size=window_size)
            self.cross_attn = MultiHeadAttention(n_state=n_state, n_head=n_head)

        self.cross_attn = MultiHeadAttention(n_state=n_state, n_head=n_head) if cross_attention else None
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
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer, window_size = 1, max_rel_dist = 1, use_hybrid_attention = False, cross_attention=True, checkpointing=False, base=10000):
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
                                            checkpointing=checkpointing) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(num_features=n_state)

    def update_base(self, new_base):
        self.combined_rotary.update_base(new_base=new_base)
        for block in self.blocks:
            if isinstance(block.attn, MultiHeadAttention, CombinedRotaryEmbedding):
                block.attn.update_base(new_base)
            if block.cross_attn and isinstance(block.cross_attn, MultiHeadAttention, CombinedRotaryEmbedding):
                block.cross_attn.update_base(new_base)

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
    def __init__(self, vocab_size, n_ctx, n_state, n_head, n_layer, window_size=1, max_rel_dist=1, use_hybrid_attention=False, cross_attention=True, checkpointing=False, base=10000):
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
                                    checkpointing=checkpointing)
            for _ in range(n_layer)
        ])
        
        self.ln = LayerNorm(num_features=n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(value=-np.inf).triu_(diagonal=1)
        self.register_buffer(name="mask", tensor=mask, persistent=False)

        self.attention_block = copy.deepcopy(self.blocks[0].attn) if use_hybrid_attention else None

    def update_base(self, new_base):
        self.combined_rotary.update_base(new_base=new_base)
        for block in self.blocks:
            if isinstance(block.attn, MultiHeadAttention, CombinedRotaryEmbedding):
                block.attn.update_base(new_base)
            if block.cross_attn and isinstance(block.cross_attn, MultiHeadAttention, CombinedRotaryEmbedding):
                block.cross_attn.update_base(new_base)

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

class Echo(WhisperPreTrainedModel, PyTorchModelHubMixin):
    config_class = WhisperConfig

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.config = config

        self.config.num_mel_bins = config.n_mels
        self.config.max_source_positions = config.n_audio_ctx
        self.config.d_model = config.n_audio_state
        self.config.encoder_attention_heads = config.n_audio_head
        self.config.encoder_layers = config.n_audio_layer
        self.config.vocab_size = config.vocab_size
        self.config.max_target_positions = config.n_text_ctx
        self.config.d_model = config.n_text_state
        self.config.decoder_attention_heads = config.n_text_head
        self.config.decoder_layers = config.n_text_layer
        self.config.checkpointing = config.checkpointing
        self.config.max_rel_dist = config.max_rel_dist
        self.config.use_hybrid_attention = config.use_hybrid_attention
        self.config.cross_attention = config.cross_attention
        self.config.base = config.base
        self.config.window_size = config.window_size

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

        self.best_loss = float('inf')
        self.base = 10000 

    def update_base(self, new_base):
        self.encoder.combined_rotary.update_base(new_base=new_base)
        self.decoder.combined_rotary.update_base(new_base=new_base)

        for name, module in self.encoder.named_modules():
            if isinstance(module, (MultiHeadAttention, CombinedRotaryEmbedding)):
                module.update_base(new_base=new_base)

        for name, module in self.decoder.named_modules():
            if isinstance(module, (MultiHeadAttention, CombinedRotaryEmbedding)):
                module.update_base(new_base=new_base)

    def adjust_base(self, loss, factor=1.05):
        if loss < self.best_loss:
            new_base = self.base * factor
        else:
            new_base = self.base / factor

        self.update_base(new_base=new_base)
        self.best_loss = loss
        return new_base  

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
                    input_ids=labels, pad_token_id=self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
                )

        encoded_features = self.encoder(input_features).to(device)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(encoded_features.device).long() 

            logits = self.decoder(dec_input_ids, encoded_features, loss=None)  
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

            adjusted_base = self.adjust_base(loss.item())
            # print(f"Adjusted base: {adjusted_base}")

        logits = self.decoder(dec_input_ids, encoded_features, loss=loss.item() if loss is not None else None) 

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
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

    # def get_encoder(self):
    #     return self.encoder
    
    # def get_decoder(self):
    #     return self.decoder, self.config.decoder_start_token_id
    
    # def get_output_embeddings(self):
    #     return self.decoder.token_embedding, self.decoder.positional_embedding, self.decoder.combined_rotary
    
    # def get_input_embeddings(self):
    #     return self.encoder.combined_rotary, self.encoder.positional_embedding, self.encoder.conv1, self.encoder.conv2

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_features': input_ids}

    def _prepare_decoder_input_ids_for_generation(self, batch_size, decoder_start_token_id=None, bos_token_id=None):
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * self.config.decoder_start_token_id

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

##########

class GradientClippingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.nn.utils.clip_grad_norm_(parameters=kwargs["model"].parameters(), max_norm=0.98)

class MetricsCallback(TrainerCallback):
    def __init__(self, tb_writer, tokenizer, metric, log_every_n_steps=10):
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

class MetricsCallback(TrainerCallback):
    def __init__(self, tb_writer, tokenizer, metric, log_every_n_steps=50):
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
            for key, value in metrics.items():
                if key.startswith("eval_"):
                    self.tb_writer.add_scalar(key, value, state.global_step)
                    print(f"Step {state.global_step} - {key}: {value}")

                    loss = metrics['eval_loss']
                    adjusted_base = model.adjust_base(loss)
                    print(f"Adjusted base: {adjusted_base}")

                    if model.config.use_hybrid_attention:
                        print(f"Current Loss: {loss:.4f}, Window Scale: {model.encoder.blocks[0].attn.window_scale.item():.4f}")

        if self.predictions is not None and self.label_ids is not None:
            pred_str = self.tokenizer.batch_decode(self.predictions, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(self.label_ids, skip_special_tokens=True)

            sample_index = 1
            self.tb_writer.add_text("Prediction", pred_str[sample_index], state.global_step)
            self.tb_writer.add_text("Label", label_str[sample_index], state.global_step)

            print(f"Step {state.global_step} - Sample Prediction: {pred_str[sample_index]}")
            print(f"Step {state.global_step} - Sample Label: {label_str[sample_index]}")

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

        return {
            "cer": cer,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    return compute_metrics

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    tokenizer: Any
    feature_extractor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def get_length_of_dataset(dataset):
    length = 0
    for item in dataset:
        length += len(item["audio"]["array"]) / item["audio"]["sampling_rate"]
    return length / 3600  

def train():
    from datetime import datetime
    log_dir = os.path.join('./output/', datetime.now().strftime('%Y-%m-%d_%H'))
    os.makedirs(log_dir, exist_ok=True)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path="D:/newproject/my_extractor", feature_size=128, n_fft=1024, hop_length=256, sampling_rate=16000)
    tokenizer = WhisperTokenizerFast.from_pretrained(pretrained_model_name_or_path="D:/newproject/my_tokenizer")
    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path="D:/newproject/my_processor")

    config = WhisperConfig(
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
        max_rel_dist=128,
        base=10000,
        cross_attention = True,
        checkpointing = True,
        use_hybrid_attention = True,
        window_size = 40,
        is_encoder_decoder = True,
        init_std=0.02,
        pad_token_id = 112,
        unk_token_id = 113,
        bos_token_id = 114,
        eos_token_id = 115,
        decoder_start_token_id = 120,
        begin_suppress_tokens = [None]
        )

    name="/echo12/"
    model = Echo(config=config).to(device)
    model.apply_initialization(module)
    model.save_pretrained(log_dir+name)
    torch.save(model.state_dict(), log_dir+name+"state_dict.pt")
    
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        transcription = batch["sentence"]
        transcription = neologdn.normalize(transcription, repeat=1)
        batch["labels"] = tokenizer(transcription).input_ids
        return batch

    test = load_dataset(path="audiofolder", data_dir="D:/proj/datasets/gv_test")["train"] \
            .to_iterable_dataset(num_shards=2) \
            .map(prepare_dataset).select_columns(["input_features", "labels"])

    train = load_dataset(path="audiofolder", data_dir="D:/proj/datasets/gv")["train"] \
            .take(1000) \
            .to_iterable_dataset(num_shards=200) \
            .map(prepare_dataset).select_columns(["input_features", "labels"])

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

    metrics_callback = MetricsCallback(tb_writer=tb_writer, tokenizer=tokenizer, metric=metric, log_every_n_steps=50)
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
        max_steps=10000,
        save_steps=1000,
        eval_steps=100,
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
        # max_grad_norm = 0.98,
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

if __name__=="__main__":
    train()
