
import base64, gzip, math, os, functools, warnings, numpy as np, torch, transformers, aiohttp, torch.nn.functional as F, evaluate, json, random
from torch import Tensor, amp, optim, nn
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard.writer import SummaryWriter
from threading import Thread
from typing import Dict, Optional, Tuple, Union, List, Any
from dataclasses import dataclass
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, PretrainedConfig, TrainerCallback, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizerFast)
from torch.optim import Optimizer
import evaluate
from evaluate import module
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from datasets import load_dataset, IterableDatasetDict, Audio, load_from_disk
from torch.nn.functional import scaled_dot_product_attention
transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings(action="ignore")
warnings.warn = lambda *args, **kwargs: None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


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
    def __init__(self, base, dims, head, theta_learnable=True, rot_learnable=True,
                 matrix_learnable=False, freq_learnable=True):
        super(CombinedRotaryEmbedding, self).__init__()

        self.base = base
        self.dims = dims
        self.head = head

        self.h_dim = self.dims // self.head
        self.rot = (self.dims // self.head) // 2

        self.thetas = nn.Parameter(torch.zeros(self.rot))
        self.r_pairs = nn.Parameter(data=torch.rand(self.rot, 2) * self.h_dim)

        self.theta_scale = nn.Parameter(torch.ones(1), requires_grad=theta_learnable)
        self.rot_scale = nn.Parameter(torch.ones(1), requires_grad=rot_learnable)

        self.r_matrix = nn.Parameter(torch.eye(n=self.h_dim), requires_grad=matrix_learnable)

        freq_data = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
        self.inv_freq = nn.Parameter(freq_data, requires_grad=freq_learnable)

        self.orthogonal_reg_weight = 0.01

    def blended_rotation_matrix(self, dims, i, j, theta):
        G = torch.eye(dims).to(theta.device)
        G[i, i] = torch.cos(theta)
        G[i, j] = -torch.sin(theta)
        G[j, i] = torch.sin(theta)
        G[j, j] = torch.cos(theta)

        v = torch.zeros(dims).to(theta.device)
        v[i] = torch.cos(theta)
        v[j] = torch.sin(theta)
        H = torch.eye(dims).to(theta.device) - 2 * torch.outer(v, v) / torch.dot(v, v)

        R = torch.eye(dims).to(theta.device)
        R[i, i] = torch.cos(theta)
        R[i, j] = -torch.sin(theta)
        R[j, i] = torch.sin(theta)
        R[j, j] = torch.cos(theta)

        return (G + H + R) / 3

    def apply_blended_rotation(self, x):
        adjusted_rot = int(torch.round(self.rot_scale * self.rot))
        for k in range(adjusted_rot):
            i, j = self.r_pairs[k].long()
            theta = self.thetas[k] * self.theta_scale
            B = self.blended_rotation_matrix(dims=self.h_dim, i=i, j=j, theta=theta)
            x = torch.matmul(input=x, other=B)
        return x

    def update_base(self, new_base):
        if new_base is not None and new_base != self.base:
            self.base = new_base
            inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
            self.inv_freq.data.copy_(inv_freq)
            self.update_pairs()

    def reset_parameters(self):
        nn.init.orthogonal_(self.r_matrix)
        nn.init.zeros_(self.thetas)
        nn.init.zeros_(self.r_pairs)
        nn.init.ones_(self.theta_scale)
        nn.init.ones_(self.rot_scale)

    def orthogonal_regularization_term(self):
        loss = torch.tensor(0.0, device=self.r_matrix.device)
        if self.r_matrix.requires_grad:
            product = torch.matmul(self.r_matrix, self.r_matrix.t())
            identity = torch.eye(self.r_matrix.size(0)).to(self.r_matrix.device)
            loss = ((product - identity) ** 2).sum()
        return self.orthogonal_reg_weight * loss

    def update_pairs(self):
        pairs = []
        while len(pairs) < self.rot:
            i, j = torch.randint(0, self.h_dim - 1, (2,))
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
                raise ValueError(f"Expected dims ({dims}) to be compatible with head ({self.head}) * h_dim ({self.h_dim}={self.head * self.h_dim})")
        else:
            head, h_dim = rest
            if head != self.head or h_dim != self.h_dim:
                raise ValueError(f"For 4D input, expected head {self.head} and h_dim {self.h_dim}, but got head {head} and h_dim {h_dim}")

        x = x.view(batch_size, seq_len, self.head, self.h_dim)
        x = x.reshape(-1, self.h_dim)

        x = self.apply_blended_rotation(x)

        x = torch.matmul(input=x, other=self.r_matrix)

        x = x.view(batch_size, seq_len, self.head, self.h_dim)

        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(end=seq_len, device=x.device), self.inv_freq.to(device=x.device))
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]

        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat(tensors=[x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x = x.view(batch_size, seq_len, self.dims)

        return x

class SinusoidalEmbedding(nn.Module):
    def __init__(self, n_ctx, dims, checkpoint):
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
        self.pos_embeds = nn.Parameter(self.my_big_toe.clone())

    def forward(self, positions):
        if self.checkpoint:
            position_embeddings = checkpoint(lambda x: self.pos_embeds[x], positions)
        else:
            position_embeddings = self.pos_embeds[positions]
        return F.normalize(position_embeddings, p=2, dim=-1) 

class CombinedPositionalEmbedding(nn.Module):
    def __init__(self, base, dims, head, n_ctx, theta_learnable=True, rot_learnable=True, 
                 matrix_learnable=False, freq_learnable=True, checkpoint=False):
        super().__init__()
        self.rotary_embedding = CombinedRotaryEmbedding(base, dims, head, theta_learnable, 
                                                        rot_learnable, matrix_learnable, freq_learnable)
        self.sinusoidal_embedding = SinusoidalEmbedding(n_ctx, dims, checkpoint)

    def forward(self, x, positions, global_step=None):
        rotary_embed = self.rotary_embedding(x, global_step)
        sinusoidal_embed = self.sinusoidal_embedding(positions)
        
        combined_embedding = rotary_embed + sinusoidal_embed
        return combined_embedding

class MultiheadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, base, dims, head, max_dist):
        super().__init__()
        assert dims % head == 0, "dims must be divisible by head"
        self.head = head
        self.h_dim = dims // head
        assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

        self.query = nn.Linear(dims, dims)
        self.key = nn.Linear(dims, dims, bias=False)
        self.value = nn.Linear(dims, dims)
        self.out = nn.Linear(dims, dims)

        # self.givens_rotary = CombinedRotaryEmbedding(base=base, dims=dims, head=head)

    def forward(self, x, xa = None, mask = None, kv_cache = None):

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)

        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        # q = self.givens_rotary(q)
        # k = self.givens_rotary(k)

        wv, qk = self.qkv_attention(q=q, k=k, v=v, mask=mask)

        out = self.out(wv)
        return out, qk
    
    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        
        n_batch, n_ctx, dims = q.shape
        scale = (dims // self.head) ** -0.25
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        if MultiheadAttention.use_sdpa:
            a = scaled_dot_product_attention(query=q, key=k, value=v, is_causal=mask is not None and n_ctx > 1)
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(dtype=q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk
    
class AdaptiveSpanAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, sharpen, win_size, max_span, temp_scale=0.01):
        super().__init__()
        self.max_dist = max_dist
        self.win_size = win_size
        self.max_span = max_span
        self.temp_scale = temp_scale
        self.multihead_attn = MultiheadAttention(base=base, dims=dims, head=head, max_dist=max_dist)
        self.span_scale = nn.Parameter(torch.tensor(1.0))
        self.sharpen = sharpen

    def forward(self, query, key, value, span_scale):
        span_len = int(self.max_span * span_scale.mean().item())
        span_len = min(span_len, query.shape[1], key.shape[1], value.shape[1])
        eff_span = min(span_len, self.max_dist)

        q_span = query[:, :eff_span, :]
        k_span = key[:, :eff_span, :]
        v_span = value[:, :eff_span, :]

        batch_size, _, dims = query.shape
        scale = (dims // self.multihead_attn.head) ** -0.25

        q = q_span.view(q_span.shape[0], q_span.shape[1], self.multihead_attn.head, -1).permute(0, 2, 1, 3)
        k = k_span.view(k_span.shape[0], k_span.shape[1], self.multihead_attn.head, -1).permute(0, 2, 1, 3)
        v = v_span.view(v_span.shape[0], v_span.shape[1], self.multihead_attn.head, -1).permute(0, 2, 1, 3)

        if self.sharpen:
            temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
        else:
            temperature = 0.5 + self.temp_scale * span_scale.mean().item()

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.softmax((attn_scores / temperature) * scale, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
        attn_out = attn_out.contiguous().view(batch_size, eff_span, dims)

        return attn_out, attn_weights

class SpanPredictor(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.linear = nn.Linear(in_features=dims, out_features=1)

    def forward(self, global_out):
        scale = torch.sigmoid(self.linear(global_out))
        return scale

class HybridAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, sharpen, win_size=32, max_span=32, slid_win=32):
        super().__init__()
        self.max_dist = max_dist
        self.win_size = win_size
        self.max_span = max_span
        self.slid_win = slid_win

        self.span_pred = SpanPredictor(dims=dims)
        self.dist_local = max_dist
        self.dist_global = max_dist

        self.attn_local = AdaptiveSpanAttention(base=base, dims=dims, head=head, max_dist=max_dist, sharpen=sharpen, win_size=win_size, max_span=max_span)
        self.attn_global = MultiheadAttention(base=base, dims=dims, head=head, max_dist=self.dist_global)
        self.ln_local = LayerNorm(normalized_shape=dims)
        self.ln_global = LayerNorm(normalized_shape=dims)
        self.projection = Linear(in_features=2 * dims, out_features=dims)

    def forward(self, x, new_dist=None, new_base=None, xa=None, mask=None, kv_cache=None):
        local = self.ln_local(x)
        globe = self.ln_global(x)

        globe_out, _ = self.attn_global(globe, globe, globe)

        span_scale = self.span_pred(globe_out.mean(dim=1))

        win_size = max(1, int(self.slid_win * span_scale.mean().item()))
        span_len = max(1, int(self.max_span * span_scale.mean().item()))

        effective_max_dist = min(self.max_dist, local.size(1))
        local_max_dist = min(self.dist_local, span_len, win_size)
        globe_max_dist = effective_max_dist

        self.attn_local.max_dist = local_max_dist
        self.attn_global.max_dist = globe_max_dist

        local_out = self.slide_win(x=local, win_size=win_size, span_len=span_len, span_scale=span_scale)

        combined = torch.cat(tensors=[local_out, globe_out], dim=-1)
        x = self.projection(combined)

        return x

    def slide_win(self, x, win_size, span_len, span_scale):
        batch_size, seq_len, dims = x.size()
        out = torch.zeros_like(x, device=x.device)

        for i in range(0, seq_len, win_size):
            end = min(i + win_size, seq_len)
            query = x[:, i:end, :]

            start = max(0, i - span_len + win_size)
            key = x[:, start:i + span_len, :]
            value = x[:, start:i + span_len, :]
            attn_out, _ = self.attn_local(query, key, value, span_scale)
            out[:, i:end, :] = attn_out

        return out

class ResidualAttention(nn.Module):
    def __init__(self, base, dims, head, max_dist, win_size, max_span, hybrid, checkpoint, cross, sharpen):
        super().__init__()

        if hybrid:
            self.attn = HybridAttention(base=base, dims=dims, head=head, max_dist=max_dist, sharpen=sharpen)
            self.attn_ln = LayerNorm(normalized_shape=dims)
        else:
            self.attn = MultiheadAttention(base=base, dims=dims, head=head, max_dist=max_dist)
            self.attn_ln = LayerNorm(normalized_shape=dims)

        n_mlp = dims * 4
        self.mlp = nn.Sequential(Linear(in_features=dims, out_features=n_mlp), nn.GELU(), Linear(in_features=n_mlp, out_features=dims))
        self.mlp_ln = LayerNorm(normalized_shape=dims)

    def forward(self, x, mask=None, kv_cache=None):
        x = self._attn_forward(x=x, mask=mask, kv_cache=kv_cache)
        x = self._mlp_forward(x=x)
        return x

    def _attn_forward(self, x, mask=None, kv_cache=None):
        residual = x
        x = self.attn_ln(x)

        if isinstance(self.attn, HybridAttention):
            attn_output = self.attn(x)  

            x = residual + attn_output
        else:
            attn_output, _ = self.attn(x, mask=mask, kv_cache=kv_cache)  
            x = residual + attn_output
        return x

    def _mlp_forward(self, x):
        residual = x
        x = self.mlp_ln(x)
        return residual + self.mlp(x)

class AudioEncoder(nn.Module):
    def __init__(self, base, mels, dims, head, n_layer, n_ctx, max_dist,
                 win_size, max_span, hybrid, checkpoint, cross, sharpen):
        super().__init__()
        self.conv1 = Conv1d(in_channels=mels, out_channels=dims, kernel_size=3, padding=1)
        self.conv2 = Conv1d(in_channels=dims, out_channels=dims, kernel_size=3, stride=2, padding=1)
        self.pos_embed = SinusoidalEmbedding(n_ctx=n_ctx, dims=dims, checkpoint=checkpoint)
        self.checkpoint = checkpoint

        self.givens_rotary = CombinedRotaryEmbedding(base=base, dims=dims, head=head)
        # self.combine = CombinedPositionalEmbedding(base=base, dims=dims, head=head)
        self.blocks = nn.ModuleList(modules=[ResidualAttention(base=base, dims=dims, head=head, max_dist=max_dist, win_size=win_size, max_span=max_span, hybrid=hybrid, checkpoint=checkpoint, cross=cross, sharpen=sharpen) for _ in range(n_layer)])
        self.ln_post = LayerNorm(normalized_shape=dims)

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
    
        p = self.pos_embed(torch.arange(end=x.size(dim=1), device=x.device)).unsqueeze(0)
        x = (x + p).to(x.dtype)
        x = self.givens_rotary(x)
        # x = self.combine(x)
        return x

class TextDecoder(nn.Module):
    def __init__(self, base, vocab, dims, head, n_layer, n_ctx, max_dist,
                 win_size, max_span, hybrid, checkpoint, cross, sharpen):
        super().__init__()
        
        self.tok_embed = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        self.pos_embed = SinusoidalEmbedding(n_ctx=n_ctx, dims=dims, checkpoint=checkpoint)
        self.checkpoint = checkpoint

        self.givens_rotary = CombinedRotaryEmbedding(base=base, dims=dims, head=head)

        self.blocks = nn.ModuleList(modules=[ResidualAttention(base=base, dims=dims, head=head, max_dist=max_dist, win_size=win_size, max_span=max_span, hybrid=hybrid, checkpoint=checkpoint, cross=cross, sharpen=sharpen) for _ in range(n_layer)])

        self.ln_post = LayerNorm(normalized_shape=dims)
        self.ln = LayerNorm(normalized_shape=dims)

        mask = torch.empty(n_ctx, n_ctx).fill_(value=-np.inf).triu_(diagonal=1)
        self.register_buffer(name="mask", tensor=mask, persistent=False)
        self.mask=mask

    def forward(self, x, xa, kv_cache=None):
        if self.checkpoint:
            x = checkpoint(self._embedding_forward, x, xa, kv_cache)
        else:
            x = self._embedding_forward(x=x, xa=xa, kv_cache=kv_cache)

        for block in self.blocks:
            if self.checkpoint:
                x = checkpoint(block, x, self.mask, kv_cache)
            else:
                x = block(x, self.mask, kv_cache)

        x = self.ln(x)
        x = (x @ torch.transpose(input=self.tok_embed.weight.to(dtype=x.dtype), dim0=0, dim1=1)).float()
        return x
    
    def _embedding_forward(self, x, xa, kv_cache):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        positions = torch.arange(x.shape[1], device=x.device) + offset
        pos_emb = self.pos_embed(positions).unsqueeze(0)
        x = self.tok_embed(x) + pos_emb
        x = self.givens_rotary(x)
        return x

class EchoConfig(PretrainedConfig):
    model_type = "Echo"
    def __init__(
        self,
        checkpoint=False,
        cross=False,
        hybrid=False,
        sharpen=False,
        a_ctx=1500,
        a_head=16,
        a_layer=8,
        a_dims=1024,
        mels=128,
        t_ctx=448,
        t_head=8,
        t_layer=8,
        t_dims=1024,
        win_size=64,
        max_span=64,
        max_dist=64,
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
        self.sharpen=sharpen

class Echo(nn.Module):
    def __init__(self, config: EchoConfig):
        super().__init__()
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
            sharpen=self.config.sharpen,
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
            sharpen=self.config.sharpen,
        )

        all_heads = torch.zeros(self.config.t_layer, self.config.t_head, dtype=torch.bool) 
        all_heads[self.config.t_layer // 2:] = True
        self.register_buffer(name="alignment_heads", tensor=all_heads.to_sparse(), persistent=False)

        self.base = self.config.base
        self.win_size = self.config.win_size
        self.adjust_counter = 0
        self.best_loss = float('inf')
        self.kv_cache = {}

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

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
            self.update_window(new_window=new_window)
            self.best_loss = loss
            self.adjust_counter += 1
            return new_window
        return self.win_size

    def adjust_base(self, loss, factor=1.0025) -> float | int:
                if self.adjust_counter % 25 == 0:
                    if loss < self.best_loss:
                        new_base=self.base*factor
                    else:
                        new_base=self.base/factor
                    self.update_base(new_base=new_base)
                    self.base=new_base
                    self.best_loss=loss
                self.adjust_counter += 1
                return self.base
            
    def update_base(self, new_base):
        self.new_base=new_base
        for name, module in self.encoder.named_modules():
            if isinstance(module, (CombinedRotaryEmbedding)):
                module.update_base(new_base=self.new_base)

    @staticmethod
    def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone() 
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward(self, input_features, labels=None, dec_input_ids=None) -> dict[str, Any | None]:
        if labels is not None:
            if dec_input_ids is None:
                dec_input_ids = self.shift_tokens_right(
                    input_ids=labels, pad_token_id=self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
                )

        encoded_features = self.encoder(input_features).to(self.device)  
        logits = self.decoder(dec_input_ids, encoded_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(logits.device).long()
            loss = loss_fct(logits.view(-1, self.config.vocab), labels.view(-1))

            self.adjust_window(loss.item())
            # self.adjust_base(loss=loss.item())
        return {"loss": loss, "logits": logits}

    def reset_parameters(self):
        for name, module in self.encoder.named_modules():
            if isinstance(module, CombinedRotaryEmbedding):
                module.reset_parameters()
        
    def _initialize_weights(self, module):
            nn.init.normal_(tensor=self.decoder.tok_embed.weight, mean=0.0, std=0.02)
            nn.init.constant_(tensor=self.decoder.ln.weight, val=1)
            nn.init.constant_(tensor=self.decoder.ln.bias, val=0)
            nn.init.xavier_normal_(tensor=self.encoder.conv1.weight)
            nn.init.zeros_(tensor=self.encoder.conv1.bias)
            nn.init.kaiming_normal_(tensor=self.encoder.conv2.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(tensor=self.encoder.conv2.bias)
            nn.init.constant_(tensor=self.encoder.ln_post.weight, val=1)
            nn.init.constant_(tensor=self.encoder.ln_post.bias, val=0)

            for block in self.decoder.blocks:
                for layer in block.children():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(tensor=layer.weight)
                        nn.init.zeros_(tensor=layer.bias)
                    if isinstance(layer, LayerNorm):
                        nn.init.constant_(tensor=layer.weight, val=1)
            
            for block in self.encoder.blocks:
                for layer in block.children():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(tensor=layer.weight)
                        nn.init.zeros_(tensor=layer.bias)
                    if isinstance(layer, LayerNorm):
                        nn.init.constant_(tensor=layer.weight, val=1)

            for module in self.encoder.named_modules():
                if isinstance(module, CombinedRotaryEmbedding):
                    nn.init.constant_(tensor=module.thetas, val=1)
                    nn.init.constant_(tensor=module.r_matrix, val=1)
                    nn.init.constant_(tensor=module.r_pairs, val=1)
                    nn.init.constant_(tensor=module.inv_freq, val=1)

    def apply_initialization(self, module):
        self._initialize_weights(module=module)

from datetime import datetime
log_dir = os.path.join('./output/Echo/', datetime.now().strftime(format='%m-%d_%H'))
os.makedirs(name=log_dir, exist_ok=True)

config = EchoConfig(
    checkpoint=False,
    cross=False,
    hybrid=False,
    sharpen=False,
    audio_ctx=1500,
    audio_head=4,
    audio_layer=4,
    audio_dims=512,
    mels=128,
    text_ctx=448,
    text_head=4,
    text_layer=4,
    text_dims=512,
    win_size=16,
    max_span=16,
    max_dist=16,
    base=50000,
    pad_token_id=50257,
    unk_token_id=50257,
    vocab=51865,
    eos_token_id=50257,
    bos_token_id=50257,
    decoder_start_token_id=50258,

)

model = Echo(config=config).to(device=device)
model.apply_initialization(module=model)


feature_extractor = WhisperFeatureExtractor.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small", 
    feature_size=128, sample_rate=160000, do_normalize=True)

tokenizer = WhisperTokenizerFast.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small", 
    language="en", task="transcribe")

processor = WhisperProcessor.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small", 
    feature_size=128, sample_rate=160000, do_normalize=True, 
    language="en", task="transcribe")

class GradientClippingCallback(TrainerCallback):
    def on_step_end(self, args, dims, control, **kwargs):
        torch.nn.utils.clip_grad_norm_(parameters=kwargs["model"].parameters(), max_norm=0.98)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
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
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=config.decoder_start_token_id)

datasets = IterableDatasetDict()

datasets["train"] = load_dataset(
    path="mozilla-foundation/common_voice_17_0", token="",
    name="en", split="train", streaming=True, trust_remote_code=True).take(10000)

datasets["test"] = load_dataset(
    path="mozilla-foundation/common_voice_17_0", token="", 
    name="en", split="test", streaming=True, trust_remote_code=True).take(100)

dataset = datasets.cast_column(column="audio", feature=Audio(sampling_rate=16000))

dataset = dataset.map(function=prepare_dataset, 
                      remove_columns=list(next(iter(dataset.values())).features)).with_format(type="torch")

class MetricsCallback(TrainerCallback):
    def __init__(self, tb_writer, tokenizer, metric, optimizer, scheduler, log_every_n_steps=1):
        super().__init__()
        self.tb_writer = tb_writer
        self.tokenizer = tokenizer
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_every_n_steps = log_every_n_steps
        self.predictions = None
        self.label_ids = None

    def compute_wer(self, pred_str, label_str):
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return wer

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        if metrics is not None:
            self.eval_loss = metrics.get('eval_loss')

            current_learning_rate = self.optimizer.param_groups[0]['lr']
            if state.global_step % self.log_every_n_steps == 0:
                self.tb_writer.add_scalar('learning_rate', current_learning_rate, state.global_step)
                print(f"Learning Rate: {current_learning_rate:.8f}")

                self.tb_writer.add_scalar('eval_loss', self.eval_loss, state.global_step)

            for key, value in metrics.items():
                if key.startswith("eval_"):
                    self.tb_writer.add_scalar(key, value, state.global_step)

        if self.predictions is not None and self.label_ids is not None:
            pred_str = self.tokenizer.batch_decode(self.predictions, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(self.label_ids, skip_special_tokens=True)

            if state.global_step % self.log_every_n_steps == 0:
                total_samples = len(pred_str)
                random_indices = random.sample(range(total_samples), 1)

                for sample_index in random_indices:
                    self.tb_writer.add_text(f"Prediction_{sample_index}", pred_str[sample_index], state.global_step)
                    self.tb_writer.add_text(f"Label_{sample_index}", label_str[sample_index], state.global_step)
                    print(f"Evaluation: - Step {state.global_step} - Loss: {self.eval_loss:.2f}")
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

metric = evaluate.load(path="wer")
tb_writer = SummaryWriter(log_dir=log_dir)

training_args = Seq2SeqTrainingArguments(
    output_dir=log_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    tf32=True,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",
    max_steps=10000,
    save_steps=10000,
    eval_steps=100,
    warmup_steps=100,
    logging_steps=10,
    logging_dir=log_dir + "/logs_hf",
    report_to=["tensorboard"],
    load_best_model_at_end=False,
    metric_for_best_model="loss",
    greater_is_better=False,
    push_to_hub=False,
    disable_tqdm=False,
    save_total_limit=1,
    remove_unused_columns=False,
    label_names=["labels"],
    eval_on_start=True,
)

class MaxFactor(Optimizer):
    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0, 
                 weight_decay=0.0, gamma=0.99, eps_rms=1e-8, maximize=False):
        
        defaults = dict(lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay, 
                        gamma=gamma, eps_rms=eps_rms, maximize=maximize)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad, grads, row_vars, col_vars, v, state_steps = [], [], [], [], [], []
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
                    if p.grad.dim() > 1:
                        row_shape, col_shape = list(p.grad.shape), list(p.grad.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"], state["col_var"] = p.grad.new_zeros(row_shape), p.grad.new_zeros(col_shape)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                row_vars.append(state.get("row_var", None))
                col_vars.append(state.get("col_var", None))
                v.append(state["v"])
                state_steps.append(state["step"])
                params_with_grad.append(p)
                grads.append(grad)

            for i, param in enumerate(params_with_grad):
                grad = grads[i]

                if group["maximize"]:
                    grad = -grad
                step_t, row_var, col_var, vi = state_steps[i], row_vars[i], col_vars[i], v[i]

                if eps1 is None:
                    eps1 = torch.finfo(param.dtype).eps
                    
                step_t += 1
                step_float = step_t.item()
                one_minus_beta2_t = step_float ** group["beta2_decay"]
                rho_t = min(group["lr"], 1 / (step_float ** 0.5))
                alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

                if group["weight_decay"]!= 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

                if grad.dim() > 1:
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1))
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2))
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    var_estimate = row_var @ col_var
                    max_row_var = row_var.max(dim=-2, keepdim=True)[0]  
                    var_estimate.div_(max_row_var.clamp_(min=eps1))

                else:
                    vi.mul_(group["gamma"]).add_(1 - group["gamma"], grad ** 2)
                    var_estimate = vi
                
                update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                update = update.div_(torch.norm(update, float('inf')).clamp_(min=eps1))
                denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                param.add_(-alpha / denom * update.sign() * update.abs().max(dim=-1, keepdim=True)[0])

        return loss
    
optimizer = MaxFactor(
    model.parameters(), 
    lr=0.025,  
    beta2_decay=-0.8,
    eps=(None, 1e-4),
    d=1.0,
    weight_decay=0.0025,
    gamma=0.99, 
    eps_rms=1e-8,
    maximize=False,
    )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=training_args.max_steps,
    eta_min=0.0,
    last_epoch=-1  
)

metrics_callback = MetricsCallback(tb_writer=tb_writer, tokenizer=tokenizer, metric=metric, optimizer=optimizer, scheduler=scheduler, log_every_n_steps=10)
compute_metrics = create_compute_metrics(callback_instance=metrics_callback)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=feature_extractor,
    callbacks=[metrics_callback],
    optimizers=(optimizer, scheduler)
)

trainer.train(resume_from_checkpoint=False)

from tensorboard import program
log_dir = "D:/new/tensorboard3" 
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()
print(f"TensorBoard started at {url}")


