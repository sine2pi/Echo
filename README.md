```python         
      import base64, os, evaluate, random, gzip, math, torch, numpy as np, torch.nn.functional as F, json, warnings
      from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
      from datasets import load_dataset, IterableDatasetDict, Audio
      from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizerFast)
      import torch.nn.functional as F
      import transformers
      from itertools import chain
      from torch.utils.checkpoint import checkpoint
      from typing import Dict, Optional, Tuple
      from torch import Tensor, nn
      from dataclasses import dataclass
      from typing import Dict, Optional, Tuple, Union, List, Any
      
      from torch.nn.functional import scaled_dot_product_attention
      
      transformers.utils.logging.set_verbosity_error()
      device = torch.device(device="cuda:0" if torch.cuda.is_available() else "cpu")
      dtype = torch.float32
      torch.set_default_dtype(dtype)
      
      class Dimensions:
          def __init__(
              self,
              vocab: int,
              text_ctx: int,
              text_state: int,
              text_head: int,
              text_layerA: int,
              text_layerB: int,
              audio_ctx: int,
              audio_state: int,
              audio_head: int,
              audio_layerA: int,
              audio_layerB: int,
              mels: int,
              checkpoint: bool,
              dropout: float,
              activation: str,
          ):
              self.vocab = vocab
              self.text_ctx = text_ctx
              self.text_state = text_state
              self.text_head = text_head
              self.text_layerA = text_layerA
              self.text_layerB = text_layerB
              self.audio_ctx = audio_ctx
              self.audio_state = audio_state
              self.audio_head = audio_head
              self.audio_layerA = audio_layerA
              self.audio_layerB = audio_layerB
              self.mels = mels
              self.checkpoint = checkpoint
              self.dropout = dropout
              self.activation = activation
      
          @classmethod
          def from_dict(cls, config: dict):
              return cls(
                  vocab=config.get("vocab_size", 51865),
                  text_ctx=config.get("text_ctx", 448),
                  text_state=config.get("hidden_size", 768),
                  text_head=config.get("num_attention_heads", 12),
                  text_layerA=config.get("num_hidden_layers", 12),
                  text_layerB=config.get("text_layerB", 0),
                  audio_ctx=config.get("audio_ctx", 1500),
                  audio_state=config.get("audio_state", 768),
                  audio_head=config.get("audio_head", 12),
                  audio_layerA=config.get("num_encoder_layers", 12),
                  audio_layerB=config.get("audio_layerB", 0),
                  mels=config.get("mels", 80),
                  checkpoint=config.get("checkpoint", False),
                  dropout=config.get("dropout", 0.01),
                  activation=config.get("activation", "gelu"))
      
          def to_dict(self):
              return {
                  "vocab_size": self.vocab,
                  "text_ctx": self.text_ctx,
                  "hidden_size": self.text_state,
                  "num_attention_heads": self.text_head,
                  "num_hidden_layers": self.text_layerA,
                  "audio_ctx": self.audio_ctx,
                  "audio_state": self.audio_state,
                  "audio_head": self.audio_head,
                  "num_encoder_layers": self.audio_layerA,
                  "mels": self.mels,
                  "checkpoint": self.checkpoint,
                  "dropout": self.dropout,
                  "activation": self.activation,
              }
      
      
      class LayerNorm(nn.LayerNorm):
          def forward(self, x: Tensor) -> Tensor:
              return super().forward(x.float()).type(x.dtype)
      
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
      
      class rotary(nn.Module):
          def __init__(self, ctx, dims, heads, base=10000, theta_learnable=False, rot_learnable=False,
                       matrix_learnable=False, freq_learnable=False):
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
              self.r_matrix = nn.Parameter(torch.eye(self.head_dim), requires_grad=matrix_learnable)
      
              freq_data = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
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
                  Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=theta, u=u, v=v)
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
                      raise ValueError(f"Needed {self.heads * self.head_dim}, but got too many {dims}")
              elif len(rest) == 2:
                  heads, head_dim = rest
                  if heads != self.heads or head_dim != self.head_dim:
                      raise ValueError(f"This many heads {self.heads} and head_dims {self.head_dim} we need, got this many heads {heads} and head_dims {head_dim} we did.")
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
              div_term = torch.exp(torch.arange(0, self.dims, 2, dtype=torch.float32) * (-math.log(10000.0) / self.dims))
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
          assert channels % 2 == 0
          log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
          inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
          scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
          return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
      
      class MultiheadA(nn.Module):
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
                  with torch.autocast('cuda'):
                      a = scaled_dot_product_attention(
                          query=q,
                          key=k,
                          value=v,
                          is_causal=mask is not None and ctx > 1
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
                  with torch.autocast('cuda'):
                      a = scaled_dot_product_attention(
                          query=q,
                          key=k,
                          value=v,
                          is_causal=mask is not None and ctx > 1
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
              self.query = nn.Linear(dims, dims)
              self.key = nn.Linear(dims, dims, bias=False)
              self.value = nn.Linear(dims, dims)
              self.out = nn.Linear(dims, dims)
              
              nn.init.normal_(self.query.weight, std=scale)
              nn.init.normal_(self.key.weight, std=scale)
              nn.init.normal_(self.value.weight, std=scale)
              nn.init.zeros_(self.out.bias)
              
          def forward(self, x: Tensor, xa: Optional[Tensor] = None,
                      mask: Optional[Tensor] = None, kv_cache: Optional[Dict] = None) -> Tuple[Tensor, Optional[Tensor]]:
      
              q = self.query(x)
              
              if kv_cache is None or xa is None or self.key not in kv_cache:
                  k = self.key(x if xa is None else xa)
                  v = self.value(x if xa is None else xa)
              else:
                  k = kv_cache[self.key]
                  v = kv_cache[self.value]
      
              wv, qk = self.qkv_attention(q=q, k=k, v=v, mask=mask)
              return self.out(wv), qk
          
          def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
         
              batch, ctx, dims = q.shape
              scale = (dims // self.heads) ** -0.25
              q = q.view(batch, ctx, self.heads, self.head_dim).permute(0, 2, 1, 3)
              k = k.view(batch, ctx, self.heads, self.head_dim).permute(0, 2, 1, 3)
              v = v.view(batch, ctx, self.heads, self.head_dim).permute(0, 2, 1, 3)
      
              if self.use_sdpa and torch.cuda.is_available():
      
                  with torch.autocast('cuda'):
                      a = scaled_dot_product_attention(
                          query=q,
                          key=k,
                          value=v,
                          is_causal=mask is not None and ctx > 1
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
          
      class miniAttention(nn.Module):
          def __init__(self, dims, max_dist, heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
              super().__init__()
              
              if dims % heads != 0:
                  raise ValueError(f"dims ({dims}) must be divisible by heads ({heads})")
              if dims % 2 != 0:
                  raise ValueError(f"dims ({dims}) must be even for rotary embeddings")
              self.heads = heads
              self.head_dim = dims // heads
              self.dims = dims
              self.max_dist = max_dist
              self.scale = qk_scale or self.head_dim ** -0.5
      
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
              qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
              q, k, v = qkv[0], qkv[1], qkv[2]
              
              q = q * self.scale
      
              attn = (q @ k.transpose(-2, -1))
              attn = attn.softmax(dim=-1)
              attn = self.attn_drop(attn)
      
              x = (attn @ v).transpose(1, 2).reshape(B, N, C)
              x = self.proj(x)
              x = self.proj_drop(x)
              return x
      
      class Residual(nn.Module):
          def __init__(self, param: Dimensions, dims: int, heads: int,
                      dropout: float, activation: str):
              
              super().__init__()
              self.param = param
              self.dims = dims
      
              act_fn = nn.GELU() if activation == 'gelu' else \
                      nn.ReLU() if activation == 'relu' else \
                      nn.Sigmoid() if activation == 'sigmoid' else \
                      nn.Tanh() if activation == 'tanh' else \
                      nn.LeakyReLU() if activation == 'leaky_relu' else \
                      nn.ELU() if activation == 'elu' else \
                      nn.ReLU() 
             
              self.attn = MultiheadA(dims=dims, heads=heads)
              self.cross = MultiHeadB(dims=dims, heads=heads)
              
              mlp_dim = dims * 4
              self.mlp = nn.Sequential(
                  nn.Dropout(p=dropout),
                  Linear(in_features=dims, out_features=mlp_dim),
                  act_fn,
                  nn.Dropout(p=dropout),
                  Linear(in_features=mlp_dim, out_features=dims)
              )
      
              self.ln_a = LayerNorm(normalized_shape=dims)
              self.ln_b = LayerNorm(normalized_shape=dims)
              self.ln_c = LayerNorm(normalized_shape=dims)
              
          def forward(
              self,
              x: Tensor,
              xa: Optional[Tensor] = None,
              mask: Optional[Tensor] = None,
              kv_cache: Optional[dict] = None):
              
              y = x
              x = x + self.attn(self.ln_a(x), mask=mask, kv_cache=kv_cache)[0]
              x = x + self.cross(self.ln_b(x), xa, mask=mask, kv_cache=kv_cache)[0]
              x = x + self.mlp(self.ln_c(x))
              return x + y
      
      class AudioEncoder(nn.Module):
          
          def __init__(self, param: Dimensions, mels: int, ctx: int, dims: int, heads: int, 
                      checkpoint: bool, dropout: float, activation: str, layerA: int, layerB: int):
              super().__init__()
              
              self.checkpoint = checkpoint
              
              act_fn = nn.GELU() if activation == 'gelu' else \
                      nn.ReLU() if activation == 'relu' else \
                      nn.Sigmoid() if activation == 'sigmoid' else \
                      nn.Tanh() if activation == 'tanh' else \
                      nn.LeakyReLU() if activation == 'leaky_relu' else \
                      nn.ELU() if activation == 'elu' else \
                      nn.ReLU() 
                      
              self.rotation = rotary(ctx=ctx, dims=dims, heads=heads, base=10000)
              self.position = sinusoids(length=ctx, channels=dims)
              self.register_buffer(name="positions", tensor=self.position, persistent=False)
        
              self.convx = nn.Sequential(
                  nn.Conv1d(in_channels=mels, out_channels=dims, kernel_size=3, padding=1, bias=False),
                  nn.BatchNorm1d(num_features=dims), 
                  act_fn,  
                  nn.Dropout(p=dropout),
                  nn.Conv1d(in_channels=dims, out_channels=dims, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm1d(num_features=dims), act_fn, nn.Dropout(p=dropout))
              
              def init_weights(m):
                  if isinstance(m, nn.Conv1d):
                      nn.init.kaiming_normal_(tensor=m.weight)
              self.convx.apply(init_weights)
                      
              self.blockA = nn.ModuleList(modules=[Residual(param=param, dims=dims, heads=heads, 
                      dropout=dropout, activation=activation) 
                      for _ in range(layerB)]) if layerB > 0 else None
              
              self.blockB = nn.ModuleList(modules=[Residual(param=param, dims=dims, heads=heads, 
                      dropout=dropout, activation=activation) 
                      for _ in range(layerA)]) if layerA > 0 else None
                     
              self.ln_post = LayerNorm(normalized_shape=dims)
      
          def forward(self, x) -> Tensor:
              x = checkpoint(self._conv_forward, x, use_reentrant=True) if self.checkpoint else self._conv_forward(x)
              for block in chain(self.blockA or [], 
                                 self.blockB or []):
                  x = checkpoint(block, x, use_reentrant=True) if self.checkpoint else block(x)
              return self.ln_post(x)
      
          def _conv_forward(self, x) -> Tensor:
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
              
              self.positional_embedding = nn.Parameter(torch.empty(ctx, dims))
              nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)
              
              self.positional_encoding = PositionalEncoding(ctx=ctx, dims=dims)
              self.ln = LayerNorm(normalized_shape=dims)
              
              self.blockA = nn.ModuleList([Residual(param=param, dims=dims, heads=heads, 
                      dropout=dropout, activation=activation) 
                      for _ in range(layerB)]) if layerB > 0 else None
              
              self.blockB = nn.ModuleList([Residual(param=param, dims=dims, heads=heads, 
                      dropout=dropout, activation=activation) 
                      for _ in range(layerA)]) if layerA > 0 else None
      
              mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(diagonal=1)
              self.register_buffer("mask", mask, persistent=False)
              self.mask = mask
              
          def forward(self, x, xa: Tensor, kv_cache: Optional[dict] = None):
                
              x = checkpoint(function=self._embedding_forward, x=x, xa=xa, kv_cache=kv_cache) if self.checkpoint else self._embedding_forward(x=x, xa=xa, kv_cache=kv_cache)
              
              for block in chain(self.blockA or [], self.blockB or []):        
                  x = checkpoint(function=block, x=x, xa=xa, mask=self.mask, kv_cache=kv_cache) if self.checkpoint else block(x=x, xa=xa, mask=self.mask, kv_cache=kv_cache)
              x = self.ln(x)
                     
              x = (x @ torch.transpose(self.token_embedding.weight.to(dtype=x.dtype), dim0=0, dim1=1)).float()
              return x
          
          def _embedding_forward(self, x, xa, kv_cache):       
              offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
              x = (self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]])
              x = self.positional_encoding(x)
              x = x.to(dtype=xa.dtype)
              return x
      
      class Echo(nn.Module):
          def __init__(self, param: Dimensions):
              super().__init__()
              self.param = param
              self.device_param = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
              
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
      
              ).to(device=self.device_param)
              
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
               
              ).to(device=self.device_param)
      
              all_heads = torch.zeros(
                  self.param.text_layerA, self.param.text_head, dtype=torch.bool
              )
              all_heads[self.param.text_layerA // 2 :] = True
              self.register_buffer(name="alignment_heads", tensor=all_heads.to_sparse(), persistent=False)
      
          def set_alignment_heads(self, dump: bytes):
              array = np.frombuffer(
                  gzip.decompress(data=base64.b85decode(b=dump)), dtype=bool
              ).copy()
              mask = torch.from_numpy(ndarray=array).reshape(
                  self.param.text_layerA, self.param.text_head
              )
              self.register_buffer(name="alignment_heads", tensor=mask.to_sparse(), persistent=False)
      
          def embed_audio(self, mel: Tensor):
              return self.encoder(mel)
      
          def logits(self, tokens: Tensor, audio_features: Tensor):
              return self.decoder(tokens, audio_features)
      
          @property
          def device(self):
              return next(self.parameters()).device
      
          @property
          def is_multilingual(self):
              return self.param.vocab >= 51865
      
          @property
          def num_languages(self):
              return self.param.vocab - 51765 - int(self.is_multilingual)
      
          def install_kv_cache_hooks(self, cache: Optional[dict] = None):
      
              cache = {**cache} if cache is not None else {}
              hooks = []
      
              def save_to_cache(module, _, output):
                  if module not in cache or output.shape[1] > self.param.text_ctx:
                      cache[module] = output
                  else:
                      cache[module] = torch.cat([cache[module], output], dim=1).detach()
                  return cache[module]
      
              def install_hooks(layer: nn.Module):
                  if isinstance(layer, MultiheadA):
                      hooks.append(layer.key.register_forward_hook(save_to_cache))
                      hooks.append(layer.value.register_forward_hook(save_to_cache))
      
              self.decoder.apply(install_hooks)
              return cache, hooks
      
          detect_language_function = None
          transcribe_function = None
          decode_function = None
                  
          def save_pretrained(self, save_directory: str, save_config: bool = True, safe_serialization: bool = False):
      
              if os.path.isfile(save_directory):
                  raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
              os.makedirs(save_directory, exist_ok=True)
              weights_file = os.path.join(save_directory, "pytorch_model.bin")
              if safe_serialization:
                  try:
                      from safetensors.torch import save_file as safe_save_file
                      state_dict = self.state_dict()
                      safe_save_file(state_dict, weights_file.replace(".bin", ".safetensors"))
                  except ImportError:
                      warnings.warn("safetensors not found. Falling back to torch.save")
                      torch.save(self.state_dict(), weights_file)
              else:
                  torch.save(self.state_dict(), weights_file)
      
              if save_config:
                  config_dict = {
                      "model_config": self.param.to_dict(),
                      "architectures": [self.__class__.__name__],
                      "model_type": "whisper",
                      "vocab_size": self.param.vocab,
                      "decoder_dims": self.param.text_state,
                      "encoder_dims": self.param.audio_state,
                      "decoder_attention_heads": self.param.text_head,
                      "encoder_attention_heads": self.param.audio_head,   
                      "encoder_layers": self.param.audio_layerA,
                      "decoder_layers": self.param.text_layerA,
                      "dropout": self.param.dropout,
                      
                  }
      
                  config_file = os.path.join(save_directory, "config.json")
                  with open(config_file, "w", encoding="utf-8") as f:
                      json.dump(config_dict, f, indent=2, sort_keys=True)
      
                  model_card_file = os.path.join(save_directory, "README.md")
                  if not os.path.exists(model_card_file):
                      model_card_content = f"""---
      
          * Model Type: Echo
          * Vocabulary Size: {self.param.vocab}
          * Decoder Dimensions: {self.param.text_state}
          * Encoder Dimensions: {self.param.audio_state}  
          * Decoder Layers: {self.param.text_layerA}
          * Encoder Layers: {self.param.audio_layerA}
          * Decoder Attention Heads: {self.param.text_head}
          * Encoder Attention Heads: {self.param.audio_head}
          * Dropout: {self.param.dropout}
          """
                      with open(model_card_file, "w", encoding="utf-8") as f:
                          f.write(model_card_content)
          
          @classmethod
          def from_pretrained(
              cls,
              pretrained_model_name_or_path: str,
              device_map: Optional[str] = None,
              torch_dtype: Optional[torch.dtype] = None,
              force_cpu: bool = False
          ) -> "Echo":
      
              if os.path.isfile(pretrained_model_name_or_path):
                  raise ValueError(f"Provided path ({pretrained_model_name_or_path}) should be a directory, not a file")
      
              config_file = os.path.join(pretrained_model_name_or_path, "config.json")
              if not os.path.exists(config_file):
                  raise ValueError(f"Config file not found in {pretrained_model_name_or_path}")
      
              with open(file=config_file, mode="r", encoding="utf-8") as f:
                  config_dict = json.load(fp=f)
      
              model_config = config_dict.get("model_config", config_dict)
              model = cls(param=Dimensions.from_dict(config=model_config))
      
              if force_cpu:
                  device = torch.device("cpu")
              elif device_map is not None:
                  device = torch.device(device_map)
              else:
                  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
              if torch_dtype is not None:
                  model = model.to(torch_dtype)
      
              weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
              safetensors_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.safetensors")
              if os.path.exists(safetensors_path):
                  try:
                      from safetensors.torch import load_file as safe_load_file
                      state_dict = safe_load_file(safetensors_path)
                  except ImportError:
                      warnings.warn("safetensors not found. Falling back to torch.load")
                      state_dict = torch.load(weights_path, map_location="cpu")
              elif os.path.exists(weights_path):
                  state_dict = torch.load(weights_path, map_location="cpu")
              else:
                  raise ValueError(f"No weights found in {pretrained_model_name_or_path}")
              model.load_state_dict(state_dict)
              model = model.to(device)
              return model
              
          @staticmethod
          def shift_tokens_right(input_ids, pad_token_id=50257, decoder_start_token_id=50258):
              shifted_input_ids = input_ids.new_zeros(input_ids.shape)
              shifted_input_ids[:, 1:] = input_ids[:, :-1].clone() 
              shifted_input_ids[:, 0] = decoder_start_token_id
              shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
              return shifted_input_ids
      
          def forward(self, input_features, labels=None, dec_input_ids=None) -> dict[str, Any | None]:
              if labels is not None:
                  if dec_input_ids is None:
                      dec_input_ids = self.shift_tokens_right(
                          input_ids=labels, pad_token_id=50257, decoder_start_token_id=50258
                      )
      
              encoded_features = self.encoder(input_features).to(self.device)  
              logits = self.decoder(dec_input_ids, encoded_features)
      
              loss = None
              if labels is not None:
                  loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                  labels = labels.to(logits.device).long()
                  loss = loss_fct(logits.view(-1, self.param.vocab), labels.view(-1))    
       
              return {"loss": loss, "logits": logits}
      
          def _init_weights(self, module):
              std = 0.02
              if isinstance(module, (Linear, Conv1d)):
                  module.weight.data.normal_(mean=0.0, std=std)
                  if module.bias is not None:
                      module.bias.data.zero_()
              elif isinstance(module, nn.Embedding):
                  module.weight.data.normal_(mean=0.0, std=std)
                  if module.padding_idx is not None:
                      module.weight.data[module.padding_idx].zero_()
              elif isinstance(module, AudioEncoder):
                  module.convx.apply(fn=self._init_weights)
              elif isinstance(module, TextDecoder):
                  nn.init.normal_(tensor=module.positional_embedding, mean=0.0, std=std)
                  nn.init.normal_(tensor=module.token_embedding.weight, mean=0.0, std=std)
              elif isinstance(module, Residual):
                  for layer in module.mlp:
                      if isinstance(layer, Linear):
                          nn.init.normal_(tensor=layer.weight, std=std)
                          nn.init.zeros_(tensor=layer.bias)
                      nn.init.normal_(tensor=LayerNorm(normalized_shape=module.dims).weight, mean=0.0, std=std)
                      nn.init.normal_(tensor=LayerNorm(normalized_shape=module.dims).bias, mean=0.0, std=std) 
                      module.attn.init_weights()
                      module.cross.init_weights()
      
          def init_weights(self):
              self.apply(fn=self._init_weights)
      
      from datetime import datetime
      log_dir = os.path.join('./output/echo', datetime.now().strftime(format='%m-%d_%H'))
      os.makedirs(name=log_dir, exist_ok=True)
      
      param = Dimensions(
              mels = 80,
              audio_ctx = 1500,
              audio_head = 4,
              audio_layerA = 4,
              audio_layerB = 0,
              audio_state = 512,
              vocab = 51865,
              text_ctx = 448,
              text_head = 4, 
              text_layerA = 4,
              text_layerB = 0,
              text_state = 512,
              checkpoint = False,
              dropout = 0.01,
              activation = 'gelu',
              )
      
      model = Echo(param=param).to(device=device)
      model.init_weights()
      
      
      token=""
      
      extractor = WhisperFeatureExtractor.from_pretrained(
          pretrained_model_name_or_path="openai/whisper-small", token=token)
      
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
      
      datasets = IterableDatasetDict()
      
      datasets["train"] = load_dataset(
          path="mozilla-foundation/common_voice_17_0",
          name="en", split="train", streaming=True, 
          token=token, trust_remote_code=True)#.take(10000)
      
      datasets["test"] = load_dataset(
          path="mozilla-foundation/common_voice_17_0", 
          name="en", split="test", streaming=True, 
          token=token, trust_remote_code=True).take(500) # type: ignore
      
      dataset = datasets.cast_column(column="audio", feature=Audio(sampling_rate=16000))
      
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
          max_steps=100000,
          save_steps=1000,
          eval_steps=1000,
          warmup_steps=300,
          num_train_epochs=1,
          logging_steps=10,
          logging_dir=log_dir + "/logs_hf",
          report_to=["tensorboard"],
          push_to_hub=False,
          disable_tqdm=False,
          save_total_limit=1,
          remove_unused_columns=False,
          label_names=["labels"],
          eval_on_start=False,
          optim="adafactor",
      )
      
      trainer = Seq2SeqTrainer(
          args=training_args,
          model=model,
          train_dataset=dataset["test"],
          eval_dataset=dataset["test"],
          data_collator=data_collator,
          compute_metrics=compute_metrics,
          processing_class=extractor,
      )
      
      trainer.train(resume_from_checkpoint=False)
