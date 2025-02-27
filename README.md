```python         
     
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


@torch.jit.script
def _apply_qrotation(x: torch.Tensor, theta: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u = u / torch.norm(u)
    v = v / torch.norm(v)

    half_theta = theta / 2
    cos_ht = torch.cos(half_theta)
    sin_ht = torch.sin(half_theta)

    q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
    
    x_shape = x.shape
    x = x.view(-1, 3)

    uv_cross = torch.cross(u.unsqueeze(0), x)
    uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
    x_rot = x + 2 * (q[0] * uv_cross + uuv_cross)

    x_rot = x_rot.view(*x_shape)
    return x_rot

@torch.jit.script
def _create_rotation_matrix(dims: int, i: int, j: int, theta: torch.Tensor, device: torch.device) -> torch.Tensor:
    G = torch.eye(dims, device=device)
    c, s = torch.cos(theta), torch.sin(theta)
    G[i, i], G[j, j] = c, c
    G[i, j], G[j, i] = -s, s
    
    if dims == 3:
        u = torch.eye(dims, device=device)[i]
        v = torch.eye(dims, device=device)[j]
        x = torch.eye(dims, device=device)
        
        Q = _apply_qrotation(x, theta=theta, u=u, v=v)
        G = (G + Q) / 2
    return G

@torch.jit.script
def _apply_rope_transform(
    x: torch.Tensor, 
    sin: torch.Tensor, 
    cos: torch.Tensor
) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class rotary2(nn.Module):
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
        return _apply_qrotation(x, theta, u, v)

    def rotation_matrix(self, dims, i, j, theta):
        return _create_rotation_matrix(dims, i, j, theta, theta.device)

    @torch.jit.script_method # type: ignore
    def apply_rotations(self, x: torch.Tensor) -> torch.Tensor:
        adjusted_rot = int(self.rot_scale.item() * self.rot)
        for k in range(adjusted_rot):
            i, j = int(self.r_pairs[k, 0].item()), int(self.r_pairs[k, 1].item())
            theta = self.thetas[k] * self.theta_scale
            G = _create_rotation_matrix(self.head_dim, i, j, theta, x.device)
            x = x @ G
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if x.dim() == 3:
            if x.shape[2] != self.dims:
                raise ValueError(f"Expected dim {self.dims}, got {x.shape[2]}")
            x = x.view(batch_size, seq_len, self.heads, self.head_dim)
        elif x.dim() == 4:
            if x.shape[2] != self.heads or x.shape[3] != self.head_dim:
                raise ValueError(f"Expected {self.heads} heads and {self.head_dim} head_dim")
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

        x_flat = x.reshape(-1, self.head_dim)
        x_rotated = self.apply_rotations(x_flat)
        x_rotated = x_rotated @ self.r_matrix
        
        x = x_rotated.view(batch_size, seq_len, self.heads, self.head_dim)
        
        position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = self.inv_freq.unsqueeze(0)
        sinusoid_inp = position * div_term
        
        sin = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(2)
        cos = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(2)
        
        x = _apply_rope_transform(x, sin, cos)
        
        x = x.view(batch_size, seq_len, self.dims)
        x = x * math.sqrt(self.dims)
        return x


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
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

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
    def __init__(
        self,
        dims,
        max_dist,
        heads=1,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
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

    def forward(self, query, key, value, max_dist, max_span, span_scale, mask):
        span_mean = span_scale.mean().item()
        span_len = min(
            int(max_span * span_mean), query.shape[1], key.shape[1], value.shape[1]
        )
        eff_span = min(span_len, max_dist)

        if eff_span == 0:
            batch_size = query.shape[0]
            return (
                torch.zeros(batch_size, eff_span, self.dims, device=query.device),
                None,
            )

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
        self.mask = mask

        self.register_buffer("window_mask_template", None, persistent=False)
        self.register_buffer("focus_threshold", torch.tensor(1e-4), persistent=False)
        self.register_buffer("focus_s_factor", torch.tensor(0.1), persistent=False)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        if mask is None:
            mask = self.mask
            
        local = self.ln_a(x)
        globe = self.ln_b(x)

        globe_out, _ = self.attn_global(globe, globe, globe)
        base_scale = self.span_pred(globe_out.mean(dim=1))
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

        threshold = 1e-4
        s_factor = 0.1

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
        self.dims = dims
        num_windows = (seq_len + win_size - 1) // win_size
        output = torch.zeros_like(x)
        device = x.device
        default_mask_shape = (batch_size, self.heads, 1, 1)
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
        """ Shift input tokens right for teacher forcing. Returns: Shifted input tokens """
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

log_dir = os.path.join("./output/", datetime.now().strftime(format="%m-%d_%H"))
os.makedirs(name=log_dir, exist_ok=True)

param = Dimensions(
    mels=128,
    audio_ctx=1500,
    audio_head=8,
    audio_layerA=8,
    audio_layerB=4,
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
    
log_dir = os.path.join(os.getcwd(), "training_logs")
os.makedirs(log_dir, exist_ok=True)

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
    logging_steps=1,
    logging_dir=os.path.join(log_dir, "logs_hf"),
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
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=extractor,
)


trainer.train(resume_from_checkpoint=False)


########

##pytorch loop


import os
import csv
import time
import torch
import numpy as np
import datetime
import logging
import torchaudio
import neologdn
import whisper
import evaluate
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchaudio import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import WhisperTokenizerFast, WhisperConfig
from functools import lru_cache
from typing import Dict, List, Any, Optional, Union


@lru_cache(maxsize=128)
def load_wave(wave_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """
    Load and resample an audio file with caching to improve performance.

    Args:
        wave_path: Path to the audio file
        sample_rate: Target sample rate

    Returns:
        Loaded and resampled waveform tensor
    """
    try:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        return waveform
    except Exception as e:
        print(f"Error loading audio file {wave_path}: {e}")
        return torch.zeros(1, sample_rate)


class AudioDataset(Dataset):
    def __init__(
        self, csv_file, aud_dir, tokenizer, sample_rate=16000, max_samples=None
    ):
        """
        Dataset for audio files with text transcriptions.

        Args:
            csv_file: Path to CSV file with filenames and transcriptions
            aud_dir: Directory containing audio files
            tokenizer: Tokenizer for text processing
            sample_rate: Audio sample rate
            max_samples: Maximum number of samples to load (None for all)
        """
        self.aud_dir = aud_dir
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.samples = []

        with open(csv_file, "r", encoding="utf-8") as f:
            total_rows = sum(1 for _ in f) - 1

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)

            for i, row in enumerate(
                tqdm(reader, total=total_rows, desc="Loading dataset")
            ):
                if max_samples is not None and i >= max_samples:
                    break

                if len(row) >= 2:
                    aud_path, label = row[0], row[1]
                    full_path = os.path.join(aud_dir, aud_path)
                    if os.path.exists(full_path):
                        self.samples.append((aud_path, label))
                    else:
                        print(f"Warning: Audio file not found: {full_path}")

        print(f"Loaded {len(self.samples)} valid audio samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        aud_path, label = self.samples[idx]
        label = handle_unknown_characters(label)
        aud = os.path.join(self.aud_dir, aud_path)
        return {"input_features": aud, "labels": label, "input_ids": label}


def handle_unknown_characters(label: str) -> str:
    """
    Clean text labels by handling unknown characters and normalizing.

    Args:
        label: Input text label

    Returns:
        Cleaned text label
    """
    try:
        label = label.encode("utf-8").decode("utf-8", errors="replace")
        label = neologdn.normalize(label, repeat=1)
        return label.strip()
    except Exception as e:
        print(f"Error handling characters: {e}")
        return label.strip()


class DataCollatorWithPadding:
    def __init__(
        self, tokenizer, mels, n_fft, hop_length, sample_rate=16000, max_length=512
    ):
        """


        Args:
            tokenizer: Tokenizer for text processing
            mels: Number of mel spectrogram bands
            n_fft: FFT size
            hop_length: Hop length between FFT windows
            sample_rate: Audio sample rate
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.mels = mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length

        self.mel_spectrogram_transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            mels=self.mels,
            hop_length=self.hop_length,
        )

        self.audio_cache = {}

    def process_audio(self, audio_path):
        """Process audio with caching for repeated accesses"""
        if audio_path in self.audio_cache:
            return self.audio_cache[audio_path]

        aud = load_wave(audio_path, self.sample_rate)
        aud = whisper.pad_or_trim(aud.flatten())
        mel_spectrogram = self.mel_spectrogram_transform(aud)
        log_mel_spectrogram = torch.log(mel_spectrogram + 1e-8)

        if len(self.audio_cache) < 1000:
            self.audio_cache[audio_path] = log_mel_spectrogram

        return log_mel_spectrogram

    def __call__(self, features):
        input_features, dec_input_ids, labels = [], [], []

        for f in features:
            log_mel_spec = self.process_audio(f["input_features"])
            input_features.append(log_mel_spec)

            label = handle_unknown_characters(f["labels"])

            encoded_input = self.tokenizer.encode(
                label, max_length=self.max_length, truncation=True
            )
            encoded_label = encoded_input.copy()

            dec_input_ids.append([self.tokenizer.bos_token_id] + encoded_input)
            labels.append(encoded_label + [self.tokenizer.eos_token_id])

        input_features = torch.stack(input_features)

        input_lengths = [len(ids) for ids in dec_input_ids]
        label_lengths = [len(lab) for lab in labels]
        max_len = min(max(input_lengths + label_lengths), self.max_length)

        dec_input_ids_padded = []
        labels_padded = []

        for ids in dec_input_ids:
            length = min(len(ids), max_len)
            padded = ids[:length] + [self.tokenizer.pad_token_id] * (max_len - length)
            dec_input_ids_padded.append(padded)

        for lab in labels:
            length = min(len(lab), max_len)
            padded = lab[:length] + [-100] * (max_len - length)
            labels_padded.append(padded)

        batch = {
            "input_ids": torch.tensor(dec_input_ids_padded, dtype=torch.long),
            "labels": torch.tensor(labels_padded, dtype=torch.long),
            "input_features": input_features,
        }

        return batch


def create_logger(log_dir):
    """Create and configure logger"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("speech_training")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def compute_metrics(pred):
    """
    Compute Character Error Rate (CER) for model predictions.

    Args:
        pred: Dictionary with predictions and label IDs

    Returns:
        Dictionary with CER score
    """
    metrics_cer = evaluate.load("cer")
    pred_ids = pred["predictions"]
    label_ids = pred["label_ids"]

    label_ids_copy = label_ids.copy()
    label_ids_copy[label_ids_copy == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids_copy, skip_special_tokens=True)

    try:
        cer = 100 * metrics_cer.compute(predictions=pred_str, references=label_str)
    except Exception as e:
        print(f"Error computing CER: {e}")
        cer = 100.0

    return {"cer": cer}


def train_and_evaluate(
    model,
    train_loader,
    eval_loader,
    optimizer,
    scheduler,
    loss_fn,
    num_epochs=1,
    max_steps=None,
    device="cuda",
    accumulation_steps=1,
    clear_cache=True,
    log_interval=10,
    eval_interval=20,
    save_interval=100,
    checkpoint_dir="checkpoint_dir",
    log_dir="log_dir",
):
    """
    Train and evaluate a speech recognition model.

    Args:
        model: Model to train
        train_loader: DataLoader for training data
        eval_loader: DataLoader for evaluation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        num_epochs: Number of training epochs
        max_steps: Maximum training steps
        device: Device for training (cuda/cpu)
        accumulation_steps: Gradient accumulation steps
        clear_cache: Whether to clear CUDA cache periodically
        log_interval: Steps between logging
        eval_interval: Steps between evaluation
        save_interval: Steps between model saving
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for logs
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(log_dir)
    logger.info(f"Starting training with {num_epochs} epochs, max_steps={max_steps}")

    model.to(device)

    global_step = 0
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    best_cer = float("inf")

    for epoch in range(num_epochs):
        if max_steps is not None and global_step >= max_steps:
            logger.info(f"Reached max steps {max_steps}, stopping training")
            break

        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True
        )

        for step, batch in enumerate(progress_bar):
            if max_steps is not None and global_step >= max_steps:
                break

            start_time = time.time()

            input_features = batch["input_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            try:
                with torch.cuda.amp.autocast():
                    input_features_encoded = model.encoder(input_features)
                    decoder_output = model.decoder(input_ids, input_features_encoded)

                    logits = decoder_output.view(-1, decoder_output.size(-1))
                    loss = loss_fn(logits, labels.view(-1))
                    total_loss += loss.item()

                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0 or (
                    step + 1 == len(train_loader)
                ):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    if clear_cache and global_step % 5 == 0:
                        torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    logger.error(f"CUDA OOM at step {global_step}: {e}")
                    continue
                else:
                    logger.error(f"Runtime error at step {global_step}: {e}")
                    raise e

            global_step += 1
            end_time = time.time()

            batch_time = end_time - start_time
            samples_per_sec = input_features.size(0) / batch_time

            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                    "norm": f"{total_norm:.2f}",
                    "step": global_step,
                }
            )

            if global_step % log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]

                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("GradientNorm", total_norm, global_step)
                writer.add_scalar("LearningRate", current_lr, global_step)
                writer.add_scalar("SamplesPerSec", samples_per_sec, global_step)
                writer.add_scalar("BatchTime", batch_time, global_step)

                logger.info(
                    f"Step {global_step}: loss={loss.item():.4f}, "
                    f"lr={current_lr:.6f}, norm={total_norm:.2f}, "
                    f"samples/sec={samples_per_sec:.1f}"
                )

            if global_step % eval_interval == 0:
                logger.info(f"Evaluating at step {global_step}...")
                eval_metrics = evaluate_model(
                    model=model,
                    eval_loader=eval_loader,
                    loss_fn=loss_fn,
                    device=device,
                    tokenizer=tokenizer,
                    writer=writer,
                    global_step=global_step,
                    logger=logger,
                )

                eval_cer = eval_metrics["cer"]
                logger.info(f"Evaluation CER: {eval_cer:.4f}")

                if eval_cer < best_cer:
                    best_cer = eval_cer
                    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(
                        f"New best model with CER {best_cer:.4f} saved to {best_model_path}"
                    )

                model.train()

            if global_step % save_interval == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_step_{global_step}.pt"
                )
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss.item(),
                        "best_cer": best_cer,
                    },
                    checkpoint_path,
                )
                logger.info(
                    f"Checkpoint saved at step {global_step} to {checkpoint_path}"
                )

        scheduler.step()

        avg_epoch_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} completed: Average loss: {avg_epoch_loss:.4f}")
        writer.add_scalar("Loss/epoch", avg_epoch_loss, epoch + 1)

    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    writer.close()

    return best_cer


def evaluate_model(
    model, eval_loader, loss_fn, device, tokenizer, writer, global_step, logger
):
    """Evaluate model on test data"""
    model.eval()
    eval_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for eval_batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_features = eval_batch["input_features"].to(device)
            input_ids = eval_batch["input_ids"].to(device)
            labels = eval_batch["labels"].to(device)

            input_features_encoded = model.encoder(input_features)
            decoder_output = model.decoder(input_ids, input_features_encoded)

            logits = decoder_output.view(-1, decoder_output.size(-1))
            loss = loss_fn(logits, labels.view(-1))
            eval_loss += loss.item()

            predictions = torch.argmax(decoder_output, dim=-1)
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    eval_loss /= max(1, len(eval_loader))

    predictions = {
        "predictions": np.array(all_predictions, dtype="object"),
        "label_ids": np.array(all_labels, dtype="object"),
    }
    metrics = compute_metrics(predictions)

    writer.add_scalar("Loss/eval", eval_loss, global_step)
    writer.add_scalar("CER", metrics["cer"], global_step)

    if len(all_predictions) > 0:
        for idx in range(min(2, len(all_predictions))):
            pred_ids = [
                id for id in all_predictions[idx] if id != tokenizer.pad_token_id
            ]
            label_ids = [
                id for id in all_labels[idx] if id not in [-100, tokenizer.pad_token_id]
            ]

            pred_str = tokenizer.decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.decode(label_ids, skip_special_tokens=True)

            example_str = (
                f"Example {idx}:\n"
                f"  Reference: {label_str}\n"
                f"  Prediction: {pred_str}"
            )
            logger.info(example_str)

    return {"loss": eval_loss, "cer": metrics["cer"]}


if __name__ == "__main__":
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(os.getcwd(), f"logs_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    logger = create_logger(log_dir)
    logger.info("Starting speech recognition training")

    model_name = "openai/whisper-small"
    tokenizer = WhisperTokenizerFast.from_pretrained(model_name)
    logger.info(f"Loaded tokenizer from {model_name}")

    csv_file = "D:/proj/datasets/gf_1/metadata.csv"
    audio_dir = "D:/proj/datasets/gf_1/"

    try:
        logger.info("Creating dataset...")
        dataset = AudioDataset(csv_file, audio_dir, tokenizer)
        logger.info(f"Created dataset with {len(dataset)} samples")

        train_size = int(0.999 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset, [train_size, eval_size]
        )
        logger.info(
            f"Split into {len(train_dataset)} training and {len(eval_dataset)} validation samples"
        )

    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise

    logger.info("Creating data loaders...")
    collate_fn = DataCollatorWithPadding(
        tokenizer, n_fft=1024, hop_length=256, mels=80
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        drop_last=False,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    param = Dimensions(
        mels=128,
        audio_ctx=1500,
        audio_head=8,
        audio_layerA=8,
        audio_layerB=0,
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

    logger.info("Initializing model...")

    model = Echo(param=param)
    model.to(device=device)

    from transformers.optimization import Adafactor

    optimizer = Adafactor(
        model.parameters(),
        clip_threshold=0.99,
        weight_decay=0.025,
        scale_parameter=True,
        relative_step=False,
        warmup_init=False,
        lr=2.25e-3,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=max_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    torch.backends.cudnn.benchmark = True
    logger.info("Starting training...")
    try:
        train_and_evaluate(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            max_steps=100,
            num_epochs=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            accumulation_steps=1,
            clear_cache=True,
            log_interval=1,
            eval_interval=10,
            save_interval=100,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


```
