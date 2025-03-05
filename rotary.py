class rotary(nn.Module):
    def __init__(self, ctx, dims, heads, base=10000, theta_learnable=False,
        rot_learnable=False, matrix_learnable=False, freq_learnable=False,
        debug=False
    ):
        
        if debug == True:
            print(f"Rotary check: {ctx} {dims} {heads} {base} {theta_learnable} {rot_learnable} "
                  f"{matrix_learnable} {freq_learnable}")

        
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
            Q = self.q_rotation(
                torch.eye(dims, device=theta.device), theta=theta, u=u, v=v)
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
