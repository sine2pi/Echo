class Residual(nn.Module):
    def __init__(
        self, param: Dimensions, dims: int, heads: int, dropout: float, activation: str, debug=False
    ):
        if debug == True:
            print(f"Residual check: {param} {dims} {heads} {dropout} {activation}")
        
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
        self.cross = MultiheadB(dims=dims, heads=heads)

        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            Linear(in_features=dims, out_features=dims * 4, bias=True),
            act_fn,
            nn.Dropout(p=dropout),
            Linear(in_features=dims * 4, out_features=dims, bias=True),
        )

        self.ln_a = LayerNorm(normalized_shape=dims)
        self.ln_b = LayerNorm(normalized_shape=dims)
        self.ln_c = LayerNorm(normalized_shape=dims)

        self.init_weights()

    def init_weights(self):

        for m in self.mlp:
            if isinstance(m, Linear):
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
    
