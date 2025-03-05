class TextDecoder(nn.Module):
    def __init__(self, param: Dimensions, vocab: int, ctx: int, dims: int, heads: int, 
                checkpoint: bool, dropout: float, activation: str, layerA: int, layerB: int, debug=False):
        
        if debug == True:
            print(f"TextDecoder check: {param} {vocab} {ctx} {dims} {heads} {checkpoint} {dropout} {activation} {layerA} {layerB}")
        
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
