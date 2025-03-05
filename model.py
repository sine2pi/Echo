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


    @classmethod
    def from_dict(cls, param: dict):
        return cls(
            vocab=param.get("vocab_size", 51865),
            text_ctx=param.get("text_ctx", 448),
            text_state=param.get("hidden_size", 768),
            text_head=param.get("num_attention_heads", 12),
            text_layerA=param.get("num_hidden_layers", 12),
            text_layerB=param.get("text_layerB", 0),
            audio_ctx=param.get("audio_ctx", 1500),
            audio_state=param.get("audio_state", 768),
            audio_head=param.get("audio_head", 12),
            audio_layerA=param.get("num_encoder_layers", 12),
            audio_layerB=param.get("audio_layerB", 0),
            mels=param.get("mels", 80),
            checkpoint=param.get("checkpoint", False),
            dropout=param.get("dropout", 0.01),
            activation=param.get("activation", "gelu"),

        )

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

class Echo(nn.Module):

    def __init__(self, param: Dimensions, debug=False):
        
        if debug == True:
            print(f"Echo check: {param}")
        
        super().__init__()
        self.param = param
        self.to(self.device)

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

        self.PAD_TOKEN_ID = 50257
        self.START_TOKEN_ID = 50258

    @property
    def device(self) -> torch.device:

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def shift_tokens_right(
        input_ids: torch.Tensor,
        pad_token_id = 50257,
        decoder_start_token_id: int = 50258,
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
        input_ids: Optional[torch.Tensor] = None,
        auto_shift: bool = False,  # Shift input ids right for teacher forcing
    ) -> Dict[str, Optional[torch.Tensor]]:
        
        decoder_input_ids = input_ids
        if auto_shift and labels is not None and decoder_input_ids is None:
            decoder_input_ids = self.shift_tokens_right(
                input_ids=labels,
                pad_token_id=self.PAD_TOKEN_ID,
                decoder_start_token_id=self.START_TOKEN_ID,
            )
        
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            encoded_features = self.encoder(input_features)
            logits = self.decoder(decoder_input_ids, encoded_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(logits.device).long()
            
            flattened_logits = logits.view(-1, self.param.vocab)
            flattened_labels = labels.view(-1)
            
            loss = loss_fct(flattened_logits, flattened_labels)
        
        return {"loss": loss, "logits": logits}
            
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
                "model_type": "Echo",
                "vocab_size": self.param.vocab,
                "decoder_dims": self.param.text_state,
                "encoder_dims": self.param.audio_state,
                "decoder_attention_heads": self.param.text_head,
                "encoder_attention_heads": self.param.audio_head,   
                "encoder_layers": self.param.audio_layerA,
                "decoder_layers": self.param.text_layerA,
                "dropout": self.param.dropout,
                
            }

            config_file = os.path.join(save_directory, "param.json")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
   
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

        config_file = os.path.join(pretrained_model_name_or_path, "param.json")
        if not os.path.exists(config_file):
            raise ValueError(f"Config file not found in {pretrained_model_name_or_path}")

        with open(file=config_file, mode="r", encoding="utf-8") as f:
            config_dict = json.load(fp=f)

        model_config = config_dict.get("model_config", config_dict)
        model = cls(param=Dimensions.from_dict(param=model_config))

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


    def _init_weights(self, module):
        std = 0.02

        if isinstance(module, (Linear, Conv1d)):
            nn.init.normal_(tensor=module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(tensor=module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(tensor=module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, AudioEncoder):
            module.convx.apply(self._init_weights)

        elif isinstance(module, TextDecoder):
            nn.init.normal_(tensor=module.positional_embedding, mean=0.0, std=std)
            nn.init.normal_(tensor=module.token_embedding.weight, mean=0.0, std=std)

        elif isinstance(module, Residual):
            for layer in module.mlp:
                if isinstance(layer, Linear):
                    nn.init.normal_(tensor=layer.weight, std=std)
                    nn.init.zeros_(tensor=layer.bias)

            for ln_name in ["ln_a", "ln_b", "ln_c"]:
                if hasattr(module, ln_name):
                    ln = getattr(module, ln_name)
                    nn.init.normal_(tensor=ln.weight, mean=1.0, std=std)
                    nn.init.zeros_(tensor=ln.bias)

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
