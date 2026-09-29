from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


class WaterSheepConfig(PreTrainedConfig):
    model_type = "watersheep"

    def __init__(self, encoder_config=None, head_layers=1, max_len=512, max_question_tokens=96,
                 max_option_tokens=32, max_options=10, temperatures=None, multi_threshold=0.5, **kwargs):
        self.encoder_config = encoder_config or {}
        self.head_layers = head_layers
        self.max_len = max_len
        self.max_question_tokens = max_question_tokens
        self.max_option_tokens = max_option_tokens
        self.max_options = max_options
        self.temperatures = temperatures or {}
        self.multi_threshold = multi_threshold
        super().__init__(**kwargs)


@dataclass
class WaterSheepOutput(ModelOutput):
    logits: torch.FloatTensor = None


class WaterSheepModel(PreTrainedModel):
    config_class = WaterSheepConfig
    base_model_prefix = "watersheep"

    def __init__(self, config):
        super().__init__(config)
        enc = AutoConfig.for_model(**config.encoder_config)
        if hasattr(enc, "reference_compile"):
            enc.reference_compile = False
        self.encoder = AutoModel.from_config(enc)
        h = enc.hidden_size
        self.proj = nn.Sequential(nn.Dropout(0.0), nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h))
        self.mix = None
        if config.head_layers > 0:
            layer = nn.TransformerEncoderLayer(h, nhead=max(1, h // 64), dim_feedforward=2 * h, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.mix = nn.TransformerEncoder(layer, config.head_layers, enable_nested_tensor=False)
        self.out = nn.Linear(h, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask, option_positions, option_mask, **kwargs):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = option_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, hs.size(-1))
        x = self.proj(torch.gather(hs, 1, idx))
        if self.mix is not None:
            x = self.mix(x, src_key_padding_mask=~option_mask)
        logits = self.out(x).squeeze(-1).float()
        return WaterSheepOutput(logits=logits.masked_fill(~option_mask, -1e4))
