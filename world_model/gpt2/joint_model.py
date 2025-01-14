from typing import Dict

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import BoolTensor, FloatTensor, LongTensor

from world_model.gpt2.mup_dit import MupDiT
from world_model.gpt2.mup_gpt2 import MupGPT2
from world_model.gpt2.prepare_token_sequence import prepare_token_sequence


class JointModel(nn.Module):

    def __init__(
        self,
        gpt_config: OmegaConf,
        dit_config: OmegaConf,
    ) -> None:
        super().__init__()
        self.gpt: MupGPT2 = instantiate(gpt_config)
        self.dit: MupDiT = instantiate(dit_config)

    def visual_tokens_to_embeds(self, visual_tokens: LongTensor) -> FloatTensor:
        sequence_data = prepare_token_sequence(visual_tokens)
        spatial_pos_emb = self.gpt.transformer.wse(sequence_data["spatial_positions"])
        temporal_pos_emb = self.gpt.transformer.wte(sequence_data["temporal_positions"])
        tok_emb = self.gpt.transformer.wie(sequence_data["token_sequence"])
        visual_embeds = tok_emb + temporal_pos_emb + spatial_pos_emb
        return visual_embeds

    def forward(
        self,
        attention_mask: BoolTensor,
        position_ids_all: Dict[str, LongTensor],
        inputs_all: Dict[str, FloatTensor | LongTensor],
    ) -> FloatTensor:
        pass
