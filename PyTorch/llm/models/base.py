# Copyright (c) Microsoft Corporation.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F

from models.configs import ModelArgs, find_multiple


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.max_position_embeddings = config.block_size
        self.rope_base = config.rope_base
        self.max_batch_size = -1
        self.max_seq_length = -1

    def set_max_position_embeddings(self, max_pos_emb):
        self.max_position_embeddings = max_pos_emb

    def setup_caches(self, max_batch_size, max_seq_length):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype=self.output.weight.dtype
        for b in self.layers:
            b.attention._init_rope(self.max_position_embeddings, self.rope_base, dtype=dtype)

        self.causal_mask = torch.tril(
            torch.ones(self.config.n_head, self.max_seq_length, self.max_seq_length, dtype=torch.int32)
        )

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        mask = self.causal_mask[None, :, input_pos, :input_pos[-1].item()+1]
        x = self.tok_embeddings(idx)

        for _, layer in enumerate(self.layers):
            x = layer(x, input_pos, mask)

        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str, max_pos_emb: int = 8192):
        model = cls(ModelArgs.from_name(name))
        if "phi-3.5" in name.lower():
            model.set_max_position_embeddings(max_pos_emb)

        return model