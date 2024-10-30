# Copyright (c) Microsoft Corporation.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, Union
import torch_directml
import torch
import torch.nn as nn
from torch import Tensor

from models.configs import ModelArgs
from models.layers import RotaryEmbedding
from models.base import Transformer


class Phi2Transformer(Transformer):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.LayerNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        hidden_states = self.attention_norm(x)
        attn_outputs = self.attention(hidden_states, mask, input_pos)
        ffn_hidden_states = self.feed_forward(hidden_states)
        return attn_outputs + ffn_hidden_states + x

class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.config = config

        self.wqkv = nn.Linear(config.dim, 3 * config.dim)
        self.wo = nn.Linear(config.dim, config.dim)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.partial_rotary_factor = config.partial_rotary_factor

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: Dict[str, torch.Tensor], prefix: str, *argspy):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def _init_rope(
        self,
        max_position_embeddings: int = 4096,
        rope_base: Union[int, float] = 10000.0,
        dtype: torch.dtype = torch.float32
    ) -> None:
        self.min_position = 0
        self.past_key_tensor = None
        self.past_value_tensor = None
        self.rotary_emb = RotaryEmbedding(
            int(self.head_dim * self.partial_rotary_factor),
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
            dtype=dtype,
            config=self.config,
        )

    def forward(self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)

        q = q.reshape(bsz, seqlen, self.n_head, self.head_dim).transpose(1,2)
        k = k.reshape(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1,2)

        cos, sin = self.rotary_emb(self.min_position + seqlen)

        q, k = torch_directml.apply_rotary_position_emb(
            q, k, cos, sin, self.min_position, seqlen, self.rotary_emb.dim)

        self.min_position += seqlen

        q, k = map(lambda x: x.transpose(1, 2).reshape(bsz, -1, self.dim), (q, k))
        y, self.past_key_tensor, self.past_value_tensor = torch_directml.multi_head_attention(
            q,k,v, self.dim, self.n_head, self.past_key_tensor, self.past_value_tensor, mask
        )

        y = self.wo(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.dim)
        self.mlp = torch_directml.mlp_phi2

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x, self.w1.weight, self.w2.weight, self.w1.bias, self.w2.bias)
