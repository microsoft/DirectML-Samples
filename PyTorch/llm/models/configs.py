# Copyright (c) Microsoft Corporation.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    partial_rotary_factor: float = 1.0
    rope_scaling: Optional[Dict[str, Any]] = field(default=None)
    original_max_position_embeddings: int = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match

        if not config:
            raise ValueError(f"No configuration found for the model named '{name}'. Supported models: {list(transformer_configs.keys())}")

        return cls(**transformer_configs[config[0]])

transformer_configs = {
    "7B": dict(block_size=4096, n_layer=32, n_head=32, dim=4096),
    "phi-2": dict(block_size=2048, n_layer=32, n_head=32, dim=2560, intermediate_size=10240, rope_base=10000, vocab_size=51200, partial_rotary_factor=0.4),
    "Phi-3-mini-4k-instruct": dict(block_size=4096, n_layer=32, n_head=32, dim=3072, intermediate_size=8192, rope_base=10000, vocab_size=32064),
    "Mistral-7B": dict(block_size=4096, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "Llama-3-8B": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000.0),
    "Llama-3.1-8B": dict(
        block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096,
        intermediate_size=14336, vocab_size=128256, rope_base=500000.0,
        rope_scaling={
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }
    ),
    "Phi-3.5-mini-instruct": dict(
        block_size=8192, n_layer=32, n_head=32, dim=3072, intermediate_size=8192,
        rope_base=10000, vocab_size=32064, original_max_position_embeddings=4096,
        rope_scaling={
            "long_factor": [
                1.0800000429153442, 1.1100000143051147, 1.1399999856948853, 1.340000033378601, 1.5899999141693115,
                1.600000023841858, 1.6200000047683716, 2.620000123977661, 3.2300000190734863, 3.2300000190734863,
                4.789999961853027, 7.400000095367432, 7.700000286102295, 9.09000015258789, 12.199999809265137,
                17.670000076293945, 24.46000099182129, 28.57000160217285, 30.420001983642578, 30.840002059936523,
                32.590003967285156, 32.93000411987305, 42.320003509521484, 44.96000289916992, 50.340003967285156,
                50.45000457763672, 57.55000305175781, 57.93000411987305, 58.21000289916992, 60.1400032043457,
                62.61000442504883, 62.62000274658203, 62.71000289916992, 63.1400032043457, 63.1400032043457,
                63.77000427246094, 63.93000411987305, 63.96000289916992, 63.970001220703125, 64.02999877929688,
                64.06999969482422, 64.08000183105469, 64.12000274658203, 64.41000366210938, 64.4800033569336,
                64.51000213623047, 64.52999877929688, 64.83999633789062
            ],
            "short_factor": [ 
                1.0, 1.0199999809265137, 1.0299999713897705, 1.0299999713897705, 1.0499999523162842, 1.0499999523162842,
                1.0499999523162842, 1.0499999523162842, 1.0499999523162842, 1.0699999332427979, 1.0999999046325684,
                1.1099998950958252, 1.1599998474121094, 1.1599998474121094, 1.1699998378753662, 1.2899998426437378,
                1.339999794960022, 1.679999828338623, 1.7899998426437378, 1.8199998140335083, 1.8499997854232788,
                1.8799997568130493, 1.9099997282028198, 1.9399996995925903, 1.9899996519088745, 2.0199997425079346,
                2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346,
                2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914,
                2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0799996852874756,
                2.0899996757507324, 2.189999580383301, 2.2199995517730713, 2.5899994373321533, 2.729999542236328,
                2.749999523162842, 2.8399994373321533
            ],
            "rope_type": "longrope"
        },
    ),
}

default_models = {
    "llama-3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "phi-2": "microsoft/phi-2",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct"
}
