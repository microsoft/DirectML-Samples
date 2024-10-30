# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Microsoft Corporation.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file
from requests.exceptions import HTTPError
from huggingface_hub.utils._errors import RepositoryNotFoundError

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.configs import ModelArgs, default_models


def is_dir_empty(directory: str) -> bool:
    return not any(os.scandir(directory))

def download_model_from_hf(hf_model: str, checkpoint_dir: str, hf_token: Optional[str]) -> str:
    from huggingface_hub import snapshot_download
    checkpoint_dir = f"{checkpoint_dir}/{hf_model}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not is_dir_empty(checkpoint_dir):
        print(f"The directory {checkpoint_dir} is not empty. Skipping download.")
    else:
        try:
            snapshot_download(hf_model, local_dir=checkpoint_dir, local_dir_use_symlinks=False, token=hf_token)
            print(f"Downloaded {hf_model} successfully.")
        except HTTPError as e:
            if e.response.status_code == 401:
                print("You need to pass a valid Hugging Face token to download private models.")
            raise e
    return checkpoint_dir

def hf_download(hf_model: Optional[str] = None, hf_token: Optional[str] = None, checkpoint_dir: str = "checkpoints") -> None:
    try:
        checkpoint_dir_download = download_model_from_hf(hf_model, checkpoint_dir, hf_token)
    except RepositoryNotFoundError as e:
        # invalid repo passed, try to search for a default repo from the given hf_model
        os.rmdir(f"{checkpoint_dir}/{hf_model}")
        print(f"Couldn't find {hf_model} on HuggingFace. Searching for the closest supported match ...")
        if hf_model in default_models:
            hf_model = default_models[hf_model]
        else:
            raise ValueError(f"Please provide a valid hf_model to download from Huggingface. {hf_model} doesnt exist on Huggingface.")

        print(f"Found closest match on Huggingface: {hf_model}")
        checkpoint_dir_download = download_model_from_hf(hf_model, checkpoint_dir, hf_token)
    return checkpoint_dir_download

@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/microsoft/Phi-3-mini-4k-instruct"),
    weight_map_path: str = "config/weight_map.json",
) -> None:
    if not os.path.exists(checkpoint_dir):
        raise ValueError("Please download you model first with the hf_download function.")

    if os.path.exists(checkpoint_dir / "model.pth"):
        print(f"Converted checkpoint already exists here {checkpoint_dir / 'model.pth'}. Skipping Conversion.")
        return

    model_name = checkpoint_dir.name
    config = ModelArgs.from_name(model_name)

    with open(weight_map_path, 'r') as file:
        weight_maps = json.load(file)

    model_name = checkpoint_dir.name
    if "phi-3" in model_name.lower():
        model_name = "Phi-3-mini-4k-instruct"
    elif "phi" not in model_name:
        model_name = "llama"
    weight_map = weight_maps[model_name]

    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "model.safetensors.index.json"

    assert model_map_json.is_file()

    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    merged_result = {}
    for file in sorted(bin_files):
        state_dict = load_file(str(file))
        merged_result.update(state_dict)

    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key, count=1)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key == "":
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq.weight" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v], dim=0)

            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

        if "wq.bias" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v], dim=0)

            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download and convert HuggingFace checkpoint.')
    parser.add_argument('--hf_model', type=str, default="microsoft/Phi-3-mini-4k-instruct", help='Huggingface Repository ID to download from.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token.')
    parser.add_argument(
        '--checkpoint_dir', type=str, default="checkpoints",
        help="Directory to downloads the Huggingface repo to. The model will be downloaded and converted to '{checkpoint_dir}/{hf_model}/"
    )

    args = parser.parse_args()
    checkpoint_dir = hf_download(args.hf_model, args.hf_token, args.checkpoint_dir)
    convert_hf_checkpoint(
        checkpoint_dir=Path(checkpoint_dir),
    )
