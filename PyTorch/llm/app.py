# Copyright (c) Microsoft Corporation.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path
from typing import Union, List, Tuple, Iterator

import torch_directml
import torch

import gradio as gr

from transformers import AutoTokenizer, PreTrainedTokenizerFast

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from utils import decode_one_token, prefill, _load_model, decode_with_overlap
from scripts.download_and_convert import hf_download, convert_hf_checkpoint
from models.phi3 import Transformer as Phi3Transformer
from models.phi2 import Transformer as Phi2Transformer
from models.llama import Transformer as LlamaTransformer


device = torch_directml.device(torch_directml.default_device())

def decode_n_tokens(
    model: Union[Phi2Transformer, Phi3Transformer, LlamaTransformer],
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    tokenizer: PreTrainedTokenizerFast,
    stream_every_n: int,
    is_llama_3: bool = False,
    **sampling_kwargs
) -> Iterator[str]:
    res = tokenizer.decode(cur_token[0][0].item(), skip_special_tokens=True).strip() + " "
    yield res

    new_tokens = []
    previous_output = ""
    overlap_text = ""  # Stores the text of the overlap to avoid re-decoding
    start_pos = 0
    overlap_size = 2   # to decode the tokens from previous stream batch as well;
    last_pos = 0

    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        cur_token = next_token.view(1, -1)

        # Handle output and overlap at the specified intervals or at the last token for adding
        # the space correctly between stream batches
        if ((i + 1) % stream_every_n == 0 or i == num_new_tokens - 1):
            # Determine the range of tokens to decode, including the overlap
            from_index = max(0, start_pos - overlap_size)
            yield decode_with_overlap(tokenizer, new_tokens, from_index, overlap_text)
            last_pos = i

            # Update overlap_text to the last few characters of the current output
            overlap_text = tokenizer.decode(
                torch.IntTensor(new_tokens[-overlap_size:]).tolist(), skip_special_tokens=True) if len(new_tokens) >= overlap_size else ""
            start_pos += stream_every_n

        if next_token[-1] == tokenizer.eos_token_id or \
           next_token[-1] == tokenizer.convert_tokens_to_ids("<|end|>") or \
           (is_llama_3 and next_token[-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>")):
            if i > last_pos:
                from_index = max(0, start_pos - overlap_size)
                yield decode_with_overlap(tokenizer, new_tokens, from_index, overlap_text)
            break

@torch.no_grad()
def generate(
    model: Union[Phi2Transformer, Phi3Transformer, LlamaTransformer],
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    tokenizer: PreTrainedTokenizerFast,
    stream_every_n: int = 10,
    is_llama_3: bool = False,
    **sampling_kwargs
) -> Iterator[str]:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    # generated_tokens = decode_n_tokens(
    yield from decode_n_tokens(
        model, next_token.view(1, -1), input_pos, max_new_tokens - 1, tokenizer, stream_every_n, is_llama_3=is_llama_3, **sampling_kwargs)


class LLM_Model:
    def __init__(
        self,
        prompt: str = "Hello, my name is",
        interactive: bool = False,
        num_samples: int = 5,
        max_new_tokens: int = 100,
        top_k: int = 200,
        temperature: float = 0.01,
        hf_model: str = "microsoft/Phi-3-mini-4k-instruct",
        checkpoint_path: str = None,
        precision: str = 'float32',
        stream_every_n: int = 7,
        max_context_length: int = 3500,
        use_history: bool = False,
        max_pos_emb: int = 8192
    ):
        self.prompt = prompt
        self.interactive = interactive
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.temperature = temperature
        self.hf_model = hf_model
        self.checkpoint_path = Path(f"checkpoints/{hf_model}/model.pth") if checkpoint_path is None else Path(checkpoint_path)
        self.precision = torch.float32 if precision == 'float32' else torch.float16
        self.stream_every_n = stream_every_n
        self.max_context_length = max_context_length
        self.use_history = use_history
        self.max_pos_emb = max_pos_emb

        self.tokenizer = None
        self.model = None

    def encode_tokens(
        self,
        prompt: str,
        conversation_history: List[List[str]],
        device: torch.device = None,
        max_context_length: int = 1500,
        bos: bool = True
    ) -> torch.Tensor:
        if self.is_phi_2:
            tokens = self.format_prompt_phi2_chat_and_encode(
                prompt, conversation_history, device, max_context_length, bos
            )
        else:
            tokens = self.format_prompt_and_encode(
                prompt, conversation_history, device, max_context_length,
            )
        return tokens

    def format_prompt_and_encode(
        self,
        prompt: str,
        conversation_history: List[List[str]],
        device: torch.device = None,
        max_context_length: int = 1500,
    ) -> torch.Tensor:
        messages = []
        if len(conversation_history) and self.use_history:
            for user, assistant in conversation_history:
                user = {"role": "user", "content": user}
                assistant = {"role": "assistant", "content": assistant}
                messages.append(user)
                messages.append(assistant)
        messages.append({"role": "user", "content": prompt})
        tokens = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True)[0].to(dtype=torch.int, device=device)
        if self.use_history:
            while tokens.size(0) > max_context_length:
                print("Clipping history of conversation as it exceeds the max context length.")
                if len(messages) > 1:
                    messages.pop(0)  # Remove the oldest user message
                    messages.pop(0)  # Remove the oldest assistant message
                else:
                    break
                tokens = self.tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True)[0].to(dtype=torch.int, device=device)

        return tokens

    def format_prompt_phi2_chat_and_encode(
        self,
        prompt: str,
        conversation_history: List[List[str]],
        device: torch.device = None,
        max_context_length: int = 1500,
        bos: bool = True
    ) -> torch.Tensor:
        formatted_prompt = ""
        if self.use_history:
            for user_prompt, llm_response in conversation_history:
                formatted_prompt += f"Human: {user_prompt}\nAI:{llm_response}\n"

        formatted_prompt += f"Human: {prompt}\nAI:"
        tokens = self.tokenizer.encode(formatted_prompt)
        if self.use_history and len(tokens) > max_context_length:
            while len(tokens) > max_context_length and conversation_history:
                conversation_history.pop(0)  # Remove the oldest exchange
                formatted_prompt = ""
                for user_prompt, llm_response in conversation_history:
                    formatted_prompt += f"Human: {user_prompt}\nAI:{llm_response}\n"
                formatted_prompt += f"Human: {prompt}\nAI:"
                tokens = self.tokenizer.encode(formatted_prompt)
        if bos:
            tokens = [self.tokenizer.encode(self.tokenizer.bos_token)[0]] + tokens

        token_tensor = torch.tensor(tokens, dtype=torch.int, device=device)
        return token_tensor

    def format_prompt_phi2_qa_and_encode(
        self,
        prompt: str,
        conversation_history: List[List[str]],
        device: torch.device = None,
        max_context_length: int = 1500,
        bos: bool = True
    ) -> torch.Tensor:
        formatted_prompt = ""
        if self.use_history:
            for user_prompt, llm_response in conversation_history:
                formatted_prompt += f"Instruct: {user_prompt}\nOutput:{llm_response}\n"

        formatted_prompt += f"Instruct: {prompt}\nOutput:"

        tokens = self.tokenizer.encode(formatted_prompt)

        if self.use_history and len(tokens) > max_context_length:
            while len(tokens) > max_context_length and conversation_history:
                conversation_history.pop(0)  # Remove the oldest exchange
                formatted_prompt = ""
                for user_prompt, llm_response in conversation_history:
                    formatted_prompt += f"Instruct: {user_prompt}\nOutput:{llm_response}\n"
                formatted_prompt += f"Instruct: {prompt}\nOutput:"
                tokens = self.tokenizer.encode(formatted_prompt)

        if bos:
            tokens = [self.tokenizer.encode(self.tokenizer.bos_token)[0]] + tokens

        token_tensor = torch.tensor(tokens, dtype=torch.int, device=device)
        return token_tensor

    def download_and_convert(self) -> None:
        checkpoint_dir = hf_download(self.hf_model)
        convert_hf_checkpoint(
            checkpoint_dir=Path(checkpoint_dir),
        )
        self.checkpoint_path = Path(f"{checkpoint_dir}/model.pth")

    def load_model(self) -> None:
        if not self.checkpoint_path.is_file():
            print(f"{self.checkpoint_path} doesnt exist. Downloading and converting {self.hf_model} from huggingface hub. "
                  "Specify a different model with --hf_model or valid pre-converted checkpoint with --checkpoint_path")
            self.download_and_convert()
        print("Running app...")
        print(f"Loading model from {self.checkpoint_path}")

        self.is_llama_3 = "Llama-3" in self.checkpoint_path.parent.name
        self.is_phi_2 = "phi-2" in self.checkpoint_path.parent.name
        print(f"Using device={device}, is_llama_3={self.is_llama_3}, is_phi_2={self.is_phi_2}")
        if self.is_phi_2:
            self.precision = torch.float32

        self.model = _load_model(self.checkpoint_path, device, self.precision, max_pos_emb=self.max_pos_emb)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path.parent)
        if self.max_context_length > self.model.config.block_size - (self.max_new_tokens+1):
            raise ValueError(
                f"Expected max_context_length to be less than {self.model.config.block_size - (self.max_new_tokens+1)} but got {self.max_context_length}")

    @torch.no_grad()
    def chat(
        self,
        prompt: str,
        history: List[List[str]],
        **sampling_kwargs
    ) -> Iterator[str]:
        torch.manual_seed(1235)

        encoded = self.encode_tokens(
            prompt,
            history,
            device=device,
            max_context_length=self.max_context_length,
        )

        yield from generate(
            self.model,
            encoded,
            self.max_new_tokens,
            tokenizer=self.tokenizer,
            stream_every_n=self.stream_every_n,
            is_llama_3=self.is_llama_3,
            temperature=self.temperature,
            top_k=self.top_k,
        )


def chat(message: str, history: List[List[str]]) -> Iterator[str]:
    total_msg = ""
    for msg in llm_model.chat(message, history):
        total_msg += msg
        yield total_msg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument(
        '--hf_model',
        type=str,
        default="phi-3",
        help='Huggingface Repository ID to download from. Or one of the model name from ["phi-2", "phi-3", "llama-2", "llama-3", "mistral"]'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Converted pytorch model checkpoint path. Defaults to `checkpoints/{hf_model}/model.pth`.'
    )
    parser.add_argument(
        '--max_context_length',
        type=int,
        default=1500,
        help='Max prompt length including the history. If exceeded, history is clipped starting from the first (user, assistant) pair.'
    )
    parser.add_argument(
        '--disable_history',
        action="store_true",
        help='Whether to disable history of the chat for generation. History is enabled by default.'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='float16',
        choices=['float16', 'float32'],
        help='Precision to run the generation with.'
    )
    parser.add_argument(
        '--max_pos_emb',
        type=int,
        default=8192,
        help='Maximum Position to scale Phi-3.5 position encodings.'
    )
    args = parser.parse_args()
    if args.max_pos_emb > 131072:
        args.max_pos_emb = 131072

    llm_model = LLM_Model(prompt = "Hello",
                      interactive = False,
                      num_samples = 1,
                      max_new_tokens = 500,
                      top_k = 200,
                      temperature = 0.8,
                      hf_model = args.hf_model,
                      checkpoint_path = args.checkpoint_path,
                      precision = args.precision,
                      max_context_length = args.max_context_length,
                      use_history = not args.disable_history,
                      max_pos_emb=args.max_pos_emb)
    llm_model.load_model()

    demo = gr.ChatInterface(chat).queue()
    demo.launch()
