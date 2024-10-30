# Simple and Efficient Language Models for PyTorch with DirectML

This sample provides a simple way to download a PyTorch model, optimize it for DirectML, and run it through a Gradio app UI.

This sample is extracted from [pytorch-labs/gpt-fast](https://github.com/pytorch-labs/gpt-fast), and has been slightly changed to use `torch-directml`. The original code is Copyright (c) 2023 Meta, and is used here under the terms of the BSD 3-Clause License. See [LICENSE](./LICENSE) for more information.

- [Supported Models](#supported-models)
- [Setup](#setup)
- [Run the App](#run-the-app)
- [App Settings](#app-settings)
- [External Links & Model Licenses](#external-links-and-model-licenses)

## Supported Models

The following models are currently supported by this sample:

- [Phi-2](https://huggingface.co/microsoft/phi-2): Small Language Model with 2.7 billion parameters. Best suited for prompts using QA format, chat format, and code format.
- [Phi-3 Mini 4K](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct): Small Language Model with 3.8 billion parameters using a 4k context window. The Instruct version has been fine-tuned to follow instructions and adhere to safety measures.
- [Phi-3.5 Mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct): A lightweight, state-of-the-art open model with 3.8 billion parameters using a 128k context window. The Instruct version has been fine-tuned to ensure precise instruction adherence and robust safety measures.
- [LLaMA 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf): Large Language Model with 7 billion parameters optimized specifically for dialogue use cases.
- [LLaMA 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct): Large Language Model with 8 billion parameters. The Llama 3 instruction tuned models are optimized for dialogue use cases.
- [LLaMA 3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct): Large Language Model with 8 billion parameters. The Llama 3.1 instruction tuned models are an inprovement over Llama-3 and are optimized for dialogue use cases.
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1): Large Language Model with 7 billion parameters. The Mistral-7B-Instruct-v0.1 Large Language Model is a instruct fine-tuned version of the Mistral-7B-v0.1 generative text model using a variety of publicly available conversation datasets.

>⚠️ **NOTE**: Other variants of these models may work but they were not tested.

The various models have different VRAM requirements, the following table lists the memory requirements for the tested models.

|  Model            | fp16  | fp32  |
| ----------------- | ------| ----- |
| Phi-2             | 6GB   | 12GB  |
| Phi-3-mini-4k     | 8GB   | >16GB |
| Phi-3.5-mini      | 8GB   | >16GB |
| Llama-2-7b        | 14GB  | 28GB  |
| Meta-Llama-3-8B   | >16GB | 32GB  |
| Meta-Llama-3.1-8B | >16GB | 32GB  |
| Mistral-7B        | 15GB  | 30GB  |

## Setup
Once you've setup `torch-directml` following our [Windows](https://learn.microsoft.com/windows/ai/directml/pytorch-windows) or [WSL 2](https://learn.microsoft.com/windows/ai/directml/pytorch-wsl) guidance, install the following requirements for running app:

```
pip install -r requirements.txt
```

To use the Llama and Mistral models, you will need to go through an extra step to access their Hugging Face repository. To do so:
1. Visit
    - LLaMA 2: [https://huggingface.co/meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    - LLaMA 3: [https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
    - LLaMA 3.1: [https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
    - Mistral: [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
2. Follow the steps on the Hugging Face page to obtain access
3. Run `huggingface-cli login`
4. Paste your [Hugging Face User Access Token](https://huggingface.co/docs/hub/en/security-tokens) to login

## Run the App

Run the chatbot app using the following command:

```
> python app.py
```

The chatbot app will start with the default settings, which uses `DirectML` as the backend to run the `Phi-3` model for inference using `float16`. The app will automatically download `Phi-3-4k-instruct` on the first run from the default `hf_model` which is set to `microsoft/Phi-3-4k-instruct`.

This model is optimized to take advantage of DirectML operators and to use the custom DirectML graph implementations for Rotary Positional Embedding (RoPE), Multi-Head Attention (MHA), and the Feedforward layers (MLP).

When you run this code, a local URL will be displayed on the console along the following lines:

```
Using device=privateuseone:0, is_llama_3=False, is_phi_2=False
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Running on local URL:  http://127.0.0.1:7860
--------
```

You should see results such as this, when running for the first time and downloading the model:

```
checkpoints\microsoft\Phi-3-mini-4k-instruct\model.pth doesnt exist. Downloading and converting from huggingface hub
.gitattributes: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.52k/1.52k [00:00<?, ?B/s]
NOTICE.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.77k/1.77k [00:00<?, ?B/s]
LICENSE: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.08k/1.08k [00:00<?, ?B/s]
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17.2k/17.2k [00:00<?, ?B/s]
sample_finetune.py: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 6.34k/6.34k [00:00<?, ?B/s]
modeling_phi3.py: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 73.8k/73.8k [00:00<00:00, 5.90MB/s]
SECURITY.md: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.66k/2.66k [00:00<?, ?B/s]
CODE_OF_CONDUCT.md: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 444/444 [00:00<?, ?B/s]
Fetching 19 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.79it/s]
Saving checkpoint to checkpoints\microsoft\Phi-3-mini-4k-instruct\model.pth
Using device=privateuseone:0, is_llama_3=False, is_phi_2=False
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Running on local URL:  http://127.0.0.1:7860
--------

To create a public link, set `share=True` in `launch()`.
```

Open [http://localhost:7860](http://localhost:7860) (or the local URL you see) in a browser to interact with the chatbot.

### Change model precision

To run the model using `float32` precision, pass `--precision float32` to `app.py`.

```
> python app.py --precision float32
```

### Change the model

You can also select another model to run (`microsoft/Phi-3.5-mini-instruct`, `microsoft/Phi-3-mini-4k-instruct`, `microsoft/phi-2`, `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mistral-7B-Instruct-v0.1`).

For example to run `Mistral-7B-Instruct-v0.1` use the following command:

```
> python app.py --precision float16 --hf_model "mistralai/Mistral-7B-Instruct-v0.1"
```

You should see the result such as this:

```
> python app.py --precision float16 --hf_model "mistralai/Mistral-7B-Instruct-v0.1"
checkpoints\mistralai\Mistral-7B-Instruct-v0.1\model.pth doesnt exist. Downloading and converting from huggingface hub
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.90k/3.90k [00:00<?, ?B/s] 
model.safetensors.index.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25.1k/25.1k [00:00<?, ?B/s] 
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 571/571 [00:00<?, ?B/s] 
.gitattributes: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.52k/1.52k [00:00<00:00, 97.0kB/s] 
special_tokens_map.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 72.0/72.0 [00:00<?, ?B/s] 
pytorch_model.bin.index.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23.9k/23.9k [00:00<?, ?B/s] 
generation_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 116/116 [00:00<?, ?B/s] 
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.47k/1.47k [00:00<?, ?B/s] 
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.80M/1.80M [00:00<00:00, 8.51MB/s] 
tokenizer.model: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 493k/493k [00:00<00:00, 15.8MB/s] 
model-00002-of-00002.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.54G/4.54G [04:42<00:00, 16.1MB/s] 
pytorch_model-00002-of-00002.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.06G/5.06G [04:51<00:00, 17.4MB/s] 
model-00001-of-00002.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 9.94G/9.94G [07:17<00:00, 22.7MB/s] 
pytorch_model-00001-of-00002.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 9.94G/9.94G [07:19<00:00, 22.6MB/s] 
Fetching 14 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14/14 [07:22<00:00, 31.62s/it] 
Saving checkpoint to checkpoints\mistralai\Mistral-7B-Instruct-v0.1\model.pth
Using device=privateuseone:0, is_llama_3=False, is_phi_2=False
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

## App Settings

Following is a list of the basic settings supported by `app.py`:

| Flag                   | Description                                                  | Default |
| ---------------------- | ------------------------------------------------------------ | ------- |
| `--help`               | Show this help message. | N/A |
| `--hf_model`           | Specify the model to downloading using the Hugging Face Repository ID. | `microsoft/Phi-3-mini-4k-instruct` |
| `--precision`          | Model precision to use during generation. Options: [`float16`, `float32`] | `float16` |
| `--checkpoint_path`    | Path to converted PyTorch model checkpoint. | `checkpoints/{hf_model}/model.pth` |
| `--max_context_length` | Max prompt length including the history. If exceeded, history is clipped starting from the first (user, assistant) pair. | `1500` |
| `--disable_history`    | Disable the chat history during generation. | Enabled |
| `--max_pos_emb`        | Maximum Position to scale Phi-3.5 position encodings. | 8192 |

>⚠️ **NOTE**: The app uses the checkpoint path to determine the correct transformer model to load. The model path must specify the Hugging Face model ID included in the path name. For example:

- `checkpoints/microsoft/phi-2/model.pth`
- `checkpoints/microsoft/Phi-3-mini-4k-instruct/model.pth`
- `checkpoints/microsoft/Phi-3.5-mini-instruct/model.pth`
- `checkpoints/mistralai/Mistral-7B-v0.1/model.pth`
- `checkpoints/mistralai/Mistral-7B-Instruct-v0.1/model.pth`
- `checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth`
- `checkpoints/meta-llama/Meta-Llama-3-8B/model.pth`
- `checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/model.pth`
- `checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth`
- `checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct/model.pth`

## _[Optional]_ Prepare the Supported Models
This step is optional as `app.py` script in [Run the App](#run-the-app) section handles both downloading and optimizing a PyTorch model with DirectML.

We offer two methods for preparing PyTorch models:

### Use `download_and_convert.py` to download a language model:

```
> python .\scripts\download_and_convert.py --hf_model "microsoft/Phi-3-mini-4k-instruct"
```

After the model is downloaded and converted, you can pass the following parameter to `app.py` to run the language model:

```
> python app.py --hf_model "microsoft/Phi-3-mini-4k-instruct"
```



## External Links and Model Licenses
- [Phi-2 Hugging Face Repository](https://huggingface.co/microsoft/phi-2)    
This sample uses the Phi-2 model, which has been optimized to work with PyTorch-DirectML. This model is licensed under the [MIT license](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE).  If you comply with the license, you have the rights described therein. By using the Sample, you accept the terms.
- [Phi-3 Hugging Face Repository](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)   
This sample uses the Phi-3 model, which has been optimized to work with PyTorch-DirectML. This model is licensed under the [MIT license](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE).  If you comply with the license, you have the rights described therein. By using the Sample, you accept the terms.
- [Phi-3.5 Hugging Face Repository](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)   
This sample uses the Phi-3.5 model, which has been optimized to work with PyTorch-DirectML. This model is licensed under the [MIT license](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/LICENSE).  If you comply with the license, you have the rights described therein. By using the Sample, you accept the terms.
- [LLaMA 2 Hugging Face Repository](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)    
This sample uses the Llama-2 model, which has been optimized to work with PyTorch-DirectML. This model is licensed under the[ LLAMA 2 COMMUNITY LICENSE AGREEMENT](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/LICENSE.txt). For terms of use, please visit: Llama 2 - [Acceptable Use Policy - Meta AI](https://ai.meta.com/llama/use-policy/). If you comply with the license and terms of use, you have the rights described therein. By using the Sample, you accept the terms.
- [LLaMA 3 Hugging Face Repository](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)   
This sample uses the Llama-3 model, which has been optimized to work with PyTorch-DirectML. This model is licensed under the 
[LLAMA 3 COMMUNITY LICENSE AGREEMENT](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE). For terms of use, please visit: [Meta Llama 3 Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/). If you comply with the license and terms of use, you have the rights described therein. By using the Sample, you accept the terms.
- [LLaMA 3.1 Hugging Face Repository](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)   
This sample uses the Llama-3.1 model, which has been optimized to work with PyTorch-DirectML. This model is licensed under the 
[LLAMA 3.1 COMMUNITY LICENSE AGREEMENT](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE). For terms of use, please visit: [Meta Llama 3.1 Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy/). If you comply with the license and terms of use, you have the rights described therein. By using the Sample, you accept the terms.
- [Mistral 7B Hugging Face Repository](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)   
This sample uses the Mistral model, which has been optimized to work with PyTorch-DirectML. This model is licensed under the [Apache-2.0 license](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md). If you comply with the license, you have the rights described therein. By using the Sample, you accept the terms.
- [PyTorch gpt-fast Source Code](https://github.com/pytorch-labs/gpt-fast/)



