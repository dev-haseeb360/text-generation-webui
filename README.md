# Text generation web UI

A Gradio web UI for Large Language Models.

## Features

- Supports multiple text generation backends in one UI/API, including [Transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), and [ExLlamaV2](https://github.com/turboderp/exllamav2). [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [HQQ](https://github.com/mobiusml/hqq), and [AQLM](https://github.com/Vahe1994/AQLM) are also supported but you need to install them manually.
- OpenAI-compatible API with Chat and Completions endpoints – see [examples](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#examples).
- Automatic prompt formatting using Jinja2 templates.
- Three chat modes: `instruct`, `chat-instruct`, and `chat`, with automatic prompt templates in `chat-instruct`.
- "Past chats" menu to quickly switch between conversations.
- Free-form text generation in the Default/Notebook tabs without being limited to chat turns. You can send formatted conversations from the Chat tab to these.
- Multiple sampling parameters and generation options for sophisticated text generation control.
- Switch between different models easily in the UI without restarting.
- Simple LoRA fine-tuning tool.
- Requirements installed in a self-contained `installer_files` directory that doesn't interfere with the system environment.
- Extension support, with numerous built-in and user-contributed extensions available. See the [wiki](https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions) and [extensions directory](https://github.com/oobabooga/text-generation-webui-extensions) for details.


## Project Setup Guide

### Prerequisites
- **Python**: Ensure that Python 3.9 is installed on your system.
- **Git**: Ensure Git is installed for cloning the repository.

### How to Install

1. Clone the Repository
To get started, clone the repository to your local machine using Git:
```bash
git clone <repository-url>
```
2. Move to the Project Directory
Navigate to the project folder:
```bash
cd text-generation-webui
```
3. Set Up a Virtual Environment (Python 3.9)
It is recommended to create and activate a virtual environment to isolate dependencies. Make sure the virtual environment is created using Python 3.9.
For Linux/macOS:
```bash
python3.9 -m venv venv
source venv/bin/activate

#For Windows:
python -m venv venv
.\venv\Scripts\activate
```
4. Install the Required Dependencies
Once the virtual environment is activated, install the project dependencies from the requirements.txt file:
```bash
pip install -r requirements.txt
```
5. Run the Server
After the dependencies are installed, you can start the server by running:
```
python3 server.py
```
The application should now be up and running!

6. Once the server is running, you can access the web interface by navigating to:
```
http://127.0.0.1:8000
```

## Downloading models

Models should be placed in the folder `text-generation-webui/models`. They are usually downloaded from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).

* GGUF models are a single file and should be placed directly into `models`. Example:

```
text-generation-webui
└── models
    └── llama-2-13b-chat.Q4_K_M.gguf
```

* The remaining model types (like 16-bit transformers models and GPTQ models) are made of several files and must be placed in a subfolder. Example:

```
text-generation-webui
├── models
│   ├── lmsys_vicuna-33b-v1.3
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── pytorch_model-00001-of-00007.bin
│   │   ├── pytorch_model-00002-of-00007.bin
│   │   ├── pytorch_model-00003-of-00007.bin
│   │   ├── pytorch_model-00004-of-00007.bin
│   │   ├── pytorch_model-00005-of-00007.bin
│   │   ├── pytorch_model-00006-of-00007.bin
│   │   ├── pytorch_model-00007-of-00007.bin
│   │   ├── pytorch_model.bin.index.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.model
```

In both cases, you can use the "Model" tab of the UI to download the model from Hugging Face automatically. It is also possible to download it via the command-line with 

```
python download-model.py organization/model
```

Run `python download-model.py --help` to see all the options.
