from typing import List
from pydantic import BaseModel

class InputData(BaseModel):
    message: str

class StatusResponse(BaseModel):
    message: str

class DownloadRequest(BaseModel):
    repo_id: str
    specific_file: str

class ExtensionInput(BaseModel):
    github_url: str

class HistoryItem(BaseModel):
    internal: List[List[str]]
    visible: List[List[str]]

class ModelInput(BaseModel):
    model_name: str
    max_new_tokens: int = 512
    auto_max_new_tokens: bool = False
    max_tokens_second: int = 0
    max_updates_second: int = 0
    prompt_lookup_num_tokens: int = 0
    seed: int = -1
    temperature: float = 1.0
    temperature_last: bool = False
    dynamic_temperature: bool = False
    dynatemp_low: float = 1.0
    dynatemp_high: float = 1.0
    dynatemp_exponent: float = 1.0
    smoothing_factor: float = 0.0
    smoothing_curve: float = 1.0
    top_p: float = 1.0
    min_p: float = 0.05
    top_k: int = 0
    typical_p: float = 1.0
    epsilon_cutoff: int = 0
    eta_cutoff: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty_range: int = 1024
    encoder_repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_sequence_breakers: str = "\\n, :, \", *"
    xtc_threshold: float = 0.1
    xtc_probability: float = 0.0
    do_sample: bool = True
    penalty_alpha: float = 0.0
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    grammar_string: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 1.0
    add_bos_token: bool = True
    ban_eos_token: bool = False
    custom_token_bans: str = ""
    sampler_priority: str
    truncation_length: int = 4096
    custom_stopping_strings: str = ""
    skip_special_tokens: bool = True
    stream: bool = True
    tfs: float = 1.0
    top_a: float = 0.0
    textbox: str = ""
    start_with: str = ""
    character_menu: str = "Assistant"
    history: HistoryItem
    unique_id: str
    name1: str = "You"
    user_bio: str = ""
    name2: str = "AI"
    greeting: str = "How can I help you today?"
    context: str
    mode: str = "chat-instruct"
    custom_system_message: str = ""
    instruction_template_str: str
    chat_template_str: str
    chat_style: str = "cai-chat"
    chat_instruct_command: str
    textbox_notebook: str = "Common sense questions and answers\n\nQuestion: \nFactual answer:"
    textbox_default: str = "Common sense questions and answers\n\nQuestion: \nFactual answer:"
    output_textbox: str = ""
    prompt_menu_default: str = "QA"
    prompt_menu_notebook: str = "QA"
    loader: str = "llama.cpp"
    filter_by_loader: str = "llama.cpp"
    cpu_memory: int = 0
    auto_devices: bool = False
    disk: bool = False
    cpu: bool = False
    bf16: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    no_use_fast: bool = False
    use_flash_attention_2: bool = False
    use_eager_attention: bool = False
    load_in_4bit: bool = False
    compute_dtype: str = "float16"
    quant_type: str = "nf4"
    use_double_quant: bool = False
    wbits: str = "None"
    groupsize: str = "None"
    triton: bool = False
    desc_act: bool = False
    no_inject_fused_mlp: bool = False
    no_use_cuda_fp16: bool = False
    disable_exllama: bool = False
    disable_exllamav2: bool = False
    cfg_cache: bool = False
    no_flash_attn: bool = False
    no_xformers: bool = False
    no_sdpa: bool = False
    num_experts_per_token: int = 2
    cache_8bit: bool = False
    cache_4bit: bool = False
    autosplit: bool = False
    enable_tp: bool = False
    threads: int = 0
    threads_batch: int = 0
    n_batch: int = 512
    no_mmap: bool = False
    mlock: bool = False
    no_mul_mat_q: bool = False
    n_gpu_layers: int = 33
    tensor_split: str = ""
    n_ctx: int = 4096
    gpu_split: str = ""
    max_seq_len: int = 2048
    compress_pos_emb: int = 1
    alpha_value: float = 1.0
    rope_freq_base: int = 0
    numa: bool = False
    logits_all: bool = False
    no_offload_kqv: bool = False
    row_split: bool = False
    tensorcores: bool = False
    flash_attn: bool = False
    streaming_llm: bool = False
    attention_sink_size: int = 5
    hqq_backend: str = "PYTORCH_COMPILE"
    cpp_runner: bool = False


class ExtensionSettings(BaseModel):
    extension_names: List[str]
    active_flags: List[str]