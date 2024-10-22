import os
import re
from datetime import datetime
from pathlib import Path

from modules import github, shared
from modules.logging_colors import logger


# Helper function to get multiple values from shared.gradio
def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [shared.gradio[k] for k in keys]


def save_file(fname, contents):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path_str = os.path.abspath(fname)
    rel_path_str = os.path.relpath(abs_path_str, root_folder)
    rel_path = Path(rel_path_str)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: \"{fname}\"')
        return

    with open(abs_path_str, 'w', encoding='utf-8') as f:
        f.write(contents)

    logger.info(f'Saved \"{abs_path_str}\".')


def delete_file(fname):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path_str = os.path.abspath(fname)
    rel_path_str = os.path.relpath(abs_path_str, root_folder)
    rel_path = Path(rel_path_str)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: \"{fname}\"')
        return

    if rel_path.exists():
        rel_path.unlink()
        logger.info(f'Deleted \"{fname}\".')


def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"


def atoi(text):
    return int(text) if text.isdigit() else text.lower()


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_available_models():
    model_list = []
    for item in list(Path(f'{shared.args.model_dir}/').glob('*')):
        if not item.name.endswith(('.txt', '-np', '.pt', '.json', '.yaml', '.py')) and 'llama-tokenizer' not in item.name:
            model_list.append(item.name)

    return ['None'] + sorted(model_list, key=natural_keys)


def get_available_ggufs():
    model_list = []
    for item in Path(f'{shared.args.model_dir}/').glob('*'):
        if item.is_file() and item.name.lower().endswith(".gguf"):
            model_list.append(item.name)

    return ['None'] + sorted(model_list, key=natural_keys)


def get_available_presets():
    return sorted(set((k.stem for k in Path('presets').glob('*.yaml'))), key=natural_keys)


def get_available_prompts():
    prompt_files = list(Path('prompts').glob('*.txt'))
    sorted_files = sorted(prompt_files, key=lambda x: x.stat().st_mtime, reverse=True)
    prompts = [file.stem for file in sorted_files]
    prompts.append('None')
    return prompts


def get_available_characters():
    paths = (x for x in Path('characters').iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_instruction_templates():
    path = "instruction-templates"
    paths = []
    if os.path.exists(path):
        paths = (x for x in Path(path).iterdir() if x.suffix in ('.json', '.yaml', '.yml'))

    return ['None'] + sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_extensions():
    extensions = sorted(set(map(lambda x: x.parts[1], Path('extensions').glob('*/script.py'))), key=natural_keys)
    extensions = [v for v in extensions if v not in github.new_extensions]
    return extensions


def get_available_loras():
    return ['None'] + sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=natural_keys)


def get_datasets(path: str, ext: str):
    # include subdirectories for raw txt files to allow training from a subdirectory of txt files
    if ext == "txt":
        return ['None'] + sorted(set([k.stem for k in list(Path(path).glob('*.txt')) + list(Path(path).glob('*/')) if k.stem != 'put-trainer-datasets-here']), key=natural_keys)

    return ['None'] + sorted(set([k.stem for k in Path(path).glob(f'*.{ext}') if k.stem != 'put-trainer-datasets-here']), key=natural_keys)


def get_available_chat_styles():
    return sorted(set(('-'.join(k.stem.split('-')[1:]) for k in Path('css').glob('chat_style*.css'))), key=natural_keys)


def get_available_grammars():
    return ['None'] + sorted([item.name for item in list(Path('grammars').glob('*.gbnf'))], key=natural_keys)


def transform_settings_to_state(settings):
    state = {
        'max_new_tokens': settings.get('max_new_tokens', 512),
        'auto_max_new_tokens': settings.get('auto_max_new_tokens', False),
        'max_tokens_second': settings.get('max_tokens_second', 0),
        'max_updates_second': settings.get('max_updates_second', 0),
        'prompt_lookup_num_tokens': settings.get('prompt_lookup_num_tokens', 0),
        'seed': settings.get('seed', -1),
        'temperature': 1,
        'temperature_last': False,
        'dynamic_temperature': False,
        'dynatemp_low': 1,
        'dynatemp_high': 1,
        'dynatemp_exponent': 1,
        'smoothing_factor': 0,
        'smoothing_curve': 1,
        'top_p': 1,
        'min_p': 0.05,
        'top_k': 0,
        'typical_p': 1,
        'epsilon_cutoff': 0,
        'eta_cutoff': 0,
        'repetition_penalty': 1,
        'presence_penalty': 0,
        'frequency_penalty': 0,
        'repetition_penalty_range': 1024,
        'encoder_repetition_penalty': 1,
        'no_repeat_ngram_size': 0,
        'dry_multiplier': 0,
        'dry_base': 1.75,
        'dry_allowed_length': 2,
        'dry_sequence_breakers': '"\\n", ":", "\\""',
        'xtc_threshold': 0.1,
        'xtc_probability': 0,
        'do_sample': True,
        'penalty_alpha': 0,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'grammar_string': '',
        'negative_prompt': settings.get('negative_prompt', ''),
        'guidance_scale': 1,
        'add_bos_token': settings.get('add_bos_token', True),
        'ban_eos_token': settings.get('ban_eos_token', False),
        'custom_token_bans': settings.get('custom_token_bans', ''),
        'sampler_priority': ('repetition_penalty\npresence_penalty\nfrequency_penalty\ndry\ntemperature'
                             '\ndynamic_temperature\nquadratic_sampling\ntop_k\ntop_p'
                             '\ntypical_p\nepsilon_cutoff\neta_cutoff\ntfs\ntop_a\nmin_p'
                             '\nmirostat\nxtc\nencoder_repetition_penalty\nno_repeat_ngram'),
        'truncation_length': settings.get('truncation_length', 4096),
        'custom_stopping_strings': settings.get('custom_stopping_strings', ''),
        'skip_special_tokens': settings.get('skip_special_tokens', True),
        'stream': settings.get('stream', True),
        'tfs': 1,
        'top_a': 0,
        'textbox': '',
        'start_with': settings.get('start_with', ''),
        'character_menu': settings.get('character', 'Assistant'),
        'history': {'internal': [], 'visible': []},
        'unique_id': '',
        'name1': settings.get('name1', 'You'),
        'user_bio': settings.get('user_bio', ''),
        'name2': 'AI',
        'greeting': 'How can I help you today?',
        'context': ('The following is a conversation with an AI Large Language Model. '
                    'The AI has been trained to answer questions, provide recommendations, '
                    'and help with decision making. The AI follows user requests. '
                    'The AI thinks outside the box.'),
        'mode': settings.get('mode', 'chat-instruct'),
        'custom_system_message': settings.get('custom_system_message', ''),
        'instruction_template_str': settings.get('instruction_template_str', ''),
        'chat_template_str': settings.get('chat_template_str', ''),
        'chat_style': settings.get('chat_style', 'cai-chat'),
        'chat-instruct_command': settings.get('chat-instruct_command', ''),
        'textbox-notebook': '',
        'textbox-default': '',
        'output_textbox': '',
        'prompt_menu-default': 'None',
        'prompt_menu-notebook': 'None',
        'loader': 'llama.cpp',
        'filter_by_loader': 'llama.cpp',
        'cpu_memory': 0,
        'auto_devices': False,
        'disk': False,
        'cpu': False,
        'bf16': False,
        'load_in_8bit': False,
        'trust_remote_code': False,
        'no_use_fast': False,
        'use_flash_attention_2': False,
        'use_eager_attention': False,
        'load_in_4bit': False,
        'compute_dtype': 'float16',
        'quant_type': 'nf4',
        'use_double_quant': False,
        'wbits': 'None',
        'groupsize': 'None',
        'triton': False,
        'desc_act': False,
        'no_inject_fused_mlp': False,
        'no_use_cuda_fp16': False,
        'disable_exllama': False,
        'disable_exllamav2': False,
        'cfg_cache': False,
        'no_flash_attn': False,
        'no_xformers': False,
        'no_sdpa': False,
        'num_experts_per_token': 2,
        'cache_8bit': False,
        'cache_4bit': False,
        'autosplit': False,
        'enable_tp': False,
        'threads': 0,
        'threads_batch': 0,
        'n_batch': 512,
        'no_mmap': False,
        'mlock': False,
        'no_mul_mat_q': False,
        'n_gpu_layers': 33,
        'tensor_split': '',
        'n_ctx': 4096,
        'gpu_split': '',
        'max_seq_len': 2048,
        'compress_pos_emb': 1,
        'alpha_value': 1,
        'rope_freq_base': 0,
        'numa': False,
        'logits_all': False,
        'no_offload_kqv': False,
        'row_split': False,
        'tensorcores': False,
        'flash_attn': False,
        'streaming_llm': False,
        'attention_sink_size': 5,
        'hqq_backend': 'PYTORCH_COMPILE',
        'cpp_runner': False
    }

    return state
