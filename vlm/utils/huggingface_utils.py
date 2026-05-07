import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_tokenizer_from_huggingface(tokenizer_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except ValueError:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_llm_from_huggingface(model_name, use_quantization=False, r=16, lora_alpha=32,
                               target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"),
                               lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"):
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
    model.gradient_checkpointing_enable()

    if r:
        config = LoraConfig(
            r=r, lora_alpha=lora_alpha, target_modules=target_modules,
            lora_dropout=lora_dropout, bias=bias, task_type=task_type,
        )
        model = get_peft_model(model, config)
    return model


def convert_meta_to_tensor(state_dict, device='cpu'):
    for key, param in state_dict.items():
        if param.is_meta:
            state_dict[key] = torch.zeros_like(param, device=device)
    return state_dict
