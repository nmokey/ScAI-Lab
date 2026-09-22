"""Strict adapter-only checkpoints, including recovery of the old raw-key format."""
import re
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file


def validate_tensors(state, expected, label):
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    if missing or extra:
        raise ValueError(f"{label} keys differ: missing={sorted(missing)}, unexpected={sorted(extra)}")
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"{label} shape mismatch for {key}: {value.shape} != {expected[key].shape}")
        if value.is_meta or not torch.isfinite(value).all():
            raise ValueError(f"{label} contains unavailable or nonfinite tensor: {key}")


def canonicalize_legacy_adapter(state):
    """Only repair the known single 'default' LoRA namespace; never guess others."""
    repaired = {}
    for key, value in state.items():
        new_key = re.sub(r"\.(lora_[AB])\.default\.weight$", r".\1.weight", key)
        if new_key in repaired:
            raise ValueError(f"Colliding adapter keys after repair: {new_key}")
        repaired[new_key] = value
    return repaired


def read_adapter(directory):
    directory = Path(directory)
    safe, binary = directory / "adapter_model.safetensors", directory / "adapter_model.bin"
    if safe.exists():
        state = load_file(str(safe))
        if binary.exists():
            alternate = torch.load(binary, map_location="cpu", weights_only=True)
            validate_tensors(alternate, state, "Duplicate adapter file")
            if any(not torch.equal(state[k], alternate[k]) for k in state):
                raise ValueError("The two adapter weight files disagree; select the intended checkpoint explicitly")
        return state
    if binary.exists():
        return torch.load(binary, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights in {directory}")


def load_adapter_strict(language_model, directory, *, allow_legacy=False):
    config = PeftConfig.from_pretrained(str(directory), local_files_only=True)
    if isinstance(language_model, PeftModel):
        if set(language_model.peft_config) != {"default"}:
            raise ValueError("Checkpoint loading requires a single default adapter")
        # Loading into the prepared shell avoids wrapping injected LoRA layers twice.
        actual = language_model.peft_config["default"].to_dict()
        saved = config.to_dict()
        ignored = {"inference_mode", "base_model_name_or_path", "revision", "auto_mapping"}
        differences = [k for k in saved if k not in ignored and actual.get(k) != saved[k]]
        if differences:
            raise ValueError(f"Adapter configuration mismatch: {differences}")
    else:
        language_model = get_peft_model(language_model, config)
    state = read_adapter(directory)
    if allow_legacy:
        state = canonicalize_legacy_adapter(state)
    expected = get_peft_model_state_dict(language_model, save_embedding_layers=False)
    validate_tensors(state, expected, "Adapter")
    result = set_peft_model_state_dict(language_model, state, adapter_name="default")
    missing_adapter = [k for k in result.missing_keys if "lora_" in k]
    if missing_adapter or result.unexpected_keys:
        raise ValueError(f"Adapter load failed: {missing_adapter}, {result.unexpected_keys}")
    restored = get_peft_model_state_dict(language_model, save_embedding_layers=False)
    for key, value in state.items():
        if not torch.equal(restored[key].detach().cpu(), value):
            raise ValueError(f"Adapter tensor was not restored exactly: {key}")
    language_model.set_adapter("default")
    return language_model
