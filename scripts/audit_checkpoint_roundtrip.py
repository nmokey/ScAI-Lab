"""CPU-only, offline check of the production VLM checkpoint round trip.

Uses a tiny random Llama and deliberately nonzero adapters; downloads no weights.
Writes only temporary checkpoint files and prints measurements as JSON.
"""
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import peft
import transformers
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vlm"))
from model.vision_language_model import VisionLanguageModel


def main():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    base = LlamaForCausalLM(cfg)
    base_weights = copy.deepcopy(base.state_dict())
    lcfg = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0, task_type="CAUSAL_LM")
    lm = get_peft_model(base, lcfg)
    with torch.no_grad():
        for name, p in lm.named_parameters():
            if "lora_" in name:
                p.normal_(0, 0.15)
    kwargs = dict(vision_model=None, img_token_id=3, img_tokens=4,
                  add_multitask=True, tokenizer=SimpleNamespace(eos_token_id=2),
                  pool_at="question_eos")
    model = VisionLanguageModel(language_model=lm, **kwargs).eval()
    ids = torch.tensor([[1, 3, 7, 8, 2]])
    images = torch.randn(1, 4, 768)

    def outputs(m):
        emb, mask, _ = m.get_image_and_text_embeddings(ids, image_features=images)
        with torch.no_grad():
            out = m.language_model(inputs_embeds=emb, attention_mask=mask,
                                   output_hidden_states=True)
            feat = m._eos_hidden_state(ids, out.hidden_states, torch.tensor([ids.size(1)-1]))
            return out.logits, m.genotype_head(feat), m.tbr_regression_head(feat)

    before = outputs(model)
    expected = {k: v.detach().clone() for k, v in lm.state_dict().items() if "lora_" in k}
    with tempfile.TemporaryDirectory(prefix="scai-checkpoint-audit-") as tmp:
        model.save_pretrained(tmp)
        fresh = LlamaForCausalLM(cfg)
        fresh.load_state_dict(base_weights)
        shell = get_peft_model(fresh, lcfg)
        loaded = VisionLanguageModel.from_pretrained(tmp, language_model=shell, **kwargs).eval()
        after = outputs(loaded)
        actual = loaded.language_model.state_dict()
        errors = {k: float((v - actual[k]).abs().max()) for k, v in expected.items()}
        # Independent control: stock PEFT serialization on identical adapters.
        standard = str(Path(tmp) / "standard")
        lm.save_pretrained(standard)
        fresh_control = LlamaForCausalLM(cfg)
        fresh_control.load_state_dict(base_weights)
        control = peft.PeftModel.from_pretrained(fresh_control, standard)
        control_errors = {k: float((v - control.state_dict()[k]).abs().max())
                          for k, v in expected.items()}
        # Repair only this disposable checkpoint, never the production files.
        lm.save_pretrained(str(Path(tmp) / "lora_adapter"))
        fresh_fixed = LlamaForCausalLM(cfg)
        fresh_fixed.load_state_dict(base_weights)
        fixed = VisionLanguageModel.from_pretrained(
            tmp, language_model=get_peft_model(fresh_fixed, lcfg), **kwargs).eval()
        fixed_out = outputs(fixed)
        assert max(errors.values()) == 0, "Production reload lost trained adapter tensors"
        assert max(control_errors.values()) == 0
        assert all(torch.equal(a,b) for a,b in zip(before,after)), "Production predictions changed after reload"
        assert all(v.norm()>0 for k,v in actual.items() if "lora_B" in k)
        print(json.dumps({
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "adapter_tensor_count": len(expected),
            "production_adapter_max_errors": errors,
            "stock_peft_adapter_max_error": max(control_errors.values()),
            "production_output_max_errors": dict(zip(
                ["language_logits", "genotype_head", "tbr_head"],
                [float((a-b).abs().max()) for a, b in zip(before, after)])),
            "loaded_lora_B_norms": {k: float(v.norm()) for k, v in actual.items()
                                    if "lora_B" in k},
            "correctly_serialized_checkpoint_output_max_errors": dict(zip(
                ["language_logits", "genotype_head", "tbr_head"],
                [float((a-b).abs().max()) for a,b in zip(before, fixed_out)])),
        }, indent=2))


if __name__ == "__main__":
    main()
