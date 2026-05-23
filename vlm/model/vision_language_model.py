"""
VisionLanguageModel for pre-saved embeddings.
Adapted from NephrologyKG/model/vision_language_model.py with one key change:
  vision_hidden_dim = 768  (RAD-DINO, not the NLST encoder's 1024)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 — used in forward
from safetensors.torch import save_file as safetensors_save_file
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftModel
from utils.huggingface_utils import convert_meta_to_tensor


class VisionLanguageModel(nn.Module):
    def __init__(self, vision_model, language_model, img_token_id, img_tokens=1,
                 num_proj_layers=1, create_projection_layer=True,
                 create_self_attn_block=False, create_x_attn_block=False,
                 num_attn_layers=1, num_attn_heads=12, add_attn_mlp=True,
                 num_x_attn_heads=12, add_x_attn_mlp=True, x_attn_query="text",
                 add_multitask=False, add_multitask_unknown=False, multitask_wt=1.0,
                 tokenizer=None, image_size=(1, 3, 224, 224)):
        super().__init__()
        self.vision_model   = vision_model
        self.language_model = language_model
        self.img_token_id   = img_token_id
        self.ignore_token_id = -100
        self.img_tokens     = img_tokens
        self.add_multitask  = add_multitask
        self.add_multitask_unknown = add_multitask_unknown
        self.multitask_wt   = multitask_wt
        self.tokenizer      = tokenizer

        # RAD-DINO produces 768-d embeddings
        self.vision_hidden_dim = 768

        self.language_projection = None
        if create_projection_layer:
            self._build_projection(num_proj_layers)

        # Multitask heads: trained jointly with the language model.
        # Both heads operate on the LLM last-hidden-state at the EOS token position,
        # following the pattern in NephrologyKG/nlst_trainer.
        self.tbr_regression_head  = None
        self.genotype_head        = None
        if add_multitask:
            llm_dim = self.language_model.config.hidden_size
            # Regression: predict up to 4 future TBR values (W15/18/20 + padding slot)
            self.tbr_regression_head = nn.Sequential(
                nn.Linear(llm_dim, 256),
                nn.GELU(),
                nn.Linear(256, 4),
            )
            # Classification: predict genotype (0=WT, 1=KO) — binary BCE
            self.genotype_head = nn.Linear(llm_dim, 1)

        print(f"VisionLanguageModel: vision_dim={self.vision_hidden_dim}, "
              f"llm_dim={self.language_model.config.hidden_size}, "
              f"img_tokens={img_tokens}, multitask={add_multitask}")

    def _build_projection(self, num_proj_layers):
        llm_dim = self.language_model.config.hidden_size
        if num_proj_layers == 1:
            self.language_projection = nn.Linear(self.vision_hidden_dim, llm_dim)
        elif num_proj_layers == 2:
            self.language_projection = nn.Sequential(
                nn.Linear(self.vision_hidden_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim),
            )
        else:
            raise ValueError(f"num_proj_layers must be 1 or 2, got {num_proj_layers}")

    def _eos_hidden_state(self, input_ids, hidden_states):
        """
        Extract the LLM last-hidden-state at each sequence's LAST EOS token.

        During training input_ids is question+answer, each wrapped with bos+eos by
        _add_prompt, so there are two EOS tokens per sequence — one at the question
        end and one at the answer end. We want the answer-end EOS so the heads see
        the full processed context, not just the question.

        During inference input_ids is question-only (single EOS), so last == first.

        The combined sequence seen by the LLM has (img_tokens - 1) extra positions
        inserted where the single <image> token was, so the EOS position in the
        combined sequence is offset accordingly.

        Returns feat: (B, llm_dim)
        """
        eos_mask = (input_ids == self.tokenizer.eos_token_id)        # (B, L)
        # argmax on reversed mask → position of last EOS in original sequence
        L             = input_ids.size(1)
        last_eos_idx  = (L - 1 - eos_mask.flip(dims=[1]).float().argmax(dim=1) +
                         (self.img_tokens - 1)).long()
        batch_idx     = torch.arange(input_ids.size(0), device=input_ids.device)
        return hidden_states[-1][batch_idx, last_eos_idx]            # (B, llm_dim)

    def get_image_and_text_embeddings(self, input_ids, pixel_values=None,
                                      image_features=None, attention_mask=None, labels=None):
        # Project image features into LLM embedding space
        image_features = self.language_projection(image_features)

        # Find position of <image> token
        first_seq = input_ids[0]
        if self.img_token_id not in first_seq:
            raise ValueError("Image token not found in input_ids")
        img_pos = (first_seq == self.img_token_id).nonzero(as_tuple=False).item()

        # Split text embeddings around image token
        embed = self.language_model.get_input_embeddings()
        pre  = embed(input_ids[:, :img_pos])
        post = embed(input_ids[:, img_pos + 1:])

        combined = torch.cat([pre, image_features, post], dim=1)

        if attention_mask is not None:
            img_mask = torch.ones(image_features.shape[:2], device=image_features.device)
            attention_mask = torch.cat([
                attention_mask[:, :img_pos],
                img_mask,
                attention_mask[:, img_pos + 1:],
            ], dim=1)
        else:
            attention_mask = torch.ones(combined.shape[:2], device=combined.device)

        if labels is not None:
            img_labels = torch.full(
                (labels.size(0), self.img_tokens),
                fill_value=self.ignore_token_id, device=labels.device,
            )
            labels = torch.cat([labels[:, :img_pos], img_labels, labels[:, img_pos + 1:]], dim=1)

        return combined, attention_mask, labels

    def forward(self, input_ids, pixel_values=None, image_features=None,
                attention_mask=None, labels=None, tbr_targets=None,
                genotype_label=None, **kwargs):
        combined, attention_mask, labels = self.get_image_and_text_embeddings(
            input_ids=input_ids, image_features=image_features,
            attention_mask=attention_mask, labels=labels,
        )
        outputs = self.language_model(
            inputs_embeds=combined, attention_mask=attention_mask, labels=labels,
            output_hidden_states=self.add_multitask,
        )
        loss = outputs.loss

        if self.add_multitask:
            feat = self._eos_hidden_state(input_ids, outputs.hidden_states)  # (B, llm_dim)

            # TBR regression (MSE, masked)
            if self.tbr_regression_head is not None and tbr_targets is not None:
                tbr_pred    = self.tbr_regression_head(feat)               # (B, 4)
                tbr_targets = tbr_targets.to(feat.device, dtype=feat.dtype)
                mask        = (tbr_targets >= 0).float()
                mse         = F.mse_loss(tbr_pred * mask, tbr_targets * mask, reduction="sum")
                mse         = mse / mask.sum().clamp(min=1)
                loss        = loss + self.multitask_wt * mse

            # Genotype classification (BCE, masked — label == -1 for non-genotype records)
            if self.genotype_head is not None and genotype_label is not None:
                geno_logits = self.genotype_head(feat).squeeze(-1)          # (B,)
                geno_label  = genotype_label.to(feat.device, dtype=feat.dtype)
                geno_valid  = (geno_label >= 0)
                if geno_valid.any():
                    bce  = F.binary_cross_entropy_with_logits(
                        geno_logits[geno_valid], geno_label[geno_valid]
                    )
                    loss = loss + self.multitask_wt * bce

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(self, input_ids, attention_mask, max_new_tokens,
                 pixel_values=None, image_features=None, **decoding_kwargs):
        with torch.no_grad():
            combined, attention_mask, _ = self.get_image_and_text_embeddings(
                input_ids=input_ids, image_features=image_features,
                attention_mask=attention_mask,
            )
            gen_out = self.language_model.generate(
                inputs_embeds=combined, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, use_cache=False, **decoding_kwargs,
            )

            if not self.add_multitask:
                return {"sequences": gen_out}

            # Second forward pass to obtain hidden states for multitask head logits
            lm_out = self.language_model(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            feat = self._eos_hidden_state(input_ids, lm_out.hidden_states)  # (B, llm_dim)

            return {
                "sequences":       gen_out,
                "genotype_logits": self.genotype_head(feat).squeeze(-1),    # (B,)
                "tbr_logits":      self.tbr_regression_head(feat),          # (B, 4)
            }

    def save_pretrained(self, save_directory):
        """
        Save the model checkpoint.

        LoRA adapter weights are saved via PEFT's save_pretrained into a
        'lora_adapter' subdirectory — this is the only reliable way to checkpoint
        a 4-bit quantized model, because bitsandbytes quantization tensors cannot
        be serialized and reloaded into a non-quantized shell.

        The projection layer and multitask heads (non-LLM weights) are saved
        separately in 'other_weights.bin'.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save LoRA adapter weights only. PEFT's save_pretrained on a quantized
        # model still emits bitsandbytes state keys (absmax, quant_map, etc.) into
        # adapter_model.bin, which then can't be loaded into a fresh fp16/bf16 shell.
        # We filter them out by saving only the lora_ keys manually.
        lora_dir = os.path.join(save_directory, "lora_adapter")
        os.makedirs(lora_dir, exist_ok=True)
        # First let PEFT write its config files (adapter_config.json, etc.)
        self.language_model.save_pretrained(lora_dir)
        # Overwrite both adapter weight files with only the LoRA delta tensors,
        # stripping the bitsandbytes quantization state keys that PEFT includes
        # when saving a 4-bit model (absmax, quant_map, quant_state, etc.).
        lora_state = {k: v.contiguous().cpu()
                      for k, v in self.language_model.state_dict().items()
                      if "lora_" in k}
        torch.save(lora_state, os.path.join(lora_dir, "adapter_model.bin"))
        safetensors_save_file(lora_state, os.path.join(lora_dir, "adapter_model.safetensors"))

        # Save projection layer + multitask heads
        other = {}
        if self.language_projection is not None:
            for k, v in self.language_projection.state_dict().items():
                other[f"language_projection.{k}"] = v.cpu()
        if self.tbr_regression_head is not None:
            for k, v in self.tbr_regression_head.state_dict().items():
                other[f"tbr_regression_head.{k}"] = v.cpu()
        if self.genotype_head is not None:
            for k, v in self.genotype_head.state_dict().items():
                other[f"genotype_head.{k}"] = v.cpu()
        torch.save(other, os.path.join(save_directory, "other_weights.bin"))

    @classmethod
    def from_pretrained(cls, save_directory, vision_model, language_model, img_token_id,
                        img_tokens=1, num_proj_layers=1, create_self_attn_block=False,
                        create_x_attn_block=False, num_attn_layers=1, num_attn_heads=12,
                        add_attn_mlp=True, num_x_attn_heads=12, add_x_attn_mlp=True,
                        x_attn_query="text", add_multitask=False, add_multitask_unknown=False,
                        multitask_wt=1.0, load_projection_matrix=False, tokenizer=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        lora_dir       = os.path.join(save_directory, "lora_adapter") if save_directory else None
        other_bin      = os.path.join(save_directory, "other_weights.bin") if save_directory else None
        use_peft_load  = lora_dir is not None and os.path.isdir(lora_dir)

        if use_peft_load:
            # Reload the LoRA adapter onto the already-quantized base model (language_model
            # passed in is the fresh quantized+LoRA shell from _load_llm).  We discard
            # that shell's LoRA weights and replace them with the saved adapter.
            base_llm = language_model.base_model.model  # unwrap PeftModel → base LLM
            language_model = PeftModel.from_pretrained(base_llm, lora_dir)

        model = cls(
            vision_model=vision_model, language_model=language_model,
            img_token_id=img_token_id, img_tokens=img_tokens,
            num_proj_layers=num_proj_layers, add_multitask=add_multitask,
            add_multitask_unknown=add_multitask_unknown, multitask_wt=multitask_wt,
            tokenizer=tokenizer,
        )

        if other_bin is not None and os.path.exists(other_bin):
            other = torch.load(other_bin, map_location=device)
            other = convert_meta_to_tensor(other, device=device)

            proj_sd = {k[len("language_projection."):]: v
                       for k, v in other.items() if k.startswith("language_projection.")}
            tbr_sd  = {k[len("tbr_regression_head."):]: v
                       for k, v in other.items() if k.startswith("tbr_regression_head.")}
            geno_sd = {k[len("genotype_head."):]: v
                       for k, v in other.items() if k.startswith("genotype_head.")}

            if proj_sd and model.language_projection is not None:
                model.language_projection.load_state_dict(proj_sd, strict=True)
                model.language_projection.to(device)
            if tbr_sd and model.tbr_regression_head is not None:
                model.tbr_regression_head.load_state_dict(tbr_sd, strict=True)
                model.tbr_regression_head.to(device)
            if geno_sd and model.genotype_head is not None:
                model.genotype_head.load_state_dict(geno_sd, strict=True)
                model.genotype_head.to(device)

        return model
