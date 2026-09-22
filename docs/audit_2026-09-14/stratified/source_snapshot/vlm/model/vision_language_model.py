"""
VisionLanguageModel for pre-saved embeddings.
Adapted from NephrologyKG/model/vision_language_model.py with one key change:
  vision_hidden_dim = 768  (RAD-DINO, not the NLST encoder's 1024)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 — used in forward
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftModel
from utils.checkpoint_utils import load_adapter_strict, validate_tensors


class VisionLanguageModel(nn.Module):
    def __init__(self, vision_model, language_model, img_token_id, img_tokens=1,
                 num_proj_layers=1, create_projection_layer=True,
                 create_self_attn_block=False, create_x_attn_block=False,
                 num_attn_layers=1, num_attn_heads=12, add_attn_mlp=True,
                 num_x_attn_heads=12, add_x_attn_mlp=True, x_attn_query="text",
                 add_multitask=False, add_multitask_unknown=False, multitask_wt=1.0,
                 tokenizer=None, image_size=(1, 3, 224, 224), pool_at="question_eos"):
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
        if pool_at not in ("answer_eos", "question_eos"):
            raise ValueError(f"pool_at must be 'answer_eos' or 'question_eos', got {pool_at!r}")
        self.pool_at        = pool_at
        self.num_proj_layers = num_proj_layers
        self.create_projection_layer = create_projection_layer

        # Per-fold TBR normalization stats (shape: 4,). Set by the trainer after
        # computing mean/std over valid training-fold TBR slots. Registered as
        # buffers so they move with .to(device) and are saved in other_weights.bin.
        self.register_buffer("tbr_mean", torch.zeros(4))
        self.register_buffer("tbr_std",  torch.ones(4))

        # RAD-DINO produces 768-d embeddings
        self.vision_hidden_dim = 768

        self.language_projection = None
        if create_projection_layer:
            self._build_projection(num_proj_layers)

        # Multitask heads: trained jointly with the language model.
        # Both heads read the same explicit question boundary used at inference.
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

    def _eos_hidden_state(self, input_ids, hidden_states, question_end_index=None,
                          attention_mask=None):
        """Pool at an explicit question boundary (before image expansion).

        The historical name ``question_eos`` is retained in configs, but the
        boundary is the final question-prefix token, including any chat header.
        Legacy answer pooling is available only for explicit reproduction.
        """
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if valid.shape != input_ids.shape:
            raise ValueError("Pooling requires the original, unspliced attention mask")
        batch = torch.arange(input_ids.size(0), device=input_ids.device)
        if self.pool_at == "question_eos":
            if question_end_index is None:
                raise ValueError("question_end_index is required for question-only head supervision")
            index = torch.as_tensor(question_end_index, device=input_ids.device)
            if index.dtype not in (torch.int32, torch.int64) or index.shape != (input_ids.size(0),):
                raise ValueError("question_end_index must be an integer vector with one index per sequence")
            if ((index < 0) | (index >= input_ids.size(1))).any():
                raise ValueError("question_end_index is outside the sequence (possibly truncated)")
            if not valid[batch, index].all():
                raise ValueError("question_end_index points into padding")
        else:
            eos = (input_ids == self.tokenizer.eos_token_id) & valid
            if not eos.any(dim=1).all():
                raise ValueError("Legacy answer pooling requires an unmasked EOS in every sequence")
            positions = torch.arange(input_ids.size(1), device=input_ids.device)
            index = positions.expand_as(input_ids).masked_fill(~eos, -1).max(dim=1).values
        image_mask = input_ids == self.img_token_id
        if not (image_mask.sum(dim=1) == 1).all():
            raise ValueError("Each sequence must contain exactly one image token")
        image_index = image_mask.long().argmax(dim=1)
        if (index <= image_index).any():
            raise ValueError("Head pooling boundary must follow the image token")
        index = index + self.img_tokens - 1
        return hidden_states[-1][batch, index]

    def get_image_and_text_embeddings(self, input_ids, pixel_values=None,
                                      image_features=None, attention_mask=None, labels=None):
        if image_features is None or image_features.shape != (input_ids.size(0), self.img_tokens, self.vision_hidden_dim):
            raise ValueError(f"Expected image_features shaped (batch, {self.img_tokens}, {self.vision_hidden_dim})")
        # Project image features into LLM embedding space
        image_features = self.language_projection(image_features)

        # Find position of the <image> token.
        # F7: img_pos is derived from row 0 and applied to the whole batch, so every row
        # must place the placeholder at the same index. That holds for the fixed-length
        # prefix _add_prompt emits today, but `beg_prompt` is config-settable, and the
        # failure mode was silent -- a misplaced row kept its placeholder and lost real
        # text tokens. Assert the precondition instead of assuming it.
        img_mask_ids = input_ids == self.img_token_id
        per_row = img_mask_ids.sum(dim=1)
        if not (per_row == 1).all():
            raise ValueError(
                f"Each sequence must contain exactly one image token "
                f"(id={self.img_token_id}); got counts {per_row.tolist()}"
            )
        positions = img_mask_ids.float().argmax(dim=1)
        if not (positions == positions[0]).all():
            raise ValueError(
                f"Image token must be at the same index in every sequence of a batch; "
                f"got positions {positions.tolist()}. The splice uses a single offset "
                f"for the whole batch, so mixed positions would corrupt every row after "
                f"the first."
            )
        img_pos = int(positions[0].item())

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
                genotype_label=None, question_end_index=None, **kwargs):
        original_mask = attention_mask
        if self.add_multitask and self.pool_at == "question_eos" and labels is not None:
            if question_end_index is None:
                raise ValueError("question_end_index is required for supervised heads")
            qend = torch.as_tensor(question_end_index, device=input_ids.device)
            if qend.shape != (input_ids.size(0),) or qend.dtype not in (torch.int32, torch.int64):
                raise ValueError("question_end_index must be an integer vector")
            prefix = torch.arange(input_ids.size(1), device=input_ids.device)[None, :] <= qend[:, None]
            if ((labels != self.ignore_token_id) & prefix).any():
                raise ValueError("Question pooling boundary overlaps supervised answer tokens")
        combined, attention_mask, labels = self.get_image_and_text_embeddings(
            input_ids=input_ids, image_features=image_features,
            attention_mask=attention_mask, labels=labels,
        )
        outputs = self.language_model(
            inputs_embeds=combined, attention_mask=attention_mask, labels=labels,
            output_hidden_states=self.add_multitask,
        )
        loss = outputs.loss if outputs.loss is not None else outputs.logits.new_zeros(())

        if self.add_multitask:
            feat = self._eos_hidden_state(input_ids, outputs.hidden_states, question_end_index, original_mask)  # (B, llm_dim)

            # TBR regression (MSE on z-scored targets, masked)
            if self.tbr_regression_head is not None and tbr_targets is not None:
                tbr_pred    = self.tbr_regression_head(feat)               # (B, 4)
                tbr_targets = tbr_targets.to(feat.device, dtype=feat.dtype)
                mask        = (tbr_targets >= 0).float()
                # Normalize valid targets to ~N(0,1) using fold-level stats so
                # MSE stays O(1) regardless of raw TBR scale or multitask_wt.
                tbr_norm    = (tbr_targets - self.tbr_mean) / self.tbr_std.clamp(min=1e-6)
                mse         = F.mse_loss(tbr_pred * mask, tbr_norm * mask, reduction="sum")
                mse         = mse / mask.sum().clamp(min=1)
                loss        = loss + self.multitask_wt * mse

            # Genotype classification (BCE; -1 denotes an unavailable label)
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
                 pixel_values=None, image_features=None, question_end_index=None, **decoding_kwargs):
        original_mask = attention_mask
        # generate() accepts question-only inputs; its boundary is the last valid token.
        if question_end_index is None:
            positions = torch.arange(input_ids.size(1), device=input_ids.device)
            question_end_index = positions.expand_as(input_ids).masked_fill(~attention_mask.bool(), -1).max(dim=1).values
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
            feat = self._eos_hidden_state(input_ids, lm_out.hidden_states, question_end_index, original_mask)  # (B, llm_dim)

            return {
                "sequences":       gen_out,
                "genotype_logits": self.genotype_head(feat).squeeze(-1),    # (B,)
                "tbr_logits":      self.tbr_regression_head(feat),          # (B, 4)
            }

    def _checkpoint_config(self):
        return dict(format_version=2, img_token_id=self.img_token_id,
                    img_tokens=self.img_tokens, num_proj_layers=self.num_proj_layers,
                    create_projection_layer=self.create_projection_layer,
                    add_multitask=self.add_multitask,
                    add_multitask_unknown=self.add_multitask_unknown,
                    multitask_wt=self.multitask_wt, pool_at=self.pool_at,
                    hidden_size=self.language_model.config.hidden_size,
                    base_model_name_or_path=self.language_model.config._name_or_path)

    def _other_state(self):
        other = {}
        for name in ("language_projection", "tbr_regression_head", "genotype_head"):
            module = getattr(self, name)
            if module is not None:
                other.update({f"{name}.{k}": v for k, v in module.state_dict().items()})
        other.update(tbr_mean=self.tbr_mean, tbr_std=self.tbr_std)
        return other

    def save_pretrained(self, save_directory):
        """Save canonical PEFT weights plus all projection/head state and semantics."""
        if not isinstance(self.language_model, PeftModel):
            raise ValueError("Adapter checkpoints require a PEFT language model")
        if set(self.language_model.peft_config) != {"default"}:
            raise ValueError("Only a single default adapter is supported")
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "vlm_config.json")
        # Metadata is the completion marker: an interrupted replacement must fail
        # loading rather than mix new adapters with an older set of heads.
        if os.path.exists(config_path):
            os.remove(config_path)
        lora_dir = os.path.join(save_directory, "lora_adapter")
        self.language_model.save_pretrained(lora_dir, safe_serialization=True,
                                            save_embedding_layers=False)
        # A previous writer may have left a conflicting raw-state .bin alongside it.
        old_bin = os.path.join(lora_dir, "adapter_model.bin")
        if os.path.exists(old_bin):
            os.remove(old_bin)
        other = {k: v.detach().cpu().contiguous() for k, v in self._other_state().items()}
        validate_tensors(other, other, "Projection/head state")
        torch.save(other, os.path.join(save_directory, "other_weights.bin"))
        with open(config_path, "w") as f:
            json.dump(self._checkpoint_config(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory, vision_model, language_model, img_token_id,
                        img_tokens=None, num_proj_layers=None, create_self_attn_block=False,
                        create_x_attn_block=False, num_attn_layers=1, num_attn_heads=12,
                        add_attn_mlp=True, num_x_attn_heads=12, add_x_attn_mlp=True,
                        x_attn_query="text", add_multitask=None, add_multitask_unknown=None,
                        multitask_wt=None, load_projection_matrix=False, tokenizer=None,
                        pool_at=None, allow_legacy=False):
        if not save_directory or not os.path.isdir(save_directory):
            raise FileNotFoundError(f"VLM checkpoint directory does not exist: {save_directory}")
        config_path = os.path.join(save_directory, "vlm_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                saved = json.load(f)
            if saved.get("format_version") != 2:
                raise ValueError("Unsupported VLM checkpoint version")
            required = {"img_token_id", "img_tokens", "num_proj_layers", "create_projection_layer",
                        "add_multitask", "add_multitask_unknown", "multitask_wt", "pool_at",
                        "hidden_size", "base_model_name_or_path"}
            if required - saved.keys():
                raise ValueError(f"Incomplete VLM metadata: {sorted(required - saved.keys())}")
        elif allow_legacy and pool_at is not None:
            # Legacy files did not record pooling. Callers must supply its provenance.
            saved = {}
        else:
            raise ValueError("Missing vlm_config.json; repair legacy checkpoints with their original run config first")
        requested = dict(img_tokens=img_tokens, num_proj_layers=num_proj_layers,
                         add_multitask=add_multitask, add_multitask_unknown=add_multitask_unknown,
                         multitask_wt=multitask_wt, pool_at=pool_at)
        defaults = dict(img_tokens=1, num_proj_layers=1, add_multitask=False,
                        add_multitask_unknown=False, multitask_wt=1.0, pool_at="question_eos")
        for key, value in requested.items():
            if key in saved and value is not None and value != saved[key]:
                raise ValueError(f"Checkpoint {key}={saved[key]!r} conflicts with requested {value!r}")
            requested[key] = saved.get(key, defaults[key]) if value is None else value
        for key, value in dict(img_token_id=img_token_id,
                               hidden_size=language_model.config.hidden_size,
                               base_model_name_or_path=language_model.config._name_or_path).items():
            if key in saved and saved[key] != value:
                raise ValueError(f"Checkpoint {key} mismatch: {saved[key]!r} != {value!r}")
        other = torch.load(os.path.join(save_directory, "other_weights.bin"),
                           map_location="cpu", weights_only=True)
        language_model = load_adapter_strict(language_model,
                                             os.path.join(save_directory, "lora_adapter"),
                                             allow_legacy=allow_legacy)
        model = cls(vision_model=vision_model, language_model=language_model,
                    img_token_id=img_token_id, tokenizer=tokenizer,
                    create_projection_layer=saved.get("create_projection_layer", True), **requested)
        validate_tensors(other, model._other_state(), "Projection/head state")
        if (other["tbr_std"] <= 0).any():
            raise ValueError("TBR normalization standard deviations must be positive")
        device = language_model.get_input_embeddings().weight.device
        for name in ("language_projection", "tbr_regression_head", "genotype_head"):
            module = getattr(model, name)
            if module is not None:
                prefix = name + "."
                module.load_state_dict({k[len(prefix):]: v for k, v in other.items() if k.startswith(prefix)}, strict=True)
                module.to(device)
        model.tbr_mean = other["tbr_mean"].to(device)
        model.tbr_std = other["tbr_std"].to(device)
        for key, value in model._other_state().items():
            if not torch.equal(value.detach().cpu(), other[key]):
                raise ValueError(f"Projection/head tensor was not restored exactly: {key}")
        return model
