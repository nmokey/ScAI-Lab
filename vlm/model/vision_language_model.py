"""
VisionLanguageModel for pre-saved embeddings.
Adapted from NephrologyKG/model/vision_language_model.py with one key change:
  vision_hidden_dim = 768  (RAD-DINO, not the NLST encoder's 1024)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers.modeling_outputs import CausalLMOutputWithPast
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
                attention_mask=None, labels=None, **kwargs):
        combined, attention_mask, labels = self.get_image_and_text_embeddings(
            input_ids=input_ids, image_features=image_features,
            attention_mask=attention_mask, labels=labels,
        )
        outputs = self.language_model(
            inputs_embeds=combined, attention_mask=attention_mask, labels=labels,
        )
        return CausalLMOutputWithPast(
            loss=outputs.loss,
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
        return {"sequences": gen_out}

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, save_directory, vision_model, language_model, img_token_id,
                        img_tokens=1, num_proj_layers=1, create_self_attn_block=False,
                        create_x_attn_block=False, num_attn_layers=1, num_attn_heads=12,
                        add_attn_mlp=True, num_x_attn_heads=12, add_x_attn_mlp=True,
                        x_attn_query="text", add_multitask=False, add_multitask_unknown=False,
                        multitask_wt=1.0, load_projection_matrix=False, tokenizer=None):
        model = cls(
            vision_model=vision_model, language_model=language_model,
            img_token_id=img_token_id, img_tokens=img_tokens,
            num_proj_layers=num_proj_layers, add_multitask=add_multitask,
            add_multitask_unknown=add_multitask_unknown, multitask_wt=multitask_wt,
            tokenizer=tokenizer,
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        if save_directory is not None:
            if "checkpoint" in os.path.basename(save_directory):
                state_dict = {}
                with safe_open(os.path.join(save_directory, "model.safetensors"), framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(
                    os.path.join(save_directory, "pytorch_model.bin"), map_location="cpu"
                )
            state_dict = convert_meta_to_tensor(state_dict, device="cuda" if torch.cuda.is_available() else "cpu")

            if load_projection_matrix:
                proj_state = {
                    "weight": state_dict.get("language_projection.weight"),
                    "bias":   state_dict.get("language_projection.bias"),
                }
                if proj_state["weight"] is not None:
                    model.language_projection.load_state_dict(proj_state)
            else:
                model.load_state_dict(state_dict)

        return model
