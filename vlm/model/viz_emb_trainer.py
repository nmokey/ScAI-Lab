"""
Trainer for the mouse trajectory VLM using pre-saved RAD-DINO embeddings.
Adapted from NephrologyKG/model/viz_emb_trainer.py.
"""

import os
import json
import numpy as np
import torch
from model.base_model import BaseLLM
from model.vision_language_model import VisionLanguageModel
from data.dataset_factory import get_dataset_factory


class VizEmbTrainer(BaseLLM):

    def setup_tokenizer(self):
        super().setup_tokenizer()
        self.tokenizer.add_tokens([self.img_token])
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(self.img_token)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_train_model(self):
        llm = self._load_llm(self.params["train"])
        p   = self.params["train"]
        if p["pretrained"]:
            model = VisionLanguageModel.from_pretrained(
                p["model_name"], vision_model=None, language_model=llm,
                img_token_id=self.img_token_id,
                img_tokens=self.params["data"]["img_tokens"],
                num_proj_layers=p["num_proj_layers"],
                load_projection_matrix=p["load_projection_matrix"],
                add_multitask=p["add_multitask"],
                add_multitask_unknown=p["add_multitask_unknown"],
                multitask_wt=p["multitask_wt"],
                tokenizer=self.tokenizer,
            )
        else:
            model = VisionLanguageModel(
                vision_model=None, language_model=llm,
                img_token_id=self.img_token_id,
                img_tokens=self.params["data"]["img_tokens"],
                num_proj_layers=p["num_proj_layers"],
                add_multitask=p["add_multitask"],
                add_multitask_unknown=p["add_multitask_unknown"],
                multitask_wt=p["multitask_wt"],
                tokenizer=self.tokenizer,
            )
        if p["freeze_llm_model"]:
            for param in model.language_model.parameters():
                param.requires_grad = False
        return model, None

    def load_inf_model(self):
        llm = self._load_llm(self.params["inf"])
        p   = self.params["train"]
        inf = self.params["inf"]
        model = VisionLanguageModel.from_pretrained(
            inf["model_name"], vision_model=None, language_model=llm,
            img_token_id=self.img_token_id,
            img_tokens=self.params["data"]["img_tokens"],
            num_proj_layers=p["num_proj_layers"],
            load_projection_matrix=inf["load_projection_matrix"],
            add_multitask=p["add_multitask"],
            add_multitask_unknown=p["add_multitask_unknown"],
            multitask_wt=p["multitask_wt"],
            tokenizer=self.tokenizer,
        )
        return model, None

    def _load_llm(self, section):
        return self.load_llm_model(
            model_name=section["llm_model_name"],
            use_quantization=section["use_quantization"],
            r=section["r"], lora_alpha=section["lora_alpha"],
            target_modules=section["target_modules"],
            lora_dropout=section["lora_dropout"],
            bias=section["bias"], task_type=section["task_type"],
        )

    def load_vision_model(self):
        return None, None

    def get_vision_model_norm_params(self):
        return None, None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def get_train_data(self):
        factory    = get_dataset_factory(self.params["data"]["train_dataset"])
        shared_kw  = dict(
            tokenizer=self.tokenizer,
            prompt_type=self.params["data"]["prompt_type"],
            beg_prompt="", mid_prompt="", end_prompt="",
            img_dir=self.params["data"]["img_dir"],
            img_tokens=self.params["data"]["img_tokens"],
            pad_token_str=self.pad_token,
            img_token_str=self.img_token,
            seq_length=self.params["data"]["seq_length"],
        )
        train_data = factory.create_dataset(
            data_path=self.params["data"]["data_path"], mode="train", **shared_kw
        )
        val_data   = factory.create_dataset(
            data_path=self.params["data"]["inf_data_path"], mode="val", **shared_kw
        )
        return {"train": train_data, "test": val_data}

    def get_inf_data(self):
        factory = get_dataset_factory(self.params["data"]["inf_dataset"])
        return factory.create_dataset(
            tokenizer=self.tokenizer,
            prompt_type=self.params["data"]["prompt_type"],
            beg_prompt=self.params["inf"]["beg_prompt"],
            mid_prompt=self.params["inf"]["mid_prompt"],
            end_prompt=self.params["inf"]["end_prompt"],
            replace_prompt=self.params["inf"]["replace_prompt"],
            data_path=self.params["data"]["inf_data_path"],
            img_dir=self.params["data"]["inf_img_dir"],
            img_tokens=self.params["data"]["img_tokens"],
            img_token_str=self.img_token if self.params["inf"]["include_img"] else "",
            seq_length=self.params["data"]["seq_length"],
            mode="test",
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self):
        inf_data      = self.get_inf_data()
        model, _      = self.load_inf_model()
        model.eval()
        decoding_kwargs = self.params["inf"]["decoding_kwargs"]
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        content = []

        for i in range(len(inf_data)):
            sample   = inf_data[i]
            qid      = sample.get("qid", i)
            orig_q   = sample["question"]
            question = self.prepare_question(orig_q)
            inputs   = self.apply_tokenizer(question).to(device)
            img_feat = sample["image_features"].to(device, dtype=torch.float)

            with torch.no_grad():
                out = model.generate(
                    **inputs, image_features=img_feat,
                    max_new_tokens=self.params["inf"]["max_new_tokens"],
                    **decoding_kwargs,
                )
            gen_ids     = out["sequences"]
            answer_raw  = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0]
            answer_clean = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            answer_raw_wo_q = answer_raw.split(question)[-1]

            result = {
                "qid":                      qid,
                "pid":                      sample.get("pid"),
                "input_week":               sample.get("input_week"),
                "future_weeks":             sample.get("future_weeks"),
                "orig_question":            orig_q,
                "question":                 question,
                "answer":                   sample.get("answer"),
                "model_answer":             answer_clean,
                "model_raw_answer_wo_question": answer_raw_wo_q,
                "content_type":             sample.get("content_type"),
            }
            content.append(result)
            print(f"[{i+1}/{len(inf_data)}] {answer_clean[:80]}")

        save_path = os.path.join(self.output_dir, self.params["inf"]["save_file"])
        with open(save_path, "w") as f:
            json.dump(content, f, indent=2)
        print(f"[+] Predictions saved → {save_path}")

        self.get_eval_metrics(
            pred_file=save_path,
            train_gt_file=self.params["inf"]["train_gt_file"],
            results_file=os.path.join(self.output_dir, self.params["inf"]["results_file"]),
        )

    def get_eval_metrics(self, pred_file, train_gt_file, results_file):
        inf_data = self.get_inf_data()
        inf_data.calculate_metrics(
            train_gt_file=train_gt_file,
            pred_file=pred_file,
            results_file=results_file,
        )

    # ------------------------------------------------------------------
    # Required params declaration
    # ------------------------------------------------------------------

    @property
    def required_params(self):
        rp = super().required_params
        rp["data"] = rp["data"] + [
            "train_dataset", "inf_dataset", "base_dir", "img_dir", "inf_img_dir",
            "height", "width", "num_channels", "img_tokens", "seq_length",
            "inf_data_path", "kg_embedder_params", "prompt_type",
        ]
        rp["train"] = rp["train"] + [
            "vision_model_name", "freeze_llm_model", "freeze_vision_model", "pretrained",
            "load_projection_matrix", "num_proj_layers", "create_self_attn_block",
            "create_x_attn_block", "num_attn_layers", "num_attn_heads", "add_x_attn_mlp",
            "x_attn_query", "add_multitask", "add_multitask_unknown", "multitask_wt",
        ]
        rp["inf"] = rp["inf"] + [
            "llm_model_name", "use_quantization", "r", "lora_alpha", "target_modules",
            "lora_dropout", "bias", "task_type", "load_projection_matrix",
            "context_prompt", "replace_prompt", "max_new_tokens", "top_k",
            "similarity_threshold", "decoding_kwargs", "clean_mc", "include_img",
            "save_file", "results_file", "llm_only", "train_gt_file",
        ]
        return rp
