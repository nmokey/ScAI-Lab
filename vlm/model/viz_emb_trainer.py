"""
Trainer for the mouse trajectory VLM using pre-saved RAD-DINO embeddings.
Adapted from NephrologyKG/model/viz_emb_trainer.py.

BaseModel and BaseLLM have been merged into this file — only VizEmbTrainer
exists, so the intermediate base classes are not needed.
"""

import os
import json
import shutil
import torch
import transformers
from transformers import TrainerCallback
from model.vision_language_model import VisionLanguageModel
from data.dataset_factory import get_dataset_factory
from utils.misc_utils import load_yaml, assert_required_params_list
from utils.huggingface_utils import load_tokenizer_from_huggingface, load_llm_from_huggingface


class EvalAtStartCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("Running evaluation at start of training...")
        self.trainer.evaluate()


class VizEmbTrainer:

    def __init__(self, exp_file=None):
        self.exp_file = exp_file
        self.params = load_yaml(exp_file)
        print("params:", self.params)
        self._check_params()

    def setup(self):
        self._setup_exp_dir()
        self._setup_tokenizer()

    def run(self):
        self.train()

    # ------------------------------------------------------------------
    # Setup helpers (formerly BaseModel / BaseLLM)
    # ------------------------------------------------------------------

    def _setup_exp_dir(self):
        output_dir = self.params["exp"]["output_dir"]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.exp_file is not None:
            dst = os.path.join(output_dir, "exp.yml")
            try:
                shutil.copyfile(self.exp_file, dst)
            except shutil.SameFileError:
                pass

    def _check_params(self):
        assert isinstance(self.params, dict)
        rp = self._required_params()
        required_sections = list(rp.keys())
        assert_required_params_list(required_sections, list(self.params.keys()))
        for section in required_sections:
            assert_required_params_list(rp[section], self.params[section], header=section)

    def _required_params(self):
        return {
            "exp": ["output_dir"],
            "data": [
                "tokenizer_name", "data_path", "test_size", "data_seed",
                "train_dataset", "inf_dataset", "base_dir", "img_dir", "inf_img_dir",
                "height", "width", "num_channels", "img_tokens", "seq_length",
                "inf_data_path", "kg_embedder_params", "prompt_type",
            ],
            "train": [
                "model_name", "save_model_name", "use_quantization", "r", "lora_alpha",
                "target_modules", "lora_dropout", "bias", "task_type",
                "per_device_train_batch_size", "per_device_eval_batch_size",
                "gradient_accumulation_steps", "num_train_epochs", "learning_rate", "fp16",
                "save_total_limit", "logging_steps", "save_strategy", "evaluation_strategy",
                "eval_steps", "save_steps", "optim", "lr_scheduler_type", "warmup_ratio",
                "resume_from_checkpoint", "evaluate_start", "gen_train_outputs",
                "gen_llava_med_train_outputs", "label_names",
                "vision_model_name", "freeze_llm_model", "freeze_vision_model", "pretrained",
                "load_projection_matrix", "num_proj_layers", "create_self_attn_block",
                "create_x_attn_block", "num_attn_layers", "num_attn_heads", "add_x_attn_mlp",
                "x_attn_query", "add_multitask", "add_multitask_unknown", "multitask_wt",
            ],
            "inf": [
                "model_name", "beg_prompt", "mid_prompt", "end_prompt",
                "llm_model_name", "use_quantization", "r", "lora_alpha", "target_modules",
                "lora_dropout", "bias", "task_type", "load_projection_matrix",
                "context_prompt", "replace_prompt", "max_new_tokens", "top_k",
                "similarity_threshold", "decoding_kwargs", "clean_mc", "include_img",
                "save_file", "results_file", "llm_only", "train_gt_file",
            ],
        }

    def _setup_tokenizer(self):
        self.tokenizer = load_tokenizer_from_huggingface(self.params["data"]["tokenizer_name"])
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_tokens([self.img_token])
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(self.img_token)

    def get_data_collator(self):
        from transformers import DefaultDataCollator
        return DefaultDataCollator()

    def prepare_question(self, question):
        return question

    def apply_tokenizer(self, text):
        return self.tokenizer(text, return_tensors="pt")

    def load_llm_model(self, **kwargs):
        return load_llm_from_huggingface(**kwargs)

    def get_training_args(self):
        p = self.params["train"]
        return transformers.TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=p["per_device_train_batch_size"],
            per_device_eval_batch_size=p["per_device_eval_batch_size"],
            gradient_accumulation_steps=p["gradient_accumulation_steps"],
            num_train_epochs=p["num_train_epochs"],
            learning_rate=p["learning_rate"],
            fp16=p["fp16"],
            bf16=p.get("bf16", False),
            save_total_limit=p["save_total_limit"],
            logging_steps=p["logging_steps"],
            label_names=p["label_names"],
            save_strategy=p["save_strategy"],
            evaluation_strategy=p["evaluation_strategy"],
            eval_steps=p["eval_steps"],
            save_steps=p["save_steps"],
            optim=p["optim"],
            lr_scheduler_type=p["lr_scheduler_type"],
            warmup_ratio=p["warmup_ratio"],
            load_best_model_at_end=True,
            report_to="tensorboard",
        )

    def save_model(self, model):
        model.save_pretrained(os.path.join(self.output_dir, self.params["train"]["save_model_name"]))

    @property
    def img_token(self):
        return "<image>"

    @property
    def pad_token(self):
        return "<|finetune_right_pad_id|>"

    @property
    def pixel_values_dtype(self):
        return torch.float

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        data = self.get_train_data()
        print(f"Train samples: {len(data['train'])}, Val samples: {len(data['test'])}")
        data_collator = self.get_data_collator()
        model, image_processor = self.load_train_model()
        data["train"].update_transforms_w_processor(image_processor)
        data["test"].update_transforms_w_processor(image_processor)
        training_args = self.get_training_args()
        trainer = transformers.Trainer(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            args=training_args,
            data_collator=data_collator,
        )
        if self.params["train"].get("evaluate_start"):
            cb = EvalAtStartCallback()
            cb.trainer = trainer
            trainer.add_callback(cb)
        trainer.train(resume_from_checkpoint=self.params["train"]["resume_from_checkpoint"])
        self.save_model(model)
        # Free training model from GPU before inference loads a fresh copy
        del model, trainer
        torch.cuda.empty_cache()

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
            predicted_emb_dir=self.params["data"].get("predicted_emb_dir"),
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
            predicted_emb_dir=self.params["data"].get("predicted_emb_dir"),
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
            gen_ids      = out["sequences"]
            answer_raw   = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0]
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

            # Save multitask head outputs when present
            if "genotype_logits" in out:
                result["genotype_logit"] = float(out["genotype_logits"][0].cpu())
                result["genotype_label"] = sample.get("answer_vqa_numeric", {}).get("genotype")
            if "tbr_logits" in out:
                result["tbr_regression"] = out["tbr_logits"][0].cpu().tolist()
                result["tbr_targets"]    = sample.get("answer_vqa_numeric", {}).get("tbr")

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
