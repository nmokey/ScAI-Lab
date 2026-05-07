"""
Base model classes — adapted from NephrologyKG/model/base_model.py.
Unused callbacks and RAG mixin removed.
"""

from abc import abstractmethod, ABC
import os
import shutil
import torch
import transformers
from transformers import TrainerCallback
from utils.misc_utils import load_yaml, assert_required_params_list
from utils.huggingface_utils import load_tokenizer_from_huggingface, load_llm_from_huggingface


class BaseModel(ABC):
    def __init__(self, exp_file=None):
        self.exp_file = exp_file
        self.params = load_yaml(exp_file)
        print("params:", self.params)
        self.check_params()

    def setup(self):
        self.setup_exp_dir()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def setup_exp_dir(self):
        output_dir = self.params["exp"]["output_dir"]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.exp_file is not None:
            dst = os.path.join(output_dir, "exp.yml")
            try:
                shutil.copyfile(self.exp_file, dst)
            except shutil.SameFileError:
                pass

    def check_params(self):
        assert isinstance(self.params, dict)
        required_sections = list(self.required_params.keys())
        assert_required_params_list(required_sections, list(self.params.keys()))
        for section in required_sections:
            assert_required_params_list(
                self.required_params[section], self.params[section], header=section
            )

    @property
    def required_params(self):
        return {"exp": ["output_dir"]}


class BaseLLM(BaseModel):
    def __init__(self, exp_file=None):
        super().__init__(exp_file=exp_file)

    def setup(self):
        super().setup()
        self.setup_tokenizer()

    def run(self):
        self.train()

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

    def setup_tokenizer(self):
        self.tokenizer = load_tokenizer_from_huggingface(self.params["data"]["tokenizer_name"])
        self.tokenizer.add_bos_token = False

    def load_llm_model(self, **kwargs):
        return load_llm_from_huggingface(**kwargs)

    def get_data_collator(self):
        from transformers import DefaultDataCollator
        return DefaultDataCollator()

    def prepare_question(self, question):
        return question

    def apply_tokenizer(self, text):
        return self.tokenizer(text, return_tensors="pt")

    @abstractmethod
    def get_train_data(self):
        raise NotImplementedError()

    @property
    def required_params(self):
        rp = super().required_params
        rp["data"]  = ["tokenizer_name", "data_path", "test_size", "data_seed"]
        rp["train"] = ["model_name", "save_model_name", "use_quantization", "r", "lora_alpha",
                       "target_modules", "lora_dropout", "bias", "task_type",
                       "per_device_train_batch_size", "per_device_eval_batch_size",
                       "gradient_accumulation_steps", "num_train_epochs", "learning_rate", "fp16",
                       "save_total_limit", "logging_steps", "save_strategy", "evaluation_strategy",
                       "eval_steps", "save_steps", "optim", "lr_scheduler_type", "warmup_ratio",
                       "resume_from_checkpoint", "evaluate_start", "gen_train_outputs",
                       "gen_llava_med_train_outputs", "label_names"]
        rp["inf"]   = ["model_name", "beg_prompt", "mid_prompt", "end_prompt"]
        return rp

    @property
    def img_token(self):
        return "<image>"

    @property
    def pad_token(self):
        return "<|finetune_right_pad_id|>"

    @property
    def pixel_values_dtype(self):
        return torch.float


class EvalAtStartCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("Running evaluation at start of training...")
        self.trainer.evaluate()
