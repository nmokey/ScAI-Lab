"""
Mouse trajectory VQA dataset.
Adapted from NephrologyKG/data/vqa_dataset.py — only the MouseTrajDataset
class is needed for this project. All other dataset classes removed.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from safetensors.torch import load_file
from utils.huggingface_utils import load_tokenizer_from_huggingface


class MouseTrajDataset(Dataset):
    """
    Dataset for mouse atherosclerosis trajectory VQA.

    Each record in the JSON has:
      - pid, qid, input_week, future_weeks
      - embedding_path_ts0  — .safetensors file for the input scan, shape (1, 768)
      - question, answer
      - answer_vqa_numeric  — dict with 'genotype' key (0=WT, 1=KO)

    The model receives:
      - image_features  : (img_tokens, 768) float tensor
                          ts0 (observed Week 12) + up to 3 longitudinal-encoder
                          predicted future embeddings (ts1/ts2/ts3), zero-padded
                          when a predicted embedding is unavailable.
      - input_ids       : tokenized prompt
      - attention_mask
      - labels          : causal LM targets (question tokens masked with -100)

    Pass predicted_emb_dir=<path> to enable multi-token longitudinal input.
    When None, falls back to ts0-only (single token, original behaviour).
    """

    def __init__(self, tokenizer, prompt_type, beg_prompt, mid_prompt, end_prompt,
                 data_path, replace_prompt=None, img_dir="", img_tokens=1,
                 pad_token_str="<|finetune_right_pad_id|>", img_token_str="<image>",
                 seq_length=150, mode="train", height=224, width=224, num_channels=3,
                 filter_key=None, filter_cond=None, calculate_mae=False,
                 predicted_emb_dir=None, **kwargs):
        self.prompt_type      = prompt_type
        self.beg_prompt       = beg_prompt
        self.mid_prompt       = mid_prompt
        self.end_prompt       = end_prompt
        self.replace_prompt   = replace_prompt
        self.tokenizer        = tokenizer
        self.pad_token_id     = self.tokenizer.convert_tokens_to_ids(pad_token_str)
        self.ignore_token_id  = -100
        self.img_token_str    = img_token_str
        self.img_tokens       = img_tokens
        self.seq_length       = seq_length
        self.mode             = mode
        self.predicted_emb_dir = predicted_emb_dir
        self.data_path        = data_path
        self.data             = self._load(data_path)

    def _load(self, data_path):
        with open(data_path) as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record       = self.data[idx]
        question_text = record["question"]
        answer_text   = record["answer"]

        if self.replace_prompt is not None:
            question_text = self.replace_prompt

        image_features = self._load_embedding(record)
        question_text, answer_text = self._add_prompt(question_text, answer_text)

        item = {}
        if self.mode in ("train", "val"):
            tok_qa = self.tokenizer(question_text + answer_text)["input_ids"]
            tok_q  = self.tokenizer(question_text)["input_ids"]

            ignore  = [self.ignore_token_id] * len(tok_q)
            labels  = ignore + tok_qa[len(tok_q):]
            mask    = [1] * len(tok_qa)

            if len(tok_qa) < self.seq_length:
                pad = self.seq_length - len(tok_qa)
                tok_qa = np.pad(tok_qa, (0, pad), constant_values=self.pad_token_id)
                mask   = np.pad(mask,   (0, pad), constant_values=0)
                labels = np.pad(labels, (0, pad), constant_values=self.ignore_token_id)
            elif len(tok_qa) > self.seq_length:
                trunc  = len(tok_qa) - self.seq_length
                tok_qa = tok_qa[:-trunc]
                mask   = mask[:-trunc]
                labels = labels[:-trunc]

            item["image_features"]  = image_features
            item["input_ids"]       = tok_qa
            item["attention_mask"]  = mask
            item["labels"]          = labels
            item["tbr_targets"]     = self._tbr_targets(record)
            item["genotype_label"]  = torch.tensor(
                float(record.get("answer_vqa_numeric", {}).get("genotype", -1))
            )

        elif self.mode == "test":
            item = dict(record)
            item["question"]       = question_text
            item["image_features"] = image_features.unsqueeze(0)
            # Augment answer_vqa_numeric with parsed TBR targets so the evaluator
            # can compute regression-head metrics at inference time.
            tbr_t = self._tbr_targets(record)
            avq   = dict(item.get("answer_vqa_numeric") or {})
            avq["tbr"] = tbr_t.tolist()
            item["answer_vqa_numeric"] = avq

        return item

    def _load_embedding(self, record):
        """
        Load scan embeddings as (img_tokens, 768).

        Token 0 is always the observed Week 12 RAD-DINO embedding (ts0).
        Tokens 1-3 are LOSO-predicted future embeddings from the longitudinal
        MLP (ts1/ts2/ts3), loaded from predicted_emb_dir as .npy files.
        Missing predictions are filled with zeros.

        When predicted_emb_dir is None, returns (1, 768) — original behaviour.
        """
        ts0 = load_file(record["embedding_path_ts0"])["embeddings"]  # (1, 768)

        if self.predicted_emb_dir is None:
            return ts0

        emb_dim  = ts0.shape[-1]
        tokens   = [ts0.squeeze(0)]  # list of (768,) tensors
        pid      = record["pid"]
        week_tag = {"Week 15": "ts1", "Week 18": "ts2", "Week 20": "ts3"}

        for week, tag in week_tag.items():
            path = os.path.join(self.predicted_emb_dir, f"{pid}_{tag}.npy")
            if os.path.exists(path):
                arr = np.load(path).astype(np.float32)
                tokens.append(torch.from_numpy(arr))
            else:
                tokens.append(torch.zeros(emb_dim))

        # Pad or trim to exactly img_tokens
        while len(tokens) < self.img_tokens:
            tokens.append(torch.zeros(emb_dim))
        tokens = tokens[: self.img_tokens]

        return torch.stack(tokens)  # (img_tokens, 768)

    def _tbr_targets(self, record):
        """Pack future TBR values into a fixed-length (4,) tensor, padded with -1."""
        import re
        tbr_re = re.compile(r"Week\s+\d+:\s*(\d+(?:\.\d+)?)")
        answer = record.get("answer", "")
        # Only TBR and combined questions have numeric TBR in the answer
        if "TBR" not in record.get("question", ""):
            return torch.full((4,), -1.0)
        vals = [float(m.group(1)) for m in tbr_re.finditer(answer)]
        t = torch.full((4,), -1.0)
        for i, v in enumerate(vals[:4]):
            t[i] = v
        return t

    def _add_prompt(self, question, answer):
        bos = self.tokenizer.bos_token or ""
        eos = self.tokenizer.eos_token or ""
        if self.prompt_type == "standard":
            question = bos + f"{self.img_token_str}{self.beg_prompt}{question}{self.mid_prompt}" + eos
            answer   = bos + answer + eos
        elif self.prompt_type == "llama3":
            question = (bos + "<|start_header_id|>system<|end_header_id|>\n\n"
                        "You are a helpful medical AI assistant.<|eot_id|>"
                        "<|start_header_id|>\nuser<|end_header_id|>\n\n"
                        f"{self.img_token_str}\n{self.beg_prompt}{question}{self.mid_prompt}<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                        f"{self.end_prompt}")
            answer = answer + eos
        else:
            raise ValueError(f"Unknown prompt_type: {self.prompt_type}")
        return question, answer

    def update_transforms_w_processor(self, image_processor):
        pass  # embeddings are pre-computed; no image processor needed

    def calculate_metrics(self, train_gt_file, pred_file, results_file):
        from data.eval import calculate_mouse_metrics
        calculate_mouse_metrics(gt_file=self.data_path if hasattr(self, "data_path") else None,
                                train_gt_file=train_gt_file,
                                pred_file=pred_file,
                                out_file=results_file)
