"""Isolated VLM variants: direct visual genotype path or genotype-only loss.

Production model/trainer sources and their existing checkpoint contracts remain
unchanged. Variant checkpoints use version 3 and cannot load as stock models.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vlm'))
from model.vision_language_model import VisionLanguageModel
from model.viz_emb_trainer import VizEmbTrainer, VLMTrainer
from utils.checkpoint_utils import load_adapter_strict, validate_tensors
from utils.research_io import atomic_json
from utils.target_contract import fold_target_statistics

VARIANTS = ('direct_visual', 'genotype_only')


class GenotypeVariant(VisionLanguageModel):
    def __init__(self, *args, variant, **kwargs):
        if variant not in VARIANTS:
            raise ValueError('Unknown genotype variant')
        super().__init__(*args, **kwargs)
        if not self.add_multitask or self.img_tokens != 4 or self.pool_at != 'question_eos':
            raise ValueError('Variants require four image tokens and question-boundary heads')
        self.variant = variant
        if variant == 'direct_visual':
            self.visual_head = torch.nn.Linear(self.img_tokens * 768, 1, bias=False)
            torch.nn.init.zeros_(self.visual_head.weight)
            self.register_buffer('visual_mean', torch.zeros(self.img_tokens * 768))
            self.register_buffer('visual_std', torch.ones(self.img_tokens * 768))

    @classmethod
    def from_base(cls, base, variant):
        # Common weights AND RNG progression must match the existing VLM.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            model = cls(vision_model=base.vision_model, language_model=base.language_model,
                        img_token_id=base.img_token_id, img_tokens=base.img_tokens,
                        num_proj_layers=base.num_proj_layers, add_multitask=base.add_multitask,
                        add_multitask_unknown=base.add_multitask_unknown,
                        multitask_wt=base.multitask_wt, tokenizer=base.tokenizer,
                        pool_at=base.pool_at, variant=variant)
        for name in ('language_projection', 'genotype_head', 'tbr_regression_head'):
            getattr(model, name).load_state_dict(getattr(base, name).state_dict(), strict=True)
        model.tbr_mean.copy_(base.tbr_mean); model.tbr_std.copy_(base.tbr_std)
        for key, value in base._other_state().items():
            assert torch.equal(value.cpu(), model._other_state()[key].cpu())
        return model

    def visual_score(self, image_features):
        if image_features.shape[1:] != (4, 768):
            raise ValueError('Direct visual path requires exactly four 768-d tokens')
        x = image_features.flatten(start_dim=1).float()
        return self.visual_head((x - self.visual_mean) / self.visual_std).squeeze(-1)

    def genotype_score(self, hidden, image_features):
        language = self.genotype_head(hidden).squeeze(-1)
        return language + self.visual_score(image_features) if self.variant == 'direct_visual' else language

    def forward(self, input_ids, pixel_values=None, image_features=None,
                attention_mask=None, labels=None, tbr_targets=None,
                genotype_label=None, question_end_index=None, **kwargs):
        if self.variant == 'genotype_only':
            return super().forward(input_ids, pixel_values=pixel_values, image_features=image_features,
                attention_mask=attention_mask, labels=None, tbr_targets=None,
                genotype_label=genotype_label, question_end_index=question_end_index, **kwargs)
        outputs = super().forward(input_ids, pixel_values=pixel_values, image_features=image_features,
            attention_mask=attention_mask, labels=labels, tbr_targets=tbr_targets,
            genotype_label=None, question_end_index=question_end_index, **kwargs)
        if genotype_label is not None:
            hidden = self._eos_hidden_state(input_ids, outputs.hidden_states, question_end_index, attention_mask)
            score = self.genotype_score(hidden, image_features)
            target = genotype_label.to(hidden.device, dtype=hidden.dtype); valid = target >= 0
            if valid.any():
                outputs.loss = outputs.loss + self.multitask_wt * F.binary_cross_entropy_with_logits(score[valid], target[valid])
        return outputs

    def generate(self, input_ids, attention_mask, max_new_tokens, pixel_values=None,
                 image_features=None, question_end_index=None, **kwargs):
        result = super().generate(input_ids, attention_mask, max_new_tokens, pixel_values=pixel_values,
                                  image_features=image_features, question_end_index=question_end_index, **kwargs)
        if self.variant == 'direct_visual':
            with torch.no_grad():
                result['language_genotype_logits'] = result['genotype_logits']
                result['visual_genotype_logits'] = self.visual_score(image_features)
                result['genotype_logits'] = result['language_genotype_logits'] + result['visual_genotype_logits']
        return result

    def _checkpoint_config(self):
        return dict(super()._checkpoint_config(), format_version=3, genotype_variant=self.variant,
                    visual_normalization='unique_training_mouse_per_coordinate_standardization' if self.variant == 'direct_visual' else None)

    def _other_state(self):
        state = super()._other_state()
        if getattr(self, 'variant', None) == 'direct_visual':
            state.update({'visual_head.' + k: v for k, v in self.visual_head.state_dict().items()})
            state.update(visual_mean=self.visual_mean, visual_std=self.visual_std)
        return state

    @classmethod
    def from_pretrained(cls, directory, vision_model, language_model, img_token_id, *, variant, tokenizer=None):
        directory = Path(directory)
        saved = json.loads((directory / 'vlm_config.json').read_text())
        if saved.get('format_version') != 3 or saved.get('genotype_variant') != variant:
            raise ValueError('Checkpoint format/variant mismatch')
        for key, expected in dict(img_token_id=img_token_id, hidden_size=language_model.config.hidden_size,
                                  base_model_name_or_path=language_model.config._name_or_path).items():
            if saved.get(key) != expected:
                raise ValueError(f'Checkpoint {key} mismatch')
        fields = ('img_tokens', 'num_proj_layers', 'create_projection_layer', 'add_multitask',
                  'add_multitask_unknown', 'multitask_wt', 'pool_at')
        if set(fields) - saved.keys():
            raise ValueError('Incomplete variant checkpoint')
        language_model = load_adapter_strict(language_model, directory / 'lora_adapter', allow_legacy=False)
        model = cls(vision_model, language_model, img_token_id, tokenizer=tokenizer,
                    variant=variant, **{k: saved[k] for k in fields})
        if saved != model._checkpoint_config():
            raise ValueError('Checkpoint semantics differ from variant implementation')
        other = torch.load(directory / 'other_weights.bin', map_location='cpu', weights_only=True)
        validate_tensors(other, model._other_state(), 'Variant head/projection state')
        if (other['tbr_std'] <= 0).any() or ('visual_std' in other and (other['visual_std'] <= 0).any()):
            raise ValueError('Normalization scales must be positive')
        device = language_model.get_input_embeddings().weight.device
        names = ['language_projection', 'genotype_head', 'tbr_regression_head']
        if variant == 'direct_visual':
            names.append('visual_head')
        for name in names:
            prefix = name + '.'
            module = getattr(model, name)
            module.load_state_dict({k[len(prefix):]: v for k, v in other.items() if k.startswith(prefix)}, strict=True)
            module.to(device)
        for name in ('tbr_mean', 'tbr_std', 'visual_mean', 'visual_std'):
            if name in other:
                setattr(model, name, other[name].to(device))
        for name, value in model._other_state().items():
            if not torch.equal(value.detach().cpu(), other[name]):
                raise ValueError(f'Variant tensor failed exact reload: {name}')
        return model


class VariantHFTrainer(VLMTrainer):
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = self.model if model is None else model
        loaded = GenotypeVariant.from_pretrained(resume_from_checkpoint, target.vision_model,
            target.language_model, target.img_token_id, variant=target.variant, tokenizer=target.tokenizer)
        names = ['language_projection', 'genotype_head', 'tbr_regression_head']
        if target.variant == 'direct_visual':
            names.append('visual_head')
        for name in names:
            getattr(target, name).load_state_dict(getattr(loaded, name).state_dict(), strict=True)
        for name in ('tbr_mean', 'tbr_std', 'visual_mean', 'visual_std'):
            if hasattr(target, name):
                getattr(target, name).copy_(getattr(loaded, name))


def visual_statistics(dataset):
    unique = {r['pid']: r for r in dataset.data}
    features = np.stack([dataset._load_embedding(unique[s]).numpy().reshape(-1) for s in sorted(unique)]).astype(float)
    scaler = StandardScaler().fit(features)
    return torch.tensor(scaler.mean_, dtype=torch.float32), torch.tensor(scaler.scale_, dtype=torch.float32), sorted(unique)


def common_parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if value.requires_grad and not name.startswith('visual_head.'):
            digest.update(name.encode()); digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class VariantHelper(VizEmbTrainer):
    def __init__(self, exp_file, variant):
        self.variant = variant
        super().__init__(exp_file)

    def load_train_model(self):
        base, processor = super().load_train_model()
        return GenotypeVariant.from_base(base, self.variant), processor

    def load_inf_model(self):
        lm = self._load_llm(self.params['inf'])
        return GenotypeVariant.from_pretrained(self.params['inf']['model_name'], None, lm,
            self.img_token_id, variant=self.variant, tokenizer=self.tokenizer), None

    def _sample_output(self, model, sample, max_new_tokens):
        device = model.language_model.get_input_embeddings().weight.device
        inputs = self.apply_tokenizer(self.prepare_question(sample['question'])).to(device)
        with torch.no_grad():
            return model.generate(**inputs, image_features=sample['image_features'].to(device, dtype=torch.float),
                max_new_tokens=max_new_tokens, **self.params['inf']['decoding_kwargs'])

    def train(self):
        transformers.set_seed(int(self.params['data']['data_seed']))
        data = self.get_train_data(); model, processor = self.load_train_model()
        normal = fold_target_statistics(data['train'].data)
        model.tbr_mean.copy_(torch.tensor(normal['mean'])); model.tbr_std.copy_(torch.tensor(normal['std']))
        atomic_json(Path(self.output_dir) / 'target_statistics.json', normal)
        audit = dict(variant=self.variant, initial_common_parameter_sha256=common_parameter_digest(model),
                     proxy_supervised=self.variant != 'genotype_only')
        if self.variant == 'direct_visual':
            mean, scale, subjects = visual_statistics(data['train'])
            model.visual_mean.copy_(mean); model.visual_std.copy_(scale)
            audit['visual_normalization_subjects'] = subjects
        data['train'].update_transforms_w_processor(processor); data['test'].update_transforms_w_processor(processor)
        trainer = VariantHFTrainer(model=model, tokenizer=self.tokenizer, train_dataset=data['train'],
            eval_dataset=data['test'], args=self.get_training_args(), data_collator=self.get_data_collator())
        trainer.train(resume_from_checkpoint=self.params['train']['resume_from_checkpoint'])
        audit.update(updates=trainer.state.global_step, epochs=trainer.state.epoch)
        if audit['updates'] != 180 or audit['epochs'] != 20:
            raise ValueError('Variant training budget differs from the matched VLM')
        self.save_model(model); model.eval()
        # generate() calls the language model directly, so both checks use the
        # ordinary inference precision, unaffected by Trainer.forward wrapping.
        inf_data = self.get_inf_data()
        audit['reload_reference'] = []
        for i in (0, len(inf_data)-1):
            sample = inf_data[i]; out = self._sample_output(model, sample, 1)
            audit['reload_reference'].append(dict(index=i, genotype=float(out['genotype_logits'][0].cpu()),
                                                    tbr=out['tbr_logits'][0].cpu().tolist()))
        atomic_json(Path(self.output_dir) / 'variant_training_audit.json', audit)
        del model, trainer; torch.cuda.empty_cache()

    def evaluate(self):
        dataset = self.get_inf_data(); model, _ = self.load_inf_model(); model.eval()
        audit_path = Path(self.output_dir) / 'variant_training_audit.json'
        audit = json.loads(audit_path.read_text())
        for reference in audit['reload_reference']:
            out = self._sample_output(model, dataset[reference['index']], 1)
            if float(out['genotype_logits'][0].cpu()) != reference['genotype'] or out['tbr_logits'][0].cpu().tolist() != reference['tbr']:
                raise ValueError('Final variant checkpoint inference did not reproduce exactly')
        audit['reload_exact'] = True; atomic_json(audit_path, audit)
        rows = []
        for i in range(len(dataset)):
            sample = dataset[i]; question = self.prepare_question(sample['question'])
            out = self._sample_output(model, sample, self.params['inf']['max_new_tokens'])
            raw = self.tokenizer.batch_decode(out['sequences'], skip_special_tokens=False)[0]
            clean = self.tokenizer.batch_decode(out['sequences'], skip_special_tokens=True)[0]
            prediction = out['tbr_logits'][0].cpu() * model.tbr_std.cpu() + model.tbr_mean.cpu()
            row = dict(qid=sample.get('qid', i), head_pool_at=model.pool_at,
                target_schema_version=sample.get('target_schema_version'), target_name=sample.get('target_name'),
                train_mean_prediction=model.tbr_mean.cpu().tolist(), checkpoint_format_version=3,
                genotype_variant=self.variant, proxy_supervised=self.variant != 'genotype_only',
                pid=sample.get('pid'), input_week=sample.get('input_week'), future_weeks=sample.get('future_weeks'),
                orig_question=sample['question'], question=question, answer=sample.get('answer'),
                model_answer=clean, model_raw_answer_wo_question=raw.split(question)[-1], content_type=sample.get('content_type'),
                genotype_logit=float(out['genotype_logits'][0].cpu()),
                genotype_label=sample.get('answer_vqa_numeric', {}).get('genotype'),
                tbr_regression=prediction.tolist(), tbr_targets=sample.get('answer_vqa_numeric', {}).get('tbr'))
            for field in ('language_genotype_logits', 'visual_genotype_logits'):
                if field in out:
                    row[field.replace('_logits', '_logit')] = float(out[field][0].cpu())
            rows.append(row)
        path = Path(self.output_dir) / self.params['inf']['save_file']; atomic_json(path, rows)
        self.get_eval_metrics(str(path), self.params['inf']['train_gt_file'],
                              str(Path(self.output_dir) / self.params['inf']['results_file']))
