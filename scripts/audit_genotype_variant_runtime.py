"""Actual quantized-backbone checks before the fixed genotype variant runs."""
import copy
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import DefaultDataCollator

from vlm_genotype_variants import GenotypeVariant
from model.viz_emb_trainer import VizEmbTrainer
from utils.target_contract import fold_target_statistics
from utils.research_io import atomic_json, file_sha256

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / 'docs/audit_2026-09-14/genotype_vlm_variants'


def main():
    transformers.set_seed(0)
    helper = VizEmbTrainer('/data1/Processed_NIfTI_Test/embeddings/vlm/runs/stratified_residual_seed0/fold_00/exp.yml')
    helper._setup_tokenizer(); dataset = helper.get_train_data()['train']
    base, _ = helper.load_train_model(); base.to('cuda').eval()
    stats = fold_target_statistics(dataset.data)
    base.tbr_mean.copy_(torch.tensor(stats['mean'], device='cuda'))
    base.tbr_std.copy_(torch.tensor(stats['std'], device='cuda'))
    initial_lora = {n: p.detach().clone() for n, p in base.language_model.named_parameters() if p.requires_grad}
    data = {k: v.cuda() for k, v in DefaultDataCollator()([dataset[0], dataset[len(dataset)-1]]).items()}
    report = dict(status='running', model_source_sha256=file_sha256(REPO / 'scripts/vlm_genotype_variants.py'),
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), variants={})
    for variant in ('direct_visual', 'genotype_only'):
        for name, p in base.language_model.named_parameters():
            if name in initial_lora:
                with torch.no_grad(): p.copy_(initial_lora[name])
        base.eval(); model = GenotypeVariant.from_base(base, variant).to('cuda').eval()
        if variant == 'direct_visual':
            from vlm_genotype_variants import visual_statistics
            mean, std, subjects = visual_statistics(dataset)
            model.visual_mean.copy_(mean); model.visual_std.copy_(std)
        selected = dict(data)
        if variant == 'genotype_only':
            selected.pop('labels'); selected.pop('tbr_targets')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            expected = base(**selected).loss; actual = model(**data).loss
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        check = dict(initial_loss=float(actual), expected_loss=float(expected), objective_match=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0002, weight_decay=0.)
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**data).loss
            loss.backward(); optimizer.step()
        check['genotype_gradient_nonzero'] = bool(model.genotype_head.weight.grad.abs().sum() > 0)
        assert check['genotype_gradient_nonzero']
        if variant == 'direct_visual':
            assert model.visual_head.weight.grad.abs().sum() > 0
            check['visual_gradient_nonzero'] = True
        else:
            assert all(p.grad is None for p in model.tbr_regression_head.parameters())
            check['proxy_gradients_absent'] = True
        model.eval()
        input_keys = ('input_ids', 'attention_mask', 'image_features', 'question_end_index')
        query = {k: data[k] for k in input_keys}
        # Full input shape retains answers, but the head is explicitly question-pooled.
        with torch.no_grad():
            out = model(**query)
            hidden = model._eos_hidden_state(query['input_ids'], out.hidden_states, query['question_end_index'], query['attention_mask'])
            expected_scores = model.genotype_score(hidden, query['image_features']).detach().clone()
            changed = dict(query, input_ids=query['input_ids'].clone())
            for i, end in enumerate(query['question_end_index']):
                changed['input_ids'][i, int(end)+1:] = 7
            alt = model(**changed)
            alt_hidden = model._eos_hidden_state(changed['input_ids'], alt.hidden_states, changed['question_end_index'], changed['attention_mask'])
            torch.testing.assert_close(model.genotype_score(alt_hidden, changed['image_features']), expected_scores, atol=0, rtol=0)
        path = Path('/tmp/scai-genotype-variant-runtime') / variant
        model.save_pretrained(path)
        loaded = GenotypeVariant.from_pretrained(path, None, model.language_model, helper.img_token_id,
                                                 variant=variant, tokenizer=helper.tokenizer).eval()
        with torch.no_grad():
            restored = loaded(**query)
            hidden = loaded._eos_hidden_state(query['input_ids'], restored.hidden_states, query['question_end_index'], query['attention_mask'])
            scores = loaded.genotype_score(hidden, query['image_features'])
        torch.testing.assert_close(scores, expected_scores, atol=0, rtol=0)
        check.update(answer_invariance=True, reload_max_logit_error=float((scores-expected_scores).abs().max()))
        report['variants'][variant] = check; atomic_json(OUTPUT / 'runtime_checks.json', report)
        print(json.dumps(dict(variant=variant, **check)), flush=True)
        del model, loaded, optimizer, loss, out, alt, restored
    report['status'] = 'passed'; atomic_json(OUTPUT / 'runtime_checks.json', report)


if __name__ == '__main__': main()
