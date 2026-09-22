import copy
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM
from model.vision_language_model import VisionLanguageModel

pytestmark = pytest.mark.torch


def base():
    torch.manual_seed(123)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    lm = get_peft_model(LlamaForCausalLM(config), LoraConfig(r=2, lora_alpha=4,
        target_modules=['q_proj', 'v_proj'], lora_dropout=0, task_type='CAUSAL_LM'))
    return VisionLanguageModel(None, lm, 63, img_tokens=4, add_multitask=True,
        multitask_wt=5, tokenizer=SimpleNamespace(eos_token_id=2)).eval()


def batch():
    ids = torch.tensor([[1, 63, 7, 2, 12, 2], [1, 63, 7, 2, 13, 2]])
    labels = ids.clone(); labels[:, :4] = -100
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        image_features=torch.randn(2, 4, 768), question_end_index=torch.tensor([3, 3]),
        labels=labels, genotype_label=torch.tensor([0., 1.]),
        tbr_targets=torch.tensor([[1., 2., -1., -1.], [2., 3., 4., -1.]]))


def test_initial_direct_visual_loss_and_common_initialization_match(scripts):
    module = scripts('vlm_genotype_variants'); original = base(); data = batch()
    rng = torch.get_rng_state().clone()
    direct = module.GenotypeVariant.from_base(original, 'direct_visual').eval()
    assert torch.equal(torch.get_rng_state(), rng)
    assert module.common_parameter_digest(direct) == module.common_parameter_digest(original)
    torch.testing.assert_close(direct(**data).loss, original(**data).loss, rtol=0, atol=0)
    direct(**data).loss.backward()
    assert direct.visual_head.weight.grad.abs().sum() > 0
    assert direct.genotype_head.weight.grad.abs().sum() > 0


def test_genotype_only_matches_declared_loss_and_ignores_other_targets(scripts):
    module = scripts('vlm_genotype_variants'); original = base(); data = batch()
    model = module.GenotypeVariant.from_base(original, 'genotype_only').eval()
    selected = dict(data); selected.pop('labels'); selected.pop('tbr_targets')
    expected = original(**selected).loss
    torch.testing.assert_close(model(**data).loss, expected, atol=0, rtol=0)
    changed = dict(data, tbr_targets=torch.full_like(data['tbr_targets'], 9999.), labels=torch.zeros_like(data['labels']))
    torch.testing.assert_close(model(**changed).loss, expected, atol=0, rtol=0)
    model(**data).loss.backward()
    assert all(p.grad is None for p in model.tbr_regression_head.parameters())
    assert model.genotype_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('variant', ['direct_visual', 'genotype_only'])
def test_variant_checkpoint_and_resume_preserve_heads_buffers_and_predictions(scripts, tmp_path, variant):
    m = scripts('vlm_genotype_variants'); original = base()
    model = m.GenotypeVariant.from_base(original, variant).eval()
    if variant == 'direct_visual':
        with torch.no_grad():
            model.visual_head.weight.normal_(0, .01)
            model.visual_mean.normal_(0, .1); model.visual_std.uniform_(.5, 2.)
    data = batch(); question = {k: data[k] for k in ('input_ids', 'attention_mask', 'image_features', 'question_end_index')}
    question['input_ids'] = question['input_ids'][:, :4]; question['attention_mask'] = question['attention_mask'][:, :4]
    before = model.generate(**question, max_new_tokens=1, do_sample=False)
    model.save_pretrained(tmp_path)
    restored = m.GenotypeVariant.from_pretrained(tmp_path, None, base().language_model, 63,
        variant=variant, tokenizer=original.tokenizer).eval()
    after = restored.generate(**question, max_new_tokens=1, do_sample=False)
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match='version'):
        VisionLanguageModel.from_pretrained(tmp_path, None, base().language_model, 63)
    with pytest.raises(ValueError, match='variant mismatch'):
        m.GenotypeVariant.from_pretrained(tmp_path, None, base().language_model, 63,
            variant='genotype_only' if variant == 'direct_visual' else 'direct_visual')
    ids = {n: id(p) for n, p in restored.named_parameters()}
    trainer = object.__new__(m.VariantHFTrainer); trainer.model = restored
    trainer._load_from_checkpoint(tmp_path)
    assert ids == {n: id(p) for n, p in restored.named_parameters()}
    for key, value in model._other_state().items():
        torch.testing.assert_close(value, restored._other_state()[key], atol=0, rtol=0)


def test_visual_checkpoint_rejects_missing_or_invalid_normalization(scripts, tmp_path):
    m = scripts('vlm_genotype_variants'); model = m.GenotypeVariant.from_base(base(), 'direct_visual')
    model.save_pretrained(tmp_path)
    path = tmp_path / 'other_weights.bin'; original = torch.load(path, weights_only=True)
    for damage in ('missing', 'zero', 'nan'):
        state = copy.deepcopy(original)
        if damage == 'missing': del state['visual_head.weight']
        if damage == 'zero': state['visual_std'].zero_()
        if damage == 'nan': state['visual_mean'].fill_(float('nan'))
        torch.save(state, path)
        with pytest.raises(ValueError):
            m.GenotypeVariant.from_pretrained(tmp_path, None, base().language_model, 63, variant='direct_visual')


def test_visual_statistics_use_unique_training_mice_only(scripts):
    m = scripts('vlm_genotype_variants')
    class Data:
        data = [dict(pid='a')] * 3 + [dict(pid='b')] * 9
        def _load_embedding(self, row):
            return torch.full((4, 768), 2. if row['pid'] == 'a' else 4.)
    mean, scale, subjects = m.visual_statistics(Data())
    assert subjects == ['a', 'b'] and (mean == 3.).all() and (scale == 1.).all()
