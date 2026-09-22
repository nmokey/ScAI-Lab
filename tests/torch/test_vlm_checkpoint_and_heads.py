"""Behavioral regression checks of the production VLM adapter and head paths."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM

from model.vision_language_model import VisionLanguageModel

pytestmark = pytest.mark.torch


def build(img_tokens=4, pool_at="question_eos", seed=123):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    lm = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0,
        task_type="CAUSAL_LM"))
    return VisionLanguageModel(None, lm, img_token_id=63, img_tokens=img_tokens,
                               add_multitask=True, pool_at=pool_at,
                               tokenizer=SimpleNamespace(eos_token_id=2)).eval()


def inputs(img_tokens=4):
    return dict(input_ids=torch.tensor([[1, 63, 7, 8, 2]]),
                image_features=torch.randn(1, img_tokens, 768),
                attention_mask=torch.ones(1, 5, dtype=torch.long),
                question_end_index=torch.tensor([4]))


def outputs(model, batch):
    with torch.no_grad():
        out = model(**batch)
        feat = model._eos_hidden_state(batch['input_ids'], out.hidden_states,
                                      batch['question_end_index'], batch['attention_mask'])
        generated = model.generate(**batch, max_new_tokens=2, do_sample=False)
    return (out.logits, model.genotype_head(feat), model.tbr_regression_head(feat),
            generated['genotype_logits'], generated['tbr_logits'], generated['sequences'])


@pytest.mark.parametrize('img_tokens', [1, 4])
def test_roundtrip_preserves_all_outputs_and_state(tmp_path, img_tokens):
    model = build(img_tokens)
    with torch.no_grad():
        for name, p in model.language_model.named_parameters():
            if 'lora_' in name:
                p.normal_(0, .15)
        model.tbr_mean.copy_(torch.tensor([1., 2., 3., 4.]))
        model.tbr_std.copy_(torch.tensor([.3, .5, .7, 1.]))
    batch = inputs(img_tokens)
    before = outputs(model, batch)
    model.save_pretrained(tmp_path)
    fresh = build(img_tokens)
    loaded = VisionLanguageModel.from_pretrained(
        tmp_path, None, fresh.language_model, 63, tokenizer=fresh.tokenizer).eval()
    for a, b in zip(before, outputs(loaded, batch)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for k, v in model._other_state().items():
        torch.testing.assert_close(v, loaded._other_state()[k], atol=0, rtol=0)
    for k, v in get_peft_model_state_dict(model.language_model).items():
        torch.testing.assert_close(v, get_peft_model_state_dict(loaded.language_model)[k], atol=0, rtol=0)
    assert loaded.pool_at == 'question_eos'
    assert not (tmp_path / 'lora_adapter/adapter_model.bin').exists()
    assert all('.default.' not in k for k in load_file(str(tmp_path / 'lora_adapter/adapter_model.safetensors')))


@pytest.mark.parametrize('damage', ['missing_adapter', 'extra_adapter', 'adapter_shape', 'adapter_nan',
                                     'missing_head', 'missing_std', 'zero_std', 'head_nan', 'pooling', 'lora_scale'])
def test_corrupt_checkpoint_is_rejected(tmp_path, damage):
    model = build(); model.save_pretrained(tmp_path)
    adapter = tmp_path / 'lora_adapter/adapter_model.safetensors'
    other_path = tmp_path / 'other_weights.bin'
    if damage in ('missing_adapter', 'extra_adapter', 'adapter_shape', 'adapter_nan'):
        state = load_file(str(adapter)); key = next(iter(state))
        if damage == 'missing_adapter': del state[key]
        if damage == 'extra_adapter': state['unknown.weight'] = torch.ones(1)
        if damage == 'adapter_shape': state[key] = state[key][:1]
        if damage == 'adapter_nan': state[key].fill_(float('nan'))
        save_file(state, str(adapter))
    elif damage in ('missing_head', 'missing_std', 'zero_std', 'head_nan'):
        state = torch.load(other_path, weights_only=True)
        if damage == 'missing_head': del state['genotype_head.weight']
        if damage == 'missing_std': del state['tbr_std']
        if damage == 'zero_std': state['tbr_std'].zero_()
        if damage == 'head_nan': state['genotype_head.weight'].fill_(float('nan'))
        torch.save(state, other_path)
    elif damage == 'lora_scale':
        p = tmp_path / 'lora_adapter/adapter_config.json'
        cfg = json.loads(p.read_text()); cfg['lora_alpha'] *= 2; p.write_text(json.dumps(cfg))
    kwargs = {'pool_at': 'answer_eos'} if damage == 'pooling' else {}
    with pytest.raises(ValueError):
        VisionLanguageModel.from_pretrained(tmp_path, None, build().language_model, 63, **kwargs)


def test_missing_checkpoint_never_creates_random_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        VisionLanguageModel.from_pretrained(tmp_path / 'missing', None, build().language_model, 63)


def test_known_legacy_adapter_keys_are_recoverable(tmp_path):
    model = build()
    with torch.no_grad():
        for n, p in model.language_model.named_parameters():
            if 'lora_' in n: p.normal_(0, .1)
    model.save_pretrained(tmp_path)
    path = tmp_path / 'lora_adapter/adapter_model.safetensors'
    raw = {k: v.contiguous() for k, v in model.language_model.state_dict().items() if 'lora_' in k}
    save_file(raw, str(path))
    with pytest.raises(ValueError, match='keys differ'):
        VisionLanguageModel.from_pretrained(tmp_path, None, build().language_model, 63)
    loaded = VisionLanguageModel.from_pretrained(tmp_path, None, build().language_model, 63, allow_legacy=True)
    for k, v in raw.items():
        torch.testing.assert_close(v, loaded.language_model.state_dict()[k], atol=0, rtol=0)


@pytest.mark.parametrize('img_tokens', [1, 4])
def test_heads_ignore_answers_padding_and_chat_eos(img_tokens):
    model = build(img_tokens)
    # Earlier EOS in system context; actual question boundary is the assistant header.
    question = [1, 10, 2, 63, 7, 2, 11, 12]
    features = torch.randn(2, img_tokens, 768)
    prefix = dict(input_ids=torch.tensor([question] * 2), image_features=features,
                  attention_mask=torch.ones(2, len(question), dtype=torch.long),
                  question_end_index=torch.tensor([len(question)-1] * 2))
    expected = outputs(model, prefix)
    for answers in ([[20, 2], [30, 31, 32, 2]], [[44, 45, 46, 2], [22, 2]]):
        ids = [question+a+[2]*(5-len(a)) for a in answers]
        masks = [[1]*(len(question)+len(a))+[0]*(5-len(a)) for a in answers]
        batch = dict(prefix, input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks))
        out = model(**batch)
        feat = model._eos_hidden_state(batch['input_ids'], out.hidden_states,
                                      batch['question_end_index'], batch['attention_mask'])
        torch.testing.assert_close(model.genotype_head(feat), expected[1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(model.tbr_regression_head(feat), expected[2], atol=1e-6, rtol=1e-6)
    generated = model.generate(**{k:v for k,v in prefix.items() if k != 'question_end_index'},
                               max_new_tokens=1, do_sample=False)
    torch.testing.assert_close(generated['tbr_logits'], expected[2], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('index', [None, [-1], [5], [1], [3.5]])
def test_bad_question_boundaries_fail(index):
    model = build(); batch = inputs(); batch['question_end_index'] = index
    with pytest.raises(ValueError): model(**batch)


def test_padding_and_image_count_fail():
    model = build(); batch = inputs(); batch['attention_mask'][0,4] = 0
    with pytest.raises(ValueError, match='padding'): model(**batch)
    batch = inputs(1)
    with pytest.raises(ValueError, match='image_features'): model(**batch)


def test_heads_and_adapters_learn_image_signal_and_survive_reload(tmp_path):
    model = build(img_tokens=1).train()
    y = torch.tensor([0., 1.] * 4)
    image = (2*y-1)[:, None, None].expand(-1, 1, 768).contiguous()
    ids = torch.tensor([[1, 63, 7, 2, 1, 20, 2]] * len(y))
    labels = ids.clone(); labels[:, :4] = -100
    targets = torch.stack([1+y, 2+y, 3+y, -torch.ones_like(y)], dim=1)
    batch = dict(input_ids=ids, image_features=image, attention_mask=torch.ones_like(ids),
                 labels=labels, question_end_index=torch.full((len(y),), 3),
                 genotype_label=y, tbr_targets=targets)
    before = {k:v.detach().clone() for k,v in model.named_parameters() if v.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.005)
    for step in range(65):
        opt.zero_grad(); loss=model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        if step == 1:
            for name in ('language_projection', 'genotype_head', 'tbr_regression_head', 'lora_A', 'lora_B'):
                assert any(name in k and p.grad is not None and p.grad.abs().sum()>0
                           for k,p in model.named_parameters())
        opt.step()
    for name in ('language_projection', 'genotype_head', 'tbr_regression_head', 'lora_A', 'lora_B'):
        assert any(name in k and not torch.equal(before[k], p) for k,p in model.named_parameters() if k in before)
    model.eval(); model.save_pretrained(tmp_path)
    loaded = VisionLanguageModel.from_pretrained(tmp_path, None, build(1).language_model, 63).eval()
    question = dict(input_ids=ids[:, :4], attention_mask=torch.ones_like(ids[:, :4]), image_features=image,
                    question_end_index=torch.full((len(y),), 3))
    a, b = outputs(model, question), outputs(loaded, question)
    for x,z in zip(a,b): torch.testing.assert_close(x,z,atol=0,rtol=0)
    assert ((b[1].squeeze()>0).float() == y).all()
    assert torch.mean((b[2][:,:3]-targets[:,:3])**2) < .03
    shuffled = dict(question, image_features=image.flip([0]))
    assert ((outputs(loaded,shuffled)[1].squeeze()>0).float() != y).all()


@pytest.mark.parametrize('prompt_type', ['standard', 'llama3'])
def test_dataset_prefix_matches_inference_and_truncation_preserves_boundary(tmp_path, prompt_type):
    from data.vqa_dataset import MouseTrajDataset
    from transformers import DefaultDataCollator
    # Character tokenizer deliberately makes concatenated tokenization differ at a
    # boundary. Separate answer tokenization must preserve the inference prefix.
    class Tokenizer:
        bos_token = '^'; eos_token = '$'
        def convert_tokens_to_ids(self, text): return 0
        def __call__(self, text, add_special_tokens=True):
            ids = [3 + ord(c) % 50 for c in text]
            if add_special_tokens: ids = [1] + ids
            return {'input_ids': ids}
    tokenizer=Tokenizer()
    records=[dict(question='TBR?', answer='Week 3: 2.0', answer_vqa_numeric={'genotype':1})]
    path=tmp_path/'records.json';path.write_text(json.dumps(records))
    dataset=MouseTrajDataset(tokenizer,prompt_type,'','','',str(path),seq_length=512)
    dataset._load_embedding=lambda _:torch.zeros(1,768)
    train=dataset[0]
    question,_=dataset._add_prompt(records[0]['question'],records[0]['answer'])
    expected=tokenizer(question)['input_ids'];end=train['question_end_index']
    assert list(train['input_ids'][:end+1]) == expected
    assert (train['labels'][:end+1] == -100).all()
    batch=DefaultDataCollator()([train, train])
    assert batch['question_end_index'].tolist()==[end,end]
    dataset.seq_length=end+3
    with pytest.warns(RuntimeWarning, match='Question head boundary is preserved'):
        truncated=dataset[0]
    assert truncated['question_end_index']==end
    assert list(truncated['input_ids'][:end+1])==expected
    dataset.seq_length=end+1
    with pytest.raises(ValueError,match='question boundary'):dataset[0]


def test_trainer_resume_and_best_checkpoint_restore_all_state(tmp_path):
    from model.viz_emb_trainer import VLMTrainer
    from transformers import TrainingArguments, TrainerCallback
    class StopAfterTwo(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step==2:control.should_training_stop=True
    row=dict(input_ids=torch.tensor([1,63,7,2,1,20,2]),image_features=torch.ones(1,768),
             attention_mask=torch.ones(7,dtype=torch.long),labels=torch.tensor([-100]*4+[1,20,2]),
             question_end_index=3,genotype_label=torch.tensor(1.),tbr_targets=torch.tensor([1.,2.,3.,-1.]))
    dataset=[row]*4
    def args(path):
        return TrainingArguments(output_dir=str(path),per_device_train_batch_size=2,max_steps=4,
                                 save_steps=2,save_strategy='steps',learning_rate=.001,report_to=[],
                                 use_cpu=True,disable_tqdm=True,seed=123,
                                 label_names=['labels','tbr_targets','genotype_label'])
    continuous=build(1)
    VLMTrainer(model=continuous,args=args(tmp_path/'continuous'),train_dataset=dataset).train()
    partial=build(1)
    first=VLMTrainer(model=partial,args=args(tmp_path/'resumed'),train_dataset=dataset,callbacks=[StopAfterTwo()])
    first.train()
    checkpoint=tmp_path/'resumed/checkpoint-2'
    assert (checkpoint/'lora_adapter/adapter_model.safetensors').exists()
    assert not (checkpoint/'model.safetensors').exists()
    resumed=build(1)
    second=VLMTrainer(model=resumed,args=args(tmp_path/'resumed'),train_dataset=dataset)
    second.train(resume_from_checkpoint=str(checkpoint))
    assert second.state.global_step==4
    for k,v in continuous.state_dict().items():
        torch.testing.assert_close(v,resumed.state_dict()[k],atol=0,rtol=0)
    second.state.best_model_checkpoint=str(checkpoint)
    second._load_best_model()
    for k,v in partial.state_dict().items():
        torch.testing.assert_close(v,resumed.state_dict()[k],atol=0,rtol=0)
