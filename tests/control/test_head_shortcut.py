"""
R3 -- do the multitask heads learn from the image, or from the teacher-forced answer?

Background
----------
`VisionLanguageModel._eos_hidden_state` pools the LLM hidden state at the *last* EOS
token. During training `input_ids` is question+answer, so that is the answer-terminal
EOS -- the head sees a representation that has attended over the ground-truth answer.
docs/design_decisions.md states this as intent ("heads read context that includes the
generated answer").

For genotype records the answer *is* the label:

    "The eventual mouse status: the mouse will develop atherosclerosis."      (KO)
    "The eventual mouse status: the mouse will not develop atherosclerosis."  (WT)

So a trivially available shortcut exists: read the label off the answer tokens and
ignore the image entirely. At inference the answer is absent (question-only input), so
whatever the head learned from the answer is unavailable and it must fall back on an
image pathway it was never pushed to learn.

This is not test-set leakage -- inference is clean -- but it is a training pathology,
and it is the most likely explanation for the ts0-only baseline landing *below* chance
(AUROC 0.163), which the README currently reads as "no stable single-scan genotype
signal to learn."

The probes
----------
Three synthetic conditions, all run through the real VisionLanguageModel on a small
randomly-initialised Llama:

  IMAGE_ONLY  image encodes genotype, answer text identical for both classes
  TEXT_ONLY   image is noise, answer text states the label
  BOTH        image encodes genotype AND answer states it   <- the real dataset

Each is trained the same way, then evaluated the way inference actually works:
question-only input, read the genotype head.

What the runs found
-------------------
Not the crowding-out effect originally hypothesised. Two different results, both
consequential:

 1. The index arithmetic in `_eos_hidden_state` is CORRECT -- it pools the final
    position in both modes. The `(img_tokens - 1)` offset is right. Nobody should
    "fix" it. (test_eos_index_arithmetic_is_correct)

 2. But the head's *input distribution* differs between train and inference: at
    training the pooled state has attended over the ground-truth answer, at inference
    it has not. The consequence is not that ranking degrades -- AUROC reaches 1.000 --
    it is that the decision threshold does not transfer. Accuracy sits at ~0.500 in
    every condition tested, at every training length, even when AUROC is perfect.

    `vlm/data/eval.py` thresholds at `genotype_logit > 0.0`, so the reported genotype
    *accuracies* (0.219 / 0.531 / 0.719 / 0.344) are readings of a miscalibrated head.
    An accuracy of 0.219 on a balanced binary task -- far *below* the 0.5 floor -- is
    the signature of a systematically shifted threshold, not of inverted signal.
    AUROC is the only genotype metric here that survives the mismatch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = [pytest.mark.torch, pytest.mark.slow]

VOCAB, HIDDEN, LAYERS = 64, 32, 2
IMG_TOKEN_ID, BOS, EOS = 63, 1, 2
IMG_TOKENS, EMB_DIM = 4, 768
N_SUBJECTS, EPOCHS, LR = 32, 150, 3e-3

Q_TOKENS = [10, 11, 12]      # "what will be the eventual status"
ANSWER_KO = [20, 21, 22]     # "...will develop atherosclerosis"
ANSWER_WT = [30, 31, 32]     # "...will not develop atherosclerosis"
ANSWER_SAME = [40, 41, 42]   # uninformative answer, identical for both classes


class _Tok:
    eos_token_id = EOS


def _build_model(seed=0):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from model.vision_language_model import VisionLanguageModel

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
        num_hidden_layers=LAYERS, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256, bos_token_id=BOS, eos_token_id=EOS, pad_token_id=0,
    )
    return VisionLanguageModel(
        vision_model=None, language_model=LlamaForCausalLM(cfg),
        img_token_id=IMG_TOKEN_ID, img_tokens=IMG_TOKENS,
        add_multitask=True, multitask_wt=5.0, tokenizer=_Tok(),
    )


def _make_data(condition, seed=0):
    """
    Returns (train_ids, infer_ids, image_features, labels).

    Sequence layout mirrors MouseTrajDataset._add_prompt with prompt_type="standard":
        question = BOS <image> q... EOS
        answer   = BOS a... EOS
    """
    import torch

    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(EMB_DIM).astype(np.float32)
    direction /= np.linalg.norm(direction)

    labels = np.repeat([0, 1], N_SUBJECTS // 2)
    rng.shuffle(labels)

    feats, train_ids, infer_ids = [], [], []
    for lbl in labels:
        base = rng.standard_normal((IMG_TOKENS, EMB_DIM)).astype(np.float32) * 0.3
        if condition in ("IMAGE_ONLY", "BOTH"):
            base += (1.0 if lbl == 1 else -1.0) * direction  # genotype in the image
        feats.append(base)

        if condition == "IMAGE_ONLY":
            ans = ANSWER_SAME
        else:
            ans = ANSWER_KO if lbl == 1 else ANSWER_WT  # genotype in the answer text

        question = [BOS, IMG_TOKEN_ID] + Q_TOKENS + [EOS]
        train_ids.append(question + [BOS] + ans + [EOS])
        infer_ids.append(question)

    return (
        torch.tensor(train_ids),
        torch.tensor(infer_ids),
        torch.tensor(np.stack(feats)),
        torch.tensor(labels, dtype=torch.float32),
    )


def _train_and_eval(condition, seed=0):
    """Train the real model, then score the genotype head under inference conditions."""
    import torch
    from sklearn.metrics import roc_auc_score

    model = _build_model(seed=seed)
    train_ids, infer_ids, feats, labels = _make_data(condition, seed=seed)

    # Causal-LM labels with the question span masked, as MouseTrajDataset does.
    lm_labels = train_ids.clone()
    q_len = 2 + len(Q_TOKENS) + 1
    lm_labels[:, :q_len] = -100

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        out = model(
            input_ids=train_ids, image_features=feats,
            attention_mask=torch.ones_like(train_ids), labels=lm_labels,
            tbr_targets=torch.full((len(labels), 4), -1.0), genotype_label=labels,
        )
        out.loss.backward()
        opt.step()

    # Inference: question only. The answer -- and any shortcut through it -- is gone.
    model.eval()
    with torch.no_grad():
        combined, mask, _ = model.get_image_and_text_embeddings(
            input_ids=infer_ids, image_features=feats,
            attention_mask=torch.ones_like(infer_ids),
        )
        hidden = model.language_model(
            inputs_embeds=combined, attention_mask=mask, output_hidden_states=True
        ).hidden_states
        logits = model.genotype_head(model._eos_hidden_state(infer_ids, hidden)).squeeze(-1)

    y = labels.numpy().astype(int)
    scores = logits.numpy()
    return float(roc_auc_score(y, scores)), float(((scores > 0).astype(int) == y).mean())


@pytest.fixture(scope="module")
def scores():
    results = {}
    for cond in ("IMAGE_ONLY", "TEXT_ONLY", "BOTH"):
        auc, acc = _train_and_eval(cond)
        results[cond] = (auc, acc)
        print(f"\n  [{cond:10s}] inference-time genotype AUROC = {auc:.3f}  acc = {acc:.3f}")
    return results


def test_eos_index_arithmetic_is_correct():
    """
    The `(img_tokens - 1)` offset in `_eos_hidden_state` is right: in both training
    (question+answer, two EOS) and inference (question only, one EOS) it lands on the
    final position of the spliced sequence.

    Recorded explicitly so this is not mistaken for an off-by-one and "fixed". The
    problem is what that position *means*, not where it is.
    """
    import torch

    model = _build_model()
    train_ids = torch.tensor([[BOS, IMG_TOKEN_ID] + Q_TOKENS + [EOS, BOS] + ANSWER_KO + [EOS]])
    infer_ids = torch.tensor([[BOS, IMG_TOKEN_ID] + Q_TOKENS + [EOS]])

    for name, ids in (("train", train_ids), ("inference", infer_ids)):
        eos_mask = ids == EOS
        L = ids.size(1)
        pooled = int(
            (L - 1 - eos_mask.flip(dims=[1]).float().argmax(dim=1) + (IMG_TOKENS - 1)).item()
        )
        combined_len = L - 1 + IMG_TOKENS  # one <image> token expands to IMG_TOKENS
        print(f"\n  {name:10s} seq_len={L:2d} combined_len={combined_len:2d} pooled_at={pooled}")
        assert pooled == combined_len - 1, (
            f"{name}: pooled position {pooled} is not the final position {combined_len - 1}"
        )


@pytest.mark.probe
def test_head_input_is_independent_of_the_answer_during_training():
    """
    EXPECTED TO FAIL.

    The mechanism, shown without any training at all: hold the image fixed and change
    only the answer tokens. If the pooled hidden state moves, then the head's input is
    a function of the ground-truth answer -- which for genotype records *is* the label.
    At inference that input is not available, so whatever the head calibrated to is gone.
    """
    import torch

    model = _build_model()
    model.eval()
    feats = torch.randn(1, IMG_TOKENS, EMB_DIM)

    states = {}
    for name, answer in (("KO answer", ANSWER_KO), ("WT answer", ANSWER_WT)):
        ids = torch.tensor([[BOS, IMG_TOKEN_ID] + Q_TOKENS + [EOS, BOS] + answer + [EOS]])
        with torch.no_grad():
            combined, mask, _ = model.get_image_and_text_embeddings(
                input_ids=ids, image_features=feats, attention_mask=torch.ones_like(ids)
            )
            hidden = model.language_model(
                inputs_embeds=combined, attention_mask=mask, output_hidden_states=True
            ).hidden_states
            states[name] = model._eos_hidden_state(ids, hidden)

    delta = float((states["KO answer"] - states["WT answer"]).abs().max())
    print(f"\n  max|h(KO answer) - h(WT answer)| with identical image = {delta:.6f}")

    assert delta < 1e-6, (
        f"The multitask heads' input depends on the ground-truth answer during training "
        f"(max abs difference {delta:.4f} from changing only the answer tokens). For "
        f"genotype records the answer states the label, so the head is calibrated on a "
        f"context it never sees at inference. Fix: pool at the question-terminal EOS."
    )


def test_image_pathway_can_be_learned(scores):
    """
    Reported, not gated. With no text shortcut available the head does pick up *some*
    image signal, but weakly (AUROC ~0.6) -- in this condition the LM objective is
    uninformative about the class, so only the BCE head loss drives the image pathway.
    """
    auc, _ = scores["IMAGE_ONLY"]
    print(f"\n  IMAGE_ONLY inference AUROC = {auc:.3f}")
    assert auc > 0.5, f"no image signal learned at all (AUROC={auc:.3f})"


def test_answer_text_does_not_crowd_out_the_image_pathway(scores):
    """
    An honest negative on the original hypothesis. BOTH (the real dataset's shape:
    image *and* answer carry the label) does not score worse than IMAGE_ONLY at
    inference -- it scores better. Making the answer label-bearing gives the LM
    objective a reason to route class information through the image tokens, which
    helps rather than hurts the ranking.
    """
    img_auc, _ = scores["IMAGE_ONLY"]
    both_auc, _ = scores["BOTH"]
    drop = img_auc - both_auc
    print(f"\n  IMAGE_ONLY={img_auc:.3f}  BOTH={both_auc:.3f}  drop={drop:+.3f}")
    assert drop < 0.10, (
        f"Adding the label to the answer text costs {drop:+.3f} inference AUROC; the "
        f"head would then be reading the teacher-forced answer instead of the image."
    )


@pytest.mark.probe
def test_genotype_head_threshold_transfers_to_inference(scores):
    """
    EXPECTED TO FAIL -- and this is the consequential finding.

    In the BOTH condition the head ranks genotype perfectly (AUROC 1.000) yet
    `sigmoid(logit) > 0.5` classifies at chance, because every inference logit lands on
    one side of zero. The ranking survives the train/inference context shift; the
    threshold does not.

    `vlm/data/eval.py` reports accuracy with exactly that threshold, so every genotype
    accuracy in README.md and docs/experiments.md inherits the problem.
    """
    auc, acc = scores["BOTH"]
    print(f"\n  BOTH: AUROC = {auc:.3f}  but accuracy at logit>0 = {acc:.3f}")

    assert not (auc > 0.9 and acc < 0.6), (
        f"Genotype head ranks perfectly (AUROC={auc:.3f}) but classifies at chance "
        f"(acc={acc:.3f}): the decision threshold does not transfer from training to "
        f"inference. Reported genotype accuracies are uninterpretable as-is -- an "
        f"accuracy of 0.219 on a balanced binary task is a shifted threshold, not "
        f"inverted signal. Either calibrate the threshold on training-fold logits, or "
        f"report AUROC only."
    )
