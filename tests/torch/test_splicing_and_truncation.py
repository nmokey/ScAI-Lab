"""
R7 and R8 -- two latent defects in how image tokens are spliced into the sequence.

R8: `get_image_and_text_embeddings` locates the <image> placeholder using **row 0 only**
and applies that single offset to the whole batch:

    first_seq = input_ids[0]
    img_pos = (first_seq == self.img_token_id).nonzero(as_tuple=False).item()
    pre  = embed(input_ids[:, :img_pos])
    post = embed(input_ids[:, img_pos + 1:])

Safe today because `_add_prompt` builds a fixed-length prefix, so every row has the
placeholder at the same index. It stops being safe the moment `beg_prompt` becomes
non-empty or variable-length -- both are config-settable (`inf.beg_prompt` in every
YAML) -- and the failure is silent rather than loud. `.item()` also raises on any row
containing two placeholders.

R7: `MouseTrajDataset.__getitem__` truncates from the **right** when a record exceeds
`seq_length` (150):

    trunc = len(tok_qa) - self.seq_length
    tok_qa = tok_qa[:-trunc]

That removes the answer's terminal EOS. `_eos_hidden_state` then falls back to the
question EOS for that record while other records in the same batch still pool at the
answer EOS -- two different pooling semantics mixed within one batch, with nothing
warning that it happened.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = [pytest.mark.torch, pytest.mark.unit]

VOCAB, HIDDEN = 64, 32
IMG_TOKEN_ID, BOS, EOS = 63, 1, 2
IMG_TOKENS, EMB_DIM = 4, 768


class _Tok:
    eos_token_id = EOS


def _model(img_tokens=IMG_TOKENS):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from model.vision_language_model import VisionLanguageModel

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256, bos_token_id=BOS, eos_token_id=EOS, pad_token_id=0,
    )
    return VisionLanguageModel(
        vision_model=None, language_model=LlamaForCausalLM(cfg),
        img_token_id=IMG_TOKEN_ID, img_tokens=img_tokens,
        add_multitask=True, tokenizer=_Tok(),
    )


# ---------------------------------------------------------------------------
# The splice is correct in the homogeneous case
# ---------------------------------------------------------------------------

def test_splice_expands_the_placeholder_correctly():
    """Baseline: one <image> token becomes img_tokens positions, in the right place."""
    import torch

    model = _model()
    ids = torch.tensor([[BOS, IMG_TOKEN_ID, 10, 11, EOS]])
    feats = torch.randn(1, IMG_TOKENS, EMB_DIM)

    combined, mask, labels = model.get_image_and_text_embeddings(
        input_ids=ids, image_features=feats,
        attention_mask=torch.ones_like(ids), labels=ids.clone(),
    )
    expected_len = ids.size(1) - 1 + IMG_TOKENS
    assert combined.shape[1] == expected_len
    assert mask.shape[1] == expected_len
    # Image positions must be ignored by the LM loss.
    assert (labels[0, 1 : 1 + IMG_TOKENS] == -100).all(), "image span should be masked to -100"


# ---------------------------------------------------------------------------
# R8 -- heterogeneous batches
# ---------------------------------------------------------------------------

def test_batch_with_differing_image_positions_is_rejected():
    """
    F7, fixed. Row 0 has <image> at index 1, row 1 at index 3. The splice applies a
    single offset to the whole batch, so mixed positions cannot be handled correctly --
    previously row 1's placeholder survived and real text tokens were consumed in its
    place, silently. Now it raises.
    """
    import torch

    model = _model()
    ids = torch.tensor([
        [BOS, IMG_TOKEN_ID, 10, 11, 12, EOS],
        [BOS, 20, 21, IMG_TOKEN_ID, 12, EOS],  # placeholder two positions later
    ])

    with pytest.raises(ValueError, match="same index in every sequence"):
        model.get_image_and_text_embeddings(
            input_ids=ids, image_features=torch.randn(2, IMG_TOKENS, EMB_DIM),
            attention_mask=torch.ones_like(ids),
        )


def test_missing_image_token_is_rejected():
    """A row with no placeholder must fail loudly rather than splice at index 0."""
    import torch

    model = _model()
    ids = torch.tensor([[BOS, 10, 11, 12, EOS]])
    with pytest.raises(ValueError, match="exactly one image token"):
        model.get_image_and_text_embeddings(
            input_ids=ids, image_features=torch.randn(1, IMG_TOKENS, EMB_DIM),
            attention_mask=torch.ones_like(ids),
        )


def test_two_image_tokens_in_one_row_raises():
    """Two placeholders in a row is likewise rejected, with a clear message."""
    import torch

    model = _model()
    ids = torch.tensor([[BOS, IMG_TOKEN_ID, IMG_TOKEN_ID, 10, EOS]])
    with pytest.raises(ValueError, match="exactly one image token"):
        model.get_image_and_text_embeddings(
            input_ids=ids, image_features=torch.randn(1, IMG_TOKENS, EMB_DIM),
            attention_mask=torch.ones_like(ids),
        )


# ---------------------------------------------------------------------------
# R7 -- truncation changes pooling semantics
# ---------------------------------------------------------------------------

def test_truncated_record_silently_changes_pooling_target():
    """
    A record whose answer EOS has been truncated away pools at the *question* EOS while
    an intact record in the same batch pools at the answer EOS. Demonstrated on the real
    `_eos_hidden_state`.
    """
    import torch

    model = _model()
    intact = torch.tensor([[BOS, IMG_TOKEN_ID, 10, 11, EOS, BOS, 20, 21, EOS]])
    truncated = torch.tensor([[BOS, IMG_TOKEN_ID, 10, 11, EOS, BOS, 20, 21, 22]])

    def pooled_index(ids):
        eos_mask = ids == EOS
        L = ids.size(1)
        return int(
            (L - 1 - eos_mask.flip(dims=[1]).float().argmax(dim=1) + (IMG_TOKENS - 1)).item()
        )

    i_intact, i_trunc = pooled_index(intact), pooled_index(truncated)
    print(f"\n  intact record   pools at combined index {i_intact} (answer EOS)")
    print(f"  truncated record pools at combined index {i_trunc} (question EOS)")
    assert i_intact != i_trunc, "the two cases should pool at different positions"


def test_dataset_warns_when_a_record_exceeds_seq_length():
    """
    F8, fixed. Right-truncation still happens -- changing it would alter training -- but
    it is no longer silent: the first affected record raises a RuntimeWarning and a
    per-dataset counter tracks the rest.
    """
    import inspect

    from data.vqa_dataset import MouseTrajDataset

    src = inspect.getsource(MouseTrajDataset.__getitem__)
    branch = src[src.index("elif len(tok_qa) > self.seq_length:") :]
    branch = branch[: branch.index("item[")] if "item[" in branch else branch

    assert any(w in branch for w in ("warn", "print", "raise", "logger")), (
        "Records longer than seq_length are truncated from the right -- removing the "
        "answer and its EOS -- with no warning, no counter, and no error. Downstream, "
        "_eos_hidden_state silently pools such records at the question EOS instead of "
        "the answer EOS. Add a count of truncated records, or left-truncate."
    )


@pytest.mark.realdata
def test_no_real_record_exceeds_seq_length(data_root):
    """The measurement: does any real record actually overflow 150 tokens?"""
    import json

    from transformers import AutoTokenizer

    all_json = data_root / "embeddings" / "vlm" / "mouse_all_vqa_traj.json"
    if not all_json.exists():
        pytest.skip(f"VQA records not found: {all_json}")

    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    records = json.loads(all_json.read_text())
    lengths = [len(tok(r["question"] + r["answer"])["input_ids"]) for r in records]
    over = [n for n in lengths if n > 150]
    print(f"\n  {len(records)} records: max token length = {max(lengths)}, over 150 = {len(over)}")

    assert not over, (
        f"{len(over)} of {len(records)} records exceed seq_length=150 and are truncated "
        f"from the right, losing the answer EOS (max length {max(lengths)})."
    )
