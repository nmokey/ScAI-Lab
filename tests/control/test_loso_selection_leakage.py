"""
R1 -- does the VLM's LOSO protocol select checkpoints using the held-out subject?

The wiring
----------
  vlm/run/run_mouse_vlm_loso.py::run_fold
      params["data"]["inf_data_path"] = val_path        # the held-out subject
  vlm/model/viz_emb_trainer.py::get_train_data
      val_data = factory.create_dataset(data_path=params["data"]["inf_data_path"], mode="val")
      -> returned as data["test"], passed to transformers.Trainer as eval_dataset
  vlm/model/viz_emb_trainer.py::get_inf_data
      inf_data = factory.create_dataset(data_path=params["data"]["inf_data_path"], mode="test")
      -> the records that become the reported LOSO predictions
  vlm/model/viz_emb_trainer.py::get_training_args
      load_best_model_at_end=True     (hardcoded)
  yaml: evaluation_strategy: epoch, save_strategy: epoch

Both the Trainer's eval_dataset and the held-out evaluation set read the *same*
config key, so every fold picks its checkpoint by loss on the subject it is then
scored on. Every reported VLM LOSO number is affected: genotype 0.719 / AUROC 0.762,
all TBR metrics, and the whole 2x5 epoch sweep.

The fold *construction* is correct -- `write_fold_jsons` produces properly disjoint
splits. The defect is entirely in how the val split is then reused.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.synth import make_embeddings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_YAML = REPO_ROOT / "vlm" / "yaml" / "viz_emb_params_mouse.yml"


class _StubTokenizer:
    """Minimal stand-in: MouseTrajDataset only needs token-id lookup at construction."""

    eos_token = "</s>"
    bos_token = "<s>"
    eos_token_id = 2

    def convert_tokens_to_ids(self, tok):
        return 0

    def __call__(self, text, **kw):
        return {"input_ids": list(range(len(text.split())))}


def _load_loso_module():
    """
    Import vlm/run/run_mouse_vlm_loso.py.

    Previously this needed `torch.cuda.set_device` monkeypatched: the module called it
    at import time, which raises on any CPU-only machine. That call now lives in main()
    and is guarded by `torch.cuda.is_available()`, so a plain import works anywhere --
    which is what makes the wiring below testable at all.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_loso_runner", REPO_ROOT / "vlm" / "run" / "run_mouse_vlm_loso.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic_records(n_subjects=6):
    """VQA records in the shape run_mouse_vlm_loso consumes (pid + question/answer)."""
    emb = make_embeddings(n_subjects=n_subjects, signal=(), dim=8, seed=0)
    records = []
    for qid, sid in enumerate(sorted(set(emb.subject_ids.tolist()))):
        for kind in ("genotype", "tbr", "combined"):
            records.append({
                "pid": sid,
                "qid": f"{sid}_{kind}",
                "question": "What will be the eventual mouse status for atherosclerosis?",
                "answer": "The eventual mouse status: the mouse will develop atherosclerosis.",
                "answer_vqa_numeric": {"genotype": 1 if "_KO_" in sid else 0},
            })
    return records


# ---------------------------------------------------------------------------
# The folds themselves are fine
# ---------------------------------------------------------------------------

@pytest.mark.torch
def test_fold_construction_is_subject_disjoint(tmp_path):
    """`write_fold_jsons` splits correctly: train and val share no subject."""
    mod = _load_loso_module()
    records = _synthetic_records()
    subjects = sorted({r["pid"] for r in records})

    seen = set()
    for sid in subjects:
        train_path, val_path, _inner = mod.write_fold_jsons(records, sid, str(tmp_path))
        train = json.load(open(train_path))
        val = json.load(open(val_path))

        train_pids = {r["pid"] for r in train}
        val_pids = {r["pid"] for r in val}

        assert val_pids == {sid}, f"val split should be exactly the held-out subject, got {val_pids}"
        assert not (train_pids & val_pids), f"train/val overlap for fold {sid}"
        assert train_pids | val_pids == set(subjects), "fold does not cover all subjects"
        seen |= val_pids

    assert seen == set(subjects), "every subject must be held out exactly once"


@pytest.mark.torch
def test_inner_val_split_is_three_way_disjoint(tmp_path):
    """
    F2 full fix: with --inner-val, each fold has train / inner-val / held-out
    that are pairwise disjoint, cover every subject, and the inner-val subject is
    NOT the held-out one. The Trainer's eval_dataset then never sees the test
    subject, so eval curves are honest and load_best_model_at_end is legitimate.
    """
    mod = _load_loso_module()
    records = _synthetic_records()
    subjects = sorted({r["pid"] for r in records})

    inner_seen = []
    for fold_idx, sid in enumerate(subjects):
        inner_sid = mod.pick_inner_val_subject(subjects, sid, fold_idx)
        assert inner_sid != sid
        train_path, val_path, inner_path = mod.write_fold_jsons(records, sid, str(tmp_path), inner_sid)
        assert inner_path is not None
        tr = {r["pid"] for r in json.load(open(train_path))}
        va = {r["pid"] for r in json.load(open(val_path))}
        inn = {r["pid"] for r in json.load(open(inner_path))}
        assert va == {sid} and inn == {inner_sid}
        assert not (tr & va) and not (tr & inn) and not (va & inn)
        assert tr | va | inn == set(subjects)
        assert len(tr) == len(subjects) - 2, "inner-val subject must be REMOVED from training"
        inner_seen.append(inner_sid)
    # Deterministic and spread across subjects, not the same one every fold.
    assert len(set(inner_seen)) > 1


def test_runner_refuses_selection_without_inner_val(tmp_path, monkeypatch):
    """
    Belt-and-braces: if someone sets load_best_model_at_end: true in a yaml and
    runs LOSO without --inner-val, the runner must refuse rather than silently
    select checkpoints on the test subject.
    """
    import yaml

    mod = _load_loso_module()
    records = _synthetic_records()
    data_path = tmp_path / "all.json"
    data_path.write_text(json.dumps(records))
    bad_yaml = tmp_path / "bad.yml"
    params = yaml.safe_load(open(CANONICAL_YAML))
    params["train"]["load_best_model_at_end"] = True
    bad_yaml.write_text(yaml.safe_dump(params))

    monkeypatch.setattr(sys, "argv", ["x", "--yaml", str(bad_yaml), "--all-data", str(data_path),
                                      "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="inner-val"):
        mod.main()


# ---------------------------------------------------------------------------
# ...but the val split is then reused as the model-selection set
# ---------------------------------------------------------------------------

@pytest.mark.torch
@pytest.mark.probe
def test_trainer_eval_set_is_not_the_held_out_subject(tmp_path):
    """
    EXPECTED TO FAIL.

    Behavioural check: build the fold exactly as run_fold does, then ask the real
    trainer methods which files they load. If `get_train_data`'s eval dataset and
    `get_inf_data`'s inference dataset are the same records, checkpoint selection
    happens on the test fold.
    """
    import yaml

    from model.viz_emb_trainer import VizEmbTrainer

    mod = _load_loso_module()
    records = _synthetic_records()
    held_out = sorted({r["pid"] for r in records})[0]
    train_path, val_path, _inner = mod.write_fold_jsons(records, held_out, str(tmp_path))

    # Replicate run_fold's parameter patching verbatim.
    base_params = yaml.safe_load(open(CANONICAL_YAML))
    params = {k: dict(v) if isinstance(v, dict) else v for k, v in base_params.items()}
    params["exp"]["output_dir"] = str(tmp_path)
    params["data"]["data_path"] = train_path
    params["data"]["inf_data_path"] = val_path
    params["data"]["predicted_emb_dir"] = None
    params["inf"]["train_gt_file"] = train_path

    # Instantiate without __init__ so no tokenizer/model download is needed; the
    # methods under test only touch self.params and self.tokenizer.
    trainer = object.__new__(VizEmbTrainer)
    trainer.params = params
    trainer.tokenizer = _StubTokenizer()

    data = trainer.get_train_data()
    eval_pids = {r["pid"] for r in data["test"].data}
    inf_pids = {r["pid"] for r in trainer.get_inf_data().data}

    print(f"\n  Trainer eval_dataset pids : {sorted(eval_pids)}")
    print(f"  held-out inference pids   : {sorted(inf_pids)}")

    # The finding is "checkpoints are SELECTED on the held-out subject", which needs
    # both an overlapping eval set and selection turned on. The register endorses two
    # remedies -- carve a val subject out of the 31, or disable selection -- so assert
    # the conjunction rather than one particular remedy. Encoding only disjointness
    # would keep failing after a correct fix via the second route.
    overlap   = eval_pids & inf_pids
    selecting = bool(params["train"].get("load_best_model_at_end", False))

    print(f"  load_best_model_at_end    : {selecting}")

    assert not (overlap and selecting), (
        f"The Trainer's eval_dataset and the held-out evaluation set are the same "
        f"subject ({sorted(overlap)}) AND load_best_model_at_end is True. With "
        f"evaluation_strategy=epoch, every LOSO fold then selects its checkpoint by "
        f"loss on the subject it is then scored on (+0.14 to +0.20 AUROC on noise). "
        f"Fix: carve a val subject out of the 31 training subjects, or set "
        f"load_best_model_at_end=False."
    )
    if overlap:
        print("  NOTE: eval_dataset still IS the held-out subject. Harmless for the "
              "reported metrics while selection is off, but any eval_loss curve you "
              "read during training is a test-set curve.")


@pytest.mark.torch
def test_load_best_model_at_end_is_config_driven_and_defaults_off():
    """
    Regression guard on the F2 fix.

    Was `test_load_best_model_at_end_is_hardcoded`, asserting the opposite: that
    selection-on-eval was unconditional. It now IS a config choice, so this pins the
    new contract instead -- read from the `train` section, defaulting to False.

    Source-grepping for the literal assignment is deliberately avoided here: the
    warning string printed when the flag is enabled also contains
    "load_best_model_at_end=True", so a substring check silently matches the message
    rather than the assignment. Call the real method instead.

    Asserted via a stub because get_training_args() passes `evaluation_strategy=`,
    removed in transformers v5. That incompatibility is itself worth recording: the
    pipeline is pinned to transformers 4.46.3 and will not run as-is on v5.
    """
    import yaml

    from model.viz_emb_trainer import VizEmbTrainer

    params = yaml.safe_load(open(CANONICAL_YAML))

    # The canonical config must not select on the eval set.
    assert params["train"].get("load_best_model_at_end", False) is False, (
        "The canonical yaml re-enabled load_best_model_at_end. Under LOSO the "
        "eval_dataset is the held-out subject, so this restores the F2 leak."
    )

    # ...and the code must honour the config rather than hardcoding it.
    trainer = object.__new__(VizEmbTrainer)
    trainer.params = params
    trainer.output_dir = "/tmp"

    args_on  = _training_args_with(trainer, params, True)
    args_off = _training_args_with(trainer, params, False)
    assert args_on.load_best_model_at_end is True
    assert args_off.load_best_model_at_end is False, (
        "get_training_args ignores train.load_best_model_at_end -- the F2 fix is "
        "not actually wired through."
    )


def _training_args_with(trainer, params, value):
    """Build TrainingArguments with load_best_model_at_end forced to `value`."""
    import copy

    p = copy.deepcopy(params)
    p["train"]["load_best_model_at_end"] = value
    trainer.params = p
    return trainer.get_training_args()


# ---------------------------------------------------------------------------
# How much is it worth?
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.probe
@pytest.mark.parametrize("n_epochs", [10, 20, 50, 100])
def test_checkpoint_selection_on_test_fold_inflates_metrics(n_epochs):
    """
    Quantify the leak in this exact regime: 32 subjects, 3 records each, no signal
    whatsoever. Compare two protocols that differ *only* in which epoch's predictions
    are kept.

      final-epoch      -> the honest protocol
      best-on-held-out -> what load_best_model_at_end=True does

    Everything here is pure numpy/sklearn -- no LLM required, because the effect is a
    property of the selection rule, not of the model. The epoch counts are the ones in
    docs/experiments.md; inflation grows with the number of checkpoints selected over,
    so the 50- and 100-epoch runs are the most affected.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut

    rng = np.random.default_rng(0)
    n_subjects, n_records, n_features = 32, 3, 64

    subjects = np.repeat(np.arange(n_subjects), n_records)
    labels = np.repeat(rng.integers(0, 2, n_subjects), n_records)
    X = rng.standard_normal((len(subjects), n_features))  # no signal at all

    honest_true, honest_prob = [], []
    leaked_true, leaked_prob = [], []

    for tr, te in LeaveOneGroupOut().split(X, labels, subjects):
        if len(set(labels[tr])) < 2:
            continue
        clf = SGDClassifier(loss="log_loss", random_state=0, learning_rate="constant", eta0=0.01)
        classes = np.array([0, 1])

        best_loss, best_prob = np.inf, None
        for _ in range(n_epochs):
            clf.partial_fit(X[tr], labels[tr], classes=classes)
            prob = clf.predict_proba(X[te])[:, 1]
            loss = log_loss(labels[te], prob, labels=classes)
            if loss < best_loss:  # <- selection on the held-out fold
                best_loss, best_prob = loss, prob

        honest_true.extend(labels[te]); honest_prob.extend(prob)        # final epoch
        leaked_true.extend(labels[te]); leaked_prob.extend(best_prob)   # best epoch

    honest = roc_auc_score(honest_true, honest_prob)
    leaked = roc_auc_score(leaked_true, leaked_prob)
    inflation = leaked - honest
    print(
        f"\n  {n_epochs:3d} epochs, no signal present -- pooled LOSO AUROC, 32 subjects"
        f"\n    final-epoch (honest)          : {honest:.3f}"
        f"\n    best-epoch-on-held-out (leak) : {leaked:.3f}"
        f"\n    inflation                     : {inflation:+.3f}"
    )

    assert inflation < 0.05, (
        f"Selecting the checkpoint by loss on the held-out subject inflates pooled "
        f"AUROC by {inflation:+.3f} ({honest:.3f} -> {leaked:.3f}) on data containing "
        f"NO signal at all, at {n_epochs} epochs. For scale, the README's headline gap "
        f"(baseline AUROC 0.163 -> longitudinal 0.762) is being read as evidence that "
        f"longitudinal tokens carry genotype signal. Until load_best_model_at_end is "
        f"turned off (or a real val subject is carved out of the 31 training subjects), "
        f"a gap of this size is not attributable to the model."
    )
