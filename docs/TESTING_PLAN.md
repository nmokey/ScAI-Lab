# Verification & Reproducibility Plan

Status: proposal, not yet implemented. Written 2026-08-29 after a full read of
`scripts/`, `vlm/`, and `docs/`.

---

## 0. What we can actually prove

"Prove correctness" means different things at different layers, and it is worth being
precise so we don't over-claim. This plan builds five levels of evidence, weakest to
strongest:

| Level | Instrument | What it establishes |
|---|---|---|
| **L1 Oracle equivalence** | Compare our hand-rolled math to a trusted reference (sklearn / scipy) on random inputs | Our metric code computes the metric it claims to |
| **L2 Invariants** | Properties that must hold for *any* input (fold disjointness, shape/dtype contracts, permutation invariance, monotonicity) | The machinery is internally consistent |
| **L3 Negative controls** | Feed the pipeline data with **no signal** (shuffled labels, random embeddings) and assert the metric collapses to chance | *Nothing is leaking.* This is the only test that can falsify "our result is an artifact" |
| **L4 Positive controls** | Feed data with **planted, known signal** and assert the pipeline recovers it | The pipeline is sensitive enough to detect real effects |
| **L5 End-to-end reproduction** | Seeded, pinned re-run producing byte-comparable artifacts | The published numbers came from this code |

L3 is the one that matters most for this project. Every headline result here is a small-*n*
supervised metric, and the failure mode is not "the code crashes" — it is "the code
quietly reports a number that comes from leakage or from noise." Unit tests alone cannot
catch that; negative controls can.

**What tests cannot prove:** that TBR-2 measures aortic calcification, that RAD-DINO
embeddings encode biology rather than acquisition artifacts, or that *n*=32 supports the
claims. Those need the statistical work in §6 and the confound controls in §7 — which is
where I think the largest scientific risk currently sits.

---

## 1. Current state

- **6,314 lines of Python, zero tests, no CI, no test dependencies.**
- `vlm/` is not an importable package: modules use bare imports (`from utils.misc_utils
  import ...`, `from data.eval import ...`) that only resolve when CWD is `vlm/`, via a
  `sys.path.insert` in the entry-point scripts. Nothing under `vlm/` can be imported from
  a test without replicating that hack. **This is the first blocker.**
- All data paths are hardcoded absolute paths on a machine that is not present here
  (`/data1/Processed_NIfTI_Test/...`). Tests must therefore run on **synthetic fixtures**,
  with real-data checks as a separately-marked tier.
- `manifest.csv` and `mouse_manifest.csv` are referenced by the README's repo-structure
  section but are not tracked in git — the DICOM stage is currently unreproducible from a
  clean clone.
- Local dev machine has Python 3.14 with no numpy/torch. The suite needs its own pinned
  virtualenv; torch-dependent tests may need to run on the GPU server if 3.14 wheels are
  unavailable.

---

## 1b. Findings register — what the probes actually measured

> **Superseded.** Phase A is complete; the authoritative, per-finding record now lives
> in [FINDINGS.md](FINDINGS.md) (F1–F12, each backed by an executable test). The
> section below is the earlier partial write-up, kept for history.

P0 and the four diagnostic probes are implemented (`tests/`). Reading code produces
hypotheses; running the probes produces measurements, and three of the four hypotheses
in §2 came back different from how they were first written. Recorded here so the
register reflects evidence rather than suspicion.

| Risk | Status after probing | Measured |
|---|---|---|
| **R1** model selection on the held-out subject | **Confirmed, unambiguous** | Trainer `eval_dataset` and the held-out inference set resolve to the *same records*. Simulated inflation on **signal-free** data: **+0.143** AUROC at 10 epochs, **+0.158** at 20, **+0.181** at 50, **+0.201** at 100 |
| **R2** genotype leaks via MLP conditioning | **Mechanism real, magnitude unmeasured** | Not automatic: when genotype is uninformative the MLP ignores the bit (`cos(pred, pred_flipped) = 0.99992`). Influence is set by `‖embedding‖` — at 1/50 scale cosine falls to 0.922. Decided by one unmeasured quantity: the real RAD-DINO norm |
| **R3** heads read the teacher-forced answer | **Reframed — worse than hypothesised** | Crowding-out did *not* occur. Instead: head ranks perfectly (**AUROC 1.000**) while classifying at **chance (acc 0.500)**. The decision threshold does not survive the train→inference context shift |
| **R5** genotype confounded with session | **Confirmed, reproduces the reported effect size** | Acquisition structure alone, with **zero** genotype signal planted, yields genotype **AUROC 0.754** (permutation p = 0.010) under leave-one-subject-out — against the reported VLM 0.762. Leave-one-session-out: chance |

### R1 — confirmed

`test_trainer_eval_set_is_not_the_held_out_subject` builds a fold exactly as `run_fold`
does, then asks the real trainer methods which files they load. Both return
`['NaF_KO_01']`. The fold *construction* is correct — `write_fold_jsons` splits cleanly;
the defect is entirely the reuse of the val split as the model-selection set.

The magnitude simulation strips the LLM away entirely, because the effect is a property
of the selection rule, not the model: 32 subjects, 3 records each, pure noise features,
pick the epoch minimising held-out loss. Inflation is monotone in epoch count, so the
50/100-epoch rows of the sweep are the most affected. For scale, the README reads a
baseline→longitudinal jump of 0.163 → 0.762 as evidence that longitudinal tokens carry
genotype signal.

### R2 — narrowed, and I had it too strong

The plan asserted the leak "has the right signature" for the headline result. The probe
does not support that as stated. An unused input does not leak: with genotype
uninformative about the target, the MLP learns to ignore the bit almost exactly. What
governs the leak is the ratio of the unit conditioning one-hot to the embedding
magnitude, and that ratio is an empirical property of the real embeddings that nobody
has measured. `test_genotype_bit_influence_realdata` is a one-command measurement.
The cheap insurance regardless: re-run `train_longitudinal.py --no-conditioning`.

### R3 — the consequence is calibration, not crowding-out

Two corrections to §2's R3:

* The `(img_tokens - 1)` offset arithmetic in `_eos_hidden_state` is **correct** — it
  lands on the final position in both training and inference. It should not be "fixed".
  A passing test now pins that down so nobody breaks it.
* The head's *input* does depend on the answer (changing only the answer tokens moves
  the pooled hidden state by 0.53 max-abs), and for genotype records the answer states
  the label. But the damage is not to ranking — it is that the threshold is calibrated
  on a context that does not exist at inference.

This has a direct consequence for the published tables. `vlm/data/eval.py` classifies
with `genotype_logit > 0.0`. Across every condition and training length tested, accuracy
sat at ~0.500 regardless of AUROC. **An accuracy of 0.219 on a balanced binary task is
below the 0.5 floor and is the signature of a shifted threshold, not of inverted
signal** — which is how README.md currently reads it ("its head ranks genotype
anti-correlated"). Every genotype *accuracy* in README.md and docs/experiments.md is a
reading of a miscalibrated head; the AUROCs are the metrics that survive.

### R5 — confirmed at exactly the reported effect size

The synthetic acquisition signature is calibrated so leave-one-subject-out lands at
AUROC 0.754, against the reported VLM 0.762 and RAD-DINO T2c 0.869. The claim is not
that the real effect is this big — it is that an effect of the reported size is fully
reproducible from acquisition structure alone, so the current protocol cannot tell the
two apart. At higher imprint strength the synthetic confound saturates at AUROC 1.000.

Blocked on plumbing: the `.npz` contract has no `session_id`, so leave-one-session-out
cannot be run on the real embeddings until `build_nifti_dataset` records the source
session per mouse and the encoder scripts propagate it.

### A statistical artifact the probes exposed

When folds are grouped by session, every held-out fold is single-class, so pooling
probabilities across folds and computing one AUROC is not a well-behaved estimator — it
sits near 0.05 under the null, not 0.5. Every assertion in the control suite therefore
compares against a *permutation null* computed with the same estimator rather than
against a hard-coded 0.5. The pipeline pools identically (`_loso_logistic` in
`scripts/evaluate_embeddings.py`), so every "chance = 0.50" printed in its reports
inherits the same problem.

### Incidental

* `vlm/run/run_mouse_vlm_loso.py` calls `torch.cuda.set_device(0)` at **import time**,
  so the entry point cannot be imported on a CPU-only machine. Move it into `main()`.
* `viz_emb_trainer.get_training_args` passes `evaluation_strategy=`, removed in
  transformers v5 — the pipeline is hard-pinned to 4.46.3 and will not run on a modern
  stack as written.

---

## 2. Correctness risks found while reading

> These are the **hypotheses from the code read**, kept as written so the reasoning is
> auditable. Where probing has since changed the picture — R2 and R3 materially — §1b
> above is the corrected record and takes precedence.

These are the concrete things the suite must be designed to catch. Ranked by how much
they threaten the published results. Each is stated as a claim I can defend from the
code, with the test that would have caught it.

### R1 — Model selection on the held-out subject (test-set leakage) — **critical**

`viz_emb_trainer.get_training_args()` hardcodes `load_best_model_at_end=True`
(`vlm/model/viz_emb_trainer.py:139`). `get_train_data()` builds the Trainer's
`eval_dataset` from `params["data"]["inf_data_path"]`, and `run_mouse_vlm_loso.py:82`
sets `inf_data_path` to **the held-out subject's records**. With
`evaluation_strategy: epoch` and `save_strategy: epoch`, every LOSO fold selects its
checkpoint by loss on the very subject it is then evaluated on.

Every reported VLM LOSO number (genotype 0.719 / AUROC 0.762, all TBR metrics, the whole
epoch sweep) is affected. It plausibly also explains the sweep's non-monotonic instability.

> **Test:** assert that the Trainer's eval dataset and the inference dataset are disjoint
> in `pid`, for every fold. Plus a negative control: run LOSO on label-shuffled data and
> assert genotype AUROC ≈ 0.5 ± CI. Under the current config a shuffled-label run should
> still beat chance — that is the smoking gun.
>
> **Fix:** either split a val subject out of the 31 training subjects, or set
> `load_best_model_at_end=False` and take the final epoch.

### R2 — Genotype label leaks into the VLM's longitudinal image tokens — **critical**

`train_longitudinal.build_pairs()` (`scripts/train_longitudinal.py:157`) concatenates a
2-d one-hot **genotype** vector into the MLP's input. The held-out subject's own genotype
is used to produce its predicted embeddings, and `save_predicted_embeddings()` writes
those to `predicted_embeddings/<sid>_ts{1,2,3}.npy` — which `MouseTrajDataset._load_embedding`
loads as VLM image tokens 1–3.

`evaluate_encoder_vs_vlm.py:189` already documents this leak for condition B, and
`results.md` calls B "an inflated ceiling." **But the same leak flows into the VLM**, and
the README's central claim — "feeding MLP-predicted future embeddings as extra image
tokens is what carries the genotype signal" — is exactly the claim the leak would fake.
The 0.219 → 0.531 → 0.719 progression is the signature you'd expect from a label being
smuggled in through the added tokens.

> **Test:** a leakage probe. Train the MLP with genotype conditioning on synthetic data
> where the embedding carries *zero* genotype information, then linear-probe genotype
> from the predicted embeddings. If AUROC > chance, the conditioning vector is
> recoverable from the output — quantifying the leak exactly.
>
> **Fix / required ablation:** re-run the longitudinal MLP with `--no-conditioning` (or
> with genotype ablated from the conditioning vector, keeping cohort + step) and re-run
> the VLM headline. The difference between the two is the real effect size. Until that
> run exists, the headline finding is not supported.

### R3 — Multitask heads are trained on a hidden state that contains the answer — **high**

`VisionLanguageModel._eos_hidden_state` deliberately takes the **last** EOS, which during
training is the answer-terminal EOS. `design_decisions.md` states this as intent ("heads
read context that includes the generated answer"). But for genotype records the answer
literally is the label ("the mouse will develop atherosclerosis"), so during training the
genotype head learns to read the label off the teacher-forced answer tokens rather than
from the image. At inference the answer is absent, so the head is evaluated far
off-distribution.

This is not test-set leakage (inference is clean), but it means the head is trained on a
degenerate shortcut — the most likely explanation for the ts0-only baseline landing
*below chance* (AUROC 0.163), which the README currently interprets as "no stable
single-scan genotype signal."

> **Test:** a positive control on a tiny random LLM — construct records where the image
> embedding perfectly determines genotype but the answer text is uninformative, and vice
> versa. Assert the head learns from the image arm. It currently should not.
>
> **Fix:** take the hidden state at the **question-terminal** EOS (train/inference
> matched), or mask the answer span when pooling for the heads.

### R4 — Per-mouse identity assignment can silently mislabel subjects — **high**

`build_nifti_dataset.segment_animals()` skips empty or sub-1-mL quadrants, returning a
*compacted* list of bboxes that still carries the true `position` field.
`stage3_crop_and_write()` then maps `mouse_num = mouse_nums[i]` by **list index**, not by
`bbox["position"]` (`scripts/build_nifti_dataset.py:585-587`). Any session where a
quadrant is skipped, or where mice do not occupy positions 1..n in manifest order, assigns
scans to the wrong animal — wrong subject ID, wrong genotype, broken longitudinal linkage.
`DATA_MANIFEST.md` flags at least three irregular sessions (`m54225` mixed-timepoint with
mouse numbers `19,20,1,2`; `KO 2,3`; `KO 9,11,12`) where this is live.

> **Test:** synthetic 3-mouse phantom with an empty quadrant 3; assert the mouse at
> position 4 receives `mouse_nums[2]` only if the manifest genuinely means that, and that
> the code fails loudly rather than guessing when `len(bboxes) != len(mouse_nums)`.
>
> **Fix:** make `mouse_nums` an explicit position→mouse mapping in the manifest, and
> index by `bbox["position"]`.

### R5 — Genotype is perfectly confounded with acquisition session — **high (scientific, not a code bug)**

From `DATA_MANIFEST.md`: every scan session contains exclusively WT or exclusively KO
mice, and `stage3_crop_and_write` assigns one session-level `genotype` to all mice in a
session. Sessions also differ in post-injection timepoint (1h vs 3h). So a classifier can
achieve high "genotype" AUC by recognising the *session* — scanner drift, bed position,
reconstruction settings, injection timing — with no biology involved.

This affects RAD-DINO's T2c AUC = 0.869 and every VLM genotype number.

> **Test:** add `session_id` to the embedding `.npz` contract and run (a) leave-one-*session*-out
> instead of leave-one-subject-out, and (b) a session-classification probe. If sessions are
> trivially separable and LOSO-by-session collapses genotype AUC to chance, the genotype
> result is an acquisition artifact.

### R6 — Non-deterministic training; reported numbers are single unseeded runs — **high**

`data_seed: 0` appears in every YAML but is never read by anything except the required-params
check. `viz_emb_trainer.train()` constructs the model (projection layer + both multitask
heads, randomly initialised) **before** `transformers.Trainer` is instantiated, so HF's
`set_seed(args.seed)` never covers head initialisation. `train_longitudinal.py` never calls
`torch.manual_seed` at all — the longitudinal MLP is fully unseeded.

Combined with *n*=32 (SE on an accuracy ≈ 0.086) this means the sweep table's
0.625 / 0.500 / 0.375 / 0.344 / 0.375 baseline column is consistent with pure noise, and
"20 epochs is the sweet spot" is a max over a noisy 2×5 grid.

> **Fix:** thread `data_seed` through to `set_seed()` before model construction and to
> `TrainingArguments(seed=..., data_seed=...)`; seed `train_longitudinal`. Then re-run the
> headline configs across ≥5 seeds and report mean ± SD (§6).

### R7 — `_eos_hidden_state` is fragile to truncation and prompt format — **medium**

`seq_length: 150` with right-truncation `tok_qa[:-trunc]` (`vqa_dataset.py:95`) removes
tokens from the **end** — i.e. the answer and its EOS. If any record exceeds 150 tokens,
`_eos_hidden_state` silently falls back to the question EOS for that record only, mixing
two different pooling semantics within a batch. Nothing asserts records fit.

Separately, `_eos_hidden_state` assumes exactly the EOS layout of `prompt_type: standard`;
under `llama3` the question contains multiple `<|eot_id|>` tokens and the offset logic is
unverified.

> **Test:** assert max tokenised length over all records < `seq_length`; unit-test
> `_eos_hidden_state` against hand-constructed `input_ids` for both prompt types, including
> the truncated case.

### R8 — `get_image_and_text_embeddings` assumes a batch-uniform image-token position — **medium**

`img_pos` is derived from `input_ids[0]` only and applied to the whole batch
(`vision_language_model.py:112-114`), and `.item()` raises if a sequence contains more than
one `<image>` token. Currently safe because the prompt prefix is fixed-length, but nothing
enforces it, and it will break the moment `beg_prompt` becomes non-empty or variable.

> **Test:** batch with heterogeneous prompt lengths → assert either correct per-row
> splicing or a loud failure.

### R9 — TBR parsing cannot represent negative or oddly-formatted values — **medium**

Both `data/eval.py:_TBR_RE` and `vqa_dataset._tbr_targets` use `(\d+(?:\.\d+)?)` — no sign.
A negative TBR in the ground truth string would be parsed as its absolute value in the
target and silently dropped from the text metric. `-1.0` is simultaneously the padding
sentinel in `_tbr_targets`, so a genuine TBR of −1 is indistinguishable from "missing".

Also, question routing in `eval.py` is string matching (`"status" in question.lower() and
"TBR" not in question`) with inconsistent case handling — brittle to any prompt rewording.

> **Test:** property test over generated answer strings; assert round-trip
> `format_*_qa → _parse_tbr → _tbr_targets` is lossless, and assert TBR values are
> non-negative in the real CSV (so the sentinel is safe) — or change the sentinel to NaN.

### R10 — `tbr_strategy_3` has a probable axis swap — **low (unused path)**

`ys, xs = np.where(labeled == spine_id)` on an array indexed `[x, y]` returns
(first-axis, second-axis) = (x-indices, y-indices), so `spine_cx`/`spine_cy` are swapped;
the 3 mm "anterior" offset is then applied along the wrong axis
(`extract_tbr_features.py:212-218`). TBR-3 is not the column used (`TBR_COL = tbr2_p95_median`),
but `design_decisions.md` cites its failure rate as the reason for rejecting it — a
justification that may rest on a bug.

> **Test:** synthetic phantom with a known spine location and a known hot ROI; assert the
> detected ROI centre lands where it should.

### Smaller items

- `README.md` lists `manifest.csv` / `mouse_manifest.csv` in the repo structure; neither is tracked.
- `run_mouse_vlm_loso.run_fold` shallow-copies `base_params` — nested dicts are shared across
  folds. Safe today (only top-level keys are patched) but one nested edit away from cross-fold contamination.
- `viz_emb_trainer.train()` assigns CPU tensors to the `tbr_mean`/`tbr_std` buffers of a model
  that HF later moves to GPU. Works only because `Trainer` moves the model afterwards.
- `_loso_logistic` fits `LogisticRegression(C=1.0)` on unstandardised embeddings whose scale
  differs by encoder (768-d RAD-DINO vs 2048-d Merlin) — the effective regularisation
  strength differs across the encoders being compared in the headline table.
- `evaluate_embeddings.run_t4_longitudinal` uses `__import__("sklearn.linear_model", ...)`
  inline; cosmetic, but it bypasses the module-level import conventions used everywhere else.

---

## 3. Test infrastructure (prerequisite work)

1. **Make `vlm/` importable.** Add `pyproject.toml` declaring `vlm` as a package, convert
   bare imports to `vlm.`-qualified ones (or add a `conftest.py` that inserts `vlm/` on
   `sys.path` — cheaper, less invasive, and preserves the entry-point scripts as-is).
   Recommend the `conftest.py` route first so tests can land without touching research code.
2. **`requirements-dev.txt`**: `pytest`, `pytest-cov`, `hypothesis`, `scipy`. Torch-dependent
   tests import torch only inside `@pytest.mark.torch` tests.
3. **Markers** in `pyproject.toml`:
   - `unit` — pure logic, no torch, < 1 s each. Runs everywhere, gates every commit.
   - `torch` — needs torch/transformers on CPU (tiny random LLM). Minutes.
   - `realdata` — needs `/data1`. Skipped unless `SCAI_DATA_ROOT` is set.
   - `slow` — permutation/bootstrap suites, hundreds of fits. Nightly only.
4. **Synthetic fixture factory** (`tests/fixtures/synth.py`) — the backbone of the suite:
   - `make_embeddings_npz(n_subjects, weeks, dim, signal=...)` producing the exact standard
     `.npz` interface, with a `signal` knob: `"none"` (pure noise), `"week"`, `"genotype"`,
     `"subject"`, so the same fixture drives both negative and positive controls.
   - `make_vqa_records(...)` producing records matching `create_mouse_traj_dataset`'s schema.
   - `make_ct_phantom(n_mice, positions, spacing)` — a SimpleITK volume with synthetic
     "bone" blobs at chosen quadrants, for the segmentation tests.
5. **CI** (`.github/workflows/tests.yml`): `unit` on every push; `unit + torch` nightly.

---

## 4. Test inventory

### Layer 1 — Oracle equivalence (`tests/unit/`)

| File | Target | Assertions |
|---|---|---|
| `test_eval_metrics.py` | `vlm/data/eval.py` | `_auroc` vs `sklearn.roc_auc_score` over 200 random label/score sets **including heavy ties** (the tie-handling branch is hand-written and untested); `_pearson` vs `scipy.stats.pearsonr`; `_r2` vs `sklearn.r2_score`; NaN behaviour for single-class / zero-variance / n<2 |
| `test_eval_parsing.py` | `_parse_tbr`, `_tbr_targets` | round-trip from `format_tbr_qa`/`format_combined_qa`; `NA` slots stay −1; slot mapping `{3:0,6:1,8:2}` for the W15+W20-only subject; negative and malformed input (R9) |
| `test_eval_routing.py` | question-type routing in `calculate_mouse_metrics` | every question string the generator can emit routes to exactly one of geno/tbr/combined |
| `test_encoder_vs_vlm_metrics.py` | `evaluate_encoder_vs_vlm.pearson/r2/tbr_slot_metrics` | same oracles; these are a *second* hand-rolled copy of the same math and must agree with `data/eval.py` |
| `test_retrieval_metrics.py` | `_ap_at_k`, `run_t3b`, `run_t3c` | hand-computed AP on small ranked lists; self-exclusion; MRR when no relevant item exists |

### Layer 2 — Invariants (`tests/unit/`)

| File | Assertions |
|---|---|
| `test_loso_integrity.py` | For `_loso_logistic`, `run_t3a`, `train_longitudinal` LOSO, and `run_mouse_vlm_loso.write_fold_jsons`: no `pid` appears in both train and test of any fold; every sample is predicted exactly once; folds cover all subjects |
| `test_subject_id_contract.py` | `NaF_WT_03`-style IDs parse identically in all five places that parse them (`evaluate_embeddings.main`, `train_longitudinal.parse_subject_meta`, `create_mouse_traj_dataset._genotype`, `evaluate_encoder_vs_vlm` `"_KO_" in sid`, `build_nifti_dataset._build_manifest_row`) — property test over generated IDs including malformed ones |
| `test_npz_contract.py` | `load_embeddings` in all four encoder scripts + evaluator agree on keys, dtypes, shapes, and length alignment |
| `test_build_pairs.py` | `build_pairs` emits only consecutive transitions; conditioning vector layout is exactly `[emb | geno(2) | cohort(2) | step(3)]`; subjects with gaps produce the right pairs; genotype one-hot matches the scalar label |
| `test_splits.py` | `make_random_splits` is deterministic under seed, genotype-stratified, subject-disjoint, and covers all subjects |

### Layer 3 — Negative & positive controls (`tests/control/`, marker `slow`)

**This is the core of the plan.** Each runs the real analysis function on synthetic data.

| File | Design | Assertion |
|---|---|---|
| `test_null_encoder_eval.py` | random embeddings, real label structure, 200 permutations | T2b/T2c AUC 95% interval brackets 0.5; the observed-vs-null permutation p-value machinery works |
| `test_planted_signal.py` | embeddings = label-dependent mean + noise | T2c AUC > 0.9 — proves the probe is sensitive |
| `test_longitudinal_leakage.py` | **R2 probe.** Embeddings independent of genotype; train MLP *with* genotype conditioning under LOSO; linear-probe genotype from `Y_pred` | AUROC must be ≈ 0.5. Expected to **fail** on current code — that failure is the quantified leak |
| `test_loso_selection_leakage.py` | **R1 probe.** Tiny-LLM VLM LOSO on shuffled genotype labels | genotype AUROC ≈ 0.5. Expected to fail with `load_best_model_at_end=True` |
| `test_head_shortcut.py` | **R3 probe.** Two synthetic datasets: (a) label in image only, (b) label in answer text only | head must learn from (a); learning only from (b) demonstrates the shortcut |
| `test_session_confound.py` | **R5 probe.** Embeddings carrying a session offset but no genotype signal, sessions genotype-pure | leave-one-session-out AUC ≈ 0.5 while LOSO-by-subject AUC is inflated — reproduces the confound in a controlled setting |

### Layer 4 — Component behaviour under torch (`tests/torch/`, marker `torch`)

Uses `hf-internal-testing/tiny-random-LlamaForCausalLM` so everything runs on CPU in seconds.

| File | Assertions |
|---|---|
| `test_vlm_splicing.py` | `get_image_and_text_embeddings`: output length == `L - 1 + img_tokens`; image features land at the right offset; `attention_mask` and `labels` shift consistently; image positions are `-100` in labels; heterogeneous-batch behaviour (R8); >1 `<image>` token raises |
| `test_eos_pooling.py` | `_eos_hidden_state` picks the answer-terminal EOS in train mode and the prompt-terminal EOS in test mode, for both `standard` and `llama3` prompts, including the truncated-record case (R7) |
| `test_multitask_loss.py` | TBR MSE masks `-1` slots (changing a padded target does not change the loss); genotype BCE skips `label < 0`; z-scoring uses the fold's buffers; `multitask_wt` scales as documented |
| `test_dataset_shapes.py` | `MouseTrajDataset` returns `(img_tokens, 768)`; missing `.npy` → exact zeros; label masking puts `-100` on exactly the question span; **max tokenised length < `seq_length`** for every real record |
| `test_checkpoint_roundtrip.py` | `save_pretrained` → `from_pretrained` restores projection, both heads, and `tbr_mean`/`tbr_std` bit-exactly; a missing `other_weights.bin` fails loudly rather than silently using random heads |
| `test_denormalization.py` | end-to-end: a head emitting z-scored `z` yields `z*std+mean` in `tbr_regression`, matching the raw scale of `tbr_targets` |

### Layer 5 — Pipeline / data integrity (`tests/pipeline/`)

| File | Assertions |
|---|---|
| `test_segmentation.py` | `segment_animals` on phantoms: 4/3/2-mouse layouts; quadrant→position mapping; `min_voxels` rejection; **R4** — assert compacted-bbox indexing does not silently mislabel; `crop_to_physical_bbox` produces identical physical extents for CT and PET grids with different spacing |
| `test_consolidation.py` | `get_base_id`/`get_suffix_num` on all real scan-ID forms; `consolidate_sessions` picks the highest-suffix CT-Hi and PET; **flags when CT-Hi and PET come from different session versions** (a geometric-alignment risk for PET cropping) |
| `test_tbr_strategies.py` | Phantoms with known statistics: TBR-1/TBR-2 return exactly the hand-computed ratio; `<50` voxel guard; **R10** spine-localisation phantom; TBR-2 sensitivity to crop extent (how much does a ±10% z-crop move the value?) |
| `test_manifest_invariants.py` (`realdata`) | Every `mouse_manifest.csv` row has a readable NIfTI; subject IDs unique per (cohort, genotype, num); each mouse appears ≤ once per week; genotype consistent across a mouse's weeks; TBR CSV subjects ⊆ embedding subjects |

### Layer 6 — Result regression locks (`tests/regression/`, marker `realdata`)

Once R1/R2/R3/R6 are fixed and the numbers are re-generated, freeze them:

- `expected_metrics.json` holding every number that appears in `README.md`,
  `results.md`, and `experiments.md`, with tolerances.
- A test that re-runs the deterministic parts (encoder eval, longitudinal MLP with a fixed
  seed) and asserts agreement within tolerance.
- A **docs-consistency test** that parses the markdown tables and asserts every quoted
  number exists in `expected_metrics.json` — so the docs cannot drift from the artifacts.

---

## 5. Reproducibility work

1. **Seed everything.** Thread `data_seed` into `set_seed()` *before* model construction,
   into `TrainingArguments(seed=, data_seed=)`, and into `train_longitudinal` (`torch.manual_seed`,
   `np.random.seed`, and a `DataLoader` generator). Verify with a test that two runs at the
   same seed produce identical weights.
2. **Run manifests.** Every script writes a `run_meta.json` beside its outputs: git SHA,
   dirty flag, full resolved config, package versions, CUDA/driver, seed, wall time, input
   file hashes. Nothing gets cited in a paper without one.
3. **Track the manifests.** Commit `manifest.csv` (147 rows, session-level, no PHI) so the
   DICOM stage is reproducible from a clean clone.
4. **Content-hash the intermediates.** `raddino_embeddings.npz`, `tbr_features_NaF.csv`,
   `predicted_embeddings/`, and the VQA JSONs get SHA-256 recorded in the run manifest, so
   a result can be traced to the exact inputs that produced it.
5. **One canonical entry point.** A `Makefile` / `run_all.sh` that executes the pipeline
   end-to-end from `config.yaml`, so "reproducing the paper" is one command.

---

## 6. Statistical rigor (separate from, and as important as, the tests)

The published tables report point estimates with no uncertainty on *n*=32 subjects /
96 records. Minimum additions:

- **Bootstrap CIs** on every headline metric — resample subjects (not records) with
  replacement, 10k draws, report 95% percentile intervals.
- **Permutation p-values** — shuffle genotype within the LOSO harness ≥1000 times; report
  where the observed AUROC sits in the null. This is the honest test of "0.762 beats chance."
- **Seed variance** — ≥5 seeds per headline config; report mean ± SD. Given the sweep's
  observed spread I expect the epoch-choice conclusion will not survive.
- **Multiplicity** — the epoch sweep is 10 configurations × 2 metrics; the reported best
  cell needs a correction or an explicit "selected post hoc" caveat.
- **Paired comparison** — baseline vs longitudinal should be compared per-fold and paired
  (same held-out subject), not as two independent aggregates.

---

## 7. Suggested phasing

| Phase | Content | Rough size |
|---|---|---|
| **P0** | `conftest.py` + `pyproject.toml` + dev requirements + synthetic fixture factory + CI skeleton | half a day |
| **P1** | Layers 1–2 (oracle + invariant unit tests). No source changes; fast, green, immediately useful | 1–2 days |
| **P2** | Layer 3 negative controls for R1, R2, R3, R5. **These are expected to fail** — they are the diagnostic instrument, not a regression guard | 2 days |
| **P3** | Fix R1, R2, R3, R4, R6 + re-run the affected experiments. This is where the numbers change | depends on GPU time |
| **P4** | Layers 4–5 (torch component tests, pipeline/phantom tests) | 2–3 days |
| **P5** | §6 statistical layer + Layer 6 regression locks + §5 run manifests; update `README.md`, `results.md`, `experiments.md` with CIs and the corrected results | 2–3 days |

**Recommended first action:** P0 + the four probes in P2. They are cheap, and they answer
the question that actually matters — *are the headline results real?* — before we invest in
locking them in place.
