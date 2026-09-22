# Findings Register

**Current disposition (2026-09-21):** the entries below are the historical F1–F24 record. Their claims that old VLM runs are citable or establish failure are superseded. The checkpoint adapter loss, answer-conditioned heads, non-nested forecasts, fragile targets, scoring and resume defects are now repaired and validated; the original matched rerun and later authorized development comparisons are complete. These checks do not prove all bugs or methodological issues absent. See [current status and unresolved controls](STATUS.md), [all post-audit results](POST_AUDIT_RESULTS.md), [the complete closure record](audit_2026-09-14/PIPELINE_CLOSURE.md) and [current results](results.md).

Specific interpretation corrections: F3 affects head learning/ranking as well as threshold calibration; F22 does not establish a negative or below-chance corrected-VLM result; F23 demonstrates acquisition sensitivity but does not quantify a causal fraction due to scanner drift; F24 normalization affects later rollout directions as well as scale. The genotype/diet association is not disease localization. Deterministic forecast tokens can help a finite-data learner even though they add no new observed measurements. TBR-3 geometry and Merlin cache versioning are repaired, with remaining anatomical and preprocessing limits disclosed. Historical text below is preserved so prior decisions remain auditable.


Audit of the analysis pipeline for defects that could invalidate published results.
Every entry is backed by an executable test; the test's assertion message restates the
finding, its measured magnitude, and the fix.

    .venv-test/bin/pytest -m "not probe"   # passing guards, incl. regression locks on fixes
    .venv-test/bin/pytest -m probe -s      # open findings, each failing by design

**Status vocabulary.** `CONFIRMED` = demonstrated by running the real code.
`REFUTED` = hypothesised from reading, disproved by test. `BLOCKED` = mechanism
established, magnitude unmeasurable until plumbing exists. Findings are ordered by how
much of the published work they affect.

**Status vocabulary, continued.** `FIXED` = defect repaired and the probe converted into
a passing regression guard. Protocol-changing fixes (F2, F10, F13) were made switchable
and defaulted to the leak-free branch; the pre-fix behaviour is reproducible via
`--one-step`, `load_best_model_at_end: true` (+ `--inner-val`), and the archived yamls.
That statement describes the historical audit stage. The repaired model now defaults to `question_eos`; legacy `answer_eos` is only a reproduction option.

Historical entries below are retained as recorded through 2026-09-14; the current disposition above was reconciled on 2026-09-21.

---

## Timeline — how the audit unfolded

| date | what happened | findings |
|---|---|---|
| 2026-08-29 | Code audit from a full read of `scripts/` and `vlm/`; test suite and this register created. Five defects fixed (mouse identity by index, unused seeds, splice offset, silent truncation, TBR-3 axis swap); the rest left as probes. | F1–F12 |
| 2026-09-09 | Second pass, this time measuring rather than reading. The genotype label was found inside the VLM's forecast tokens (probe 0.869 on `ts1` vs 0.353 on the scan it came from). Fixed alongside F2 and F13; leak-free re-run of both VLM arms: genotype 0.762 → 0.464, TBR R² < 0. The headline did not survive. | F10 confirmed, F13–F15 |
| 2026-09-12 | Uncertainty added everywhere (CIs, permutation p). The confound unit turned out to be the *mouse-group* (all 10 genotype-pure), not the session; under leave-one-group-out, Week 12 genotype is chance and only emerges by Week 20. Encoder table rebuilt for all four encoders. TBR-3 re-run: neither TBR proxy carries the biology. Every doc number reproduced from audited code. | F5 resolved, F9 follow-up |
| 2026-09-13 | A parallel session added statistical tests, exact permutation enumeration, `content_type` routing, and a 5-seed VLM sweep (seed SD ≈ 0.2). Meanwhile, rendering the scanner beds revealed a tube segmented as a mouse in two sessions; fixed, re-cropped, everything downstream re-derived. Then a plumbing audit of the longitudinal path found forecast tokens exported at 2.7× the observed norm, and an air-region test showed early-vs-late separation is substantially scanner drift. | F16–F19, F21, F22 (other session); F20, F23, F24 |
| 2026-09-14 | Three VLM runs in flight on fully corrected data: baseline, longitudinal (scaled tokens), longitudinal + F3. Lab asked to confirm mouse placement order. | — |

**Historical interpretation — superseded by the current disposition above.** Of the three original claims, none survives as stated. What does: every
encoder separates Week 12 from Week 20 scans — but that is substantially scanner drift with a
growth component, not disease (F23); RAD-DINO alone shows a genotype difference that emerges
over the study, and that is the one encoder disease result (F5); and the VLM's failure to
forecast from a Week 12 scan is credible because the scan has nothing to forecast from. The
TBR label needs validating before any TBR modelling means anything (F9). Subject-level
results rest on a mouse-placement assumption awaiting the lab's answer (F20).

---

## Summary

| ID | Finding | Status | Invalidates | Test |
|---|---|---|---|---|
| **F1** | Reported AUROCs not distinguishable from noise at n=32 | **FIXED** (CIs + permutation p in both evaluators; seed aggregator) | Every genotype claim | `unit/test_pooled_auroc_estimator.py` |
| **F2** | Checkpoints selected on the held-out subject | **FIXED** (`--inner-val` for a real validation subject) | Every VLM number to date | `control/test_loso_selection_leakage.py` |
| **F3** | Genotype head threshold does not transfer to inference | CONFIRMED | Every genotype *accuracy* | `control/test_head_shortcut.py` |
| **F4** | Mouse identity assigned by list index, not quadrant | **FIXED** | Potentially all results, upstream | `pipeline/test_mouse_identity.py` |
| **F5** | Genotype confounded with **mouse-group** (coarser than session) | **CONFIRMED on real data** — Week-12 genotype is group recognition; late-week genotype survives | Baseline-scan genotype claims | `control/test_session_confound.py` |
| **F6** | `data_seed` declared everywhere, consumed nowhere | **FIXED** | Reproducibility of all runs | `unit/test_determinism.py` |
| **F7** | `<image>` position taken from row 0, applied batch-wide | **FIXED** | Latent; nothing today | `torch/test_splicing_and_truncation.py` |
| **F8** | Over-length records truncated silently, changing pooling | **FIXED** | Unknown until measured | `torch/test_splicing_and_truncation.py` |
| **F9** | Axis swap in `tbr_strategy_3` spine localisation | **FIXED**; re-run shows **neither TBR proxy separates genotype** | The TBR ground truth itself | `pipeline/test_tbr_strategies.py` |
| **F10** | Genotype leaks via longitudinal MLP conditioning | **CONFIRMED — re-run kills the headline** | The headline genotype claim | `control/test_longitudinal_leakage.py` |
| **F11** | TBR round-trip lossless; `-1` sentinel safe | **REFUTED** | — | `unit/test_tbr_parsing.py` |
| **F12** | Per-fold calibration offsets distort pooled AUROC | **REFUTED** | — | `unit/test_pooled_auroc_estimator.py` |
| **F13** | ts2/ts3 predicted from *observed* future scans | **FIXED** (rollout is now the default; `--one-step` to reproduce old runs) | "from a single baseline scan" | — |
| **F14** | TBR regression metrics double-count every subject | **FIXED** | Reported n; any CI built on it | — |
| **F15** | Question routing case-inconsistent; `content_type` constant | **FIXED** | Latent | — |
| **F16** | Bootstrap/permutation code behind every CI and p-value is untested | **FIXED** | Confidence in every "inside the null" claim | `unit/test_audit_statistics.py` |
| **F17** | Cited audit artifacts are untracked; reproducibility recommendation unmet | **FIXED** | Reproducibility from a clean clone | — |
| **F18** | Mouse-group permutation null has only 15–120 distinct arrangements (varies by week) | **FIXED** (exact enumeration; re-run on real data) | Precision claimed for F5's group-level p-values | `unit/test_audit_statistics.py` |
| **F19** | Backbone selection predates the mouse-group confound control | **FIXED** (documentation) | Framing of the RAD-DINO selection rationale | — |
| **F20** | **Bed tube / phantom segmented and written as a mouse** (3 crops, 2 sessions) | **FIXED** — override + re-crop done; QA gate added; downstream re-derivation in progress | KO_09/10/11 and FDG WT_01/02 at Week 15; 8 mice have no Week 12 | `scripts/qa_embedding_outliers.py` |
| **F21** | F15's `_route()` fix doesn't survive the real GPT-augmented question bank (114/166 paraphrases misroute) | **FIXED** (`_route()` and `aggregate_seeds.py` now route on `content_type`, not question wording) | Nothing published yet — only the not-yet-used augmented VQA dataset | `unit/test_openai_augmentation.py` |
| **F22** | Seed sweep confirms the negative result; longitudinal arm's pooled genotype AUROC is significantly *below* chance | **CONFIRMED** — TBR negative in all 10 runs; below-chance longitudinal AUROC unexplained | Single-seed caveat on the leak-free re-run; opens a new below-chance anomaly | `aggregate_seeds.py` output |
| **F23** | **Early-vs-late (T2b) is substantially scanner drift**: air-only voxels separate W12/W20 at 0.76; WT-only mice at 1.000 | **CONFIRMED** | "reads disease progression" (README #1) | `docs/qa_2026-09-13/air_stats.py` |
| **F24** | **Forecast tokens exported at 2.7× the observed embedding norm** (cosine loss never constrained magnitude) | **FIXED** — export rescaled to training-fold norm; longitudinal VLM re-running | Every longitudinal-VLM number; explains long < base | — |

---

## F1 — Reported AUROCs are not distinguishable from noise · CONFIRMED

**The finding.** At n=32 the null distribution of AUROC has **sd = 0.105** and a 95%
interval of **[0.297, 0.703]** — with no signal present and no per-fold pathology
required. The pipeline compares against a bare "chance = 0.5" and reports no interval.

| Reported genotype AUROC | vs no-signal null |
|---|---|
| 0.762 — longitudinal 20ep (headline) | outside, p = 0.005 |
| 0.714 — longitudinal 100ep | outside, p = 0.018 |
| 0.560 — longitudinal 10ep | **inside the null interval** |
| 0.397 — TinyLlama 10ep | **inside the null interval** |
| 0.163 — ts0 baseline 10ep | outside on the *low* side, p = 0.0004 |

**The sharper version.** 0.762 was not a pre-registered test — `docs/experiments.md`
reports it as the best cell of a 2×5 grid, and README.md calls 20 epochs "the sweet
spot". Against the distribution of the **maximum over 10 configurations**:

    single pre-registered test : p = 0.0046
    max of a 10-config sweep   : p = 0.0460      <- marginal
    max-of-10 null median      : 0.656

And that is generous — it ignores F2 entirely, whose measured inflation (+0.14 to +0.20)
would place the headline comfortably inside the null.

**Two claims this undercuts.** "Longitudinal tokens are what carry the genotype signal"
and "20 epochs is the sweet spot." Neither survives multiplicity correction as stated.

**Separately: 0.163 is genuinely anomalous** (p = 0.0004 on the low side), so unlike the
others it is *not* noise. README.md reads it as "the head ranks genotype
anti-correlated, i.e. there is no stable single-scan genotype signal to learn" — but no
signal produces 0.5, not 0.163. A systematic inversion needs a mechanism; F2 and F3 are
the candidates. Unresolved.

**Fix.** Bootstrap CIs over subjects; permutation p-values; report the sweep as a sweep.

**FIXED (2026-09-12).** `vlm/data/eval.py` now reports `genotype_auc_ci95`,
`genotype_auc_perm_p`, `genotype_auc_null95`, and subject-bootstrapped CIs on the pooled
TBR MAE and r. `scripts/eval_stats.py` provides the same for the encoder probes, wired
into `evaluate_embeddings.py` T2b/T2c (fixed-score permutation by default; `--refit-perm`
refits the probe under each relabelling). The permutation respects group structure:
label-pure groups are permuted as blocks, mixed-label groups (early-vs-late) within-group.
`scripts/aggregate_seeds.py` gives mean ± SD over seeds plus a pooled, rank-normalised,
subject-bootstrapped AUROC. Validation: the new null on the 20-epoch predictions lands at
[0.298, 0.702] against this section's independently computed [0.297, 0.703].

---

## F2 — Checkpoints are selected on the held-out subject · CONFIRMED

**The wiring.** `run_fold` sets `inf_data_path` to the held-out subject.
`get_train_data` builds the Trainer's `eval_dataset` from that same key, and
`get_inf_data` builds the reported predictions from it too. `get_training_args`
hardcodes `load_best_model_at_end=True`, with `evaluation_strategy: epoch`.

Running the real trainer methods on a real fold, both return `['NaF_KO_01']`.

**Magnitude**, simulated on data containing no signal at all (the effect is a property
of the selection rule, not the model, so no LLM is involved):

| Epochs | Final-epoch AUROC | Best-on-held-out | Inflation |
|---|---|---|---|
| 10 | 0.443 | 0.585 | **+0.143** |
| 20 | 0.437 | 0.595 | **+0.158** |
| 50 | 0.420 | 0.601 | **+0.181** |
| 100 | 0.412 | 0.613 | **+0.201** |

Inflation grows with epoch count, so the 50/100-epoch rows of the sweep are worst hit.

**Scope.** Every VLM number: genotype 0.719/0.762, all TBR metrics, the entire sweep.
Fold *construction* is correct — `write_fold_jsons` splits cleanly (verified, passing
test). The defect is entirely the reuse of the val split as the selection set.

**Fix.** Carve a val subject out of the 31 training subjects, or set
`load_best_model_at_end=False`.

**PARTIALLY FIXED (2026-09-09).** Took the second route: `load_best_model_at_end` is no
longer hardcoded, it reads `train.load_best_model_at_end` and defaults to `False`, and the
canonical yaml sets it explicitly. `test_trainer_eval_set_is_not_the_held_out_subject` now
passes, and two guards pin the contract by calling `get_training_args()` both ways.

**Residual resolved (2026-09-12).** `run_mouse_vlm_loso.py --inner-val` carves one
training subject per fold (deterministic, rotating) into `inner.json`, removes it from
training, and passes it as `data.eval_data_path`; the trainer prefers that over
`inf_data_path` and prints a loud note when it has to fall back. The runner refuses to
start if the yaml has `load_best_model_at_end: true` without `--inner-val`. Two guards:
three-way disjointness, and the refusal. Without the flag, behaviour is unchanged
(eval set = test subject, harmless while selection is off, but `eval_loss` is a test
curve).

**A trap worth recording:** the retired guard `test_load_best_model_at_end_is_hardcoded`
grepped `get_training_args` source for the literal `load_best_model_at_end=True`. The
warning printed when the flag is enabled contains that same substring, so after the fix
the test passed by matching the warning text rather than the assignment — green for the
wrong reason. Its replacement calls the method instead of grepping it.

---

## F3 — The genotype head's decision threshold does not transfer · CONFIRMED

`_eos_hidden_state` pools at the last EOS, which during training is the **answer**-terminal
one. For genotype records the answer *is* the label ("will develop atherosclerosis").
At inference the answer is absent.

Two results, one of which corrects the original hypothesis:

* **The index arithmetic is CORRECT.** The `(img_tokens - 1)` offset lands on the final
  position in both modes. A passing test pins this so it is not "fixed" by mistake.
* **The head's input does depend on the answer** — changing only the answer tokens moves
  the pooled state by 0.53 max-abs with the image held fixed.

**The consequence is calibration, not ranking.** In the condition matching the real
dataset the head reached **AUROC 1.000 while classifying at accuracy 0.500** — at every
training length tested. `vlm/data/eval.py` classifies with `genotype_logit > 0.0`.

**Scope.** Every genotype *accuracy* in README.md and `docs/experiments.md`. An accuracy
of 0.219 on a balanced binary task is below the 0.5 floor of always guessing one class —
the signature of a shifted threshold. AUROCs are unaffected by this (but see F1).

**Fix.** Pool at the question-terminal EOS, or calibrate on training-fold logits, or
report AUROC only.

---

## F4 — Mouse identity is assigned by list index, not by quadrant · CONFIRMED

`segment_animals` skips quadrants with no bone or under 1 mL, returning a **compacted**
list whose entries still carry their true `position`. `stage3_crop_and_write` then pairs
bboxes with mouse numbers **by list index**, never consulting `position`.

Phantom CT, four mice expected, quadrant 2 fails to segment:

    mouse 1 -> quadrant 1   ok
    mouse 2 -> quadrant 3   WRONG
    mouse 3 -> quadrant 4   WRONG
    mouse 4 -> dropped entirely

**Why this is the most serious structural finding.** Mouse identity is the join key for
the whole project. A shift gives a scan the wrong subject ID; genotype is parsed *from*
the subject ID (`NaF_KO_07` → "KO"), and mouse numbers are persistent longitudinal
identifiers. So a shift also mislabels genotype and splices scans into another animal's
trajectory — corrupting the encoder evaluation and the VLM alike, upstream of everything.

The count-mismatch guard is a `print`, not a `raise`; the pipeline writes the mislabeled
files anyway. `docs/DATA_MANIFEST.md` flags several irregular sessions ("KO 9,11,12",
"KO 2,3", "WT 19,20 + WT 1,2") where the ordering assumption is load-bearing.

**Blast radius is measurable today.** `mouse_manifest.csv` already records
`crop_position` per row, so any session whose positions are not a contiguous 1..n run hit
this path — `test_no_session_lost_a_quadrant` reports them. **Run this first.**

**FIXED.** `stage3_crop_and_write` now raises a `RuntimeError` naming the session, the
quadrants actually found, and the mice the manifest expects, instead of cropping
`min(len(bboxes), len(mouse_nums))` animals under shifted identities. `mouse_nums` is
documented as scanner-position-ordered, so index pairing is correct *when every expected
quadrant is found* — the fix is to refuse when it is not, rather than guess.

**Still required:** re-run the DICOM stage. Any session that previously hit the compaction
path produced mislabeled crops that are still on disk. `test_no_session_lost_a_quadrant`
identifies them from `crop_position` in `mouse_manifest.csv` without reprocessing.

---

## F5 — Genotype is perfectly confounded with acquisition session · CONFIRMED (synthetic)

Every session is genotype-pure, the same mouse group is scanned together at every week,
and sessions differ in post-injection timepoint (1h vs 3h). Leave-one-subject-out does
not control for this: the held-out mouse's session-mates stay in training carrying the
same acquisition signature *and* the same label.

With acquisition structure planted and **zero** genotype signal:

| Fold grouping | Genotype AUROC |
|---|---|
| by subject (current protocol) | **0.754** (permutation p = 0.010) |
| by session | chance |

Calibrated to match the reported effect size (VLM 0.762, RAD-DINO T2c 0.869). The claim
is not that the real effect is this big — it is that an effect of the reported size is
fully reproducible from acquisition structure alone, so the protocol cannot tell them
apart. At higher imprint strength the synthetic confound saturates at AUROC 1.000.

**Wider than genotype.** Sessions are (group, week) pairs, so **week is confounded too**.
T2b's early-vs-late AUC = 1.000 could be scanner drift across the study period rather
than disease progression.

**Week half, resolved as far as this design allows (2026-09-13).** The mouse-group control
does *not* address it — week and acquisition date coincide for every mouse, so holding out
groups leaves the date signal intact. The decisive measurement is by genotype: under
mouse-group holdout, **WT-only** early-vs-late is 1.000 for RAD-DINO (M3D 0.956, Merlin
0.925, COLIPRI 0.878). Healthy controls have no disease to progress, so T2b measures age
and/or drift, and README/results.md have been corrected to stop calling it a progression
result. An air-region test (statistics of voxels containing no mouse) can separate growth
from drift; see `docs/qa_2026-09-13/`. The KO-vs-WT contrast at late weeks is unaffected —
both genotypes share acquisition dates.

**Measured on real data (2026-09-12).** No plumbing was needed: `(subject_id, week)` joins
to `session_id` in `mouse_manifest.csv` for 229/229 rows, and `evaluate_embeddings.py
--group-by {subject,session,mousegroup}` now does it post hoc.

**Session is the wrong unit; mouse-group is the right one.** A mouse-group is a connected
component of the mouse–session graph: mice that are ever scanned together, closed over all
their sessions. There are **10 on this dataset and every one is genotype-pure** (3 WT, 7
KO). Leave-one-*session*-out leaves the held-out mice in training at their other weeks and
their group-mates in training at every week, so it barely moves anything (T2c 0.920 →
0.915). Leave-one-*mouse-group*-out is the only split where no mouse and no session of the
held-out block is seen in training.

RAD-DINO, linear probe, genotype AUROC under leave-one-mouse-group-out, by week.
**Re-run 2026-09-13 (F18) with exact permutation enumeration**, superseding the
Monte-Carlo-approximated values this table originally reported — the differences are
all within Monte Carlo noise (largest shift: Week 12 p 0.51 → 0.488) and no conclusion
below changes. The number of groups actually present differs by week (not every
mouse-group has scans at every week), so the exact null size — and therefore the
p-value's resolution — is **not uniformly 1/120**; it is given per row:

| Scans | n | AUROC | 95% CI | perm p | exact null size | resolution |
|---|---|---|---|---|---|---|
| Week 12 (the VLM's input) | 72 | **0.498** | [0.35, 0.96] | 0.488 | 84 | 1/84 |
| Week 15 | 72 | 0.630 | [0.52, 0.73] | 0.083 | 36 | 1/36 |
| Week 18 | 38 | 0.707 | [0.49, 0.92] | 0.200 | 15 | 1/15 |
| Week 20 | 47 | 0.772 | [0.65, 0.88] | **0.036** | 28 | 1/28 |
| Weeks 18+20 | 85 | 0.797 | [0.62, 0.93] | **0.028** | 36 | 1/36 |
| All weeks | 229 | 0.823 | [0.78, 0.96] | **0.008** | 120 | 1/120 |

Week 18's floor is particularly coarse (1/15): with only 15 distinct label arrangements
possible, p = 0.200 means literally 3 of 15 arrangements scored at least as high as
observed — read it as "not distinguishable from chance at this design's resolution,"
not as a precise probability. Reproduction: `f18_rerun_f5_by_week.py`-equivalent calls
into `evaluate_embeddings.run_t2c_wt_vs_ko` per week filter, using the same
`join_mouse_group_ids` groups and the fixed `permutation_auc_p`.

For comparison, the standard leave-one-subject-out on all weeks gives 0.920 (0.869 before
per-fold standardisation was added to `_loso_logistic`) — the gap to 0.823 is group
recognition. **T2b early-vs-late is 1.000 under every grouping**, including mouse-group;
week is not confounded in a way that matters.

**Two consequences.**

1. **The leak-free VLM result is consistent with this, which makes it credible.** The VLM
   forecasts from a Week 12 scan, and a *linear probe* on that scan finds no genotype
   signal that generalises across mouse groups (0.498, though the CI is [0.35, 0.96] on
   10 groups). The VLM's 0.464 is what that predicts, and the only route to 0.762 was
   the injected label (F10). This is evidence the negative result is real, **not** a
   proof that no positive result exists: a linear probe is not the ceiling for a
   nonlinear model, the VLM has never been evaluated under mouse-group holdout, it
   trains on 32 mice where the probe saw 72 scans, and TBR forecasting is a different
   question from genotype classification. Open routes: mouse-group evaluation of the
   VLM, a seed sweep with `pool_at: question_eos`, and a multi-observed-week input.
2. **The encoder half has an honest positive finding.** RAD-DINO embeddings track disease
   progression: genotype separability is absent at baseline and rises monotonically to
   ~0.8 by Week 20, under the strictest available control. That — not "genotype from a
   single scan" — is what the data support. Caveats: 10 groups is a thin null (the
   mouse-group permutation null's 95% upper bound reaches 0.789, and some weeks resolve
   as coarsely as 1/15), and Week 18/20 n is small.

`test_session_confound_realdata` now runs (it used to skip) and fails by design:
0.756 → 0.232 on Week 12 (unstandardised probe; 0.498 standardised).

---

## F6 — Declared seeds are never consumed · CONFIRMED

`data_seed: 0` appears in every YAML and reaches nothing: not `transformers.set_seed`,
not `TrainingArguments(seed=)`. It is only checked for *existence* by
`_required_params`. `scripts/train_longitudinal.py` never calls `torch.manual_seed` at
all — and its LOSO predictions become the VLM's ts1/ts2/ts3 image tokens, so the VLM's
inputs change on every regeneration.

The VLM's projection layer and both multitask heads are built in `load_train_model()`
*before* `Trainer` exists, so HF's internal seeding would come too late regardless.

**FIXED.** `train_longitudinal.py` gained a `--seed` flag and seeds **per fold**
(`seed + fold_index`), so a fold reproduces regardless of how many ran before it; the
seed is recorded in the metrics output. `viz_emb_trainer.train()` calls
`transformers.set_seed(data_seed)` *before* `load_train_model()` constructs the
projection layer and heads, and threads `seed`/`data_seed` into `TrainingArguments`.

**Scope.** Combined with F1's null sd of 0.105, an unseeded single run at n=32 is not a
measurement. Every published number is one undocumented draw.

---

## F7 / F8 — Splicing and truncation · CONFIRMED (latent)

**F7.** `get_image_and_text_embeddings` locates `<image>` from `input_ids[0]` and applies
that offset to every row. On a heterogeneous batch, row 1's placeholder survives the
splice (verified) and real text tokens are consumed instead — silently. Currently masked
because `_add_prompt` emits a fixed-length prefix, but `beg_prompt` is config-settable in
every YAML.

**FIXED.** The splice now asserts its precondition: exactly one `<image>` per row, at the
same index in every row, else `ValueError` with the offending positions. A missing
placeholder is also caught rather than silently splicing at index 0.

**F8.** Records over `seq_length: 150` are truncated from the **right**, removing the
answer and its EOS. Such a record then pools at the question EOS (combined index 7)
while an intact record pools at the answer EOS (index 11) — two pooling semantics in one
batch, with no warning, no counter, no error.

**FIXED (visibility, not behaviour).** Truncation still happens — changing it would alter
training — but the first affected record now raises a `RuntimeWarning` and a per-dataset
counter tracks the rest. `test_no_real_record_exceeds_seq_length` measures whether this
bites in practice; still unrun (needs data).

---

## F9 — Axis swap in `tbr_strategy_3` · CONFIRMED

`tbr_strategy_3` indexes CT as `[x, y, z]` but unpacks `ys, xs = np.where(...)` in the
wrong order. Phantom with a spine at (40, 30): the code computes centroid **(29.5, 39.5)**
— exactly transposed. The 3 mm "anterior to spine" offset is then applied along the wrong
anatomical axis, and the bounds check compares an X-derived value against the Y extent.
End-to-end, a 50× PET hot spot planted at the correct aortic location yields
`tbr3_tbr = nan`.

**Why it matters despite TBR-3 being unused.** `docs/design_decisions.md` rejects TBR-3
as "anatomically fragile — NaN in >50% of slices" and uses that to justify TBR-2 as the
ground truth for the entire VLM TBR task. If the rejection rests on this bug rather than
on PET resolution, the choice of ground truth deserves revisiting.

**FIXED.** Unpacking corrected to `xs, ys = np.where(...)`. With the phantom's 50x hot
spot at the correct aortic location, `tbr3_tbr` goes from **`nan` to 19.97** and the
aortic mean from 5.9 to 17.96 — the ROI now lands where the docstring says it does.

**Re-extracted with the fix (2026-09-12), all three strategies, NaF cohort, 111 scans.**
TBR-3's NaN rate is **0.0%** (was reported as >50%) — the rejection rested entirely on the
bug. Head-to-head against TBR-2 on the script's own evaluations:

| | TBR-2 (`tbr2_p95_median`, current GT) | TBR-3 (`tbr3_tbr`) |
|---|---|---|
| E1 KO-vs-WT AUC, Week 12 | 0.456 | 0.484 |
| E1 KO-vs-WT AUC, Week 20 | 0.528 | 0.451 |
| E2 per-subject Spearman(week, TBR) | +0.108 | −0.250 |
| E4 LOSO embedding→TBR R² / ρ | 0.431 / 0.522 | 0.405 / 0.615 |

**The choice between them is a wash, and that is not the important finding.** Neither
proxy separates KO from WT at any week (E1 ≈ 0.5), and neither shows the rising
longitudinal trend the source paper reports for NaF TBR in KO mice (E2 ≈ 0 or negative).
The paper's manually-drawn aortic ROIs achieved r² = 0.83 against histology and clear
genotype separation; both programmatic proxies fail to reproduce that. `design_decisions.md`
already calls TBR-2 "a population-level proxy, too noisy at the individual level" — E1
says it does not work at the population level either.

**Consequence.** The VLM's TBR task was trained and scored against a label that does not
carry the biology it is named for. The negative TBR result in the leak-free re-run (R² < 0
in both arms) is therefore not just "the model can't forecast" — there may be nothing to
forecast. Any future TBR work needs a validated ground truth first (manual ROIs on a
subset, or a registered aortic segmentation), before another model is trained on it.

TBR-1 and TBR-2 both verified correct against hand computation, and TBR-2 proved
insensitive to crop extent (0.3% change on a 17% shorter crop).

---

## F10 — Genotype leaks via longitudinal MLP conditioning · CONFIRMED, dominant

`build_pairs` concatenates a genotype one-hot into the MLP input; its outputs become the
VLM's ts1/ts2/ts3 tokens. **Measured on the real embeddings (2026-09-09), this is not a
bounded side effect — it is the most likely explanation for the headline genotype result.**

LOSO linear probe for genotype:

| Source | AUROC |
|---|---|
| **ts1 — the predicted token fed to the VLM** | **0.869** |
| Observed Week 12 scan that ts1 was predicted *from*, same 32 subjects | 0.353 |
| Observed Week 12, all 36 NaF subjects | 0.422 |

The MLP consumes an embedding with no linearly recoverable genotype and emits one where
genotype is recoverable at 0.869 — higher than the VLM's own 0.762 headline. Its only
other input is the conditioning vector, and the held-out subject's *own* genotype bit is
an explicit input to the fold model that produces its token. This is direct label
injection, and it lines up with the progression README reads as the central finding:
ts0-only 0.163 → longitudinal 0.762.

**Why the earlier verdict was wrong.** The original probe used `cos(pred, pred_flipped)`
and inferred low influence from the ratio of the unit conditioning vector to `‖emb‖`
(measured: **13.97**, implying cos ≈ 0.999 by the table below). Cosine is the wrong
instrument here: RAD-DINO embeddings sit in a cone tight enough that *different mice*
score ≈0.98, so a cosine near 1.0 still leaves a linearly separable genotype direction.
A linear probe finds it immediately; the cosine cannot see it.

| mean ‖emb‖ | cos(pred, pred_flipped) |
|---|---|
| 28.0 | 0.99992 |
| 2.80 | 0.99052 |
| 0.56 | 0.92248 |

**FIXED (opt-in).** `train_longitudinal.py --no-geno-conditioning` drops only the genotype
one-hot, keeping the useful cohort+step conditioning (`--no-conditioning` drops all three).
A 2-epoch smoke run confirms the injection stops: ts1 genotype AUROC **0.869 → 0.206**.

**RE-RUN DONE (2026-09-09), and the headline does not survive.** Full-length
`--no-geno-conditioning --rollout`: ts1 genotype probe **0.869 → 0.369** (≈ the observed
Week 12 scan's 0.353), T4a cosine 0.9807 → 0.9816 — the bit bought nothing but the leak.
Then paired VLM LOSO at 20 epochs with `load_best_model_at_end=False`, one seed each:

| | Geno AUC | TBR R² |
|---|---|---|
| OLD longitudinal (F10 + F2 live) | **0.762** | 0.105 |
| NEW baseline | 0.377 | −0.400 |
| NEW longitudinal | **0.464** | −0.669 |

Both leak-free genotype AUROCs are **inside the n=32 no-signal interval** [0.297, 0.703];
the longitudinal−baseline gap is +0.087, inside noise; TBR R² is negative in both arms.
See `docs/experiments.md` § "Leak-free re-run". Remaining protocol caveats: one seed,
`pool_at=answer_eos` (F3 still live, so read AUROC not accuracy), 20 epochs inherited from
the leaky sweep.

---

## F11 — TBR round-trip · REFUTED

Hypothesised that the unsigned regex and the `-1` padding sentinel could corrupt TBR
supervision. **They do not.** Ten tests pass: generator → answer string → target tensor
round-trips losslessly, including the awkward W15+W20-only case (slot 1 correctly left
empty); both independent parsers agree; genotype records correctly produce all-sentinel
targets; missing weeks render `NA` without populating a slot.

The sentinel collision is real but latent — a negative TBR would be silently dropped.
TBR-2 is a ratio of positive quantities so it cannot occur; `test_real_tbr_values_are_positive`
pins that as a checked invariant rather than an assumption.

---

## F12 — Per-fold calibration offsets · REFUTED

Hypothesised that pooling logits from 32 differently-calibrated fold models would widen
the null and manufacture apparent signal. **It does not.** Offsets independent of the
label add symmetric noise, and AUROC is rank-based, so the null is unchanged across two
orders of magnitude (sd 0.106 → 0.103 as offset SD goes 0 → 4).

Pooling across fold models is not, by itself, the problem. F1 — plain sampling noise at
n=32 — is, and needs no pathology to explain it.

**Real-data addendum (2026-09-09).** `test_measure_per_fold_offsets_from_loso_predictions`
now runs and fails: on the 20-epoch predictions the **between-fold logit SD is 0.686
against a within-fold SD of 0.188** — 3.6×. So the offsets the synthetic study treated as
small are in fact large. The refutation's *logic* still holds (label-independent offsets
cannot bias a rank statistic), but its premise is no longer obviously satisfied, and
**under LOSO each fold predicts exactly one subject, so a per-fold calibration offset is
not separable from that subject's score.**

The obvious artifact mechanism was tested and **refuted**: leave-one-out changes the
training class balance (13KO/18WT when holding out a KO, 14KO/17WT for a WT), but that
predicts a prior shift of only −0.325 vs −0.194 — **the opposite sign and ~10× smaller**
than the observed KO−WT gap of +1.298. Observed `corr(held-out genotype, fold mean logit)`
is **+0.452, p = 0.0094**, i.e. in the direction a working classifier produces.

So this is not demonstrated to be an artifact. It stays an open caveat on pooling raw
logits across fold models, and it is confounded with F10 — a model reading an injected
genotype label would produce exactly this separation.

---

## F13 — ts2/ts3 are predicted from *observed* future scans · FIXED (opt-in)

The longitudinal MLP is only ever one-step-ahead, so the exported tokens are
`ts1 = f(observed W12)`, `ts2 = f(observed W15)`, `ts3 = f(observed W18)`. There is no
rollout from Week 12. `design_decisions.md` argued the exports are leak-free because the
MLP uses "only past information (T_k + conditioning)" — true relative to the MLP's own
step, but `T_k` for ts2/ts3 is the *future* relative to the Week 12 baseline README claims
the VLM forecasts from ("predict … from a single baseline scan embedding").

**Measured impact looks small.** Subject-ID retrieval from the predicted tokens is at
chance (ts2 0/15, ts3 1/9) and centred cosine to the input scan is only 0.10 — the tokens
are close to week-centroids with little per-mouse content. So this is primarily a claims
problem, and it touches only the 15 and 9 subjects that have those weeks.

**FIXED (opt-in).** `train_longitudinal.py --rollout` chains W12 → pred W15 → pred W18 →
pred W20, so no exported token is a function of a future scan. It also fills every
subject: 96 exported tokens (32 × 3) against 56 (32/15/9) for one-step, which removes the
zero-padding the VLM currently applies for missing ts2/ts3. LOSO evaluation metrics are
unchanged — the flag affects only the exported embeddings.

---

## F14 — TBR regression metrics double-count every subject · FIXED

Each subject emits both a `tbr` and a `combined` record carrying **identical**
`tbr_targets`, and `eval.py` iterated records directly — pooling **134 pairs from 67
independent observations**, with per-slot `n` reported as 64/30/40 against true subject
counts of 32/15/20.

Point estimates barely move (it is roughly averaging two predictions per subject), but the
effective n is halved, so any bootstrap CI or permutation test built on these rows is
about √2 too narrow. That directly undercuts F1's prescribed remedy.

**FIXED.** `eval.py` groups by `(slot, pid)` and averages the head's predictions so each
subject contributes exactly one observation per slot; it warns if two records disagree on
the target. The same collapse is applied to the text-TBR path. Re-scoring the existing
20-epoch predictions: n 134 → 67, slots 32/15/20, MAE 4.736 unchanged, Δ3wk r 0.376 → 0.383.

---

## F15 — Question routing and `content_type` · FIXED

Two small defects in how records are classified:

* `eval.py` matched `"TBR"` case-sensitively against the raw question but `"status"`
  against the lowercased one. It routes the current fixed templates correctly (verified
  32/32/32, 0 unrouted), but a record matching neither was dropped from every metric
  silently. That becomes live the moment the GPT-reworded question bank in
  `aug_config_mouse.yaml` is used.
* `create_mouse_traj_dataset.py` hardcoded `content_type: "trajectory"` on all three
  record types, so the field could not distinguish them and text matching was the only
  route available.

**FIXED.** Routing is now case-normalised in one `_route()` helper that returns `None`
rather than silently dropping, and unrouted records are counted, warned about, and
reported as `unrouted_records`. `content_type` is now `"genotype"` / `"tbr"` /
`"combined"`. Note the existing VQA JSONs on disk still carry the old constant value —
regenerating the dataset (pipeline stage 5) picks up the fix.

---

# Audit 2 (2026-09-13)

A second pass, done after F1–F15 were fixed and the green gate was passing (45/45).
Scope: re-verify the F1–F15 fixes actually match the code (not just the register),
run the full suite, and look specifically for what a correctness-and-leakage audit
does not cover — test coverage of the audit's *own* new statistical code,
reproducibility of the evidence trail, and framing issues that don't require a test
to state. Nothing below changes any reported number; these are gaps in the
scaffolding around the numbers.

**Verification done first.** Green gate: 45/45 pass. Probe suite: 9 failed / 3 passed
/ 1 skipped, and every failure is one already on the register (F1 sweep multiplicity,
F2 magnitude simulation, F3 head shortcut, F5 synthetic session confound) — no new
failures, no surprises. Diffs for the F2 (`run_mouse_vlm_loso.py`), F10/F13
(`train_longitudinal.py`), and F1/F14/F15 (`vlm/data/eval.py`) fixes were read
line-by-line against their FINDINGS.md descriptions and match.

## F16 — The statistics behind every CI and p-value are untested · CONFIRMED

`vlm/data/eval.py::_bootstrap_ci`/`_perm_p_auroc` and `scripts/eval_stats.py::
bootstrap_auc_ci`/`permutation_auc_p`/`_permute_labels`/`join_mouse_group_ids` are the
code that produced F1's fix — every `genotype_auc_ci95`, `genotype_auc_perm_p`, and
mouse-group AUROC now cited in README.md and results.md runs through one of these
four functions. None of them has a unit test.

`tests/unit/test_pooled_auroc_estimator.py` looks like it should cover this — it is
the test file F1's row in the summary table points to — but it only imports `_auroc`
and builds its own from-scratch Monte Carlo null; it never calls `_bootstrap_ci`,
`_perm_p_auroc`, `bootstrap_auc_ci`, `permutation_auc_p`, or `_permute_labels`.
Grepping the whole suite for these four names turns up zero test files.

This is exactly the gap `TESTING_PLAN.md`'s own "L1 Oracle equivalence" layer exists
to close (`_auroc` vs `sklearn.roc_auc_score`, etc.), applied to everything *except*
the newest, most citation-critical code. Concretely untested: whether
`_permute_labels`'s block-permutation for label-pure groups vs within-group
permutation for mixed groups is doing the right thing on a hand-checkable input;
whether `bootstrap_auc_ci`'s percentile interval has near-nominal coverage on a
synthetic case with a known true AUROC; whether `_bootstrap_ci` in `eval.py` and
`bootstrap_auc_ci` in `eval_stats.py` (two independent implementations of the same
idea) agree with each other on the same input.

**Consequence.** The negative result ("both leak-free arms sit inside the no-signal
interval") is exactly as trustworthy as the interval-computing code, and that code
has never been checked against a case where the right answer is known in advance.

**FIXED.** `tests/unit/test_audit_statistics.py` (8 tests): `_permute_labels`
hand-checked on a tiny label-pure input (asserts the group-level label multiset is
preserved and all C(3,2)=3 arrangements appear over repeated draws) and a tiny
mixed-group input (within-group composition preserved); `bootstrap_auc_ci` sanity
on a perfect separator (CI collapses near 1.0) and on pure noise (CI wide, brackets
0.5); and a direct cross-agreement check between `eval.py`'s `_bootstrap_ci`/
`_perm_p_auroc` and `eval_stats.py`'s `bootstrap_auc_ci`/`permutation_auc_p` on the
same synthetic input (different RNGs, same neighbourhood). The F18 exact-enumeration
path (below) is also checked against an independent brute-force oracle here.

## F17 — Cited audit evidence is not reproducible from a clean clone · CONFIRMED

`TESTING_PLAN.md §5.3` recommends committing `manifest.csv` "so the DICOM stage is
reproducible from a clean clone." It still is not: `.gitignore` line 223 ignores
`manifest.csv`, and it remains untracked (`git ls-files` returns nothing for it).
`mouse_manifest.csv` is written to `paths.output_dir` on `/data1` and was never in
the repo at all — README.md's repository-structure section lists it as if it lives
at the repo root, which is not where `build_nifti_dataset.write_mouse_manifest`
actually puts it.

Two artifacts that FINDINGS.md itself cites by name are also untracked: `aug_config_
mouse.yaml` (named in F15, "the GPT-reworded question bank in `aug_config_mouse.yaml`")
and `docs/tbr_strategy_comparison_2026-09-12.csv` (F9's re-extraction). Neither is in
`git ls-files`. A reviewer who clones the repo to check F9 or F15 cannot find the file
being referenced.

**FIXED.** `manifest.csv`, `aug_config_mouse.yaml`, and
`docs/tbr_strategy_comparison_2026-09-12.csv` are now tracked (`.gitignore`'s blanket
`manifest.csv` rule replaced with a comment explaining why it's deliberately tracked).
README's repo-structure entry for `mouse_manifest.csv` now says it's written to
`paths.output_dir`, not the repo root, and is not itself tracked.

## F18 — F5's mouse-group permutation null has only 120 distinct arrangements · CONFIRMED

F5's leave-one-mouse-group-out control has 10 groups, 3 genotype-pure WT and 7
genotype-pure KO. `_permute_labels` (`eval_stats.py`) correctly treats label-pure
groups as the exchangeable unit and permutes the 10 group-level labels — but a binary
vector of length 10 with a fixed 3/7 split has exactly C(10,3) = **120** distinct
label arrangements, not the 10,000 the code's default `n_perm` suggests. Monte Carlo
sampling 10,000 times from a 120-point space is a fine approximation of the *exact*
permutation distribution (it oversamples each outcome ~83×), but the exact p-value
itself is quantised in steps of 1/120 ≈ 0.0083.

Reported group-level p-values (0.007, 0.026, 0.031 in the F5 table) sit right at or
near that floor. This is not a bug — the reported values are correctly approximating
the true quantised null — but the precision implied by writing "p = 0.007" overstates
what a 10-group design can resolve: the true exact p is necessarily a multiple of
roughly 0.0083, so 0.007 most likely corresponds to 0 or 1 exceedances out of 120.

**Consequence.** None of F5's conclusions change. But the paper should state the
permutation floor explicitly (something like "p-values from the 10-group control are
quantised in steps of ≈1/120; treat p ≈ 0.007–0.03 as evidence at the resolution
this design allows, not as three finely distinguishable significance levels") rather
than let 3-significant-figure p-values imply more resolution than 10 groups provide.

**FIXED.** `permutation_auc_p`/`permutation_refit_p` (`eval_stats.py`) now enumerate
every arrangement exactly whenever the groups are label-pure and C(n_groups, k) is
below a threshold (default 20,000 / 2,000 for the refit variant — comfortably above
F5's 120), returning a `meta` dict (`method`, `n_arrangements`, `resolution`) instead
of silently sampling a space smaller than `n_perm`. `evaluate_embeddings.py`'s printed
line now says `exact enum, N arrangements, resolution 1/N` in place of the old
`n=10000` when this triggers. Verified against an independent brute-force oracle in
`unit/test_audit_statistics.py`.

**Follow-up done (2026-09-13).** Re-ran F5's mouse-group table (RAD-DINO, by week) and
the encoder table's "All weeks" T2c row for all four encoders, both with this code.
Every value moved by less than Monte Carlo noise (largest shift: Week 12 permutation p
0.51 → 0.488; "All weeks" RAD-DINO 0.822/p=0.007 → 0.823/p=0.008) — no conclusion
changes. One thing the re-run surfaced that the original fix description didn't
anticipate: **the exact null size is not uniformly 120.** Not every mouse-group has
scans at every week, so the per-week rows enumerate 84/36/15/28/36/120 arrangements
respectively — Week 18's floor is as coarse as 1/15. FINDINGS.md's F5 table, and
README/results.md's citations of these numbers, now report the exact values and,
where the null size differs from 120, the row-specific resolution.

## F19 — Backbone selection predates the confound control it is now read against · CONFIRMED (framing)

RAD-DINO was selected as the frozen VLM backbone (`design_decisions.md`, "Vision
Encoder Choice") on T2b (1.000) and the by-subject T2c (0.869, later 0.920 with
per-fold standardisation). The by-subject T2c was later shown inflated by mouse-group
confounding (F5) — the group-controlled T2c is 0.823, still RAD-DINO's number and
still the best of the four encoders, so the selection is corroborated after the fact.
But the sequence matters for how the paper tells it: the criterion that would
withstand review (group-controlled T2c) was not the criterion actually used at
selection time, and the encoder table's own footnote already flags that only
RAD-DINO survives the group control while the other three were selected against
(implicitly) by a metric that was later found to be measuring group recognition for
them too.

**Consequence.** None of the results change — RAD-DINO remains the right choice on
both the original and the corrected criterion. This is a paper-writing precision
issue, not a numerical one.

**FIXED.** `design_decisions.md`'s "Vision Encoder Choice" section now states plainly
that the by-subject T2c was the criterion used at selection time, and that it is
corroborated, not originally justified, by the group-controlled re-evaluation done
later (F5).

---

## F20 — Bed tube / phantom segmented and written as a mouse · CONFIRMED by image

**How it was found.** Rendering full session beds with the segmenter's quadrant divider
and the manifest's labels overlaid (`docs/qa_2026-09-13/`), after the F4 manifest check had
passed. `build_stage3.log` already contained the evidence:

    [!] bbox count (4) ≠ n_mice (3) for m54406      ... Proceeding with min(4, 3) crops.
    [!] bbox count (4) ≠ n_mice (2) for m54520      ... Proceeding with min(4, 2) crops.

**Mechanism.** `segment_animals` counts any quadrant with ≥ 1 mL of >300 HU voxels as a
mouse. The bed's curved fluid line and a rectangular phantom both clear that bar (1.6–2.6 M
bone voxels, comparable to a mouse). In a session with fewer than four mice they occupy the
"empty" quadrants, the count comes out at 4, and the pre-F4 code truncated to the manifest's
count by list index — writing the objects under real mouse IDs and discarding the real mice
at the end of the list. Bone extent does not discriminate (the curved tube's bbox is *larger*
than a mouse's).

**What was written, and what is actually there** (side-view bone projections in
`docs/qa_2026-09-13/dropped.png`):

| Session | pos 1 | pos 2 | pos 3 | pos 4 |
|---|---|---|---|---|
| m54406 (wk15 NaF, KO 9/10/11) | **tube** → written as KO_09 | mouse → written as KO_10 | mouse → written as KO_11 | **mouse — dropped** |
| m54520 (wk15 FDG, WT 1/2) | **tube** → written as WT_01 | **phantom** → written as WT_02 | **mouse — dropped** | **mouse — dropped** |

Taking `mouse_nums` as position-ordered over the *real* mice: m54406 → pos2 = KO_09,
pos3 = KO_10, pos4 = KO_11; m54520 → pos3 = WT_01, pos4 = WT_02. So `NaF_KO_10/wk15` and
`NaF_KO_11/wk15` on disk are really KO_09 and KO_10, KO_11's Week 15 scan was discarded,
and both FDG_WT crops are junk with the real mice discarded.

**Confirmed by embedding.** RAD-DINO cosine-to-centroid: median 0.989 (MAD 0.005). The three
junk crops sit at 0.78–0.86, z = −25 to −40, with a clean gap to the next crop at z = −5.6.
M3D independently puts FDG_WT_02/wk15 at cosine **−0.09**. `scripts/qa_embedding_outliers.py`
is the resulting gate — run on RAD-DINO's `.npz` (COLIPRI's anisotropy makes its robust-z
unusable; Merlin is insensitive).

**Downstream.** Genotype labels are unaffected (session-level, sessions genotype-pure).
Subject identity at Week 15 is wrong for five mice. `NaF_KO_09` is one of the 32 VLM
subjects: its Week 15 embedding, its Week 15 TBR (extracted from a tube's PET), and its
W12→W15 MLP pair are all garbage. The FDG crops affect only the encoder evaluations (3 of
229 scans; none of them in the T2b W12-vs-W20 comparison).

**Separately, two sessions failed segmentation entirely** (m54232: KO 5–8 wk12; m54267:
FDG WT 17–20 wk12 — 6 to 32 bone voxels in the whole volume), so **eight mice have no
Week 12 scan**. That is why the VLM cohort is 32 and not 36. m54232's manifest note says
the XIF uses range notation `m54232-m54250`; the real DICOM may be filed elsewhere and the
four KO mice may be recoverable.

**Interaction with the running seed sweep.** The 4-seed leak-free VLM sweep launched
2026-09-13 uses KO_09's Week-15 entries as-is. Under LOSO, KO_09's own predictions come from
folds that never saw it, so the junk affects only (a) its TBR Δ3wk target (1 of 32) and (b)
one of 56 MLP training pairs in the other folds' token generation. Genotype AUROC is
essentially untouched. The sweep is worth finishing; its TBR slot-0 numbers carry this caveat.

**Identity of the *real* mice — what was checked (2026-09-13).** Two questions are
distinct here. (A) Is each crop a mouse and does the count match — the failure above,
verifiable from images, now bounded. (B) Among real mice, is each labelled with the right
ID. The pipeline pairs quadrants with `mouse_nums` in list order, and the XIF lists mice in
ascending numerical order in 146 of 147 sessions, so the labeling *is* the assumption
"the lab places the lowest-numbered mouse in position 1, and so on." Checks made:

* *Coordinate frame.* All 63 session volumes carry identical `ImageOrientationPatient` /
  `ImagePositionPatient`; the quadrant convention is applied in one fixed frame. No
  per-session flip.
* *Position consistency across weeks.* Of 151 consecutive-week transitions, 74 are
  regroupings (different companions — the lab re-cohorted between weeks 15 and 18), 75
  keep the same companions in the same positions, and 2 are WT_01/02 leaving the
  mixed-timepoint session m54225 (explained). Note this consistency is tautological with
  the ascending XIF listing and does *not* verify physical placement. The one
  non-ascending XIF (`19;20;1;2`) is consistent with listing order = bed order.
* *Image-based identity.* Rigid registration of Week 12 → Week 15 bone masks, whole
  skeleton and skull-only, on the WT_03–06 group: same-mouse Dice is indistinguishable
  from different-mouse Dice (0.14 vs 0.21; 0.22 vs 0.23). Method failure — pose changes
  defeat whole-body registration and the skull extraction is unreliable — not evidence
  either way. A proper attempt needs validated skull segmentation, MI/surface
  registration and a positive control. Scripts in `docs/qa_2026-09-13/`.

**Status of (B): unverified, and unverifiable from CT with a quick method.** The
definitive check is the lab's placement protocol / records. Exposure is bounded: genotype
labels are session-level and sessions are genotype-pure, so (B) errors can only scramble
subject identity within a genotype — affecting T3b, the MLP's longitudinal pairs, and the
VLM's per-mouse TBR trajectories, not any genotype claim.

**RE-CROPPED (2026-09-13).** `quadrant_overrides.yaml` (repo root, tracked) declares the
position → mouse map per affected session; `stage3_crop_and_write` applies it in place of
list-order pairing and refuses unless every found quadrant is listed and every listed mouse
is assigned exactly once. Stage 3 re-run for m54406 and m54520:

    m54406: pos1->skip, pos2->9, pos3->10, pos4->11
    m54520: pos1->skip, pos2->skip, pos3->1, pos4->2

All five re-cropped Week 15 entries are skeletons on inspection (`docs/qa_2026-09-13/
verify_recrop.png`); KO_11's Week 15 is recovered. Pre-fix crops, manifest, embeddings,
TBR and VQA JSONs are in `/data1/Processed_NIfTI_Test/_pre_f20_backup/`.

**Downstream re-derived (2026-09-13).** All four encoders re-extracted; a diff of the new
RAD-DINO `.npz` against the pre-fix one shows **exactly the five re-cropped entries changed
and nothing else**. The QA gate now flags **0** crops. TBR re-extracted: the three KO
Week-15 values shifted by exactly one slot, confirming the diagnosis — old "KO_10" 36.08 →
KO_09, old "KO_11" 31.35 → KO_10, KO_11 gets 25.70, and the tube's 7.55 leaves the training
targets. Leak-free MLP re-run into `longitudinal_nogeno_f20/` (T4a 0.9816 → 0.9850 with the
garbage pair gone). VQA JSONs regenerated (post-F21 schema). VLM arms relaunched on the
corrected data (`*_ep20_f20`), and the encoder tables are being recomputed
(`eval_audit_2026-09-13_f20/`). The 4-seed sweep completed the same day by the other
session was trained on pre-fix data; its genotype numbers are essentially unaffected, its
TBR slot-0 numbers include the tube target for KO_09.

**Fix (original plan, for the record).** (1) A per-session quadrant override in the manifest so stage 1 can be
told "pos1 is not a mouse; pos2/3/4 are 9/10/11" and re-cropped for these two sessions
with the F4 refusal in place; (2) re-derive embeddings, TBR, MLP pairs and VQA records
for the five affected (mouse, week) entries; (3) run `qa_embedding_outliers.py` as a
gate after every stage-2 extraction; (4) locate the m54232 / m54267 DICOM.

---

## F21 — F15's question-routing fix does not survive the actual GPT-augmented question bank · CONFIRMED, FIXED

**Context.** `crump_aug_dataset.csv` (the OpenAI paraphrase bank generated per
`aug_config_mouse.yaml`, ~60-65 reworded Q/A pairs per one of the three fixed templates
— genotype / tbr / combined) has now been imported to build the augmented VQA dataset
(`scripts/dataset/create_mouse_vqa_dataset_from_openai.py`, adapted from Arvind's BraTS
script of the same shape). F15 claimed `_route()` in `vlm/data/eval.py` was already
case-normalised and safe for "the GPT-reworded question bank." That claim was verified
only against the fixed templates, not the actual delivered CSV.

**How it was found.** Parsing all 166 real paraphrases (58 genotype / 64 tbr / 44
combined) and running each through the old `_route()`: **114/166 (69%) misroute.** The
combined-question pool is worst — 36/44 use a synonym for "status" ("outcome", "result",
"condition", "state", "classification", ...) with no literal "status" substring, so
`has_geno` is `False` and they're misclassified as pure `"tbr"` (silently losing the
genotype half of that record's signal — `is_geno` never becomes `True` for it). 6/58
genotype-only paraphrases use "categorized"/"classified"/"classification" instead of
"status" with no "tbr" substring either, so `_route()` returns `None` and the record is
dropped from every metric, not just miscounted.

**Mechanism.** `_route()` reasoned from question *wording*, which is exactly the axis the
augmentation randomizes. Word-level matching cannot be made robust to an open-ended
paraphrase bank — any synonym list is a losing game against future re-generation.

**Fix.** `create_mouse_traj_dataset.py`'s records already carry a ground-truth
`content_type` field (`"genotype"`/`"tbr"`/`"combined"`, added by F15) that survives
paraphrasing untouched — `create_mouse_vqa_dataset_from_openai.py` only rewrites
`question`/`answer` text, and `viz_emb_trainer.py` already threads `content_type`
through to prediction records unchanged. `_route()` now takes `content_type` and uses it
first, falling back to the old text heuristic only for prediction files that predate the
field. `scripts/aggregate_seeds.py`'s independent genotype-only filter (a second,
uncoordinated copy of the same "status"/"tbr" substring check) got the equivalent fix.
Verified: all 166 real paraphrases now route correctly
(`tests/unit/test_openai_augmentation.py::test_route_uses_content_type_and_survives_every_real_paraphrase`);
TBR numeric targets are unchanged by paraphrasing since `{trajectory}` is always
substituted as an intact `"Week N: value, ..."` span
(`test_tbr_targets_identical_before_and_after_augmentation`).

**Scope.** Only matters once the augmented VQA JSONs (`*_vqa_traj_aug.json`) are actually
used for training or evaluation — nothing currently reported in `README.md` /
`docs/results.md` used reworded questions, so no existing number is affected.

---

## F22 — Seed sweep confirms the negative result; longitudinal arm's pooled AUROC is significantly below chance · CONFIRMED

> **Explained (2026-09-13/14) — see F24 and the seed-spread note below.** Two things
> resolve the "below chance, unexplained" reading. (1) *Mechanism:* the forecast tokens
> were exported at 2.7× the norm of the observed token (F24), so the longitudinal arm's
> LLM saw three loud uninformative tokens beside one quiet informative one — a concrete
> reason for long < base. (2) *Statistics:* the pooled CI [0.29, 0.45] rank-pools five
> seeds and bootstraps subjects, which treats the seeds as agreeing; they don't (per-seed
> longitudinal AUROC 0.11 / 0.21 / 0.46 / 0.53 / 0.63). The honest interval from the seed
> spread is 0.39 ± 0.22 → SE ≈ 0.10 → roughly [0.2, 0.6], and the refit null for pooled
> LOSO AUROC sits near 0.42, not 0.5 (F2's own simulation). Not distinguishable from
> baseline or from chance. The longitudinal arm is re-running on rescaled tokens with and
> without `pool_at: question_eos` (F3); results will replace this section's numbers.

**The question.** F1 flagged every VLM number as a single unseeded draw against a null
with sd ≈ 0.105 at n=32. The leak-free re-run (F2/F10 fixed) reported one seed each for
baseline and longitudinal — not enough to know whether 0.377/0.464 were representative.

**What was run (2026-09-13).** `data_seed` 0–4 (seed 0 = the existing re-run), both
leak-free arms (`viz_emb_params_mouse_nogeno_{base,long}[_seed{1..4}].yml`), 20 epochs,
`load_best_model_at_end=False`, no other protocol change. 8 new 32-fold LOSO runs
(~4.85 h each) in tmux, GPUs 1/6 in parallel. `aggregate_seeds.py --rescore` over all 5
seeds per arm.

**Per-seed genotype AUROC:**

| Arm | seed0 | seed1 | seed2 | seed3 | seed4 | mean ± SD | pooled² | 95% CI | perm p |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (ts0) | 0.377 | 0.595 | 0.687 | 0.313 | 0.548 | 0.504 ± 0.155 | 0.536 | [0.444, 0.627] | 0.250 |
| Longitudinal (4-token) | 0.464 | 0.111 | 0.532 | 0.627 | 0.206 | 0.388 ± 0.220 | 0.369 | [0.292, 0.451] | **0.997**¹ |

¹ One-sided p = P(null AUROC ≥ observed); 0.997 means the observed AUROC is *lower* than
99.7% of label-permuted draws — the low-tail significance is 1 − 0.997 = 0.003.
² Pooled by rank-normalising each seed's logits within-seed (avoids the F12
cross-model-calibration concern) then subject-bootstrapping across all 5×32 rows.

**TBR R², all 5 seeds, both arms:** baseline −0.400 / −0.152 / −0.202 / −0.100 / −0.232
(mean −0.217 ± 0.114); longitudinal −0.669 / −0.047 / −0.165 / −0.251 / −0.215
(mean −0.269 ± 0.236). **Negative in all 10 runs.**

**Reading.**

1. **The negative result is now robust, not a one-seed accident.** Baseline's pooled
   AUROC (0.536) sits comfortably at chance, matching F1's null. TBR R² is negative in
   every one of 10 runs across both arms — this closes F1's single-seed caveat
   definitively for TBR.
2. **New: the longitudinal arm's pooled AUROC is significantly *below* chance** (0.369,
   95% CI excludes 0.5, low-tail p ≈ 0.003), not merely "inside the null." This echoes
   F1's earlier flagged anomaly (the 0.163 ts0-baseline-10ep result — "no signal produces
   0.5, not 0.163 ... a systematic inversion needs a mechanism; F2 and F3 are the
   candidates. Unresolved.") — except this time it is the *longitudinal* arm, it
   *survives* both major leak fixes (F2, F10), and it is consistent across 5 independent
   seeds, so F2/F10 cannot be the explanation this time. F3 (the multitask head's
   train/inference pooling mismatch) is the live candidate: the longitudinal arm's extra
   image tokens shift the `(img_tokens - 1)` EOS-pooling offset used everywhere
   downstream, which is exactly F3's mechanism.
3. **No reported conclusion changes.** "No positive genotype or TBR signal survives the
   leak-free protocol" still holds — a below-chance AUROC is not a positive result — but
   "below chance" and "no signal" are different failure modes, and only one is currently
   explained.

**Not yet done.** Diagnose the below-chance longitudinal signal: re-run under
`pool_at: question_eos` (F3's still-unrun ablation); check whether per-seed AUROC
correlates with anything about that seed's random LoRA/head initialisation; run a
shuffled-label control through the same longitudinal architecture to see if the
below-chance pattern is a property of the architecture itself rather than of genotype
specifically.

---

## F23 — Early-vs-late separation is substantially scanner drift · CONFIRMED

README's result #1 read T2b (Week 12 vs Week 20, AUC 1.000) as "reads disease progression."
Two measurements (2026-09-13) say otherwise:

* **WT-only mice separate at 1.000** under mouse-group holdout (M3D 0.956, Merlin 0.925,
  COLIPRI 0.878). Healthy controls have no disease to progress.
* **Air-only voxels separate at 0.759** — corners of each crop containing no mouse, no
  holder. Air mean HU: −1053 ± 13 (W12) → −1040 ± 5 (W20). The scanner's calibration
  changed between acquisition periods. Body-size features alone give 0.663 (bone volume
  +10%, modest growth); full body HU statistics 0.908.

Week and acquisition date coincide for every mouse, so no CV grouping can separate them;
the air test is the only handle. Conclusion: T2b is an acquisition-period detector with a
growth component. **It is not a disease result and should not be cited as one.** The
KO-vs-WT contrast at late weeks is unaffected — both genotypes share acquisition dates —
and remains the one encoder disease result. The MLP's learned "trajectory" also includes
the drift, which is harmless to the negative VLM result but should be noted.

---

## F24 — Forecast tokens exported at 2.7× the observed embedding norm · FIXED

**Found** during a plumbing audit of the longitudinal VLM path (2026-09-13), prompted by the
leak-free longitudinal arm scoring *below* the baseline arm across seeds. Tracing the actual
tensors handed to the model:

| token | norm |
|---|---|
| ts0 — observed RAD-DINO embedding | 13.86 ± 0.17 |
| ts1 / ts2 / ts3 — MLP forecasts | 37.3 – 37.9 |

`train_longitudinal.py` trains with a **cosine loss**, which is scale-invariant, so output
magnitude is unconstrained and drifted to ~37. Every MLP evaluation (T4a cosine, T4b
retrieval) is also cosine-based, so nothing detected it. The VLM's single shared
`Linear(768→4096)` then projected forecast tokens into the LLM at 2.7× the magnitude of the
observed token: three loud, near-identical, uninformative tokens beside one quiet one. The
old leaky exports had the same norm (37.05), so this has degraded the longitudinal arm from
the start — consistent with the old longitudinal TBR never beating baseline even with the
label leaked. Also observed: the rollout converges to a fixed point (step norms 23.7 → 3.3 →
1.2; cos(ts1, ts2) = 0.995), so ts2/ts3 add almost nothing beyond ts1.

Everything else on the path checked clean: all 96 forecast files present, no NaN, no
silent zero-padding; `MouseTrajDataset` hands the model exactly the file contents
(row-by-row equality verified); `genotype_label` matches every pid; `_auroc` agrees with
sklearn to 2×10⁻¹⁶ over 500 tied cases.

**FIXED.** `--export-scale observed` (default) rescales each exported vector to the mean norm
of the *training fold's* real target embeddings (LOSO-clean). Exported norms are now 13.97;
LOSO metrics are unchanged by construction. `--export-scale raw` reproduces old exports.
Longitudinal VLM arm relaunched on the rescaled tokens (`mouse_vlm_nogeno_long_ep20_f22`).
Open: F3 (answer-EOS pooling) remains the other structural suspect for seed instability.

---

## Recommended order

**Fixed so far** (defect repair only; no change to scientific protocol): F4, F6, F7, F8,
F9, F14, F15, F16, F17, F18 (code), F19, F21, plus the import-time
`torch.cuda.set_device(0)` that made the LOSO entry point unimportable off-GPU.

**Fixed as opt-in flags** (available, but changing the protocol is the owner's call):
F10 (`--no-geno-conditioning`), F13 (`--rollout`), F2 (`train.load_best_model_at_end`,
now defaulting to `false`), F3 (`train.pool_at`, defaulting to the original
`answer_eos`).

**F4 blast radius: my first check was wrong — see F20.** The contiguity check on
`crop_position` can only detect a *skipped* quadrant. The failure that actually occurred
is the opposite — a non-mouse object filling an *empty* quadrant — which leaves positions
contiguous. `build_stage3.log` had flagged it all along (`bbox count (4) ≠ n_mice`). Two
sessions are corrupted and two more failed outright; the DICOM stage needs a targeted
re-run for those.

**Open, in priority order (2026-09-13):**

1. ~~Encoder table under mouse-group~~ **Done 2026-09-12.** Only RAD-DINO retains genotype
   signal (0.823, p=0.009); COLIPRI 0.473, Merlin 0.458, M3D 0.541 are at chance. T2b
   survives in all four. Artifacts in `embeddings/eval_audit_2026-09-12/`; TBR re-extraction in `tbr_features_audit_2026-09-12/`.
2. ~~Seed sweep~~ **Done 2026-09-13 (F22).** 5 seeds/arm aggregated: baseline pooled AUROC
   0.536 [0.444, 0.627] (chance, as expected), longitudinal pooled AUROC **0.369
   [0.292, 0.451] — significantly below chance**, unexplained (F3 the live candidate).
   TBR R² negative in all 10 runs. The single-seed caveat on the leak-free re-run is
   closed; a new below-chance anomaly is open (see F22's "Not yet done").
3. ~~F16~~ **Done 2026-09-13.** `unit/test_audit_statistics.py`, 8 tests.
4. ~~F17~~ **Done 2026-09-13.** `manifest.csv`, `aug_config_mouse.yaml`,
   `docs/tbr_strategy_comparison_2026-09-12.csv` tracked; README fixed.
5. ~~F18~~ **Done 2026-09-13.** Exact enumeration in `eval_stats.py`; F5's table and the
   encoder table's mouse-group row re-run with it (numbers moved within Monte Carlo
   noise; the exact null size varies by week, 15–120, not uniformly 120).
6. ~~F19~~ **Done 2026-09-13.** Sequencing note added to `design_decisions.md`.
7. ~~Rewrite README Key Takeaways.~~ **Done 2026-09-12.**
8. ~~F3 untested~~ **Running (2026-09-14)** as `mouse_vlm_nogeno_long_ep20_f22_qeos` — on
   rescaled, crop-fixed tokens, so it is a clean single-variable test against
   `mouse_vlm_nogeno_long_ep20_f22` (same tokens, `answer_eos`). Note the "below-chance
   anomaly" it was meant to explain is now accounted for by F24 + seed spread; F3's
   remaining question is whether matched pooling reduces seed instability.
9. **TBR ground truth** — F9's re-run shows neither TBR-2 nor TBR-3 separates KO from WT
   or trends with time, and F22 shows the negative TBR result is robust across seeds, not
   a fluke. Before any further TBR modelling: validate a proxy against manual ROIs on a
   subset, or replace it.
10. ~~F20 re-crop~~ **Done 2026-09-13.** Override + re-crop + full downstream re-derivation
    (embeddings ×4, TBR, MLP, VQA). Still to chase: the m54232 / m54267 DICOM (eight mice
    with no Week 12; four are KO in the VLM cohort).
12. ~~F24 export scale~~ **Done 2026-09-13.** `--export-scale observed` default;
    `longitudinal_nogeno_f22/` exported at norm 13.97.
13. **In flight (2026-09-14), all on corrected crops:** `mouse_vlm_nogeno_base_ep20_f20`
    (baseline), `mouse_vlm_nogeno_long_ep20_f22` (longitudinal, scaled tokens),
    `mouse_vlm_nogeno_long_ep20_f22_qeos` (+ F3). Until they land, every VLM number in the
    docs is pre-F20/F24 and is marked as such.
14. **Air-drift follow-up (F23).** The scanner's calibration shifted between acquisition
    periods. Anything that compares across weeks — T2a/T2b, the MLP's learned trajectory —
    inherits it. Worth a calibration phantom or an intensity-harmonisation step before any
    cross-week claim is made.
11. **`scripts/evaluate_encoder_vs_vlm.py` is unaudited.** It carries its own copy of the
    linear probe (unstandardised, `LogisticRegression(C=1.0)` at line 222) and its own
    metric code. The A2 and TBR-probe rows it produced are withdrawn from `results.md`
    until it is rebuilt on `eval_stats.py` / `_loso_logistic` and its numbers agree with
    the audited evaluator. Its A1 (0.353) coincides with the audited probe's value but
    that is not provenance.
12. **F22 diagnosis** — re-run the longitudinal arm under `pool_at: question_eos`; check
    per-seed AUROC against LoRA/head init; shuffled-label control through the same
    architecture.

**Reproduction status (2026-09-12).** Every metric now cited in `README.md`,
`docs/results.md`, and `docs/experiments.md` was produced by the audited code from
artifacts on `/data1`: encoder T1–T3 (`embeddings/eval_audit_2026-09-12/`), longitudinal
MLP (`embeddings/longitudinal_nogeno/`), VLM (`vlm/runs/mouse_vlm_nogeno_{base,long}_ep20/`,
re-scored), TBR strategies (`tbr_features_audit_2026-09-12/`). T1 was reproduced for
RAD-DINO (ARI −0.0102 vs documented −0.010); the other encoders' T1 values are untouched
by any audit change and were not re-run. Pre-audit values are in git history and, where
kept for comparison, are italicised or labelled "leaky, archived".
