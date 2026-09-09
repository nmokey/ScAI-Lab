# Findings Register

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
a passing regression guard. Fixes so far are confined to defect repair that does not
change scientific protocol; F2 and F3 are deliberately left open because their fixes
change how models are trained/scored and that is a call for the project owner.

Last updated 2026-08-29.

---

## Summary

| ID | Finding | Status | Invalidates | Test |
|---|---|---|---|---|
| **F1** | Reported AUROCs not distinguishable from noise at n=32 | CONFIRMED | Every genotype claim | `unit/test_pooled_auroc_estimator.py` |
| **F2** | Checkpoints selected on the held-out subject | CONFIRMED | Every VLM number | `control/test_loso_selection_leakage.py` |
| **F3** | Genotype head threshold does not transfer to inference | CONFIRMED | Every genotype *accuracy* | `control/test_head_shortcut.py` |
| **F4** | Mouse identity assigned by list index, not quadrant | **FIXED** | Potentially all results, upstream | `pipeline/test_mouse_identity.py` |
| **F5** | Genotype perfectly confounded with acquisition session | CONFIRMED (synthetic) | Genotype + week claims | `control/test_session_confound.py` |
| **F6** | `data_seed` declared everywhere, consumed nowhere | **FIXED** | Reproducibility of all runs | `unit/test_determinism.py` |
| **F7** | `<image>` position taken from row 0, applied batch-wide | **FIXED** | Latent; nothing today | `torch/test_splicing_and_truncation.py` |
| **F8** | Over-length records truncated silently, changing pooling | **FIXED** | Unknown until measured | `torch/test_splicing_and_truncation.py` |
| **F9** | Axis swap in `tbr_strategy_3` spine localisation | **FIXED** | A design justification | `pipeline/test_tbr_strategies.py` |
| **F10** | Genotype leaks via longitudinal MLP conditioning | BLOCKED | Unknown; bounded | `control/test_longitudinal_leakage.py` |
| **F11** | TBR round-trip lossless; `-1` sentinel safe | **REFUTED** | — | `unit/test_tbr_parsing.py` |
| **F12** | Per-fold calibration offsets distort pooled AUROC | **REFUTED** | — | `unit/test_pooled_auroc_estimator.py` |

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

**Blocked on plumbing — but less than expected.** `mouse_manifest.csv` *already* records
`session_id` (`_build_manifest_row`), and the encoder scripts already read the full row
dict — they just don't propagate it. This is a few lines per encoder script plus a
`--group-by` flag on `evaluate_embeddings.py`, not a pipeline rebuild.

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

**Open question this reopens:** TBR-3's rejection in `design_decisions.md` rested on a
>50% slice failure rate that was measured with the axis swap in place. TBR-3 should be
re-run before TBR-2 is accepted as the ground truth for the VLM's entire TBR task.

TBR-1 and TBR-2 both verified correct against hand computation, and TBR-2 proved
insensitive to crop extent (0.3% change on a 17% shorter crop).

---

## F10 — Genotype leaks via longitudinal MLP conditioning · BLOCKED

`build_pairs` concatenates a genotype one-hot into the MLP input; its outputs become the
VLM's ts1/ts2/ts3 tokens. The mechanism is real, but the magnitude is **not automatic**,
and the original write-up overstated it:

* When genotype is uninformative the MLP simply ignores the bit —
  `cos(pred, pred_flipped) = 0.99992`. An unused input does not leak.
* Influence is governed by the ratio of the unit conditioning vector to `‖embedding‖`:

| mean ‖emb‖ | cos(pred, pred_flipped) |
|---|---|
| 28.0 | 0.99992 |
| 2.80 | 0.99052 |
| 0.56 | 0.92248 |

So this reduces to one unmeasured quantity: the real RAD-DINO embedding norm.
`test_genotype_bit_influence_realdata` measures it in one command. Cheap insurance
regardless: re-run `train_longitudinal.py --no-conditioning` and compare.

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

---

## Recommended order

**Fixed so far** (defect repair only; no change to scientific protocol): F4, F6, F7, F8,
F9, plus the import-time `torch.cuda.set_device(0)` that made the LOSO entry point
unimportable off-GPU. Each has a passing regression guard.

**Open, in priority order:**

1. **F4 blast radius** — one query against `mouse_manifest.csv`
   (`pytest -m realdata -k lost_a_quadrant`). The fix stops *new* mislabeling; it does
   not repair crops already on disk. If sessions lost quadrants, everything downstream
   needs re-derivation, so this gates the rest.
2. **F2** — needs your decision: carve a val subject out of the 31, or
   `load_best_model_at_end=False` with a fixed epoch count. At n=32 a single val subject
   is a noisy selection signal, so the second is likely more defensible.
3. **F3** — needs your decision: pool at the question-terminal EOS, calibrate the
   threshold on training-fold logits, or report AUROC only.
4. **F1** — bootstrap CIs and permutation nulls on whatever the re-runs produce.
   Non-negotiable at n=32, and it determines whether any claim survives.
5. **F5 plumbing** — propagate `session_id` from `mouse_manifest.csv` (where it already
   exists) into the `.npz`, then re-run grouped by session. The one that decides whether
   the genotype result is biology.
6. **F10 measurement** — `pytest -m realdata -k genotype_bit_influence`.
7. **F9 follow-up** — re-run TBR-3 now that its ROI works, and revisit the TBR-2 choice.
