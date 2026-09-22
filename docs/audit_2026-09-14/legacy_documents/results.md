# Encoder Evaluation Results

> [!NOTE]
> **Audit status (2026-09-12).** The results in this file are post-audit. Two defects
> inflated every VLM number reported before 2026-09-09: the genotype label leaked into the
> longitudinal tokens (F10; probe AUROC 0.869 vs 0.353 from the observed scan), and
> checkpoints were selected on the held-out subject (F2; +0.14–0.20 AUROC on noise). The
> leak-free re-run gives genotype AUROC 0.464 and TBR R² < 0. Separately, all 10
> mouse-groups in the study are genotype-pure (F5): under leave-one-mouse-group-out,
> Week-12 genotype is chance (0.49) and rises to 0.82 across all weeks. Every finding is
> backed by a test in `../tests/`; register and evidence in [FINDINGS.md](FINDINGS.md).

> Encoder caveats, now measured. **Genotype** is confounded with mouse-group (all 10
> groups are genotype-pure); by-subject genotype numbers are inflated by group recognition,
> and only the leave-one-mouse-group-out column is citable. **Week** is confounded with
> acquisition date for every mouse, and no grouping can remove that: T2b early-vs-late
> stays at 1.000 under every control *including for WT-only mice* (2026-09-13), so it
> measures age and/or scanner drift, not disease. It should not be cited as a disease
> result. The KO-vs-WT contrast at late weeks is the one encoder result protected from
> date confounding.


Zero-shot evaluation of pretrained vision encoders on the mouse atherosclerosis CT dataset (229 scans, 78 mice, 4 timepoints). No fine-tuning. All supervised tasks use Leave-One-Subject-Out (LOSO) cross-validation.

**Dataset:** 80 mice (NaF cohort + FDG cohort, WT + KO), imaged at Weeks 12, 15, 18, 20. Only CT-Hi modality evaluated here. See [DATA_MANIFEST.md](DATA_MANIFEST.md) for full inventory.

---

## Summary Table (Paper-Relevant Results)

**Scope:** 32 NaF subjects, LOSO CV, chance = 0.5. This is the primary evaluation cohort for the paper — all encoder and VLM results below use the same subjects and the same tasks. Encoder-only results use a linear probe (LogisticRegression for genotype, Ridge for TBR); the VLM uses its multitask head on LLM last hidden state.

> **Critical evaluation consistency requirement:** Encoder-only tasks and VLM tasks must be evaluated identically. The original encoder benchmarks (T2c/full encoder comparison table) used 78 subjects across both cohorts and all 4 timepoints — those numbers are not directly comparable to the VLM, which is restricted to 32 NaF subjects with Week 12 as fixed input. The table below is the authoritative comparison; do not cite T2c=0.869 from the full encoder table alongside VLM results without noting the population mismatch.

| Condition | Input | Geno AUC | Geno acc¹ | TBR overall r | TBR overall R² |
|---|---|---|---|---|---|
| **A1** RAD-DINO linear probe, Week 12 | by subject | 0.353 | — | — | — |
| **A1'** RAD-DINO linear probe, Week 12 | **by mouse-group** | 0.498 | — | — | — |
| **VLM baseline — leak-free, seed 0** | ts0 only | 0.377 [0.18, 0.59] | 0.531 | −0.141 | −0.400 |
| **VLM baseline — leak-free, 5 seeds** | ts0 only | 0.50 ± 0.16 (seed SD) | 0.53 ± 0.08 | −0.03 ± 0.07 | −0.22 ± 0.11 |
| **VLM longitudinal — leak-free, seed 0** | ts0 + MLP forecasts (no genotype bit, rollout; **unscaled tokens, pre-crop-fix**) | 0.464 [0.26, 0.67] | 0.469 | −0.030 | −0.669 |
| **VLM longitudinal — leak-free, 5 seeds** | same, unscaled tokens | 0.39 ± 0.22 (seed SD) | 0.44 ± 0.16 | −0.01 ± 0.11 | −0.27 ± 0.24 |
| *VLM baseline / longitudinal / longitudinal+F3 — crop-fixed, scaled tokens* | *in flight 2026-09-14* | — | — | — | — |
| *B — longitudinal probe, leaky* | *ts0 + MLP-predicted with genotype bit* | *0.718* | *0.750* | *0.225* | — |
| *VLM longitudinal 20ep, leaky* | *ts0 + MLP-predicted with genotype bit; checkpoint on test subject* | *0.762* | *0.719* | *0.339* | *0.105* |

¹ VLM accuracy uses a fixed logit > 0 threshold that does not transfer from training to inference (F3); read AUROC. Brackets are subject-bootstrapped 95% CIs. The n=32 no-signal AUROC interval is [0.297, 0.703].

A1/A1' are from the audited `evaluate_embeddings.py` probe (RAD-DINO, Week-12 scans of the 32 NaF + FDG Week-12 mice). The earlier A2 and TBR-probe columns came from `evaluate_encoder_vs_vlm.py`, a second hand-rolled copy of the probe that is unstandardised and unaudited; they are withdrawn until that script is brought under the same evaluator.

**Reading.** The leak-free rows are the only citable VLM numbers, and the 5-seed rows supersede the seed-0 rows: seed SD (0.16–0.22) exceeds the n=32 sampling noise, so single-seed values are not interpretable. The longitudinal rows carry two since-fixed penalties — tokens at 2.7× the observed norm (F24) and five tube crops in the training data (F20) — and will be replaced by the in-flight runs. Both genotype AUROCs sit inside the no-signal interval; TBR R² is negative in both. Condition B and the leaky VLM row are kept, italicised, to show what the label leak (F10) and test-subject checkpoint selection (F2) were worth: B's 0.718 is the genotype one-hot being read back out of the predicted embedding (a LOSO probe on `ts1` alone recovers genotype at 0.869 vs 0.353 from the observed scan it was predicted from), and the VLM matched it at 20 epochs because it had learned to read the same injected label. A1 vs A1' shows the same input evaluated by subject (0.353) and by mouse-group (0.498) — chance either way at Week 12.

The earlier "signal-loss gap" narrative — that the VLM underfit condition B at 10 epochs and caught up at 20 — was a comparison between two ways of reading the same leaked label and is withdrawn.
---

## Full Encoder Comparison Table (78 mice, both cohorts, all 4 timepoints)

> Re-evaluated 2026-09-12 with the audited evaluator: per-fold feature standardisation in
> the linear probe, subject-level bootstrap CIs and permutation p-values, and a
> `--group-by mousegroup` control. Artifacts: `embeddings/eval_audit_2026-09-12/`.
> These use all 229 scans and are NOT directly comparable to the 32-subject Week-12 VLM
> evaluation above. Pre-audit values (unstandardised probe) are in git history.

| Metric | Chance | COLIPRI | Merlin | RAD-DINO | M3D |
|---|---|---|---|---|---|
| **T2a Accuracy** (4-class week, by subject) | 0.25 | 0.712 | 0.755 | **0.926** ⭐ | 0.738 |
| **T2b AUC-ROC** (early vs late, by subject) | 0.50 | 0.987 | 0.958 | **1.000** ⭐ | 0.993 |
| **T2b AUC-ROC** (early vs late, **by mouse-group**) | 0.50 | 0.983 | 0.958 | **1.000** ⭐ | 0.983 |
| **T2c Accuracy** (WT vs KO, by subject) | 0.50 | 0.633 | 0.624 | **0.847** ⭐ | 0.668 |
| **T2c AUC-ROC** (WT vs KO, by subject) | 0.50 | 0.666 | 0.636 | **0.920** ⭐ | 0.736 |
| **T2c AUC-ROC** (WT vs KO, **by mouse-group**)⁵ | — | 0.473 (p=0.650) | 0.458 (p=0.833) | **0.823 (p=0.008)** ⭐ | 0.541 (p=0.208) |
| **T2d AUC-ROC** (NaF vs FDG cohort, by subject) | 0.50 | 0.593 | 0.540 | **0.715** ⭐ | 0.612 |
| **T3a** Pairwise temporal ordering acc | 0.50 | 0.685 | 0.770 | **0.926** ⭐ | 0.755 |
| **T3b** Subject retrieval Recall@1 | — | 0.017 | **0.044** ⭐ | 0.013 | 0.022 |
| **T3c** Week retrieval mAP@5 | — | 0.659 | 0.692 | **0.844** ⭐ | 0.690 |

**Reading.** Time is trivially readable by every encoder — T2b by mouse-group is
0.96–1.00 for all four — but **so is it for healthy mice alone** (WT-only, mouse-group
holdout: RAD-DINO 1.000, M3D 0.956, Merlin 0.925, COLIPRI 0.878). Air-only voxels separate
the two periods at 0.76 (air mean HU −1053 → −1040, variance collapsed), so this is
substantially scanner drift with a modest growth component (size-only AUC 0.66). Do not
read it as a progression result — see FINDINGS.md F23. Genotype is a different story — by
subject all four look informative (0.64–0.92), but under leave-one-mouse-group-out only
RAD-DINO survives (0.823, p = 0.008); the three 3D encoders fall to chance. All 10
mouse-groups are genotype-pure, so the by-subject genotype numbers measure group
recognition. Standardisation moved the 3D encoders' by-subject numbers most (COLIPRI T2b
0.683 → 0.987) — their anisotropic embeddings were being under-read by the unscaled probe —
and then the clean control removed the genotype part of that gain.

RAD-DINO's surviving genotype signal — the one disease result — emerges over the study (mouse-group holdout,
Week 12: 0.498; Week 20: 0.772; all weeks: 0.823 — FINDINGS.md F5). T1 unsupervised metrics
(ARI, silhouette, within-subject consistency) are unchanged by the audit and reported in
the encoder notes below.

⁵ F18: 10 groups split 3 WT / 7 KO gives exactly C(10,3) = 120 distinct label
arrangements, so these p-values are now computed by exact enumeration, not Monte Carlo
sampling — quantised in steps of 1/120 ≈ 0.0083, not the finer resolution a bare
decimal implies. The by-week breakdown (FINDINGS.md F5) has fewer groups present at
some weeks and a correspondingly coarser floor (as loose as 1/15 at Week 18).

---

## Encoder Notes

### COLIPRI (`microsoft/colipri`)
- **Architecture:** 3D ViT, pre-trained on human chest CT + radiology reports
- **Preprocessing:** Resampled to 2 mm isotropic, resized to 192³, HU clipped ±1000
- **Embedding:** 768-d CLS token (`pool=True, project=True`)
- **Verdict:** Weakest encoder. Severe anisotropy (all cosine sims ≈ 0.997) — embeddings are packed into a tiny cone. Temporal signal exists but is geometrically very subtle. Chest-focused pre-training likely mismatches with abdominal mouse anatomy.

### Merlin (`stanfordmimi/Merlin`)
- **Architecture:** 3D ViT, pre-trained on human abdominal CT + EHR
- **Preprocessing:** Handled internally by `merlin.data.DataLoader`; preprocessed tensors cached to `embeddings/merlin/cache/`
- **Embedding:** 2048-d (`ImageEmbedding=True`)
- **Verdict:** Clear improvement over COLIPRI on all supervised tasks. Abdominal pre-training is a better domain match. The larger embedding dimension (2048) retains more task-relevant information. Still suffers from anisotropy but less severely.

### RAD-DINO (`microsoft/rad-dino`)
- **Architecture:** 2D ViT-Base/14, pre-trained on 882k chest X-rays (DINOv2 self-supervised)
- **Preprocessing:** 32 evenly-spaced axial slices per volume; HU window [-160, 240] (soft-tissue); scaled to uint8 [0, 255]; converted to RGB PIL Image; RAD-DINO's `BitImageProcessor` handles resize (518px), center crop (518×518), and normalization (mean=0.5307, std=0.2583) internally
- **Embedding:** Mean-pool of 32 per-slice 768-d CLS tokens → 768-d volume embedding
- **Verdict:** Dominant encoder across nearly all metrics. Only encoder with positive silhouette score. T2b AUC = 1.000 (perfect early vs. late separation). T3a = 0.926 (temporal ordering). The 2D slice + mean-pool adaptation is surprisingly effective — more so than native 3D processing. Note: subject retrieval (T3b) is weakest of the four; mean-pooling erases individual anatomy, so the model knows *when* but not *who*.

### M3D (`GoodBaiBai88/M3D-CLIP`)
- **Architecture:** 3D ViT (0.2B params), pre-trained on ~120k medical image-text pairs across 11 modalities via contrastive learning (CLIP objective); used in advisor's prior paper
- **Preprocessing:** HU clipped [-160, 240]; trilinear resampled to (32, 256, 256); min-max normalized to [0, 1]
- **Embedding:** 768-d CLS token via `model.encode_image(tensor)[:, 0]`
- **Verdict:** Best 3D encoder on unsupervised geometry (T1b ARI 0.114, NMI 0.150, T1d Δ 0.026) — significantly better than Merlin and COLIPRI at separating weeks without labels. Strong T2b AUC (0.963) and T3a (0.755). However, weak genotype signal (T2c AUC 0.567 ≈ near-chance) and cohort-blind (T2d AUC 0.502). The CLIP pre-training on diverse 3D medical data gives strong temporal structure but insufficient body composition sensitivity. Positioned between Merlin and RAD-DINO overall.

---

## Key Observations

> **Re-read against the 2026-09-12 table above.** These observations were written against
> the unstandardised probe. The qualitative ranking (RAD-DINO first; M3D the best 3D
> encoder; subject retrieval poor everywhere) survives re-evaluation, but the specific
> numbers quoted below are the old ones and several magnitudes changed — COLIPRI and
> Merlin are no longer "blind" by subject (T2b 0.987 / 0.958, T2c 0.666 / 0.636), though
> their genotype signal vanishes under the mouse-group control. Items 2, 4 and 5 are
> amended inline; treat the rest as directional.

1. **RAD-DINO wins on every supervised and longitudinal task.** The 2D slice + mean-pool approach (32 axial slices, mean-pool CLS tokens) outperforms all native 3D encoders across the entire evaluation battery. Self-supervised ViT pre-training on large 2D radiology corpora transfers more effectively than 3D models trained with text supervision on smaller human CT datasets.

2. **RAD-DINO is the only encoder with genotype (WT vs KO) signal that survives the mouse-group control — and it emerges over the study, not at baseline.** Under leave-one-mouse-group-out, RAD-DINO genotype AUROC is 0.498 at Week 12, 0.772 at Week 20, 0.823 across all weeks. Under the same control the three 3D encoders are at chance (COLIPRI 0.473, Merlin 0.458, M3D 0.541; all p > 0.2). By subject (inflated by group recognition), T2c AUC: RAD-DINO 0.920 vs. M3D 0.567 vs. Merlin 0.491 (chance) vs. COLIPRI 0.282 (below chance). COLIPRI's sub-chance AUC indicates its compressed embedding geometry actively anti-predicts genotype — the logistic boundary learned on training subjects inverts on held-out subjects, a hallmark of anisotropy-induced overfitting. This is the most clinically important result: only RAD-DINO could support a genotype classification application.

3. **M3D is the best encoder for unsupervised geometry (T1b/T1d).** M3D achieves the highest ARI (0.114), NMI (0.150), and T1d Δ (0.026) — meaning its embedding space is more intrinsically organised by timepoint and identity than any other encoder, without any supervision. This reflects the diversity of its CLIP pre-training across 11 modalities. However, this geometric quality does not translate to supervised genotype discrimination.

4. **[Amended] Anisotropy under-reads COLIPRI and Merlin with an unscaled probe; with per-fold standardisation they separate disease stage well (T2b 0.987 / 0.958) and show by-subject genotype signal (T2c 0.666 / 0.636) — but that genotype signal is entirely group recognition (mouse-group T2c 0.473 / 0.458, p > 0.6).** Original text: Both 3D encoders have T2c AUC ≈ 0.48–0.49 and T2e OvR AUC ≈ 0.50–0.65, meaning their embeddings contain essentially no linearly separable genotype signal. The near-identical cosine similarities (≈0.997) pack all representations into a tiny cone where only the strongest signal (time) is recoverable, and even that is weak for COLIPRI (T2b AUC = 0.68). M3D avoids this failure mode (T1d Δ = 0.026 vs. COLIPRI 0.0003).

5. **[Reversed, 2026-09-13] RAD-DINO's T2b = 1.000 is NOT validated as biological.** Per-cohort agreement (NaF and FDG both 1.000) rules out a tracer effect but not time itself, and the mouse-group control cannot separate week from acquisition date because they coincide for every mouse. The decisive check: **WT-only mice also separate at 1.000.** Healthy controls have no disease to progress, so T2b is reading age and/or scanner drift. Tested (F23): air-only regions separate the periods at 0.76 — it is substantially scanner drift, with modest growth on top. Not a disease result. Original text: Per-cohort conditioned T2b: NaF AUC = 1.000, FDG AUC = 1.000. Both cohorts use different scanners/tracers but both achieve perfect separation — scanner drift as the sole explanation is ruled out. The signal reflects real body composition/soft-tissue changes over 8 weeks of high-fat diet. M3D conditioned T2b (NaF 0.933, FDG 0.934) is consistent within cohorts at a high level, second only to RAD-DINO.

6. **RAD-DINO has moderate but non-trivial cohort sensitivity (T2d AUC = 0.709).** COLIPRI (0.488), Merlin (0.476), and M3D (0.502) are all at chance for NaF vs. FDG — they cannot tell which radiotracer was used. Only RAD-DINO can, at AUC 0.709. This could reflect genuine differences in animal preparation/body condition between cohorts rather than scanner artifacts, but warrants monitoring for cross-cohort downstream tasks.

7. **T2e accuracy is misleading for COLIPRI, Merlin, and M3D — look at F1.** T2e accuracy: M3D 0.515, COLIPRI 0.520, Merlin 0.515 (all slightly above chance of 0.20). But macro-F1: all three ≈ 0.14–0.16 ≈ chance. These encoders predict "WT" (majority class) for nearly everything, inflating accuracy. RAD-DINO (F1 = 0.621) is the only encoder learning meaningful stage boundaries.

8. **T3b (subject retrieval) is universally poor — the critical gap.** Recall@1: Merlin 0.044 ⭐, M3D 0.022, COLIPRI 0.018, RAD-DINO 0.013. All well below a useful threshold. Mean-pooling in RAD-DINO erases individual anatomy. A custom encoder with longitudinal contrastive loss (pulling same-mouse scans together) is needed to close this gap.

9. **M3D is the best 3D encoder overall.** Compared to Merlin: M3D wins on T1b/T1d (unsupervised geometry), T2b AUC (0.963 vs. 0.852), T2e OvR AUC (0.701 vs. 0.652), and T3a (0.755 vs. 0.770 — roughly equal). Merlin retains the edge on T3b subject retrieval (MRR 0.096 vs. 0.069). The diverse CLIP pre-training of M3D on 11 modalities provides stronger structural organisation than Merlin's abdominal CT + EHR objective.

---

## Conditioned Analysis (per cohort)

*Run T2a and T2b separately for NaF and FDG to disentangle biological signal from scanner/tracer confounds.*

| Metric | Cohort | COLIPRI | Merlin | RAD-DINO | M3D |
|---|---|---|---|---|---|
| **T2a acc (week, 4-class)** | NaF | 0.622 | 0.640 | **0.928** | 0.586 |
| **T2a acc (week, 4-class)** | FDG | 0.644 | 0.661 | **0.864** | 0.627 |
| **T2a OvR AUC (week, 4-class)** | NaF | 0.836 | 0.818 | **0.992** | 0.732 |
| **T2a OvR AUC (week, 4-class)** | FDG | 0.822 | 0.852 | **0.966** | 0.834 |
| **T2b acc (early vs late)** | NaF | 0.883 | 0.833 | **1.000** | 0.950 |
| **T2b acc (early vs late)** | FDG | 0.881 | 0.932 | **1.000** | 0.966 |
| **T2b AUC (early vs late)** | NaF | 0.955 | 0.888 | **1.000** | 0.981 |
| **T2b AUC (early vs late)** | FDG | 0.940 | 0.967 | **1.000** | 1.000 |

---

## VLM Results

See Summary Table above for primary reported metrics. Detailed breakdown below.

> The VLM longitudinal numbers here use the default 10-epoch config (`mouse_vlm_loso`). A subsequent epoch ablation found **20 epochs is the best configuration** — genotype acc 0.719, overall TBR MAE 4.736, and the only positive overall R² (+0.104); 50/100 epochs overfit back to ~0.625 genotype. See [experiments.md](experiments.md) for the full run log (baseline, epoch sweep, TinyLlama backbone).

**Setup:** LOSO CV over 32 NaF subjects × 3 question types = 96 records. All metrics are from multitask heads — text-match and text-parsed TBR are omitted (LLM did not generate well-formatted output in <2% of held-out records). Genotype: sigmoid(logit) > 0.5 threshold on the Linear(4096,1) head. TBR: Linear(4096,256)→GELU→Linear(256,4) regression head, MSE-trained with slot masking (−1 sentinel for missing future weeks).

### TBR Regression Head — Full Breakdown

Slot counts: Δ3wk n=64 (all subjects with Week 15), Δ6wk n=30 (subjects with Week 18), Δ8wk n=40 (subjects with Week 20).

| Metric | Longitudinal (4-token) | Baseline (1-token) |
|---|---|---|
| Overall MAE | **5.393** | 6.241 |
| Overall Pearson r | **0.276** | −0.054 |
| Overall R² | −0.076 | −0.357 |
| Δ3wk MAE | **6.123** | 6.806 |
| Δ3wk Pearson r | **0.447** | −0.090 |
| Δ3wk R² | **0.131** | −0.088 |
| Δ6wk Pearson r | **−0.093** | −0.243 |
| Δ8wk Pearson r | **0.207** | 0.069 |

Δ6wk and Δ8wk metrics are weaker, partly due to smaller sample sizes and zero-padding for missing weeks introducing noise. Negative R² overall indicates neither model beats a constant mean predictor across all slots — consistent with noisy programmatic TBR labels (see design_decisions.md).

## Longitudinal Encoder Results (LOSO CV, RAD-DINO embeddings, 32 NaF subjects, 56 pairs)

MLP (773 → 512 → 512 → 768, two hidden layers with LayerNorm+GELU) predicting T_{k+1} from T_k with cosine similarity loss. Leak-free configuration (`--no-geno-conditioning --rollout`, seed 0; `longitudinal_nogeno/`): the conditioning vector is cohort + step only (the genotype one-hot was the F10 leak), and exported ts2/ts3 are rolled forward from Week 12 (F13). Retrained on 32 NaF subjects only to match the VLM evaluation population.

| Metric | Leak-free | (leaky, archived) | Notes |
|---|---|---|---|
| T4a cosine sim | 0.982 | 0.981 | Unchanged — the genotype bit bought nothing for prediction quality |
| T4b Recall@1 | 0.018 | 0.036 | Chance-level subject retrieval; predicts week cluster, not identity |
| T4b MRR | 0.196 | 0.185 | Correct subject ranks ~5th of 32 |
| T4c improvement rate | 0.696 | 0.661 | Beats returning T_k unchanged ~70% of the time |
| Genotype probe on `ts1` | **0.369** | **0.869** | The leak, closed: now ≈ the observed Week 12 scan's own 0.353 |

---

## Future Directions

### Near-term (run next)
- [x] **Run notebook on all 3 encoders** — all T2a OvR AUC, T2c, T2d, T2e, conditioned analysis complete
- [x] **M3D encoder** (`GoodBaiBai88/M3D-CLIP`) — implemented `get_m3d_embeddings.py`, extracted 229 embeddings, full evaluation complete
- [x] **Longitudinal MLP encoder** — LOSO CV complete, predicted embeddings exported for VLM input
- [x] **VLM baseline** — LOSO CV complete (single ts0 token); genotype acc=0.219, TBR Δ3wk r=−0.090
- [x] **Genotype classification head** — implemented on LLM hidden state with BCE loss; replaces text-match
- [x] **VLM with longitudinal 4-token input + classification head** — LOSO CV complete; genotype acc=0.531, TBR Δ3wk r=0.447, overall TBR r=0.276

### Medium-term
- [ ] **Custom encoder — MAE baseline**: 3D ViT trained from scratch on this dataset with masked autoencoder objective; cheap to train and directly comparable to Merlin
- [ ] **Custom encoder — Contrastive + Reconstruction**: pull same-mouse embeddings across weeks together (longitudinal contrastive loss) + MAE reconstruction loss; designed to close the T3b (subject retrieval) gap

### Long-term (encoder paper scope)
- [ ] **DINO-style self-distillation on RAD-DINO backbone**: fine-tune the RAD-DINO ViT-B/14 backbone using DINOv2 objective on this dataset; may be challenging to reimplement but DINOv2 is open-source
- [ ] **Rigorous custom encoder evaluation**: if pursuing an encoder-specific paper, all ablations (loss terms, architecture variants, data augmentation) must be documented
