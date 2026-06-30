# Encoder Evaluation Results

Zero-shot evaluation of pretrained vision encoders on the mouse atherosclerosis CT dataset (229 scans, 78 mice, 4 timepoints). No fine-tuning. All supervised tasks use Leave-One-Subject-Out (LOSO) cross-validation.

**Dataset:** 80 mice (NaF cohort + FDG cohort, WT + KO), imaged at Weeks 12, 15, 18, 20. Only CT-Hi modality evaluated here. See [DATA_MANIFEST.md](DATA_MANIFEST.md) for full inventory.

---

## Summary Table (Paper-Relevant Results)

**Scope:** 32 NaF subjects, LOSO CV, chance = 0.5. This is the primary evaluation cohort for the paper — all encoder and VLM results below use the same subjects and the same tasks. Encoder-only results use a linear probe (LogisticRegression for genotype, Ridge for TBR); the VLM uses its multitask head on LLM last hidden state.

> **Critical evaluation consistency requirement:** Encoder-only tasks and VLM tasks must be evaluated identically. The original encoder benchmarks (T2c/full encoder comparison table) used 78 subjects across both cohorts and all 4 timepoints — those numbers are not directly comparable to the VLM, which is restricted to 32 NaF subjects with Week 12 as fixed input. The table below is the authoritative comparison; do not cite T2c=0.869 from the full encoder table alongside VLM results without noting the population mismatch.

| Condition | Input | Geno acc | Geno AUC | TBR overall r | TBR Δ3wk r | TBR Δ3wk R² |
|---|---|---|---|---|---|---|
| **A1** RAD-DINO linear probe | ts0 only (real) | 0.406 | 0.353 | −0.153 | −0.250 | −0.200 |
| **A2** RAD-DINO linear probe | ts0+ts1+ts2+ts3 (all real) | 0.406 | 0.401 | **0.432** | 0.350 | 0.085 |
| **B** Longitudinal linear probe | ts0 real + ts1/ts2/ts3 MLP-predicted | **0.750** | **0.718** | 0.225 | 0.030 | −0.323 |
| **VLM baseline** | ts0 only | 0.219 | — | −0.054 | −0.090 | −0.088 |
| **VLM longitudinal** | ts0 real + ts1/ts2/ts3 MLP-predicted | 0.531 | — | 0.276 | **0.447** | **0.131** |

**Notes on condition B:** The longitudinal linear probe (B) genotype acc=0.750/AUC=0.718 is likely inflated. The longitudinal MLP conditioning vector includes genotype as an explicit input feature, so MLP-predicted embeddings implicitly encode the label. This is not signal the VLM can equivalently exploit. Condition B should be treated as an upper-bound reference, not a fair comparison. The MLP was retrained on the 32 NaF subjects only (matching the VLM population) to remove an additional source of advantage from the prior 78-subject mixed-cohort training.

**The key discrepancy:** The VLM longitudinal model (genotype acc=0.531) underperforms condition B (acc=0.750, AUC=0.718) by ~19 AUC points despite receiving the same MLP-predicted ts1/ts2/ts3 embeddings as input. This gap — image domain signal surviving in a linear probe but degrading through the LLM pathway — is an open investigation question (see signal loss investigation system prompt).

---

## Full Encoder Comparison Table (historical — 78 subjects, both cohorts)

> **Warning:** These results use 78 subjects (NaF + FDG), all 4 timepoints, and are NOT directly comparable to the 32-subject VLM evaluation above. They are retained for encoder selection justification only.

| Metric | Chance | COLIPRI | Merlin | RAD-DINO | M3D |
|---|---|---|---|---|---|
| **T2a Accuracy** (4-class week, LOSO) | 0.25 | 0.454 | 0.563 | **0.878** ⭐ | 0.607 |
| **T2b AUC-ROC** (early vs late, LOSO) | 0.50 | 0.683 | 0.852 | **1.000** ⭐ | 0.963 |
| **T2c Accuracy** (WT vs KO, LOSO) | 0.50 | 0.411 | 0.459 | **0.786** ⭐ | 0.563 |
| **T2c AUC-ROC** (WT vs KO, LOSO) | 0.50 | 0.282 | 0.491 | **0.869** ⭐ | 0.567 |
| **T3b** Subject retrieval Recall@1 | — | 0.018 | **0.044** ⭐ | 0.013 | 0.022 |

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

1. **RAD-DINO wins on every supervised and longitudinal task.** The 2D slice + mean-pool approach (32 axial slices, mean-pool CLS tokens) outperforms all native 3D encoders across the entire evaluation battery. Self-supervised ViT pre-training on large 2D radiology corpora transfers more effectively than 3D models trained with text supervision on smaller human CT datasets.

2. **RAD-DINO is the only encoder with meaningful genotype (WT vs KO) signal.** T2c AUC: RAD-DINO 0.869 vs. M3D 0.567 vs. Merlin 0.491 (chance) vs. COLIPRI 0.282 (below chance). COLIPRI's sub-chance AUC indicates its compressed embedding geometry actively anti-predicts genotype — the logistic boundary learned on training subjects inverts on held-out subjects, a hallmark of anisotropy-induced overfitting. This is the most clinically important result: only RAD-DINO could support a genotype classification application.

3. **M3D is the best encoder for unsupervised geometry (T1b/T1d).** M3D achieves the highest ARI (0.114), NMI (0.150), and T1d Δ (0.026) — meaning its embedding space is more intrinsically organised by timepoint and identity than any other encoder, without any supervision. This reflects the diversity of its CLIP pre-training across 11 modalities. However, this geometric quality does not translate to supervised genotype discrimination.

4. **Anisotropy renders COLIPRI and Merlin blind to genotype and disease stage.** Both 3D encoders have T2c AUC ≈ 0.48–0.49 and T2e OvR AUC ≈ 0.50–0.65, meaning their embeddings contain essentially no linearly separable genotype signal. The near-identical cosine similarities (≈0.997) pack all representations into a tiny cone where only the strongest signal (time) is recoverable, and even that is weak for COLIPRI (T2b AUC = 0.68). M3D avoids this failure mode (T1d Δ = 0.026 vs. COLIPRI 0.0003).

5. **RAD-DINO's T2b = 1.000 is validated as biological, not scanner drift.** Per-cohort conditioned T2b: NaF AUC = 1.000, FDG AUC = 1.000. Both cohorts use different scanners/tracers but both achieve perfect separation — scanner drift as the sole explanation is ruled out. The signal reflects real body composition/soft-tissue changes over 8 weeks of high-fat diet. M3D conditioned T2b (NaF 0.933, FDG 0.934) is consistent within cohorts at a high level, second only to RAD-DINO.

6. **RAD-DINO has moderate but non-trivial cohort sensitivity (T2d AUC = 0.709).** COLIPRI (0.488), Merlin (0.476), and M3D (0.502) are all at chance for NaF vs. FDG — they cannot tell which radiotracer was used. Only RAD-DINO can, at AUC 0.709. This could reflect genuine differences in animal preparation/body condition between cohorts rather than scanner artifacts, but warrants monitoring for cross-cohort downstream tasks.

7. **T2e accuracy is misleading for COLIPRI, Merlin, and M3D — look at F1.** T2e accuracy: M3D 0.515, COLIPRI 0.520, Merlin 0.515 (all slightly above chance of 0.20). But macro-F1: all three ≈ 0.14–0.16 ≈ chance. These encoders predict "WT" (majority class) for nearly everything, inflating accuracy. RAD-DINO (F1 = 0.621) is the only encoder learning meaningful stage boundaries.

8. **T3b (subject retrieval) is universally poor — the critical gap.** Recall@1: Merlin 0.044 ⭐, M3D 0.022, COLIPRI 0.018, RAD-DINO 0.013. All well below a useful threshold. Mean-pooling in RAD-DINO erases individual anatomy. A custom encoder with longitudinal contrastive loss (pulling same-mouse scans together) is needed to close this gap.

9. **M3D is the best 3D encoder overall.** Compared to Merlin: M3D wins on T1b/T1d (unsupervised geometry), T2b AUC (0.963 vs. 0.852), T2e OvR AUC (0.701 vs. 0.652), and T3a (0.755 vs. 0.770 — roughly equal). Merlin retains the edge on T3b subject retrieval (MRR 0.096 vs. 0.069). The diverse CLIP pre-training of M3D on 11 modalities provides stronger structural organisation than Merlin's abdominal CT + EHR objective.

---

## Conditioned Analysis (per cohort)

*Run T2a and T2b separately for NaF and FDG to disentangle biological signal from scanner/tracer confounds.*

| Metric | Cohort | COLIPRI | Merlin | RAD-DINO | M3D |
|---|---|---|---|---|---|
| **T2a acc** (week, 4-class) | NaF | 0.460 | 0.505 | **0.802** | 0.604 |
| **T2a acc** (week, 4-class) | FDG | 0.449 | 0.517 | **0.814** | 0.568 |
| **T2a OvR AUC** (week, 4-class) | NaF | 0.610 | 0.660 | **0.954** | 0.729 |
| **T2a OvR AUC** (week, 4-class) | FDG | 0.634 | 0.724 | **0.958** | 0.729 |
| **T2b acc** (early vs late) | NaF | 0.600 | 0.617 | **1.000** | 0.767 |
| **T2b acc** (early vs late) | FDG | 0.610 | 0.678 | **1.000** | 0.864 |
| **T2b AUC** (early vs late) | NaF | 0.617 | 0.719 | **1.000** | 0.933 |
| **T2b AUC** (early vs late) | FDG | 0.574 | 0.779 | **1.000** | 0.934 |

---

## VLM Results

See Summary Table above for primary reported metrics. Detailed breakdown below.

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

MLP (775 → 512 → 512 → 768, two hidden layers with LayerNorm+GELU) predicting T_{k+1} from T_k with cosine similarity loss. Retrained on 32 NaF subjects only to match the VLM evaluation population (previously 78 subjects across both cohorts).

| Metric | Value | Notes |
|---|---|---|
| T4a cosine sim | 0.981 | High directional accuracy; reflects tight embedding cluster geometry |
| T4b Recall@1 | 0.036 | Near-chance subject retrieval; predicts week cluster, not individual identity |
| T4b MRR | 0.185 | Correct subject ranks ~5th out of 32 on average |
| T4c improvement rate | 0.661 | Beats returning T_k unchanged 66% of the time |

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
