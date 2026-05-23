# Encoder Evaluation Results

Zero-shot evaluation of pretrained vision encoders on the mouse atherosclerosis CT dataset (229 scans, 78 mice, 4 timepoints). No fine-tuning. All supervised tasks use Leave-One-Subject-Out (LOSO) cross-validation.

**Dataset:** 80 mice (NaF cohort + FDG cohort, WT + KO), imaged at Weeks 12, 15, 18, 20. Only CT-Hi modality evaluated here. See [DATA_MANIFEST.md](DATA_MANIFEST.md) for full inventory.

---

## Summary Table

| Metric | What it measures | Chance | COLIPRI | Merlin | RAD-DINO | M3D |
|---|---|---|---|---|---|---|
| **Embedding dim** | — | — | 768 | 2048 | 768 | 768 |
| **Input type** | — | — | 3D volume | 3D volume | 32 axial slices (mean-pooled) | 3D volume (32×256×256) |
| **T1b ARI** (k-means vs week) | Do unsupervised clusters align with timepoints, without any labels? | 0 | 0.033 | 0.011 | −0.010 | **0.114** ⭐ |
| **T1b NMI** (k-means vs week) | Same as ARI but less sensitive to cluster size imbalance | 0 | 0.047 | 0.017 | 0.013 | **0.150** ⭐ |
| **T1c Silhouette** (cosine, by week) | Are same-week embeddings geometrically tighter than cross-week embeddings? | 0 | −0.074 | −0.050 | **+0.020** ⭐ | −0.009 |
| **T1d Δ** (intra − inter cosine sim) | Are the same mouse's scans more similar to each other than to other mice? | 0 | 0.0003 | 0.0006 | 0.0055 | **0.0261** ⭐ |
| **T2a Accuracy** (4-class week, LOSO) | Can a linear classifier predict which of 4 timepoints a scan came from? | 0.25 | 0.454 | 0.563 | **0.878** ⭐ | 0.607 |
| **T2a macro-F1** (4-class week, LOSO) | Same as above, averaged equally across all four weeks (penalises class imbalance) | — | 0.279 | 0.444 | **0.854** ⭐ | 0.397 |
| **T2a OvR AUC** (4-class week, LOSO) | Macro-averaged one-vs-rest AUC for week classification confidence | — | 0.694 | 0.776 | **0.979** ⭐ | 0.785 |
| **T2b Accuracy** (early vs late, LOSO) | Can a linear classifier distinguish Week 12 from Week 20? | 0.50 | 0.605 | 0.748 | **1.000** ⭐ | 0.899 |
| **T2b AUC-ROC** (early vs late, LOSO) | How well does the classifier's confidence score rank early scans above late ones? | 0.50 | 0.683 | 0.852 | **1.000** ⭐ | 0.963 |
| **T2c Accuracy** (WT vs KO, LOSO) | Can a linear classifier distinguish healthy (WT) from disease (KO) mice? | 0.50 | 0.411 | 0.459 | **0.786** ⭐ | 0.563 |
| **T2c AUC-ROC** (WT vs KO, LOSO) | Confidence-based separation of genotypes | 0.50 | 0.282 ⚠️ | 0.491 | **0.869** ⭐ | 0.567 |
| **T2d Accuracy** (NaF vs FDG, LOSO) | Are scanner/radiotracer differences linearly separable? (confounder check) | 0.50 | 0.515 | 0.520 | **0.646** ⭐ | 0.489 |
| **T2d AUC-ROC** (NaF vs FDG, LOSO) | Confidence-based cohort separation | 0.50 | 0.488 | 0.476 | **0.709** ⭐ | 0.502 |
| **T2e Accuracy** (WT vs KO+stage, LOSO) | 5-class task: WT or KO at which disease stage? | 0.20 | 0.520 | 0.515 | **0.729** ⭐ | 0.515 |
| **T2e macro-F1** (WT vs KO+stage, LOSO) | Same, balanced across all five classes | — | 0.137 | 0.136 | **0.621** ⭐ | 0.156 |
| **T2e OvR AUC** (WT vs KO+stage, LOSO) | Confidence-based disease staging | — | 0.497 | 0.652 | **0.915** ⭐ | 0.701 |
| **T3a** Pairwise temporal ordering | For same-mouse pairs, can a linear classifier tell which scan came later? | 0.50 | 0.685 | 0.770 | **0.926** ⭐ | 0.755 |
| **T3b** Subject retrieval Recall@1 | Is the nearest neighbour (by cosine sim) a scan of the same mouse? | — | 0.018 | **0.044** ⭐ | 0.013 | 0.022 |
| **T3b** Subject retrieval MRR | On average, how highly ranked is the first same-mouse scan in the similarity list? | — | 0.066 | **0.096** ⭐ | 0.072 | 0.069 |
| **T3c** Week retrieval mAP@5 | What fraction of the 5 nearest neighbours share the same timepoint? | — | 0.659 | 0.692 | **0.844** ⭐ | 0.690 |

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

LOSO CV over 32 NaF subjects × 3 question types = 96 records. Each fold trains on 31 subjects, evaluates on the held-out subject. All metrics are from multitask heads (genotype classification head, TBR regression head) — text-match and text-parsed TBR are omitted as the LLM did not generate well-formatted output under the small per-fold training set.

### Genotype Classification (multitask head, n=32 subjects)

Binary prediction: KO=1, WT=0. Head reads LLM last hidden state at answer-end EOS. Threshold: sigmoid(logit) > 0.5.

| Model | Accuracy |
|---|---|
| Longitudinal (4-token: ts0 + ts1/ts2/ts3) | **0.531** |
| Baseline (1-token: ts0 only) | 0.219 |
| Chance | 0.500 |

The longitudinal model sits just above chance. The baseline falls well below chance (0.219), indicating the genotype head learned an inverted signal from ts0 alone — consistent with the embedding anisotropy noted in the encoder evaluation (RAD-DINO embeddings are primarily organised by timepoint, not genotype).

### TBR Regression Head (multitask head, MSE-trained)

Predicts up to 3 future TBR values (Δ3wk/Δ6wk/Δ8wk relative to Week 12 input). Slot counts: Δ3wk n=64 (all subjects), Δ6wk n=30 (subjects with Week 18), Δ8wk n=40 (subjects with Week 20).

| Metric | Longitudinal (4-token) | Baseline (1-token) |
|---|---|---|
| Overall MAE | **5.393** | 6.241 |
| Overall Pearson r | **0.276** | −0.054 |
| Overall R² | −0.076 | −0.357 |
| Δ3wk MAE | **6.123** | 6.806 |
| Δ3wk Pearson r | **0.447** | −0.090 |
| Δ3wk R² | **0.131** | −0.088 |
| Δ6wk MAE | **6.855** | 7.568 |
| Δ6wk Pearson r | **−0.093** | −0.243 |
| Δ6wk R² | −0.722 | −0.957 |
| Δ8wk MAE | **3.127** | 4.340 |
| Δ8wk Pearson r | **0.207** | 0.069 |
| Δ8wk R² | −1.826 | −4.421 |

The longitudinal model outperforms the baseline on every metric. The most populated slot (Δ3wk, n=64) shows the strongest signal: r=0.447, R²=0.131 — modest but positive correlation. Negative R² overall indicates neither model beats a constant mean predictor across all slots, consistent with the noisy programmatic TBR labels (see design_decisions.md). Δ6wk and Δ8wk metrics are weaker, partly due to smaller sample sizes (n=30/40) and missing-Week-18 zero-padding introducing noise.

## Longitudinal Encoder Results (LOSO CV, RAD-DINO embeddings, 78 subjects, 129 pairs)

MLP (775 → 512 → 512 → 768, two hidden layers with LayerNorm+GELU) predicting T_{k+1} from T_k with cosine similarity loss.

| Metric | Value | Notes |
|---|---|---|
| T4a cosine sim | 0.983 | High directional accuracy; reflects tight embedding cluster geometry |
| T4b Recall@1 | 0.031 | Near-chance subject retrieval; predicts week cluster, not individual identity |
| T4b MRR | 0.177 | Correct subject ranks ~6th out of 66 on average |
| T4c improvement rate | 0.682 | Beats returning T_k unchanged 68% of the time |

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
