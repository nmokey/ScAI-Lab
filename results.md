# Encoder Evaluation Results

Zero-shot evaluation of pretrained vision encoders on the mouse atherosclerosis CT dataset (229 scans, 78 mice, 4 timepoints). No fine-tuning. All supervised tasks use Leave-One-Subject-Out (LOSO) cross-validation.

**Dataset:** 80 mice (NaF cohort + FDG cohort, WT + KO), imaged at Weeks 12, 15, 18, 20. Only CT-Hi modality evaluated here. See `DATA_MANIFEST.md` for full inventory.

---

## Summary Table

| Metric | What it measures | Chance | COLIPRI | Merlin | RAD-DINO |
|---|---|---|---|---|---|
| **Embedding dim** | — | — | 768 | 2048 | 768 |
| **Input type** | — | — | 3D volume | 3D volume | 32 axial slices (mean-pooled) |
| **T1b ARI** (k-means vs week) | Do unsupervised clusters align with timepoints, without any labels? | 0 | 0.033 | 0.011 | −0.010 |
| **T1b NMI** (k-means vs week) | Same as ARI but less sensitive to cluster size imbalance | 0 | 0.047 | 0.017 | 0.013 |
| **T1c Silhouette** (cosine, by week) | Are same-week embeddings geometrically tighter than cross-week embeddings? | 0 | −0.074 | −0.050 | **+0.020** |
| **T1d Δ** (intra − inter cosine sim) | Are the same mouse's scans more similar to each other than to other mice? | 0 | 0.0003 | 0.0006 | **0.0055** |
| **T2a Accuracy** (4-class week, LOSO) | Can a linear classifier predict which of 4 timepoints a scan came from? | 0.25 | 0.454 | 0.563 | **0.878** |
| **T2a macro-F1** (4-class week, LOSO) | Same as above, averaged equally across all four weeks (penalises class imbalance) | — | 0.279 | 0.444 | **0.854** |
| **T2b Accuracy** (early vs late, LOSO) | Can a linear classifier distinguish Week 12 from Week 20? | 0.50 | 0.605 | 0.748 | **1.000** |
| **T2b AUC-ROC** (early vs late, LOSO) | How well does the classifier's confidence score rank early scans above late ones? | 0.50 | 0.683 | 0.852 | **1.000** |
| **T3a** Pairwise temporal ordering | For same-mouse pairs, can a linear classifier tell which scan came later? | 0.50 | 0.685 | 0.770 | **0.926** |
| **T3b** Subject retrieval Recall@1 | Is the nearest neighbour (by cosine sim) a scan of the same mouse? | — | 0.018 | **0.044** | 0.013 |
| **T3b** Subject retrieval MRR | On average, how highly ranked is the first same-mouse scan in the similarity list? | — | 0.066 | **0.096** | 0.072 |
| **T3c** Week retrieval mAP@5 | What fraction of the 5 nearest neighbours share the same timepoint? | — | 0.659 | 0.692 | **0.844** |

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
- **Verdict:** Dominant encoder across nearly all metrics. Only encoder with positive silhouette score. T2b AUC = 1.000 (perfect early vs. late separation). T3a = 0.926 (temporal ordering). The 2D slice + mean-pool adaptation is surprisingly effective — more so than native 3D processing. Note: subject retrieval (T3b) is weakest of the three; mean-pooling erases individual anatomy, so the model knows *when* but not *who*.

---

## Key Observations

1. **RAD-DINO wins despite being 2D.** The 2D slice + mean-pool approach outperforms native 3D encoders (COLIPRI, Merlin) on every longitudinal and classification task. This suggests that ViT features trained with self-supervision on large 2D radiology datasets transfer better than 3D models trained on smaller supervised datasets with text supervision.

2. **Anisotropy is a fundamental problem for COLIPRI and Merlin.** All cosine similarities cluster around 0.997, making unsupervised clustering impossible. RAD-DINO's cosine sim range (0.970–0.975) is far healthier and explains why its unsupervised metrics are the only ones that aren't near-random.

3. **The temporal signal is strong but subject identity is not.** T3b (subject retrieval) is poor for all three encoders — none can reliably find scans from the same mouse. This is the key gap a custom-trained model should close.

4. **RAD-DINO's T2b = 1.000 warrants scrutiny.** Perfect LOSO separation of Week 12 vs. Week 20 is striking for a zero-shot model. Possible explanations:
   - Genuine biological signal (body composition, calcification, soft tissue changes over 8 weeks of HFD)
   - Global image statistics correlated with scan date (scanner drift/gain)
   - To disambiguate: check whether T2b signal holds separately within each cohort (NaF vs FDG), since scanner drift would affect both equally while biology predicts FDG signal degrades at late timepoints

5. **Merlin is the best 3D encoder.** If 3D processing is required (e.g., for the custom MAE baseline), Merlin is the preferred starting point over COLIPRI.

---

## Planned Evaluations

- [ ] Cohort classification (NaF vs FDG) — linear probe, all three encoders
- [ ] Genotype classification (WT vs KO) — linear probe, all three encoders
- [ ] Custom MAE (3D ViT, trained from scratch on this dataset)
- [ ] RAD-DINO confound check: per-cohort T2b to rule out scanner drift
