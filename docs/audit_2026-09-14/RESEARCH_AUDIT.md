# Independent research pipeline audit — 2026-09-14

> Historical pre-repair audit snapshot. The defects below were subsequently repaired and checked. See [current status](../STATUS.md), [all completed results](../POST_AUDIT_RESULTS.md), and [evidence scope](README.md) for the present state.

Implementation follow-through: [master closure plan](../MASTER_PLAN.md), with dependencies, acceptance checks, and publication gates for every finding below.

**The current results do not establish that a correctly trained longitudinal VLM fails. The production checkpoint path discards its trained LoRA weights before evaluation.** This is an additional defect beyond F1–F24. Separately, the present experiment has substantial limitations in its labels, information content, and independent sample size. Repairing the implementation is necessary; a positive forecasting result is not guaranteed.

This audit inspected the working tree, pipeline code, installed preprocessing/loading implementations, current data artifacts, experiment configurations, historical results, and active tmux run logs. It ran the existing CPU correctness suite, an independent checkpoint round trip, matched-population probes and simple forecasting baselines, and an exact group-permutation test with full probe refits. Production code, original data, existing results, and active training jobs were not modified. New files are audit scripts and evidence only. The working tree already contained extensive uncommitted audit changes; HEAD alone does not identify this experiment.

The user confirmed the working identity heuristic: mice listed in filenames occupy lower-left, lower-right, upper-left, upper-right positions in that order. Use this rule with the existing explicit tube/phantom overrides; external placement confirmation is not a prerequisite for continuing under this assumption. Original aortic ROI measurements and histology labels are unavailable. Biological validation of the proxy consequently remains unresolved.

**1. Critical: trained adapters are lost during checkpoint reload — reproduced and present in live logs.**

`vlm/model/vision_language_model.py:273–281` first calls PEFT's correct serializer, then overwrites its files with raw `language_model.state_dict()` entries. Those entries contain adapter names such as `lora_A.default.weight`. PEFT checkpoints omit the adapter name; the loader inserts it. Overwriting the correct files creates incompatible keys. `from_pretrained` subsequently loads a fresh shell, leaves trained adapters missing, and proceeds after a warning. The separately saved projection and heads survive, but now operate with an unadapted LLM. This is a different model from the one trained.

The CPU diagnostic used the production save/load methods and a tiny random Llama with nonzero adapters, in the project's environment: torch 2.10.0, transformers 4.46.3, PEFT 0.18.1. All four adapter tensors differed after reload; both loaded B matrices had norm zero. Genotype and TBR predictions changed. Replacing **only the disposable test checkpoint** with ordinary PEFT serialization gave exactly zero output difference for language logits and both heads. This agrees with the [official PEFT checkpoint contract](https://huggingface.co/docs/peft/v0.18.0/developer_guides/checkpoint).

Live baseline, normalized-longitudinal, and question-EOS logs each list **128 missing adapter tensors per completed fold**, covering q/v A/B matrices across 32 layers. The bad overwrite dates to commit `f3cc568f` on 2026-05-23. Historical runs using this path need re-evaluation; present-run logs directly establish the problem for the active experiments. This can plausibly contribute to seed instability and poor text generation, but those effect sizes have not been isolated in a full-size controlled run.

Action: retain proper PEFT serialization, fail on missing/unexpected adapter keys, and require pre-save versus post-load head-output equivalence on an identical batch before any sweep. Verify this once on the actual quantized backbone, including buffers and projection. Preserve checkpoints until this verification passes; the LOSO runner currently deletes completed-fold weights, preventing later recovery of those folds from prediction files alone.

Evidence: [checkpoint round trip](checkpoint_roundtrip.log), [live warnings](live_adapter_warnings.json); executable: `scripts/audit_checkpoint_roundtrip.py`.

**2. Critical protocol issue: separate MLP and VLM LOSO splits are not a nested evaluation.**

`scripts/train_longitudinal.py:674–728` generates each mouse j's forecast with an MLP trained on all mice except j. `vlm/run/run_mouse_vlm_loso.py:112–120` changes JSON splits but leaves one global forecast directory shared across VLM folds.

For VLM test mouse i, the forecasts used to train on every other mouse j therefore come from an MLP trained using i's longitudinal images. i's own test token excludes i, but the **VLM training features** do not. Training-fold token norm references also include i when generated for j. This is a confirmed dependency on outer-test data, not a measured estimate of its metric inflation. It affects even the no-genotype, normalized rollout configuration.

Action: split the outer evaluation first. Within each outer training set, generate training tokens through inner cross-fitting; fit an outer-training-only forecaster to generate outer-test tokens. Every forecaster, normalization statistic, hyperparameter selection, and cache must exclude the outer test unit. Apply the same exclusion to inner validation used for selection. A genuinely independent frozen forecaster trained on separate animals would also solve this dependency.

**3. Critical design constraint: the VLM population contains only three independent scanning components.**

The current 96 records are 32 mice, split **KO 4, KO 10, WT 18** across three connected mouse/session groups. All WT animals belong to one component. A group holdout of that component leaves only KO training examples. A normal two-class classifier cannot fit; skipping that fold leaves only KO test labels, so pooled AUROC is undefined. This was encountered directly during the matched-population probe.

The often-cited ten groups refer to **both cohorts**, not the NaF VLM population. Adding a `--group-by mousegroup` flag alone cannot yield a valid NaF-only genotype experiment. A constant prediction fallback does not create missing training-class support. Subject LOSO can describe performance on additional mice within this acquisition structure, but cannot substantiate generalization to independent WT groups.

The exact VLM-population Week-12 standardized probe gives subject-LOSO AUROC **0.6190**, not the earlier 0.353/0.498 figures cited alongside VLM results. This is descriptive and confounded, not a rescued genotype claim. It establishes that the old population comparison was unsuitable for concluding that the baseline contains no information.

Evidence: [artifact snapshot](artifact_snapshot.json), `vlm_data` and `matched_week12_genotype_probe_auc`.

**4. High: answer-conditioned head training is a shortcut for both tasks, not merely a threshold problem.**

`vision_language_model.py:85–119,175–212` trains the heads on the last EOS after the ground-truth answer by default; `generate:237–249` evaluates them on question-only context. Answers contain the genotype statement and numerical TBR targets. The heads can learn to decode those answers rather than learn image-to-outcome prediction. AUROC is invariant to a fixed monotone score transformation; it is not immune to a different hidden-state distribution caused by removing the answer. TBR regression has the same training/inference mismatch.

Use `question_eos` for the current standard prompt, or an explicit question-boundary index. Require head-input invariance to changes in answer text. The active question-EOS run addresses this shortcut, but still has the checkpoint and nested-split problems. A fair comparison needs **both baseline and longitudinal arms** with the same pooling. One seed cannot demonstrate reduced seed instability. The alternative `llama3` prompt needs separate boundary tests: a first-EOS heuristic must not select a system-message boundary.

**5. High: the modeled endpoint does not reproduce the source paper's measurement.**

The source study reports localized NaF **SUVmean**, using a 5×4×2 mm ROI above the heart, a bone boundary, and a 0.6 SUV threshold. Its CT endpoint uses a thoracic ROI and a 350 HU threshold. NaF peaks at Week 18; CT becomes discriminative later. It also includes planned euthanasia, so reduced follow-up is not automatically disease dropout. These methods differ materially from the repository's account of manually drawn aortic TBR. [Source methods and results](https://d-nb.info/1370555385/34).

`extract_tbr_features.py:135–157` computes P95 of positive voxels in the middle third of a PET crop divided by a trimmed median. There is no aortic target ROI, anatomical background ROI, or validation against source measurements. The numerator uses the original mid-band P95; hotspot exclusion affects the denominator. It is a PET distribution statistic, not a validated aortic measurement. Poor prediction of this endpoint does not establish inability to predict disease. Conversely, predicting it well would not validate atherosclerosis monitoring. TBR is not inherently restricted to NaF as a concept; this implementation simply has no validated FDG endpoint.

Without original labels, retain it explicitly as an exploratory PET proxy or develop a blinded, anatomically checked replacement. Do not select a proxy because it maximizes genotype separation on the same evaluation animals.

**6. High: TBR-3 still ignores CT/PET physical geometry.**

The F9 axis fix does not fix registration. `extract_tbr_features.py:177–181,225–241` maps voxels using array-shape ratios, discarding NIfTI affines. On all 111 NaF CT/PET pairs, the shape-based map differs from the affine-based map at the CT crop center by **median 0.507 mm, maximum 0.755 mm**. This is material at mouse-vessel scales. It measures coordinate error, not the resulting label error.

Four Week-15 KO crops merge CT from `m54407_2` with PET from `m54407`; matching physical coordinates does not prove the animal stayed aligned between acquisitions. Use physical-space resampling and check overlays. The current TBR-2 label does not use this mapping, so this finding applies to TBR-3 and proposed anatomical replacements. Most NaF rows specify 3 h; two carry mixed `1+3` timing and need per-mouse resolution if reconstructing the source endpoint.

**7. High: no demonstrated personalized benefit from the corrected forecast embeddings.**

Current scaled exports have norm about 13.97, consistent with observed features. Against actual future images, however:

| Forecast target | MLP rollout cosine | Subject-excluded week centroid cosine | n |
|---|---:|---:|---:|
| Week 15 | 0.98643 | 0.98702 | 32 |
| Week 18 | 0.98470 | 0.98531 | 15 |
| Week 20 | 0.98471 | 0.98621 | 20 |

Group-excluded centroids also have higher average cosine at every horizon. Centroids are restricted to the same 32 VLM subjects and exclude the evaluated mouse or its entire group. They use available training-week observations; later horizons include observations excluded by the MLP's consecutive-pair restriction, so this is a practical comparator rather than an identical-training-pair ablation. It still demonstrates why near-0.99 cosine is insufficient evidence of personalized forecasting. No significance is claimed for these small differences.

The MLP has about 56 training pairs before holdout and a large network; its cosine objective rewards the common embedding direction. Train-mean, persistence, mean-transition, and residual prediction baselines are necessary. Evaluate **rollout** per horizon: the saved T4 scores remain one-step scores even when exports are rolled out. Scaling now happens before feedback into the next step, so F24 changes later token directions too; it is not purely a VLM input-amplitude ablation. LLaMA normalization also means a 2.7× input norm difference alone does not prove a 2.7× causal attention penalty.

Conditioned on fixed training data, predicted tokens are deterministic functions of the baseline. They add no new held-out measurements, but can still improve a finite-data learner by supplying a useful learned representation. Thus neither “they must help” nor “they cannot help” follows from determinism.

**8. High: human-scale preprocessing weakens the encoder comparison.**

Installed COLIPRI 0.1.0 uses SAR orientation, **2 mm resampling followed by crop/pad to 192³**. It does not resize the mouse to fill the cube, as the repository says. Installed Merlin uses 1.5×1.5×3 mm resampling, then padding/cropping to 224×224×160. A roughly 40 mm-wide crop becomes only about 20–27 voxels across before padding. This removes fine anatomy and fills much of the model input with padding. RAD-DINO/M3D instead resize sampled slices or the volume into the target dimensions. This is a substantial preprocessing difference, not an isolated test of 2D versus 3D architecture or self-supervision versus text supervision.

RAD-DINO and M3D additionally clip CT at 240 HU, while the source CT calcification endpoint thresholds above 350 HU. Saturation and whole-volume pooling do not remove all morphological signal, but weaken the connection to local calcification density. Test scale-aware preprocessing and localized/dual-window features before claiming that native 3D encoders lack disease information. Document architecture from actual models; the current comparison table's generic “3D ViT” descriptions should not be assumed correct.

**9. High: Merlin reuses stale path-keyed caches after recropping — reproduced.**

The installed `merlin/data/dataloaders.py:19–35` hashes the input path dictionary, not file content or modification time. Replacing a NIfTI at the same path does not invalidate its cached tensor. All five F20-corrected mouse/week crops have corresponding cache files older than the corrected CT. For `NaF_KO_09/week_15`, fresh preprocessing differs from the cached tensor in **20,768 voxels**, with maximum absolute difference **1.0** on the normalized intensity scale. The standard extraction script reuses the stale file. This is an actual cache inconsistency, not just a timestamp concern. The claim that all four encoders were fully regenerated on corrected data therefore needs verification, even though each `.npz` has 229 finite, uniquely keyed rows. Use versioned/content-hashed caches; record input hashes in encoder artifacts. Evidence: [cache probe](merlin_cache_probe.log); executable `scripts/audit_merlin_cache.py`. This does not directly affect the RAD-DINO-based VLM.

**10. Statistics and reporting need a more precise estimand.**

- Fixed-score relabeling of cross-validated predictions is not a full training-procedure permutation test. Overlapping training folds make scores depend on other test labels. For example, a featureless LOSO classifier predicting training prevalence has AUROC 0: holding out a positive lowers its training prior. Relabeling those fixed scores does not reproduce that procedure. Computational expense does not make the cheaper test equivalent.
- The five-seed rank pool retains subject clustering, so it is not simply 160 independent mice. But it estimates neither mean per-seed AUROC nor an ordinary subject-level ensemble AUROC, and its conditional bootstrap does not quantify new-training-run variability. Report those estimands separately, with paired arm comparisons, subject/group uncertainty, and seed variation distinguished. Five seeds are not five independent datasets.
- Negative ordinary R² is relative to the evaluated targets' mean, not automatically the training-fold mean. Reconstructing a per-horizon training-mean baseline on the ten saved seed runs gives pooled R² **−0.0482**, MAE **5.4739**. Longitudinal seed 1 has R² −0.047 and therefore narrowly exceeds that MSE comparator despite negative R²; its MAE remains worse. This is not meaningful rescued evidence, but the blanket claim that every run is worse than a deployable mean predictor is technically false.
- Current-data per-horizon ridge probes with fixed alpha=100 perform worse than simple means. This is one exploratory baseline, not an optimized-model impossibility result. Pooling future slots also mixes between-week variation with individual trajectory prediction; use per-horizon performance and errors in within-mouse change.
- The runner appends predictions on resume without validating unique fold/qid coverage or data/config hashes. Existing inspected files have no duplicate pid/qid records, but the path permits duplicates and mixed-protocol aggregates. Incomplete runs must not be scored as completed studies.
- `_tbr_targets` still checks literal uppercase `TBR`, even though evaluation routing uses `content_type`; current paraphrases pass, but structured numerical targets would be safer than reparsing answers. `genotype_acc_calibrated=False` is hardcoded even for question-EOS runs.

**What positive evidence survives?**

I independently reran the current RAD-DINO all-week genotype probe on **229 scans, 78 mice, ten groups**, with full refitting for all **120** group-label arrangements. AUROC = **0.82819**, exact one-sided p = **0.00833**; refit-null 95% interval = [0.08394, 0.74700]. The result survives the stronger test. Evidence: [exact refit log](group_refit.log), executable `scripts/audit_group_permutation.py`.

This is an association with the combined genotype/diet assignment across acquisition groups. Genotype and diet were not separated experimentally; neither whole-body embeddings nor group holdout localizes the mechanism to aortic plaque. Four-encoder selection and other exploratory choices also constrain confirmatory wording. Age/week separation and the existing air-region analysis support acquisition sensitivity; air AUC alone does not quantify how much of the encoder's performance is caused by scanner drift. The strong claims “one disease result” and “substantially scanner drift” exceed what these controls alone identify.

**A practical route to a defensible longitudinal VLM result.**

1. First repair and verify the checkpoint round trip on the actual backbone. Run one small, known-signal overfit check with question-only heads and compare predictions before/after reload. Verify image dependence with zeroed/shuffled-image controls. This distinguishes broken optimization/inference from weak biology.
2. Freeze a single estimand, endpoint, split, and pooling scheme before another sweep. Use nested forecast construction. Preserve dataset/config/environment hashes and completed checkpoints until verified. For NaF genotype, acknowledge the three-component limitation or enlarge the population with justified endpoints and independent class support.
3. On a validated or explicitly exploratory endpoint, compare identical heads and splits across baseline-only, repeated-baseline tokens, training-week-centroid tokens, predicted tokens, and a direct small regressor. Add a matched simple temporal regressor. Equalize token count where relevant; otherwise extra tokens may act as extra computation or prompt context. Use paired subject-level losses as the primary comparison.
4. If the scientific aim is early prognosis, baseline PET plus CT is a better-motivated information source than whole-body CT alone. The source finds early disease with PET before CT. If the aim is longitudinal updating, compare an observed W12+W15 input against W12 alone for W18/W20, restricted to the same eligible mice; then compare the VLM with simple models using those same observations. This is a revised task, not retrospective confirmation of the original baseline-only claim.
5. Without source measurements, a small blinded anatomical annotation/measurement exercise and repeatability checks are more valuable than a broad VLM hyperparameter search. If no anatomical target can be validated, frame the paper as an encoder/protocol study with an exploratory VLM component. A deterministic forecast-token benefit remains a testable representation hypothesis, not an established biological result.

The active runs are useful debugging artifacts, but their adapters currently fail to reload. Continuing them cannot answer the intended trained-VLM question. I did not stop or change them during this audit. Do not interpret their partial genotype summaries: the longitudinal runs initially contain only KO subjects because folds are sorted by identifier.

**Verification limits and reproducibility.**

The existing CPU gate returned **59 passed, 1 failed, 18 deselected** in 52.93 s. The failure is `test_count_mismatch_is_a_hard_error`, a brittle source-string assertion left behind after the override refactor; current source retains an explicit count guard. It is not evidence of a new real-data count-mapping failure. The broader probe suite, full DICOM reconversion, new encoder extraction, and a full repaired VLM training run were not performed. This audit does not certify biological ground truth or every raw acquisition.

See [the codebase map](CODEBASE_MAP.md), [machine-readable symbol index](code_index.json), [artifact measurements](artifact_snapshot.json), and [test output](existing_tests.log). Reproduction commands use the existing `.venv-test` environment, restrict CPU thread counts, and keep GPUs free. Scripts only read production artifacts and write disposable checkpoints or audit outputs. Additional runs/fixes should use new output directories so these measurements remain attributable.
