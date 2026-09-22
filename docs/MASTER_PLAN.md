# Existing-pipeline validation and metrics plan

Current status updated 2026-09-21. The original 2026-09-14 scope was to **validate the existing pipeline first, within roughly one or two days, and produce defensible baseline-versus-longitudinal metrics**. Its checklist below is complete for the declared checks. Subsequent model-development experiments were separately authorized and are recorded at the end; proposed additional ablations are not authorized or launched. See [current status](STATUS.md) and [all completed results](POST_AUDIT_RESULTS.md).

Sources: [independent audit](audit_2026-09-14/RESEARCH_AUDIT.md), [codebase map](audit_2026-09-14/CODEBASE_MAP.md), and [earlier findings](FINDINGS.md).

This replaces the broader initial closure plan. Implementation status is tracked below. Checkpoint/head repairs are documented in [the repair report](audit_2026-09-14/CHECKPOINT_HEAD_REPAIRS.md); the remaining implementation repairs and validation evidence are in [the closure report](audit_2026-09-14/PIPELINE_CLOSURE.md). All six paired runs and final aggregation are complete; [verified results](results.md).

## Original repair scope and fixed decisions

Keep the existing RAD-DINO features, NaF population, Week-12 baseline input, future visits, current numerical PET target, LLaMA backbone, and baseline-versus-predicted-token question. Correct defective implementation and evaluation; do not introduce a new endpoint, modality, architecture, observed-visit task, loss search, or encoder-preprocessing experiment.

Use the accepted filename mouse-order rule: lower-left, lower-right, upper-left, upper-right, with existing tube/phantom overrides. Original ROI/histology labels are unavailable. Report the current target accurately as the existing PET intensity proxy; validating its computation is possible, establishing aortic biological validity is not possible from the available evidence alone.

Both arms must use correctly restored adapters, the same question-only head context, matched target masks, identical outer folds and seed sets, and the same training settings. The longitudinal arm additionally needs genuinely nested forecast construction. Keep the existing fixed 20-epoch setting initially; do not tune on held-out results. Retain no-genotype conditioning, baseline-only rollout, and training-fold normalization.

**Desired outcome:** a reproducible longitudinal improvement over baseline. **Acceptance criterion:** the comparison is valid and complete, regardless of which arm wins. Fixing the checkpoint may recover substantial performance, but fixes also remove leakage and affect both arms. Their effect on the difference cannot be assumed positive.

## Execution schedule and gates

These are planning windows, not promises. Benchmark the repaired pipeline and check available GPUs before scheduling. Preserve correctness requirements if work takes longer; reduce repeated runs or withhold a claim rather than label an invalid run validated.

| Window | Work | Required output before proceeding |
|---|---|---|
| First 0–4 hours | Preserve existing evidence; repair checkpoint reload and head context; run focused tests | G1: the actual quantized model preserves its predictions across save/reload and uses identical head context |
| Roughly 4–10 hours | Validate existing data/targets; implement nested forecasts; lock metrics and resume checks | G2: no outer-test dependencies; inputs and targets agree; one fold runs correctly end to end |
| Remaining day 1 / day 2 | Run matched baseline and longitudinal seeds using available GPUs | G3: complete paired fold outputs, verified checkpoints, no mixed-version aggregates |
| Final 2–4 hours | Recompute tables, assess differences, reconcile documentation | Final metrics package with evidence, limitations, and a precise validity statement |

Data checks, statistical corrections, and evidence preparation can proceed alongside model repair. This permits concurrent work but does not require additional agents. Do not start expensive full sweeps before G1/G2.

Previous 32-fold VLM runs took about five GPU-hours per arm/seed. Three paired seeds mean about **30 GPU-hours**; five mean about **50 GPU-hours**, excluding nested forecasters and repair overhead. Three available GPUs could make those training portions roughly 10 or 17 elapsed hours under comparable conditions. Actual availability, repaired inference speed, and checkpoint I/O may change that substantially.

Current evidence: [run disposition](audit_2026-09-14/run_disposition.json), [closure report](audit_2026-09-14/PIPELINE_CLOSURE.md), and [fixed protocol](audit_2026-09-14/EXPERIMENT_PROTOCOL.json). Seeds 0, 1 and 2 are the committed comparison; no extension to seeds 3/4 is planned.

## Must-do implementation checklist

### V01 — Preserve evidence and stop spending on invalid evaluation

- [x] Inspect current runs rather than rely on the audit's old progress counts.
- [x] Preserve logs, configs, predictions, and any usable adapter/head checkpoints before cleanup or stopping superseded jobs during execution.
- [x] Mark existing affected outputs as invalid for evaluating the intended trained VLM; retain them for provenance.
- [x] Put corrected runs in fresh, versioned output directories. Do not mix old folds with repaired folds.
- [x] Keep checkpoints through verification and final aggregation. Deleted weights cannot be recovered from prediction JSONs.

**Acceptance:** a compact run manifest identifies which outputs are historical, recoverable, partial, or eligible for rescoring. This plan itself does not terminate jobs.

### V02 — Fix checkpoint serialization and prove equivalence

Implemented and verified: [repair report](audit_2026-09-14/CHECKPOINT_HEAD_REPAIRS.md), [actual 4-bit round trip](audit_2026-09-14/quantized_checkpoint_fixed.json). Exact reload errors are zero; both recent recoveries load strictly.

**Files:** `vlm/model/vision_language_model.py`, `vlm/model/viz_emb_trainer.py`, `vlm/utils/huggingface_utils.py`, focused checkpoint tests.

- [x] Remove the incompatible raw-state overwrite; use supported adapter serialization and the intended base-model/adapter load structure.
- [x] Reject missing, unexpected, or incompatible adapter/head/projection state. Verify TBR normalization buffers and pooling configuration.
- [x] Turn `scripts/audit_checkpoint_roundtrip.py` into a meaningful regression check.
- [x] Test one real quantized-LLaMA checkpoint with identical inputs before save and after reload, including language logits and both heads.
- [x] Confirm trained adapter tensors survive, including nonzero B matrices, and deliberately damaged checkpoints fail loudly.

**Acceptance:** exact adapter restoration and matching evaluation-mode outputs. If quantized execution needs numerical tolerance, establish it from repeated same-model inference first and record it; do not choose tolerance to excuse a mismatched model. Verify any checkpoint-selection/resume load path actually used by the run. No full sweep until this passes.

### V03 — Match training and inference context for both heads

Implemented and verified in the repaired defaults/configurations; historical answer-pooling configs remain explicit reproduction inputs. All 96 real prefixes agree, synthetic image-signal learning survives reload, and actual-model answer perturbation changes neither head. Calibration metadata now says not assessed.

**Files:** model, dataset, trainer, baseline/longitudinal YAMLs.

- [x] Use the same question-terminal position in both arms and for genotype and regression heads. Retain normal answer-token supervision for the causal language loss.
- [x] Test image-token offsets, padding, batches, and truncation. Fail on sequences that would change pooling semantics.
- [x] For the current standard prompt, changing the supplied answer must not change evaluation-mode head features. Question-only and question-plus-answer passes must agree at the pooling position.
- [x] Test or explicitly disallow other prompt formats for this experiment; a first-EOS heuristic may select the wrong boundary in chat templates.
- [x] Verify gradients/updates and a tiny known-signal image-to-target fit through the actual training/save/reload/inference path.

**Acceptance:** the model learns the simple plumbing control, preserves it across reload, and its head predictions depend on its image inputs rather than supplied answers. This is an implementation test, not a new biological experiment or evidence of generalization. Correct the hardcoded calibration-status metadata; matching context alone is not proof of probabilistic calibration.

### V04 — Validate current data and freeze the experiment inputs

- [x] Confirm the 32 subject / 96 record population, unique `(subject, visit)` keys, finite embeddings, and exact equality of ts0 safetensors with their source RAD-DINO features.
- [x] Confirm corrected NaF crops and target artifacts are the ones the run consumes. Apply the accepted mapping/override rules; do not require unavailable external placement confirmation.
- [x] Replace the stale mouse-count source-string assertion with behavioral tests of mismatch rejection and override coverage.
- [x] Check numerical target values and missing-slot masks against the current source CSV; ensure paraphrases/question routing cannot silently change supervision.
- [x] Prefer existing structured values, or add minimal structured numeric fields, so supervision does not depend on a literal uppercase word in the question. Avoid a general dataset-framework rewrite.
- [x] Request the same fixed future horizons for every subject, with missing observations affecting supervision/scoring only. A held-out mouse's actual future attendance must not choose its inference prompt.
- [x] Compute training normalization from unique eligible training observations under a documented convention.
- [x] Save input, target, config, source-code/working-tree, tokenizer/model, and resolved environment fingerprints, including PEFT and bitsandbytes.

**Acceptance:** each subject has the same input/target contract in both arms, no future metadata alters baseline-only inference, and the run can be tied to immutable inputs. Existing proxy values remain the endpoint; this does not certify their biological meaning.

### V05 — Remove cross-stage leakage with nested forecasts

**Files:** `scripts/train_longitudinal.py`, LOSO runner, forecast loader, split/provenance tests.

- [x] Define each outer subject holdout before fitting any forecaster or scale reference.
- [x] Generate VLM training tokens through fixed inner cross-fitting on outer-training subjects only.
- [x] Generate outer-test tokens with a forecaster fitted only on the outer-training partition; roll out from that test mouse's observed Week-12 embedding.
- [x] Keep genotype conditioning disabled and never load the test mouse's observed future scans for rollout.
- [x] If inner validation selects any setting, exclude that validation subject from the transforms used to train its predictor as well. Prefer fixed settings here to avoid extra tuning layers.
- [x] Record forecaster fit membership and normalization provenance with each token artifact. Fail on missing required tokens instead of silently inserting zeros.
- [x] Cache valid fold-specific exports and reuse them across VLM seeds; keep the forecaster seed fixed and disclosed for this comparison.

Five fixed inner subject folds plus one outer-training forecaster require about **192 forecaster fits for 32 outer folds**, rather than 1,024 for inner leave-one-subject-out. This is a compute-conscious cross-fitting choice, not an optimization sweep. Check that each partition supplies the required transitions.

**Acceptance:** changing any outer-test future image or label leaves outer training tokens, normalization, and the baseline-only test forecast unchanged. Include a permitted-training-data perturbation as a positive control so the exclusion test cannot pass through disconnected code. A single global OOF forecast directory is not acceptable. This repair cannot be skipped to meet the deadline.

### V06 — Check the actual rollout and token normalization

- [x] Evaluate the exported baseline rollouts at Week 15/18/20, not only one-step predictions from observed intermediate visits.
- [x] Check shape, finiteness, per-fold scale references, and observed/predicted/projected token norm distributions.
- [x] Run inexpensive embedding-level persistence and training-week-centroid comparisons on the same eligible animals as diagnostics.
- [x] State that normalizing recurrent feedback changes later forecast directions as well as final input magnitude. Do not claim a causal explanation from the 2.7× norm ratio alone.

**Acceptance:** the VLM consumes the intended normalized, baseline-only forecasts with valid provenance. A forecast need not beat a centroid for the implementation to be correct; the paired VLM comparison still tests whether the additional representation helps. No new forecast architecture, objective search, or full centroid-token VLM arm is required in this timebox.

### V07 — Make metrics and uncertainty defensible

**Files:** `vlm/data/eval.py`, `scripts/eval_stats.py`, `scripts/aggregate_seeds.py`, statistical regression tests.

- [x] Preserve subject/slot deduplication and reject conflicting targets, missing required outputs, and NaNs.
- [x] Define the primary TBR comparison before rerunning: longitudinal minus baseline mean absolute error, averaging valid prespecified horizon errors within each mouse and then across mice. Negative difference favors longitudinal. Also reproduce the existing observation-weighted MAE for continuity, clearly labeled.
- [x] Report each horizon's MAE/MSE and ordinary R², plus errors relative to a genuinely training-fold mean predictor. Ordinary negative R² alone does not establish inferiority to that predictor.
- [x] Report genotype AUROC by seed as a secondary result; keep threshold accuracy separate and document how the threshold was set.
- [x] Pair seeds and held-out subjects across arms. Report per-seed results and paired differences; keep any explicitly defined per-subject ensemble separate. Remove the misleading rank-pooled headline, and fix tie handling wherever ranks remain.
- [x] Distinguish variability across initializations from uncertainty across animals/groups. Never treat seeds as additional animals.
- [x] Do not label fixed-score label shuffling as a full training-procedure permutation test. Include a cheap prior-only LOSO diagnostic that exposes this distinction. Label fixed-prediction resampling intervals conditional/descriptive and avoid unsupported significance language.

**Acceptance:** saved raw predictions regenerate the same tables; known numerical examples, ties, missing classes, duplicate rows, paired masks, and true training-mean comparators pass. The current NaF population has only three scanning groups and one WT group: subject-held-out scores can be reported with that dependence, but neither a narrow subject bootstrap nor more seeds establishes independent-group genotype generalization. No NaF group-holdout AUROC is manufactured by silently dropping the WT fold.

### V08 — Run only the corrected paired comparison

| Arm | Input | Shared settings |
|---|---|---|
| Baseline | Observed W12, one token | Same subjects, targets, questions, heads, pooling, backbone, epochs, folds, seeds |
| Longitudinal | Observed W12 plus three nested normalized forecasts | Same shared settings |

- [x] Run one fold as an end-to-end correctness check before launching full CV. Its score is not a basis for choosing settings or subjects.
- [x] Freeze the run specification and commands. Start with seeds 0, 1, 2 for **both** arms; add seeds 3 and 4 only if the measured runtime and deadline allow completing both arms. Decide that extension from resources, not whether initial results look favorable.
- [x] Run matching folds/settings for all selected seeds; repeat failed folds for technical reasons, not to replace low scores.
- [x] Make writes atomic and resumes idempotent. Check expected `(fold, pid, qid)` coverage and reject mismatched fingerprints.
- [x] Aggregate only complete matched arms/seeds; a partially completed extra seed is not a final result.
- Not needed: the single-pair deadline fallback was not used; all three prescribed pairs completed.

**Acceptance:** G1/G2 remain true for every consumed artifact; full paired prediction files and verification logs exist. The corrected result supports a claim about adding the predicted-token input to this baseline on this dataset. Without token-count/centroid VLM controls, do not claim the gain proves personalized future modeling or identifies its mechanism. Those additional research arms are outside the current request.

### V09 — Final metrics package and documentation

- [x] Produce one authoritative baseline/longitudinal table with per-seed values, paired differences, per-horizon outcomes, population/missingness counts, and relevant descriptive uncertainty.
- [x] Include an explicit check against training-fold mean predictions and a short error analysis. Avoid selecting only winning horizons, seeds, or metrics.
- [x] Reconcile README, findings, results, experiment log, methods notes, and misleading YAML comments. Retire hardcoded legacy VLM summaries from the publication path.
- [x] Run the relevant CPU gate and real-data checks, including retained leakage, determinism, splicing, routing, and target-mask guards. Record actual pass/fail results and remaining exclusions.
- [x] Save reproducible scoring commands, environment/config fingerprints, and links from every table to its prediction file.

**Acceptance:** the package identifies exactly which behavior was validated and which claims remain limited. The audit's old scores are never presented as the corrected model's metrics. Improvement is reported if supported; an inconclusive or negative difference is reported just as clearly. Failure of a critical gate is stated instead of certifying the pipeline.

## Disposition of every surfaced finding

This table distinguishes issues that must be repaired before VLM metrics are usable from issues that can be handled by limiting what those metrics claim. Deferred components are not falsely marked validated.

| Independent audit finding | Treatment in this timebox | Closure evidence / restriction |
|---|---|---|
| §1 Adapter reload failure | Must fix: V01–V03 | Actual quantized-model checkpoint equivalence |
| §2 Non-nested MLP/VLM splits | Must fix: V05 | Fold membership plus held-out perturbation tests |
| §3 Three NaF groups / one WT group | Document; retain the existing subject-level question | No independent-group genotype generalization claim; unavailable classes never silently skipped |
| §4 Answer-conditioned heads | Must fix: V03 | Same answer-independent question context in both arms |
| §5 Unvalidated PET target | Verify current computation: V04; narrow naming/claims | Existing proxy forecasting only; no new biological endpoint or claim of aortic validation |
| §6 TBR-3 physical mapping / mixed acquisitions | Physical-coordinate repair implemented and tested | Anatomy and mixed-acquisition alignment remain unvalidated; TBR-3 excluded from the VLM endpoint |
| §7 Rollout, normalization, centroid comparison | Validate plumbing: V05/V06; evaluate original two-arm question | Actual rollout diagnostics; no new forecaster search or individualized-mechanism claim |
| §8 Human-scale encoder preprocessing | Outside the fixed RAD-DINO VLM comparison; correct methods/framing | Existing off-the-shelf pipelines only; no architectural superiority claim or new preprocessing experiments |
| §9 Stale Merlin cache | Cache versioning repaired; all 229 embeddings regenerated | Fresh/cached equality and invalidation checks passed; refreshed metrics complete, with exact full-refit genotype inference |
| §10 Statistics, R² comparator, rank pooling | Must fix: V07/V09 | Paired subject results, actual mean baseline, explicit uncertainty estimand |
| Resume / partial aggregates | Must fix: V01/V08 | Atomic, fingerprint-checked, complete unique fold coverage |
| Target parsing / routing / future attendance | Must fix: V04 | Target invariance, correct masks, fixed-horizon inference |
| Stale count test / identity assumption | Must fix behavioral guard; retain agreed heuristic | Tested mismatch/override behavior and documented mapping rule |
| Environment, legacy defaults, conflicting docs | V04/V09 | Recorded environment and a single authoritative metrics ledger |
| Surviving RAD-DINO encoder association | Existing audit evidence; no new research needed | Report as genotype/diet association with selection/confounding caveats, not disease localization |

For the optional-in-publication Merlin repair, the acceptance check is that fresh and cached preprocessing agree on corrected crops, changing the source at the same path invalidates the cache, and downstream metrics point to the regenerated features. This is a repair of existing results, not a new encoder experiment. Do not delay the primary VLM path for it unless the four-encoder table is also needed within the deadline.

Carry forward earlier F1–F24 guards without treating them as all freshly reverified: checkpoint selection, seed consumption, image splicing, truncation, no genotype conditioning, no observed-future inputs, deduplication, and routing must still pass on the final path. Refuted findings remain refuted unless new evidence changes them. Update historical statements about calibration and “nothing to forecast from” to match the new evidence.

## Final definition of done

Known critical implementation defects on the declared path have repair and targeted-check evidence, and metrics are reproducible from complete paired outputs. This is not an exhaustive proof of correctness or a declaration that all methodological issues are closed. Confounding, repeated use of development folds, proxy validity, conditional mouse identities, training/test forecast-representation differences, and missing attribution controls remain explicit. A validated disease endpoint, new independent animals, extra modalities, extra observed visits, or a guaranteed positive difference are not promised by this two-day plan.

## Final closure — 2026-09-14

All V01–V09 implementation checks completed for the original fixed protocol. V07 was subsequently reopened after the user's below-chance concern: pooled LOSO AUROC can be severely biased by fold-specific scoring differences. The saved-model follow-up confirmed reversed cross-fold rankings even for identical inputs, leading to the completed evaluation amendment below. See [the diagnosis and closure rationale](audit_2026-09-14/GENOTYPE_EVALUATION_FOLLOWUP.md). The original 192 VLM folds finished; all retained checkpoints, source/input fingerprints and 576 prediction records passed aggregation checks. The original final CPU gate passed 114 tests, with 18 exploratory/real-data tests excluded and separate actual-data/model checks documented. Original artifacts are preserved.

The user authorized the fixed four-fold amendment. Its implementation and leakage/scoring checks passed; all 24 nested forecaster fits completed and all 384 exported tokens passed real-input checks. **All 24 VLM fits and both scoring implementations are complete. V07's identified cross-model AUROC pooling defect is closed by the amended within-fold estimator.** Genotype performance remains poor, with no consistent longitudinal gain. Direct diagnostics of all 24 checkpoints find the intended training class direction but frequent held-out reversals; this is a remaining model-performance limitation, not a claim that every possible data or optimization issue has been excluded. See [the verified amendment](audit_2026-09-14/stratified/README.md) and [the mechanism investigation](audit_2026-09-14/stratified/GENOTYPE_MECHANISM.md).

The authorized real-data capacity control is also complete: both objectives learned the same eight existing training mice, with image dependence and exact matched-precision reload verified. The existing combined objective can learn this subset; no loss-weight or architecture change is justified by a claim that it fundamentally prevents learning. This does not establish that the full-data schedule is adequate or that longer training improves held-out results. [Control results](audit_2026-09-14/real_mouse_learning_control/RESULTS.md).

The primary result does not favor longitudinal: seed-mean subject MAE 5.6494 versus baseline 5.6112, paired difference +0.0381. W18 shows a small secondary MAE improvement in all three seeds, on 15 observed mice. Both arms are close to the training-fold mean predictor. The verified tables are in [results.md](results.md). Biological target validity and independent-acquisition genotype generalization remain claim restrictions, not falsely closed data problems.

## Subsequent authorized model-development experiment

The fixed residual full-rollout experiment is complete, separately from audit closure. All 24 forecasters, 384 tokens, and 20 classifier fits passed artifact and independent-metric checks. Residual-token genotype AUROC improves to 0.6774 versus baseline 0.6290 and original forecast tokens 0.4516, but threshold accuracy is lower than baseline and future-embedding fidelity is worse than the original forecaster and training-week centroid. The predeclared advancement criterion fails. No VLM training was included in that experiment; a subsequent authorized follow-up is recorded below. No auxiliary genotype supervision was added to the forecaster. [Results and interpretation](audit_2026-09-14/residual_rollout_experiment/README.md).

The user subsequently explicitly authorized running the residual tokens through the VLM despite the earlier advancement criterion. This separate development comparison is complete: all 12 new fits finished, and all nine runs passed artifact and independent-scoring checks. Residual VLM genotype AUROC is 0.5484 in all three seeds, above both VLM comparators each time, but accuracy remains below the always-WT comparator. Mean primary proxy MAE is 5.6564 versus baseline 5.5506; no primary gain is established. Week-20 proxy MAE improves in all three seeds (20 mice, secondary descriptive result). Training settings, supervision, and metrics remained fixed. [Results and interpretation](audit_2026-09-14/residual_vlm/README.md).

### Fixed genotype VLM variants — complete

The user authorized two further comparisons: add direct image access to the genotype head, or retain the original architecture with genotype-only supervision. Both used the completed residual tokens and exactly the same four folds, three seeds, 20 epochs, optimizer, and fixed classification threshold. The two variants are separate experiments; neither combines both changes. All 24 fits and final-checkpoint inference reloads completed, and independent scoring reproduced all 12 new/comparator runs. Training-only normalization and matching shared initial parameters passed verification. Six focused tests and actual quantized-backbone objective, gradient, answer-invariance, and exact-reload checks passed; input/source preflight passed before launch. Queues ran on GPUs 1 and 6, with the prescribed genotype-only seed 2 parallelized on GPU 4. [Frozen protocol and evidence](audit_2026-09-14/genotype_vlm_variants/README.md).

Direct-visual mean AUROC improves to 0.6882 versus residual 0.5484, but accuracy remains 0.5000 in all three seeds. Genotype-only mean AUROC is 0.5806 and accuracy 0.5208, with inconsistent ranking gains. Both predict WT for all mice in two folds in every seed; the direct-visual model ranks fold 03 perfectly while keeping every score below threshold. Fold 01 still has below-chance ranking. The remaining performance problem involves both score offsets and generalization, and 70% accuracy has not been achieved. Genotype-only proxy predictions are unsupervised and excluded from performance claims. Any gain from direct image access does not by itself establish a benefit from longitudinal tokens: a matched baseline-input version of that architecture has not been run. No additional warmup, threshold tuning, forecaster variants, or parameter search was performed.

## Proposed controls — not launched

Repeated-baseline and training-derived mean-future VLM controls were recommended to test token-count and generic-future explanations. A matched baseline-input hybrid is needed to isolate a forecast benefit in that architecture. An isolated residual-connection ablation has also not been run: the completed residual experiment changed anchoring, rollout training, initialization and regularization together. These remain recommendations, not authorized jobs. See [status and scope](STATUS.md).
