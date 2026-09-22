# Audit evidence and document scope

Updated 2026-09-21. Start with [current status](../STATUS.md), [all results](../POST_AUDIT_RESULTS.md), and [current code navigation](../CODEBASE_CURRENT.md). This directory preserves dated evidence from several stages; an older report is not a claim about every later experiment.

| Evidence | Scope and later developments |
|---|---|
| [Initial research audit](RESEARCH_AUDIT.md), [initial code map](CODEBASE_MAP.md) | Pre-repair snapshot; the adapter and other defects described there were subsequently repaired. Live-session counts are historical. |
| [Checkpoint/head repairs](CHECKPOINT_HEAD_REPAIRS.md) | First repair stage; later full-pipeline and model checks are separate. |
| [Original closure](PIPELINE_CLOSURE.md), [original interpretation](FINAL_INTERPRETATION.md), [paired study](paired_vlm_results.md) | Original LOSO study and amendment notices. Broad “validated” wording means targeted checks of that path, not proof of bug-free computation or complete methodological validity. |
| [Genotype follow-up](GENOTYPE_EVALUATION_FOLLOWUP.md), [four-fold results](stratified/RESULTS.md) | Pooled-LOSO problem and completed evaluation amendment. Current genotype reporting uses within-model comparisons. |
| [Mechanism report at completion](stratified/GENOTYPE_MECHANISM.md) | Frozen report matching diagnostic_completion.json. |
| [Mechanism report with later notes](stratified/GENOTYPE_MECHANISM_FOLLOWUPS.md) | Preserves the subsequently added capacity/probe paragraphs separately from the frozen version. |
| [Learning control](real_mouse_learning_control/RESULTS.md) | Training-only learning of eight mice; no generalization claim. |
| [Raw embedding probe](stratified/embedding_probe/RESULTS.md) | Baseline/original future representations, fixed logistic classifier. |
| [Residual forecaster](residual_rollout_experiment/README.md) | Completed encoder experiment; its frozen “no VLM rerun” statement describes its completion time. A later VLM run was separately authorized and completed. |
| [Residual VLM](residual_vlm/README.md) | That subsequent downstream VLM comparison. |
| [Genotype variants](genotype_vlm_variants/README.md) | Completed direct-visual hybrid and genotype-only models. Neither demonstrates 70% accuracy. |

The proposed repeated-baseline, mean-future VLM, matched baseline-input hybrid, and isolated residual-connection controls have not been run. Current limitations include acquisition/diet confounding, repeated development folds, unavailable biological labels, conditional identities, and possible forecast representation shifts.

Frozen RESULTS/protocol/completion files and source snapshots remain evidence of their original stage. Updates go into current guides or separately named addenda. The 2026-09-21 documentation reconciliation restored the mechanism report to its exact completion hash and retained the annotated version separately; no predictions, model weights, metrics, or original completion manifest were changed.

Model weights, optimizer checkpoints, and machine configuration are local artifacts and are not shipped in Git. Their provenance and verification reports remain available here; full reproduction additionally requires the recorded data/model paths on the research machine.
