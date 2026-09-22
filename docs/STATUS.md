# Current research status — 2026-09-21

This is the current entry point for research status, interpretation, and outstanding work. The [full post-audit result ledger](POST_AUDIT_RESULTS.md) contains all completed result families; [methods](design_decisions.md) distinguish their protocols. This update changes documentation, not models, predictions, thresholds, or experiments.

## What validation establishes

Known critical implementation defects were repaired, and the declared computations passed targeted verification. Checkpoint restoration, answer-independent heads, nested forecast exclusions, structured targets/masks, coverage, and independently reproduced scoring have direct evidence. The original CPU gate passed 114 tests, with additional focused and actual-data/backbone checks recorded separately. The current pre-commit CPU gate on 2026-09-21 passed 135 tests; 18 probe/real-data cases were deselected ([verification record](../tests/README.md#current-pre-commit-gate-2026-09-21)). These are completed checks, not a claim that every possible test has been run or that the pipeline is provably bug-free.

Remaining scientific and evaluation limitations are unresolved: genotype/diet/acquisition confounding; only three NaF acquisition components and one WT component; repeated development on the same 32 mice and folds; no source aortic ROI or histology labels; identities conditional on the accepted filename-position heuristic and overrides; and possible distribution differences between inner-cross-fitted training tokens and outer-fit test tokens. The PET endpoint is a computationally reproduced distributional proxy, not validated aortic disease severity.

## Completed experiments and the claims they support

| Experiment | Completed evidence | Current interpretation |
|---|---|---|
| Original repaired LOSO VLM | 192 fits; baseline/longitudinal mean proxy MAE 5.6112/5.6494 | No primary gain; secondary Week-18 gain. Its pooled genotype AUROC is superseded as a discrimination estimate. |
| Four-fold evaluation amendment | 24 VLM fits; genotype AUROC compares only within-model pairs | Baseline/original-longitudinal mean AUROC 0.4785/0.4409; no consistent original-longitudinal gain. |
| Real-mouse learning control | Both objectives learn 8/8 training mice; exact reload and image dependence | Learning plumbing works for that subset; this is not held-out performance. |
| Direct embedding classification | Fixed classifier: baseline AUROC 0.6290, original futures 0.4516, residual futures 0.6774 | Residual ranking improves modestly; accuracy is lower than baseline (19/32 versus 21/32). |
| Residual forecaster | 24 fits, 384 tokens; genotype/fidelity gate failed | Better genotype ranking does not establish better forecasting: future cosine is worse than original and training-week centroid. |
| Residual-input VLM | 12 additional fits, unchanged VLM architecture; mean AUROC 0.5484, accuracy 52.08% | Modest ranking gain in all seeds; no overall proxy gain. Secondary Week-20 MAE improves about 5%. |
| Direct-visual hybrid and genotype-only VLM | 24 additional fits; exact final reloads and independent scoring | Hybrid AUROC 0.6882/accuracy 50%; genotype-only 0.5806/52.08%. No 70% accuracy result. |

There are 252 completed research VLM fits in these comparisons, excluding capacity and smoke checks. The completed experiments are not a set of independent replications: seeds and follow-up models reuse the same animals.

“Longitudinal encoder performance” can refer to two different measurements: direct genotype classification of its embeddings, or fidelity against observed future embeddings. The VLM and PET-proxy outcomes are downstream evaluations and must be named separately.

“Direct image access” means a linear classifier on the same four standardized embeddings whose logit is added to the VLM genotype logit. It introduces no observed follow-up image. This is a hybrid VLM/classifier, and its direct branch retains nearly all combined ranking performance (mean AUROC 0.6828 versus 0.6882). The result does not establish improved language-model reasoning or isolate a longitudinal benefit in that architecture.

## Ablations and controls

Completed: baseline versus forecast-token inputs; baseline/future-only/combined direct classifiers; combined versus genotype-only objectives; adding the direct visual branch; and evaluation-time removal of hybrid components. The latter is a component diagnostic, not separately retrained ablation. The residual forecaster changed several ingredients together, so it does not isolate one ingredient's effect.

The following controls were recommended in discussion but have **not been implemented, launched, or authorized as new runs**:

| Proposed control | Question it would address |
|---|---|
| Original VLM with four copies of baseline | Whether extra token count/repeated baseline information explains the forecast-token gain |
| Original VLM with baseline plus training-derived mean futures | Whether personalized forecasts help beyond generic future representations |
| Matched baseline-input hybrid | Whether forecast inputs benefit the hybrid architecture |
| Isolated residual-connection comparison | Whether baseline anchoring itself explains the residual bundle's effect |

The first two would add 24 fits at four folds × three seeds × two arms. Training-derived controls must respect the existing nested exclusions. No threshold calibration, additional warmup, or parameter search has been performed. Documentation of these proposals is not authorization to launch them.

## Reading the documentation

| Document | Role |
|---|---|
| [Post-audit results](POST_AUDIT_RESULTS.md) | Consolidated numeric ledger with protocol distinctions and source fingerprints |
| [Detailed results](results.md) | Original repaired study plus subsequent development tables |
| [Methods](design_decisions.md) | Current protocol definitions and claim limits |
| [Master plan](MASTER_PLAN.md) | Original completed repair checklist, authorized follow-ups, and outstanding proposals |
| [Experiment log](experiments.md) | Chronological execution and reproduction record |
| [Current code navigation](CODEBASE_CURRENT.md) | Entry points for each completed protocol |
| [Historical evidence guide](audit_2026-09-14/README.md) | Scope of the frozen audit and experiment reports |
| [Tests](../tests/README.md) | Recorded checks, environment guidance, and exclusions |

Archived findings, audit-time code maps, and frozen experiment reports are dated evidence. They are not continuously rewritten as later work finishes. In particular, the residual forecaster's frozen “no VLM rerun” statement describes its own completion; the separately authorized residual VLM study subsequently ran and completed. Read this page and the result ledger for the latest state.
