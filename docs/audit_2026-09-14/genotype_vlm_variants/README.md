# Fixed genotype VLM comparisons

**Complete:** all 24 new fits and exact final-checkpoint reloads passed; independent scoring reproduced all 12 new/comparator runs. [Full per-seed results](RESULTS.md), [verification](completion.json), and [descriptive score diagnostics](score_diagnostics.json).

| Model | Mean within-fold AUROC | Mean accuracy | Mean balanced accuracy |
|---|---:|---:|---:|
| Baseline VLM | 0.4785 | 0.5000 | 0.4603 |
| Residual-input VLM | 0.5484 | 0.5208 | 0.4868 |
| Residual + direct visual head | 0.6882 | 0.5000 | 0.4815 |
| Residual + genotype-only loss | 0.5806 | 0.5208 | 0.4974 |

Direct visual access improves AUROC in every seed (0.6935, 0.6774, 0.6935), but all three classify only 16/32 mice correctly. Genotype-only scores 0.6613, 0.5484, 0.5323 AUROC with 17/32, 16/32, 17/32 correct; its advantage is inconsistent and it has no accuracy gain over the residual VLM. Neither reaches the requested accuracy aspiration, and both remain below always-WT accuracy (18/32). The fixed residual-embedding logistic classifier alone remains competitive: AUROC 0.6774, accuracy 19/32. The direct-visual component of the jointly trained model scores AUROC 0.6774, 0.6774, 0.6935, so the combined VLM does not demonstrate a substantial additional ranking benefit over that component.

The saved scores clarify the accuracy problem. Both new variants predict WT for every mouse in folds 00 and 03 in every seed. The direct-visual model nevertheless achieves AUROC 1.0 in fold 03 each time: all KO scores rank above all WT scores, but all remain below the decision threshold (maximum KO probability 0.0616–0.0771 across seeds). Fold 01 still ranks classes poorly, with AUROC 0.3125–0.375. Thus threshold placement/score offsets and weak generalization both remain concerns; this is not evidence that threshold tuning alone would recover 70% accuracy.

Only 6/32 direct-visual scores per seed lie in 0.4–0.6, while 19–21/32 have raw confidence at least 0.9. Mean confidence on mistakes is about 0.83; genotype-only is about 0.85–0.93. These are raw sigmoid scores, not validated probabilities. The variants have not simply resolved cautious guessing: they also make confident mistakes. These post-run descriptions change no predictions or thresholds and are reproducible with `scripts/describe_genotype_variant_scores.py`.

Direct-visual mean proxy MAE is 5.5550 versus residual 5.6564, baseline 5.5506, and training mean 5.6106; it does not establish a gain over baseline. Genotype-only proxy outputs are unsupervised and excluded. These remain descriptive results from repeatedly inspected folds with acquisition/diet confounding.

The user authorized two comparisons against the completed residual-input VLM: direct visual access for the genotype head, and genotype-only supervision. Both retain the existing residual forecast tokens, three seeds, four subject folds, fixed final 20-epoch budget, optimizer, batching, and inference threshold.

Direct visual access adds a zero-initialized linear score from the concatenated four image embeddings to the existing language-state genotype score. Per-coordinate standardization uses the 24 unique training mice only. The combined language, proxy, and genotype objective remains unchanged.

Genotype-only supervision retains the original architecture and genotype loss weight but removes language-generation and proxy-regression losses. Its proxy head is unsupervised; any saved proxy outputs are diagnostic artifacts and will not be presented as predictive performance.

Variant code is isolated in new scripts; existing model sources and prior run fingerprints remain intact. Variant checkpoints have a distinct version and strict state/normalization checks. The direct-embedding classifier remains a comparator, and direct-visual branch components are retained to distinguish the visual contribution from the language-state contribution.

Both queues started on 2026-09-16, on GPUs 1 and 6, and completed successfully. Six focused tests passed. [Actual-backbone checks](runtime_checks.json) verified loss semantics, gradients, answer invariance, and exact checkpoint reload for both variants. [Preflight](preflight.json) verified the existing inputs and settings before launch. [Protocol](PROTOCOL.json) and [source snapshots](source_snapshot/) remain frozen.

The prescribed genotype-only seed 2 also ran on free GPU 4, using the same runner/configuration and its separate seed directory. The original queue subsequently verified and skipped that completed run. This changed concurrency only, not the number of fits or the protocol.

The final comparison includes all 24 new fits and independent metric verification. These experiments test changes to the VLM on fixed residual inputs. They do not isolate a longitudinal-token benefit within the new architectures, which would require matched baseline-input variants. Accuracy near 70% was an aspiration, not an acceptance criterion or guaranteed outcome.

Reproduction from the repository root (select two available GPUs):

```bash
CUDA_VISIBLE_DEVICES='' .venv-test/bin/python scripts/evaluate_genotype_variants.py --prepare-only
bash scripts/run_genotype_variant_worker.sh 1 direct_visual
bash scripts/run_genotype_variant_worker.sh 6 genotype_only
.venv-test/bin/python scripts/report_genotype_variant_progress.py
CUDA_VISIBLE_DEVICES='' .venv-test/bin/python scripts/evaluate_genotype_variants.py
```

Run the two worker commands in separate persistent sessions. Each worker resumes its own incomplete folds and validates completed artifacts before skipping them. The second completed queue automatically runs the final evaluator. `completion.json` is written only after the full comparison passes; progress counts alone are not final verification.

The final evaluator checks run/source identities, fold prediction coverage and hashes, the 180-update budget, exact final inference reload, matching initial shared parameters between variants, and direct-visual normalization against the independently saved embedding-probe statistics. A separate standard-library scorer reproduces genotype pair counts, accuracy, and supervised proxy arithmetic. Unsupervised genotype-only proxy arithmetic is checked solely for artifact consistency.

The direct-visual model's language-state component still receives image tokens through the language model; it is not a text-only control. Its separate visual component accesses the original embeddings before language projection. Reported classification scores come from the supervised head, not generated answer text. Removing generation loss in the genotype-only variant therefore changes training but does not change how the classification metric is obtained.
