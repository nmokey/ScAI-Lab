# Residual rollout development experiment

**Complete and independently verified:** all 24 forecasters, 384 forecast tokens, and 20 classifier fits finished. [Full results](RESULTS.md) and [independent verification](independent_verification.json).

Predicted-token genotype AUROC improves from 0.4516 to 0.6774; baseline is 0.6290. Adding baseline to the residual tokens gives the same aggregate AUROC and accuracy. The improvement over baseline represents three additional correctly ordered pairs out of 62, while threshold accuracy is lower (19/32 versus 21/32). The four residual fold AUROCs are 0.6875, 0.3750, 0.6667, and 1.0000. This is improved descriptive genotype ranking under the fixed readout, with substantial fold variation.

Future-embedding fidelity does not improve: subject-mean cosine is 0.9808 versus 0.9827 for the current forecaster and 0.9866 for the training-week centroid (persistence 0.9759). The residual model is below the current forecaster and centroid at every horizon. It therefore fails the predeclared advancement criterion, and this experiment does not justify a VLM rerun or a claim of improved disease forecasting. The result is consistent with retaining more useful baseline distinctions, but the bundled changes do not establish that mechanism.

The user authorized one fixed forecaster change after the corrected direct genotype probe. This experiment keeps a baseline connection in every predicted embedding and trains the complete sequence of future predictions from baseline. It uses the existing dataset and frozen four subject folds. It is a development experiment, not an additional audit repair or an untouched confirmatory evaluation.

The [protocol](PROTOCOL.json) was written before real-data fitting. Source snapshots preserve the exact training and evaluation code. The current forecaster, previous probes, and VLM runs are preserved.

## Model and fitting

Each predicted direction is the normalized sum of the normalized baseline and a learned residual. The residual network receives the preceding predicted direction, cohort, and transition step. It has the existing two hidden layers of width 512, with its final layer initialized to zero. Initial predictions therefore have the baseline direction at every horizon.

Training unrolls all three horizons without feeding observed future embeddings into the model. Available future scans supervise their corresponding predicted horizons; missing scans contribute no target loss. Each mouse has equal weight, then its available visits have equal weight. The objective is masked cosine loss plus a fixed 0.01 penalty on the squared residual norm. No genotype or PET labels enter training. The fixed budget is 300 full-batch Adam updates, learning rate 0.001, with the final checkpoint retained.

Every outer fold fits five inner forecasters for training-query tokens and one outer-training forecaster for test tokens: 24 forecasters and 384 tokens in total. Fit seeds remain 0 through 5 by fit role. All query animals are excluded from their supplying forecaster. Export magnitudes match the original forecaster's normalization statistic for the identical fit membership.

## Evaluation

The same fixed standardized logistic classifier is fitted to baseline, current future tokens, current combined tokens, residual future tokens, and residual combined tokens. The three previous input conditions must reproduce their saved predictions exactly. All five conditions are reported. Classification scaling uses training mice only; AUROC compares only KO-WT pairs within one fitted classifier.

Forecast similarity is evaluated against actual held-out future embeddings, with the current forecaster, persistence, and outer-training week centroid as comparators. All comparators use the same observed targets. The primary forecast summary averages available visits within each mouse and then averages mice. Every horizon is also reported.

The predeclared advancement criterion requires both residual genotype readouts to meet baseline AUROC and residual subject-mean forecast cosine to exceed all three comparators. It is a practical development gate, not a statistical significance test. Passing genotype alone would not establish improved forecasting, and improved genotype readability would not establish disease severity information.

## Verification and reproduction

Four focused tests passed before fitting. They cover initialization, gradients through the full rollout, masked future visits, equal subject weighting, exclusion of outer/query future scans, sensitivity to permitted training targets, genotype-name invariance within floating-point scale roundoff, checkpoint reload, and artifact corruption rejection.

```bash
CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-test/bin/python scripts/run_residual_rollout_experiment.py --device cuda
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-test/bin/python scripts/check_residual_rollout_experiment.py
```

The runner verifies the frozen protocol before resuming completed folds. The independent checker imports neither the experiment scorer nor sklearn; it reconstructs features, scaler statistics, linear scores, within-fold pair counts, forecast similarities, and aggregation from retained artifacts. Checkpoints and forecasts are stored at the separate data path recorded in the protocol.

The first launch was rejected by automatic approval review because of an account usage limit. The user subsequently requested continuation; a new GPU availability check passed and the launch was retried. Network-filesystem startup delays cleared, and the full run completed. The fitting log is `run.log`. All 24 checkpoints reproduced their exported forecasts exactly after reload. Independent verification reproduced all 20 classifier outputs (maximum numerical difference approximately 1.1e-14), their metrics, and all 67 observed-future comparisons. Previous classifier predictions and their underlying input/source hashes were preserved.
