# Real-mouse learning control — training diagnostics only

Eight existing training mice (four KO, four WT), 24 question records, the longitudinal inputs, seed 0, identical initial parameters and 300 updates per objective. No held-out performance was measured. Both final checkpoints reproduce all control logits exactly after strict reload.

The control changes only the objective: the combined run uses the existing language + proxy + genotype losses; genotype-only retains the same weighted genotype loss and removes the other two. Both use the same batches, learning rate, schedule and extended training budget.

| Objective | Correct genotype labels | Mean genotype BCE | Lowest true-class score | Capacity criterion met |
|---|---:|---:|---:|---|
| combined | 8/8 | 0.000120 | 0.9997 | True |
| genotype_only | 8/8 | 0.000001 | 1.0000 | True |

The declared capacity criterion is all eight genotype-question labels correct at a zero-logit threshold and mean BCE ≤ 0.1. It is a debugging criterion, not a statistical test.

| Update | Combined correct / 8 | Combined BCE | Genotype-only correct / 8 | Genotype-only BCE |
|---|---:|---:|---:|---:|
| 0 | 4 | 0.927497 | 4 | 0.927497 |
| 30 | 5 | 0.657441 | 4 | 0.691742 |
| 60 | 6 | 0.389476 | 5 | 0.621396 |
| 120 | 8 | 0.029601 | 6 | 0.554426 |
| 180 | 8 | 0.000639 | 8 | 0.000738 |
| 240 | 8 | 0.000137 | 8 | 0.000001 |
| 300 | 8 | 0.000120 | 8 | 0.000001 |

Both objectives can learn this small set of actual image/label pairs. The combined objective does not prevent memorization under this extended budget. This weakens a gross training-plumbing failure as an explanation of the full-study result, but does not establish adequate optimization or generalization on the full dataset.

Passing this extended training-only control does not show that increasing the full-study epoch count would improve held-out performance.

The subset was fixed by sorted subject order before fitting. Its KO mice all belong to one acquisition component, so successful fitting could use acquisition-related cues. The controls establish only the ability to learn these selected labels. The original held-out scores, model checkpoints and training settings remain unchanged.

The 300-update budget corresponds to 100 passes through this tiny subset. The measurements at 60 updates correspond to 20 subset epochs, but the learning-rate schedule is defined over the full 300 updates; they are not a reproduction of the original 20-epoch experiment.

[Protocol](PROTOCOL.json), [input fingerprints](frozen_inputs.json), [combined report](combined/report.json), [genotype-only report](genotype_only/report.json). Each report includes learning curves, gradient checks, parameter changes and exact checkpoint-reload results.

## Measurement precision check

The initial reload assertion compared the Trainer’s automatic BF16 forward/FP32 readout with plain inference precision. It therefore stopped after both models had already finished all 300 updates and saved their final weights. The control probe now makes its precision explicit. Replaying the saved models at the original probe precision reproduces every recorded endpoint logit exactly; their tensors also equal the checkpoint at update 300. No training was repeated or model weights changed. Ordinary inference separately retains 8/8 correct labels in both controls.

[Precision repair evidence](precision_repair.json) preserves the measured discrepancy and checks. Original unfinished reports and execution source are retained; `run_real_mouse_learning_control_source.py` is the source corresponding to the launch-time fingerprint. The current script includes the measurement correction.

## Image dependence

Swapping the mouse image-token inputs makes the outputs follow the donor images, while leaving questions, supplied answer tokens, masks and labels unchanged. Giving every mouse identical image tokens removes between-mouse genotype-score differences. This confirms that the learned distinction depends on the image inputs. [Checks](image_dependence.json).
