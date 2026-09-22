# Training-only control on existing mice

**Both controls completed all 300 updates and met the capacity criterion.** The combined objective passed by the measurement at update 120; genotype-only passed by update 180. See [results](RESULTS.md). The final diagnostic reload check required an explicit precision correction, documented below and in the results.

The user authorized a real-data learning control using only the current dataset. “Real-data” means actual mouse embeddings and existing labels, not additional acquisitions or annotations.

The [frozen protocol](PROTOCOL.json) selects the first four KO and first four WT IDs from the amended fold-0 training population. It uses the same 24 records and longitudinal tokens in both controls. The selected subjects are KO 01/02/03/04 and WT 02/03/04/07. The KO mice are all in one acquisition component; a successful capacity check would not establish genotype learning independent of acquisition cues.

Two fresh models use seed 0 and identical initial parameters. One uses the current combined language/proxy/genotype objective. The other removes only the language and proxy losses, retaining genotype supervision on the same three question types with the original weight of 5. Both retain the production model, projection, adapters, head boundary, batch size, accumulation, optimizer and learning rate. Both receive 300 updates (100 passes through the eight-mouse subset) to deliberately test capacity. No held-out model selection or performance evaluation is performed.

Accuracy and loss are measured on these training mice at updates 0, 30, 60, 120, 180, 240 and 300. The declared capacity criterion is 8/8 genotype-question classifications correct at zero logit and mean genotype BCE ≤ 0.1. Outcomes are interpreted as diagnostics under this one budget; a failed criterion does not automatically prove a code bug, and a pass does not establish generalization.

Inputs and forecast exclusions are verified in [frozen_inputs.json](frozen_inputs.json). The script checks the intended objective numerically before training, records gradients and actual parameter changes, and strictly reloads the final models to verify identical logits. The generic production loader emits a message describing its inference-set fallback as a test set; in these controls both configured data paths point to the same training-only subset, and automatic evaluation is disabled.

Execution: sessions `scai_capacity_combined` (GPU 1) and `scai_capacity_genotype` (GPU 6), with logs `combined.log` and `genotype_only.log`. Models and reports are saved in the corresponding subdirectories. These outputs are separate from both completed research evaluations.

After both controls finish, run:

```bash
.venv-test/bin/python scripts/summarize_real_mouse_learning_control.py --directory docs/audit_2026-09-14/real_mouse_learning_control
```

The summary independently checks accuracy and BCE from all saved training predictions, verifies matching initialization and target statistics, and writes `RESULTS.md` and `summary.json`.

The original probe relied on Accelerate's automatic mixed precision during training, then used plain precision after reload. This caused a numerical comparison failure after the models had completed training and saved. `finalize_learning_control_precision.py` verified final tensors against checkpoint 300 and reproduced the saved learning-curve endpoints exactly at matched precision, without any further training or parameter changes. `precision_repair.json` records the check. The source used during fitting is preserved as `run_real_mouse_learning_control_source.py` and matches the launch-time source fingerprint; the current runnable script explicitly aligns probe precision. Data/config/input fingerprints remain unchanged.
