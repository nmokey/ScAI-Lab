"""Read-only compact progress for the three queued residual VLM runs."""
import json
from pathlib import Path

ROOT = Path('/data1/Processed_NIfTI_Test/embeddings/vlm/runs')
REPORT = Path(__file__).resolve().parents[1] / 'docs/audit_2026-09-14/residual_vlm'
rows = []
for seed in (0, 1, 2):
    run = ROOT / f'stratified_residual_seed{seed}'
    complete = []
    active = None
    for i in range(4):
        fold = run / f'fold_{i:02d}'
        if (fold / 'complete.json').exists():
            complete.append(fold.name)
        elif fold.exists():
            checkpoints = sorted(fold.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
            states = [p / 'trainer_state.json' for p in checkpoints if (p / 'trainer_state.json').exists()]
            state = json.loads(states[-1].read_text()) if states else {}
            active = dict(fold=fold.name, saved_update=state.get('global_step', 0),
                          saved_epoch=state.get('epoch', 0), final_model_saved=(fold / 'mouse_vlm_mdl/vlm_config.json').exists())
    rows.append(dict(seed=seed, complete_folds=complete, active=active))
print(json.dumps(dict(completed_fits=sum(len(r['complete_folds']) for r in rows), expected_fits=12,
                      final_verified=(REPORT / 'completion.json').exists(), runs=rows), indent=2))
