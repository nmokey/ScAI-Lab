"""Read-only progress for the two fixed genotype VLM variants."""
import json
from pathlib import Path

ROOT = Path('/data1/Processed_NIfTI_Test/embeddings/vlm/runs')
OUTPUT = Path(__file__).resolve().parents[1] / 'docs/audit_2026-09-14/genotype_vlm_variants'
rows = []
for variant in ('direct_visual', 'genotype_only'):
    for seed in (0, 1, 2):
        run = ROOT / f'stratified_{variant}_seed{seed}'
        complete = []; active = None
        for i in range(4):
            fold = run / f'fold_{i:02d}'
            if (fold / 'complete.json').exists():
                complete.append(fold.name)
            elif fold.exists():
                states = [p/'trainer_state.json' for p in sorted(fold.glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1])) if (p/'trainer_state.json').exists()]
                state = json.loads(states[-1].read_text()) if states else {}
                active = dict(fold=fold.name, saved_epoch=state.get('epoch', 0),
                              final_saved=(fold/'mouse_vlm_mdl/vlm_config.json').exists())
        rows.append(dict(variant=variant, seed=seed, completed=len(complete), active=active))
print(json.dumps(dict(completed_fits=sum(r['completed'] for r in rows), expected_fits=24,
    final_verified=(OUTPUT/'completion.json').exists(), runs=rows), indent=2))
