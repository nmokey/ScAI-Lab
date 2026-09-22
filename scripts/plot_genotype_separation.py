"""Plot the observed training/held-out class gaps; no fitting or score correction."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / 'docs/audit_2026-09-14/stratified'
data = json.loads((root/'genotype_separation.json').read_text())
assert data['status']=='complete' and len(data['folds'])==24
fig, axes = plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax, arm in zip(axes, ('base','long')):
    for membership, color, offset, label in [('train','#4477AA',-.17,'Training mice'),('test','#CC6677',.17,'Held-out mice')]:
        values = np.array([[100*f[membership]['probability_ko_minus_wt'] for f in data['folds']
                            if f['arm']==arm and f['fold']==f'fold_{i:02d}'] for i in range(4)])
        x=np.arange(4)+offset
        ax.bar(x,values.mean(axis=1),width=.3,color=color,alpha=.7,label=label)
        for seed in range(3):
            ax.scatter(x+(seed-1)*.06,values[:,seed],s=22,c=color,edgecolor='white',linewidth=.4,zorder=3)
    ax.axhline(0,color='#444444',linewidth=.8)
    ax.set_xticks(range(4),['Fold 0','Fold 1','Fold 2','Fold 3'])
    ax.set_title('Baseline' if arm=='base' else 'Longitudinal')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('Mean KO score − mean WT score\n(probability percentage points)')
axes[1].legend(frameon=False)
fig.suptitle('The class separation learned in training often reverses on held-out mice',fontsize=12)
fig.text(.5,.01,'Bars: mean of the three seeds. Dots: individual seeds. Positive gaps have the intended KO-positive direction.\nTraining gaps are in-sample diagnostics; these are not additional performance estimates.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.1,1,.94))
for ext in ('png','pdf'):
    fig.savefig(root/f'genotype_separation.{ext}',dpi=180,bbox_inches='tight')
