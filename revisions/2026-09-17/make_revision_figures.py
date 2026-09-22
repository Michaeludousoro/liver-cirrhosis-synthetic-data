"""Reproduce revision figures from the saved real training partition only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'paper' / 'figures'
df = pd.read_csv(ROOT / 'output/data/train_real.csv')
assert len(df) == 193 and int(df.Status.sum()) == 78
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300})
rng = np.random.default_rng(42)
fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.6), constrained_layout=True)
for ax, (col, label, log) in zip(axs, [('Bilirubin', 'Bilirubin (mg/dL)', True),
                                      ('Albumin', 'Albumin (g/dL)', False),
                                      ('Prothrombin', 'Prothrombin time (s)', False)]):
    for pos, status, color in [(0, 0, '#2166ac'), (1, 1, '#b35806')]:
        vals = df.loc[df.Status == status, col].to_numpy(float)
        transformed = np.log10(vals) if log else vals
        grid = np.linspace(transformed.min(), transformed.max(), 220)
        density = gaussian_kde(transformed)(grid)
        y = 10 ** grid if log else grid
        ax.fill_betweenx(y, pos + .06, pos + .06 + .32 * density / density.max(),
                         facecolor=color, alpha=.35, linewidth=.8, edgecolor=color)
        ax.scatter(pos - .15 + rng.uniform(-.065, .065, len(vals)), vals,
                   s=9, color=color, alpha=.46, edgecolors='none')
        ax.boxplot([vals], positions=[pos], widths=.09, showfliers=False,
                   patch_artist=True, boxprops={'facecolor':'white', 'edgecolor':color},
                   medianprops={'color':'black'}, whiskerprops={'color':color},
                   capprops={'color':color})
    if log:
        ax.set_yscale('log')
    ax.set_xticks([0, 1], ['Censored /\ntransplanted\n(n = 115)', 'Deceased\n(n = 78)'])
    ax.set_ylabel(label)
    ax.grid(axis='y', alpha=.16)
    ax.set_xlim(-.35, 1.45)
fig.savefig(OUT / 'fig13_training_raincloud.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.5)
ax.axis('off')
def box(x, y, w, h, text, face='#e9f0f6', size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                               facecolor=face, edgecolor='#536779', linewidth=1))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=size,
            linespacing=1.45, color='#172a3a')
def arrow(x1,y1,x2,y2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops={'arrowstyle':'-|>', 'color':'#536779', 'lw':1.3})
box(.15, 4.15, 2.25, 1.0, '418 PBC records\n276 complete cases')
box(3.05, 4.15, 2.15, 1.0, '193 training patients\nFit preprocessing')
box(7.2, 4.15, 3.45, 1.0, '83 held-out real patients\nEvaluation only', '#e5f1e8')
arrow(2.48,4.65,2.98,4.65)
ax.plot([1.275, 1.275, 8.925], [5.22, 5.4, 5.4], color='#536779', lw=1.3)
arrow(8.925,5.4,8.925,5.22)
ax.text(5.8,5.27,'Fixed train / test split',ha='center',fontsize=8)
box(.15, 2.55, 2.5, 1.0, 'Core: GAN / cGAN / VAE\nAdditional: CTGAN\nand masked-loss VAE', size=9.5)
box(3.05, 2.55, 2.15, 1.0, 'Training-derived\nIQR filtering')
box(5.8, 2.55, 2.15, 1.0, 'Consensus from\nthree core pools')
box(8.55, 2.55, 2.1, 1.0, 'Real-only / augmented\n/ synthetic-only\nSMOTE comparator', size=9)
arrow(3.8,4.08,1.4,3.63)
arrow(2.73,3.05,2.98,3.05)
arrow(5.28,3.05,5.73,3.05)
arrow(8.03,3.05,8.48,3.05)
ax.plot([10.72, 10.88, 10.88], [4.65, 4.65, 1.25], color='#536779', lw=1.3)
arrow(10.88,1.25,10.72,1.25)
ax.text(1.4,2.23,'Masked VAE: +142 partial training records',ha='center',fontsize=8)
box(.15, .8, 3.15, .9, 'Fidelity\nContinuous moments\nand feature distributions', size=9)
box(3.85, .8, 3.15, .9, 'Disclosure screening\nNearest-neighbour proximity', size=9)
box(7.55, .8, 3.1, .9, 'Predictive utility\nStatus / 3-year / time-to-event', '#e5f1e8',9)
arrow(4.12,2.48,1.72,1.78)
arrow(6.8,2.48,5.42,1.78)
arrow(9.6,2.48,9.1,1.78)
ax.text(5.5,.28,'Primary comparison: held-out real patients | Cross-validation: independently refit generators and filters in each training fold',
        ha='center',fontsize=9,color='#4b5563')
fig.savefig(OUT / 'fig14_study_overview.png', bbox_inches='tight', pad_inches=.12)
plt.close(fig)
print('Created raincloud and study-overview figures from the saved training data.')
