"""Render recorded exploratory sweeps without unsupported privacy labels."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
sigma = [0,.1,.2,.3,.5,.7,1.0]
axes[0].plot(sigma,[35.7,33.9,28.1,17.6,5.8,1.4,.4],'-o',color='#246a92',markersize=4)
axes[0].set(xlabel='Output noise (training SD units)',ylabel='Near-duplicate rate (%)',
            title='Empirical proximity screening')
axes[1].plot(sigma,[.2280,.2095,.1720,.1402,.0804,.0481,.0319],'-o',color='#246a92',markersize=4)
axes[1].set(xlabel='Output noise (training SD units)',ylabel='Tabular Frechet distance',
            title='Continuous-feature moments')
axes[2].plot([.5,.8,1,1.2,1.5,2,3],[210.4,89.2,61.2,46,33.6,23.9,14],'-o',color='#805795',markersize=4)
axes[2].set(xlabel='Assumed DP-SGD noise multiplier',ylabel='Approximate epsilon',
            title='Unvalidated accounting projection')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.15)
fig.suptitle('Exploratory analyses: neither output perturbation nor these projections certify privacy',fontsize=10)
fig.savefig(Path(__file__).resolve().parent/'paper/figures_supplementary/figS15_privacy_enhancement.png',dpi=220)
