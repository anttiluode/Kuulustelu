"""Figure: error vs number of questions, graded soma vs spiking soma."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {'adaptive': '#2a78d6', 'open': '#eb6834', 'random': '#1baf7a', 'adaptive_b0': '#eda100'}
NAME = {'adaptive': 'adaptive (uses answers)', 'open': 'open-loop (fixed schedule)',
        'random': 'random', 'adaptive_b0': 'adaptive, threshold fixed at 0'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def main():
    res = json.load(open('results/summary.json'))['results']
    panels = [('traces|graded|eta=0.0|sigma=0.03', 'Graded soma: y = gᵀm + noise'),
              ('traces|spiking|eta=0.0|sigma=0.03', 'Spiking soma: y = 1[gᵀm + b + noise > 0]')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (kk, title) in zip(axes, panels):
        curves = res[kk]['curves']
        ks = range(len(curves['adaptive']))
        order = ('random', 'adaptive', 'open') if 'graded' in kk else \
            ('random', 'adaptive_b0', 'open', 'adaptive')
        for pol in order:
            ls = (0, (3, 3)) if (pol == 'open' and 'graded' in kk) else '-'
            ax.plot(ks, curves[pol], color=COL[pol], lw=2, ls=ls, label=NAME[pol])
        ax.axvline(8, color=GRID, lw=1, zorder=0)
        ax.text(8.15, 6.5e-4, 'K = 8', color=MUTED, fontsize=9)
        ax.set_yscale('log')
        ax.set_ylim(5e-4, 2)
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.set_xlabel('questions asked', color=MUTED)
        ax.grid(True, which='major', axis='y', color=GRID, lw=.8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel('error on unseen questions (1 = knows nothing)', color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, loc='upper right', labelcolor=INK)
    axes[0].text(2.3, 0.14, 'adaptive = open-loop:\nsame questions in 512/512 worlds',
                 color=INK, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9, loc='lower left', labelcolor=INK)
    fig.text(0.01, -0.02, '512 held-out worlds · 12 leaky branch traces · σ = 0.03 · reads do not write (η = 0)',
             color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig('results/graded_vs_spiking.svg', bbox_inches='tight')
    fig.savefig('results/graded_vs_spiking.png', dpi=160, bbox_inches='tight')


if __name__ == '__main__':
    main()
