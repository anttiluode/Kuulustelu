"""Figure for protocol 3: learned receivers centre their thresholds."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {'adaptive': '#2a78d6', 'open': '#eb6834'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def style(ax):
    ax.grid(True, axis='y', color=GRID, lw=.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def main():
    d = json.load(open('results/protocol3_diagnostics.json'))
    r = json.load(open('results/protocol3.json'))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

    for pol, lab in (('adaptive', 'learned adaptive'), ('open', 'learned open-loop')):
        for i in range(3):
            rows = d['spread'][f'spiking-{pol}|{i}']
            a1.plot(range(2, 9), [x['median_abs_z'] for x in rows[1:]], color=COL[pol],
                    lw=2, alpha=.9, marker='o', ms=4, label=lab if i == 0 else None)
    a1.axhline(r['reference']['bayes_adaptive_median_abs_z'], color=MUTED, lw=1, ls=(0, (3, 3)))
    a1.text(6.3, r['reference']['bayes_adaptive_median_abs_z'] * 0.72,
            'programmed Bayesian\nadaptive receiver', color=MUTED, fontsize=8.5, va='top')
    a1.set_yscale('log')
    a1.set_ylim(0.01, 40)
    a1.set_xlabel('read', color=MUTED)
    a1.set_ylabel('median |z|  (0 = threshold at current best guess)', color=MUTED)
    a1.set_title('A  Where the learned receivers put the threshold', color=INK, fontsize=11, loc='left')
    a1.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK)
    style(a1)

    names = ['spiking\nadaptive', 'spiking\nopen-loop']
    vals = [[r['runs'][f'spiking-{p}|{i}']['R'] for i in range(3)] for p in ('adaptive', 'open')]
    for j, (v, pol) in enumerate(zip(vals, ('adaptive', 'open'))):
        a2.scatter([j] * 3, v, color=COL[pol], s=60, zorder=3, edgecolor='white', lw=1.5)
    a2.axhline(0.0381, color=COL['adaptive'], lw=1, ls=(0, (3, 3)))
    a2.axhline(0.1216, color=COL['open'], lw=1, ls=(0, (3, 3)))
    a2.text(1.45, 0.0395, 'programmed adaptive 0.038', color=MUTED, fontsize=8.5, va='bottom')
    a2.text(1.45, 0.123, 'programmed fixed 0.122', color=MUTED, fontsize=8.5, va='bottom')
    a2.set_xticks([0, 1])
    a2.set_xticklabels(names, color=INK)
    a2.set_xlim(-0.5, 2.4)
    a2.set_ylim(0, 0.14)
    a2.set_ylabel('error on unseen questions, 8 reads', color=MUTED)
    a2.set_title('B  Learned receivers, three inits each', color=INK, fontsize=11, loc='left')
    style(a2)

    fig.text(0.01, -0.02, 'Spiking soma · 512 held-out worlds · σ = 0.03 · η = 0 · '
             'trained only to minimise prediction error', color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig('results/learned_receivers.svg', bbox_inches='tight')
    fig.savefig('results/learned_receivers.png', dpi=160, bbox_inches='tight')


if __name__ == '__main__':
    main()
