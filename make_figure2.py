"""Figure for protocol 2: revisiting (A) and coupling (B)."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {'adaptive': '#2a78d6', 'open': '#eb6834', 'random': '#1baf7a'}
NAME = {'adaptive': 'adaptive (uses answers)', 'open': 'open-loop (fixed schedule)', 'random': 'random'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def style(ax):
    ax.grid(True, axis='y', color=GRID, lw=.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def main():
    p = json.load(open('results/protocol2.json'))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

    A = p['A']
    for pol in ('random', 'open', 'adaptive'):
        c = A['curves'][pol]
        a1.plot(range(len(c)), c, color=COL[pol], lw=2, label=NAME[pol])
    a1.axvspan(0, 12, color=GRID, alpha=.45, lw=0, zorder=0)
    a1.text(0.6, 0.075, 'fresh axes left:\nidentical questions', color=MUTED, fontsize=9)
    a1.text(13.2, 0.075, 'gap opens at read 13', color=INK, fontsize=9)
    a1.set_yscale('log')
    a1.set_ylim(0.06, 1.15)
    a1.set_xlim(0, 36)
    a1.set_xlabel('questions asked', color=MUTED)
    a1.set_ylabel('error on unseen questions', color=MUTED)
    a1.set_title('A  Isotropic prior: twelve uncorrelated directions', color=INK, fontsize=11, loc='left')
    a1.legend(frameon=False, fontsize=9, loc='upper right', labelcolor=INK)
    style(a1)

    B = p['B']
    alphas = sorted(B, key=float)
    x = [float(a) for a in alphas]
    g = [100 * B[a]['gap_fraction'] for a in alphas]
    lo = [g[i] - 100 * B[a]['gap_fraction_ci95'][0] for i, a in enumerate(alphas)]
    hi = [100 * B[a]['gap_fraction_ci95'][1] - g[i] for i, a in enumerate(alphas)]
    a2.errorbar(x, g, yerr=[lo, hi], color=COL['adaptive'], lw=2, marker='o', ms=7,
                capsize=3, elinewidth=1.2)
    for xi, gi, a in zip(x, g, alphas):
        pr = B[a]['participation_ratio']
        a2.annotate(f'{gi:.0f}%\ndim {pr:.1f}', (xi, gi), textcoords='offset points',
                    xytext=(-14, 10) if xi < 0.7 else ((-46, -2) if xi < 1 else (-40, -4)),
                    color=INK, fontsize=8.5)
    a2.set_xlim(-0.05, 1.05)
    a2.set_ylim(-5, 85)
    a2.set_xlabel('coupling α  (0 = isotropic, 1 = real branch traces)', color=MUTED)
    a2.set_ylabel('error reduction vs fixed schedule, 8 reads (%)', color=MUTED)
    a2.set_title('B  Coupling between branches makes answers steer questions',
                 color=INK, fontsize=11, loc='left')
    style(a2)

    fig.text(0.01, -0.02, 'Spiking soma · 512 held-out worlds · σ = 0.03 · η = 0 · '
             '"dim" = participation ratio of the prior · bars = 95% bootstrap CI',
             color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig('results/revisit_and_coupling.svg', bbox_inches='tight')
    fig.savefig('results/revisit_and_coupling.png', dpi=160, bbox_inches='tight')


if __name__ == '__main__':
    main()
