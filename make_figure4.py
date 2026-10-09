"""Figure for protocol 4: the question label (A) and the shared slow field (B)."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'


def style(ax, grid_axis='y'):
    ax.grid(True, axis=grid_axis, color=GRID, lw=.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def ci(x, rng, n=2000):
    x = np.asarray(x)
    m = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return np.quantile(m, .025), np.quantile(m, .975)


def main():
    r = json.load(open('results/protocol4.json'))
    pw = json.load(open('results/protocol4_per_world.json'))
    A, B = r['A'], r['B']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.05, 1]})

    # ---- A
    xs = [0, 0.05, 0.3, 0.7, 1.0]
    ys = [A['K8'][f'field|tag{t}'] for t in (0.0, 0.05, 0.3, 0.7)] + [A['K8']['field|oracle']]
    a1.plot(xs, ys, color=BLUE, lw=2, marker='o', ms=7, zorder=3)
    a1.axhline(A['K8']['open|oracle'], color=ORANGE, lw=2, zorder=2)
    a1.text(0.02, A['K8']['open|oracle'] * 1.08, 'fixed thresholds (no adaptation)', color=INK, fontsize=9)
    a1.axhline(A['K8']['self|oracle'], color=MUTED, lw=1.5, ls='--', zorder=2)
    a1.text(0.02, A['K8']['self|oracle'] * 1.1,
            'threshold moved by its own spikes only:\nany reader replays it, label free', color=MUTED, fontsize=8.5)
    ex = A['exact']['8']['R_unlabeled']
    a1.plot([0], [ex], marker='D', ms=8, color=INK, ls='none', zorder=4)
    a1.annotate('best possible reader\nwithout the label', (0, ex), textcoords='offset points',
                xytext=(-4, -30), color=INK, fontsize=8.5)
    a1.annotate('in-vivo waveform level\n(R² ≈ 0.05)', (0.05, ys[1]), textcoords='offset points',
                xytext=(14, 4), color=INK, fontsize=8.5)
    a1.annotate('reader knows the threshold', (1.0, ys[-1]), textcoords='offset points',
                xytext=(-128, -16), color=INK, fontsize=8.5)
    a1.set_yscale('log')
    a1.set_ylim(0.025, 1.2)
    a1.set_xlim(-0.04, 1.06)
    a1.set_xlabel('how much of the threshold the waveform tag reveals (R²)', color=MUTED)
    a1.set_ylabel('distant reader\'s error on unseen questions, 8 reads', color=MUTED)
    a1.set_title('A  A threshold moved by the neighbours\' field: the bit needs its question',
                 color=INK, fontsize=10.5, loc='left')
    style(a1)

    # ---- B: paired error reduction vs "own spikes only", same worlds
    rng = np.random.default_rng(7)
    rows = [('metric|spiking', 'shuffled', 'field, wrong geometry'),
            ('metric|spiking', 'field_r1', 'field, nearest neighbours'),
            ('metric|spiking', 'field_r2', 'field, two neighbours each side'),
            ('metric|spiking', 'broadcast', 'every cell hears every cell'),
            None,
            ('permuted|spiking', 'field_r1', 'field, nearest neighbours'),
            ('permuted|spiking', 'addressed', 'wired to most-correlated cells'),
            ('permuted|spiking', 'broadcast', 'every cell hears every cell')]
    labels, y = [], 0
    for row in rows:
        if row is None:
            y += 0.8
            continue
        w, arm, lab = row
        a = np.array(pw[f'{w}|{arm}'])
        b = np.array(pw[f'{w}|private'])
        idx = rng.integers(len(a), size=(2000, len(a)))
        boot = 100 * (1 - a[idx].mean(1) / b[idx].mean(1))
        g = 100 * (1 - a.mean() / b.mean())
        lo, hi = np.quantile(boot, [.025, .975])
        col = BLUE if arm == 'field_r1' else (MUTED if arm == 'shuffled' else INK)
        a2.errorbar([g], [y], xerr=[[g - lo], [hi - g]], color=col, marker='o', ms=7, capsize=3, lw=1.6)
        a2.text(hi + 0.4, y, f'{g:.1f}%', va='center', color=INK, fontsize=8.5)
        labels.append((y, lab))
        y += 1
    a2.axvline(0, color=MUTED, lw=1)
    a2.axvline(10, color=ORANGE, lw=1.5, ls='--')
    a2.text(10.3, 8.15, 'pre-registered bar (10%)', color=INK, fontsize=8.5, va='center')
    a2.set_yticks([p for p, _ in labels])
    a2.set_yticklabels([l for _, l in labels], color=INK, fontsize=9)
    a2.set_ylim(8.5, -0.9)
    a2.set_xlim(-4, 19)
    a2.text(-3.8, -0.65, 'world correlated by distance', color=MUTED, fontsize=8.5)
    a2.text(-3.8, 4.15, 'world correlated by a shuffled map', color=MUTED, fontsize=8.5)
    a2.set_xlabel('error reduction vs each cell using only its own spikes (%, paired 95% CI)', color=MUTED)
    a2.set_title('B  Sharing neighbours\' answers helps, but there is little to share',
                 color=INK, fontsize=10.5, loc='left')
    style(a2, 'x')
    fig.text(0.01, -0.03,
             'Spiking soma · 512 held-out worlds · σ = 0.03 · gates fixed, thresholds adapt. '
             'B: 8 cells × 8 reads; own-spikes-only already cuts error 67% vs fixed thresholds.',
             color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig('results/label_and_field.svg', bbox_inches='tight')
    fig.savefig('results/label_and_field.png', dpi=160, bbox_inches='tight')


if __name__ == '__main__':
    main()
