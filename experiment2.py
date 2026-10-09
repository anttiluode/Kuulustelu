"""Protocol 2: revisiting (experiment A) and coupling (experiment B)."""
import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

import kuulustelu as k
from experiment import TRAIN, HELDOUT, gap_ci

SIGMA, ETA = 0.03, 0.0
ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


def reads_to_match(adaptive_curve, target):
    """Reads the adaptive policy needs to reach `target`, interpolating log R."""
    c = np.log(np.asarray(adaptive_curve))
    t = np.log(target)
    for i in range(1, len(c)):
        if c[i] <= t:
            return float(i - 1 + (c[i - 1] - t) / (c[i - 1] - c[i]))
    return None


def compare(prior, K, primary, train, heldout):
    sched = k.run('open', 'spiking', ETA, SIGMA, prior, train, K, train=True)['schedule']
    out = {p: k.run(p, 'spiking', ETA, SIGMA, prior, heldout, K,
                    schedule=sched if p == 'open' else None)
           for p in ('adaptive', 'open', 'random')}
    ra, ro = out['adaptive']['R'], out['open']['R']
    diff = ro[:, primary] - ra[:, primary]
    same_by_K = [float(np.mean(np.all(out['adaptive']['actions'][:, :t] ==
                                      out['open']['actions'][:, :t], axis=1)))
                 for t in range(1, K + 1)]
    mean_diff = np.abs(ra.mean(0) - ro.mean(0))
    opened = [t for t in range(K + 1) if mean_diff[t] > 1e-9]
    S = k.prior_cov(prior)
    w = np.linalg.eigvalsh(S)
    curves = {p: out[p]['R'].mean(0).tolist() for p in out}
    return {
        'prior': prior, 'K': K, 'primary_K': primary,
        'participation_ratio': float(w.sum() ** 2 / (w ** 2).sum()),
        'curves': curves,
        'primary': {p: float(out[p]['R'][:, primary].mean()) for p in out},
        'gap_fraction': float(1 - ra[:, primary].mean() / ro[:, primary].mean()),
        'gap_fraction_ci95': gap_ci(ro[:, primary], ra[:, primary], np.random.default_rng(7)),
        'wilcoxon_p': float(wilcoxon(diff, alternative='greater').pvalue) if np.any(diff != 0) else 1.0,
        'identical_sequences_up_to_K': same_by_K,
        'first_K_mean_R_differs': opened[0] if opened else None,
        'reads_adaptive_needs_to_match_open_at_primary':
            reads_to_match(curves['adaptive'], curves['open'][primary]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='results/protocol2.json')
    args = ap.parse_args()
    res = {'A': compare('iso', 36, 24, TRAIN, HELDOUT), 'B': {}}
    a = res['A']
    print(f"A iso K=24: adaptive {a['primary']['adaptive']:.4f} open {a['primary']['open']:.4f} "
          f"gap {a['gap_fraction']:+.3f} {a['gap_fraction_ci95']} p {a['wilcoxon_p']:.2e} "
          f"first differs at K={a['first_K_mean_R_differs']}", flush=True)
    for alpha in ALPHAS:
        r = compare(f'mix:{alpha}', 12, 8, TRAIN, HELDOUT)
        res['B'][str(alpha)] = r
        print(f"B alpha={alpha:<4} PR {r['participation_ratio']:5.2f} adaptive {r['primary']['adaptive']:.4f} "
              f"open {r['primary']['open']:.4f} gap {r['gap_fraction']:+.3f} {r['gap_fraction_ci95']} "
              f"p {r['wilcoxon_p']:.2e} reads-to-match {r['reads_adaptive_needs_to_match_open_at_primary']}",
              flush=True)
    gaps = [res['B'][str(al)]['gap_fraction'] for al in ALPHAS]
    drops = [gaps[i] - gaps[i + 1] for i in range(len(gaps) - 1) if gaps[i + 1] < gaps[i]]
    b1 = res['B']['1.0']['primary']
    r1 = json.load(open('results/summary.json'))['results']['traces|spiking|eta=0.0|sigma=0.03']['K8']
    res['gates'] = {
        'A1_identical_up_to_12': all(x == 1.0 for x in a['identical_sequences_up_to_K'][:12]),
        'A2_gap_at_24': a['gap_fraction'] >= 0.10 and a['wilcoxon_p'] < 1e-3,
        'A3_opens_at_13_to_16': a['first_K_mean_R_differs'] is not None and 13 <= a['first_K_mean_R_differs'] <= 16,
        'B1_alpha1_replicates_round1': abs(b1['adaptive'] - r1['adaptive']) <= 1e-12 and abs(b1['open'] - r1['open']) <= 1e-12,
        'B2_gap_nondecreasing_in_alpha': len(drops) <= 1 and all(d <= 0.03 for d in drops),
        'B3_gap_at_half': res['B']['0.5']['gap_fraction'] >= 0.10 and res['B']['0.5']['wilcoxon_p'] < 1e-3,
    }
    for name, ok in res['gates'].items():
        print(name, 'PASS' if ok else 'FAIL')
    with open(args.output, 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
