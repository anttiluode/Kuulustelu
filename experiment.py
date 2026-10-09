"""Run every frozen configuration on held-out worlds and evaluate gates G1-G5."""
import argparse
import itertools
import json

import numpy as np
from scipy.stats import wilcoxon

import kuulustelu as k

TRAIN = range(3000, 3512)
HELDOUT = range(2000, 2512)
K = 12
PRIMARY_K = 8
SOMAS = ('graded', 'spiking')
ETAS = (0.0, 1.0, 0.25)
SIGMAS = (0.03, 0.1, 0.3)
PRIORS = ('traces', 'iso')


def key(prior, soma, eta, sigma):
    return f'{prior}|{soma}|eta={eta}|sigma={sigma}'


def gap_ci(r_open, r_adapt, rng, n=2000):
    idx = rng.integers(len(r_open), size=(n, len(r_open)))
    g = 1 - r_adapt[idx].mean(1) / r_open[idx].mean(1)
    return [float(np.quantile(g, .025)), float(np.quantile(g, .975))]


def run_config(prior, soma, eta, sigma, train, heldout):
    sched = k.run('open', soma, eta, sigma, prior, train, K, train=True)['schedule']
    pols = ['adaptive', 'open', 'random'] + (['adaptive_b0'] if soma == 'spiking' else [])
    out = {p: k.run(p, soma, eta, sigma, prior, heldout, K,
                    schedule=sched if p == 'open' else None) for p in pols}
    ra, ro, rr = (out[p]['R'][:, PRIMARY_K] for p in ('adaptive', 'open', 'random'))
    diff = ro - ra
    p = float(wilcoxon(diff, alternative='greater').pvalue) if np.any(diff != 0) else 1.0
    stats = {
        'curves': {p_: out[p_]['R'].mean(0).tolist() for p_ in pols},
        'K8': {p_: float(out[p_]['R'][:, PRIMARY_K].mean()) for p_ in pols},
        'K8_receiver_predicted': {p_: float(out[p_]['R_predicted'][:, PRIMARY_K].mean())
                                  for p_ in pols},
        'gap_fraction_vs_open': float(1 - ra.mean() / ro.mean()),
        'gap_fraction_ci95': gap_ci(ro, ra, np.random.default_rng(7)),
        'gap_fraction_vs_random': float(1 - ra.mean() / rr.mean()),
        'wilcoxon_p_open_minus_adaptive': p,
        'identical_action_sequences': float(np.mean(np.all(
            out['adaptive']['actions'] == out['open']['actions'], axis=1))),
        'max_abs_R_diff_any_K': float(np.max(np.abs(out['adaptive']['R'] - out['open']['R']))),
        'open_schedule': sched,
        'adaptive_bias_index_hist': np.bincount(out['adaptive']['bias_index'][:, :PRIMARY_K].ravel(),
                                                minlength=k.N_BIAS).tolist(),
        'open_bias_index': [int(i % k.N_BIAS) for i in sched[:PRIMARY_K]],
    }
    per_world = {p_: out[p_]['R'][:, PRIMARY_K].tolist() for p_ in pols}
    return stats, per_world


def gates(res):
    g = {}
    g1 = [res[key('traces', 'graded', eta, .03)] for eta in (0.0, 1.0)]
    g['G1_graded_open_loop_is_optimal'] = all(
        r['identical_action_sequences'] == 1.0 and r['max_abs_R_diff_any_K'] <= 1e-9 for r in g1)
    s = res[key('traces', 'spiking', 0.0, .03)]
    g['G2_spike_gap'] = (s['K8']['adaptive'] <= .85 * s['K8']['open']
                         and s['wilcoxon_p_open_minus_adaptive'] < 1e-3)
    g['G3_beats_random'] = s['K8']['adaptive'] <= .85 * s['K8']['random']
    g['G4_noise_shrinks_gap'] = (res[key('traces', 'spiking', 0.0, .3)]['gap_fraction_vs_open']
                                 < s['gap_fraction_vs_open'])
    g['G5_destruction_shrinks_gap'] = (res[key('traces', 'spiking', 1.0, .03)]['gap_fraction_vs_open']
                                       < s['gap_fraction_vs_open'])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='results/summary.json')
    ap.add_argument('--details', default='results/per_world_K8.json')
    ap.add_argument('--smoke', action='store_true', help='pilot seeds only, small batches')
    args = ap.parse_args()
    train, heldout = (range(3000, 3032), range(1000, 1032)) if args.smoke else (TRAIN, HELDOUT)
    res, details = {}, {}
    for prior, soma, eta, sigma in itertools.product(PRIORS, SOMAS, ETAS, SIGMAS):
        kk = key(prior, soma, eta, sigma)
        res[kk], details[kk] = run_config(prior, soma, eta, sigma, train, heldout)
        r = res[kk]
        print(f"{kk:38s} adaptive {r['K8']['adaptive']:.4f} open {r['K8']['open']:.4f} "
              f"random {r['K8']['random']:.4f} gap {r['gap_fraction_vs_open']:+.3f} "
              f"same-seq {r['identical_action_sequences']:.3f}", flush=True)
    summary = {
        'config': {'train_seeds': [train.start, train.stop - 1],
                   'heldout_seeds': [heldout.start, heldout.stop - 1],
                   'K': K, 'primary_K': PRIMARY_K, 'n_actions': len(k.actions(k.prior_cov())[1]),
                   'smoke': args.smoke},
        'results': res,
        'gates': None if args.smoke else gates(res),
    }
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=1)
    if args.details:
        with open(args.details, 'w') as f:
            json.dump(details, f)
    if summary['gates']:
        for name, ok in summary['gates'].items():
            print(name, 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
