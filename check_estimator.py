"""Post-hoc check (added after the held-out run, not a gate).

Is the spiking gap about the INFORMATION each policy collected, or an artefact of
the approximate (ADF) receiver? Re-score the same held-out reads with the exact
posterior mean, computed by importance sampling from the prior:

  E[m0 | answers] ~ sum_i w_i m_i,  w_i = prod_t Phi(s_t (g~_t^T m_i + b_t)/sigma),
  m_i = F h_i, h_i ~ N(0, I).
"""
import argparse
import json

import numpy as np
from scipy.special import log_ndtr

import kuulustelu as k
from experiment import TRAIN, HELDOUT, PRIMARY_K


def exact_R(res, sigma, samples, K):
    F = k.trace_filter()
    Q = k.questions()
    S0 = k.prior_cov('traces')
    denom = np.einsum('jn,nk,jk->', Q, S0, Q)
    Rs, ess = [], []
    for w in range(len(res['m0'])):
        g, b, s = res['asked'][w, :K], res['bias'][w, :K], res['heard'][w, :K]
        logw = log_ndtr(s[None, :] * (samples @ g.T + b[None, :]) / sigma).sum(1)
        wt = np.exp(logw - logw.max())
        wt /= wt.sum()
        mu = wt @ samples
        ess.append(1 / np.sum(wt ** 2))
        Rs.append(np.sum(((mu - res['m0'][w]) @ Q.T) ** 2) / denom)
    return np.array(Rs), np.array(ess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worlds', type=int, default=128)
    ap.add_argument('--samples', type=int, default=400_000)
    ap.add_argument('--output', default='results/estimator_check.json')
    args = ap.parse_args()
    sigma, eta = .03, 0.0
    held = range(HELDOUT.start, HELDOUT.start + args.worlds)
    sched = k.run('open', 'spiking', eta, sigma, 'traces', TRAIN, PRIMARY_K, train=True)['schedule']
    samples = (k.trace_filter() @ np.random.default_rng(99).normal(
        size=(k.HISTORY, args.samples))).T
    out = {'worlds': args.worlds, 'samples': args.samples, 'K': PRIMARY_K,
           'config': 'traces|spiking|eta=0.0|sigma=0.03'}
    for pol in ('adaptive', 'open', 'random'):
        res = k.run(pol, 'spiking', eta, sigma, 'traces', held, PRIMARY_K,
                    schedule=sched if pol == 'open' else None)
        R_exact, ess = exact_R(res, sigma, samples, PRIMARY_K)
        out[pol] = {'R_adf': float(res['R'][:, PRIMARY_K].mean()),
                    'R_exact_posterior': float(R_exact.mean()),
                    'ess_median': float(np.median(ess)), 'ess_min': float(ess.min())}
        print(pol, {kk: round(v, 4) for kk, v in out[pol].items()})
    out['gap_fraction_adf'] = 1 - out['adaptive']['R_adf'] / out['open']['R_adf']
    out['gap_fraction_exact'] = 1 - out['adaptive']['R_exact_posterior'] / out['open']['R_exact_posterior']
    print('gap adf', round(out['gap_fraction_adf'], 3), 'gap exact', round(out['gap_fraction_exact'], 3))
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
