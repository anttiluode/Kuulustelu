"""Post-hoc diagnostics for protocol 3 (added after the held-out run, not gates).

1. Do the learned adaptive receivers' questions actually vary with the answers?
   Per read, the spread of gate directions and of standardised thresholds across
   held-out worlds. A receiver whose questions don't vary is a fixed schedule.
2. What is the best *fixed* continuous-gate schedule for a graded soma? Greedy
   on the exact covariance recursion (answers never enter). This is the bound
   that L4's open-loop learner was supposed to approach.
"""
import json

import numpy as np
from scipy.optimize import minimize

import kuulustelu as k
import learned as L


def load(arm, init, W):
    pre = f'{arm}|{init}|'
    return {kk[len(pre):]: W[kk] for kk in W.files if kk.startswith(pre)}


def question_spread(p, arm):
    m0, noise, _ = k.worlds(range(2000, 2512), 'traces', L.K)
    _, g, b, s = L.forward(p, m0, noise, arm, record=True)
    _, z = L.bayes_replay(g, b, s, m0, arm.startswith('spiking'))
    out = []
    for t in range(L.K):
        mean_dir = g[:, t].mean(0)
        mean_dir /= np.linalg.norm(mean_dir)
        cos = np.abs(g[:, t] @ mean_dir)
        out.append({'read': t + 1,
                    'gate_mean_abs_cos_to_average': float(cos.mean()),
                    'gate_min_abs_cos_to_average': float(cos.min()),
                    'bias_std': float(b[:, t].std()),
                    'median_abs_z': float(np.median(np.abs(z[:, t])))})
    return out


def best_fixed_graded(K, sigma, restarts=8):
    S = k.prior_cov('traces')
    Q = k.questions()
    denom = np.einsum('jn,nk,jk->', Q, S, Q)
    rng = np.random.default_rng(0)
    gates = []
    for _ in range(K):
        def neg(x):
            g = x / np.linalg.norm(x)
            Sg = S @ g
            return -(Sg @ Sg) / (g @ Sg + sigma ** 2)
        best = min((minimize(neg, rng.normal(size=12), method='BFGS') for _ in range(restarts)),
                   key=lambda r: r.fun)
        g = best.x / np.linalg.norm(best.x)
        Sg = S @ g
        S = S - np.outer(Sg, Sg) / (g @ Sg + sigma ** 2)
        gates.append(g)
    return float(np.einsum('jn,nk,jk->', Q, S, Q) / denom)


def main():
    W = np.load('results/protocol3_weights.npz')
    res = json.load(open('results/protocol3.json'))
    out = {'spread': {}, 'bayes_rescored': {}}
    for arm in L.ARMS:
        for init in (0, 1, 2):
            key = f'{arm}|{init}'
            out['spread'][key] = question_spread(load(arm, init, W), arm)
            out['bayes_rescored'][key] = res['runs'][key]['R_bayes_rescored']
    out['best_fixed_graded_continuous_expected_R'] = best_fixed_graded(L.K, L.SIGMA)
    for key, rows in out['spread'].items():
        if 'adaptive' in key:
            print(key, 'gate |cos| to average, reads 2-8:',
                  [round(r['gate_mean_abs_cos_to_average'], 3) for r in rows[1:]],
                  ' median|z|:', [round(r['median_abs_z'], 2) for r in rows[1:]])
    print('best fixed continuous graded schedule, expected R:',
          round(out['best_fixed_graded_continuous_expected_R'], 5))
    print('graded learned arms, Bayes-rescored R:',
          {kk: round(v, 5) for kk, v in out['bayes_rescored'].items() if kk.startswith('graded')})
    with open('results/protocol3_diagnostics.json', 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
