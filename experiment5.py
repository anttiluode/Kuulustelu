"""Protocol 5, part C: the population reader."""
import argparse
import json
import time

import numpy as np
from scipy.stats import wilcoxon

import round4 as r4
import round5 as r5
from experiment import TRAIN
from experiment4 import delta_stats

HELD = range(2000, 2256)
MS = (1, 2, 4, 8)
PARTICLES = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--output', default='results/protocol5.json')
    args = ap.parse_args()
    held, train = (range(1000, 1016), range(3000, 3032)) if args.smoke else (HELD, TRAIN)
    seeds = list(held)
    m_t, V_t = delta_stats(train)
    m0, noise, nn = r5.worlds_C(held, max(MS))
    nb = r4.neighbour_bits(m0, nn)
    res, pw = {}, {}
    for M in MS:
        t0 = time.time()
        s, b = r5.run_population('field', m0, noise[:M], nb)
        ss, bs = r5.run_population('self', m0, noise[:M], nb)
        R_or, mu_o, S_o = r5.adf_decode_pop(s, b, r4.SIGMA, m0)
        bt, sig = r5.tag0_thresholds(s, m_t, V_t)
        R_t0, mu_t, S_t = r5.adf_decode_pop(s, bt, sig, m0)
        R_self = r5.adf_decode_pop(ss, bs, r4.SIGMA, m0)[0]
        trees = r5.trees_pop(s)
        R_ex, ess = [], []
        for w, sd in enumerate(seeds):
            tw = [[tr[t][w] for t in range(r4.K)] for tr in trees]
            mu, e = r4.importance_mean(sd, mu_t[w], S_t[w], lambda x: r5.tree_loglik_pop(x, s[w], tw), PARTICLES)
            R_ex.append(r4.R_single(mu, m0[w])); ess.append(e)
        R_ex, ess = np.array(R_ex), np.array(ess)
        d = R_self - R_ex
        res[M] = {'field_oracle': float(R_or.mean()), 'field_tag0': float(R_t0.mean()),
                  'field_exact_unlabeled': float(R_ex.mean()), 'self_driven': float(R_self.mean()),
                  'penalty': float(R_ex.mean() / R_or.mean()),
                  'exact_unlabeled_lt_self_p': float(wilcoxon(d, alternative='greater').pvalue),
                  'ess_median': float(np.median(ess)), 'ess_min': float(ess.min())}
        pw[M] = {'field_oracle': R_or.tolist(), 'field_tag0': R_t0.tolist(),
                 'field_exact_unlabeled': R_ex.tolist(), 'self_driven': R_self.tolist()}
        print(M, {kk: round(v, 4) for kk, v in res[M].items()}, f'{time.time() - t0:.0f}s', flush=True)
    out = {'config': {'heldout': [held.start, held.stop - 1], 'particles': PARTICLES, 'Ms': MS, 'smoke': args.smoke},
           'results': {str(M): v for M, v in res.items()}}
    if not args.smoke:
        r4pw = json.load(open('results/protocol4_per_world.json'))
        n = len(seeds)
        c1 = (np.max(np.abs(np.array(pw[1]['field_oracle']) - np.array(r4pw['field|oracle'][:n]))) <= 1e-12
              and np.max(np.abs(np.array(pw[1]['field_tag0']) - np.array(r4pw['field|tag0.0'][:n]))) <= 1e-12)
        P = [res[M]['penalty'] for M in MS]
        out['gates'] = {
            'C1_M1_replicates_round4': bool(c1),
            'C2_population_recovers_label': bool(all(P[i + 1] < P[i] for i in range(len(P) - 1)) and P[-1] <= .5 * P[0]),
            'C3_field_beats_self_for_far_reader_at_M8': bool(res[8]['field_exact_unlabeled'] < res[8]['self_driven']
                                                            and res[8]['exact_unlabeled_lt_self_p'] < 1e-3),
        }
        for kk, v in out['gates'].items():
            print(kk, 'PASS' if v else 'FAIL')
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=1)
    if not args.smoke:
        with open('results/protocol5_per_world.json', 'w') as f:
            json.dump({str(M): v for M, v in pw.items()}, f)


if __name__ == '__main__':
    main()
