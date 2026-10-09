"""Protocol 4: the question label (part A) and a shared slow field (part B)."""
import argparse
import json
import time

import numpy as np
from scipy.stats import wilcoxon

import kuulustelu as k
import round4 as r4
from experiment import TRAIN, HELDOUT, gap_ci

TAGS = (0.0, 0.05, 0.3, 0.7)
PARTICLES = 8192


def round1_open():
    """Round 1's held-out open-loop R at K = 8 (spiking, sigma 0.03, eta 0)."""
    with open('results/summary.json') as f:
        return json.load(f)['results']['traces|spiking|eta=0.0|sigma=0.03']['K8']['open']


def one_sided(a, b):
    """p for a < b (paired)."""
    d = b - a
    return float(wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0


def summarise(curves):
    return {name: c.mean(0).tolist() for name, c in curves.items()}


# ----------------------------------------------------------------- part A
def delta_stats(train):
    m0, ns, nn = r4.worlds_A(train)
    nb = r4.neighbour_bits(m0, nn)
    s, b = r4.run_sender('field', m0, ns, nb)
    d = b - r4.replay_self(s)
    return d.mean(0), d.var(0)


def tag_reader(s, b, seeds, m_t, V_t, rho2):
    bhat = r4.replay_self(s)
    delta = b - bhat
    if rho2 > 0:
        nu = np.array([np.random.default_rng([int(x), 6]).normal(size=r4.K) for x in seeds])
        tau = delta + nu * np.sqrt(V_t * (1 - rho2) / rho2)[None]
        btil = bhat + m_t + rho2 * (tau - m_t)
    else:
        btil = bhat + m_t
    sig = np.sqrt(r4.SIGMA ** 2 + V_t * (1 - rho2))
    return btil, np.repeat(sig[None], len(s), 0)


def exact_readers(seeds, m0, s, b, centre, upto, particles):
    """Exact-unlabeled (tree) and exact-labeled posterior R at read `upto`, by IS."""
    tree = r4.threshold_tree(s, upto=upto)
    Ru, Rl, essu, essl = [], [], [], []
    (mu_u, S_u), (mu_l, S_l) = centre
    for w, sd in enumerate(seeds):
        tw = [tr[w] for tr in tree]
        mu, ess = r4.importance_mean(sd, mu_u[w], S_u[w],
                                     lambda x: r4.tree_loglik(x, s[w], tw, upto), particles)
        Ru.append(r4.R_single(mu, m0[w])); essu.append(ess)
        mu, ess = r4.importance_mean(sd, mu_l[w], S_l[w],
                                     lambda x: r4.labeled_loglik(x, s[w], b[w], upto), particles)
        Rl.append(r4.R_single(mu, m0[w])); essl.append(ess)
    return np.array(Ru), np.array(Rl), np.array(essu), np.array(essl)


def part_A(train, held, particles):
    seeds = list(held)
    m_t, V_t = delta_stats(train)
    m0, ns, nn = r4.worlds_A(held)
    nb = r4.neighbour_bits(m0, nn)
    snd = {mode: r4.run_sender(mode, m0, ns, nb) for mode in ('open', 'self', 'field')}
    R = {}
    for mode, (s, b) in snd.items():
        R[f'{mode}|oracle'] = r4.adf_decode(s, b, r4.SIGMA, m0)[0]
        if mode != 'open':
            R[f'{mode}|naive'] = r4.adf_decode(s, r4.replay_self(s), r4.SIGMA, m0)[0]
    s, b = snd['field']
    for rho2 in TAGS:
        bt, sig = tag_reader(s, b, seeds, m_t, V_t, rho2)
        R[f'field|tag{rho2}'] = r4.adf_decode(s, bt, sig, m0)[0]
    out = {'delta_mean_by_step': m_t.tolist(), 'delta_var_by_step': V_t.tolist(),
           'curves': summarise(R), 'K8': {n: float(c[:, r4.K].mean()) for n, c in R.items()},
           'replay_max_abs_err_self': float(np.max(np.abs(r4.replay_self(snd['self'][0]) - snd['self'][1]))),
           'naive_minus_oracle_self_max': float(np.max(np.abs(R['self|naive'] - R['self|oracle'])))}
    exact = {}
    for upto in (4, 8):
        bt0, sig0 = tag_reader(s, b, seeds, m_t, V_t, 0.0)
        _, mu_t0, S_t0 = r4.adf_decode(s[:, :upto], bt0[:, :upto], sig0, m0)
        _, mu_o, S_o = r4.adf_decode(s[:, :upto], b[:, :upto], r4.SIGMA, m0)
        t0 = time.time()
        Ru, Rl, eu, el = exact_readers(seeds, m0, s, b, ((mu_t0, S_t0), (mu_o, S_o)), upto, particles)
        exact[upto] = {'unlabeled': Ru, 'labeled': Rl, 'ess_u': eu, 'ess_l': el}
        print(f'  exact readers K={upto}: {time.time() - t0:.0f}s, ESS median {np.median(eu):.0f}/{np.median(el):.0f}', flush=True)
    # exact posterior for the open sender (same sampler, labeled) — robustness line for A4
    so, bo = snd['open']
    _, mu_oo, S_oo = r4.adf_decode(so, bo, r4.SIGMA, m0)
    Ro_exact, ess_o = [], []
    for w, sd in enumerate(seeds):
        mu, e = r4.importance_mean(sd, mu_oo[w], S_oo[w],
                                   lambda x: r4.labeled_loglik(x, so[w], bo[w]), particles)
        Ro_exact.append(r4.R_single(mu, m0[w])); ess_o.append(e)
    Ro_exact = np.array(Ro_exact)
    out['exact'] = {str(u): {'R_unlabeled': float(e['unlabeled'].mean()), 'R_labeled': float(e['labeled'].mean()),
                             'ess_unlabeled_median': float(np.median(e['ess_u'])), 'ess_unlabeled_min': float(e['ess_u'].min()),
                             'ess_labeled_median': float(np.median(e['ess_l'])), 'ess_labeled_min': float(e['ess_l'].min())}
                    for u, e in exact.items()}
    out['exact']['open_K8'] = {'R_labeled': float(Ro_exact.mean()), 'ess_median': float(np.median(ess_o))}
    ro, rs, rf = R['open|oracle'][:, 8], R['self|oracle'][:, 8], R['field|oracle'][:, 8]
    ru, rt0 = exact[8]['unlabeled'], R['field|tag0.0'][:, 8]
    out['stats'] = {
        'self_vs_open_gap': float(1 - rs.mean() / ro.mean()), 'self_vs_open_ci95': gap_ci(ro, rs, np.random.default_rng(7)),
        'self_vs_open_p': one_sided(rs, ro),
        'field_oracle_vs_open_gap': float(1 - rf.mean() / ro.mean()),
        'field_oracle_vs_self_oracle_p_two_sided': float(wilcoxon(rf - rs).pvalue),
        'exact_unlabeled_vs_open_gap': float(1 - ru.mean() / ro.mean()),
        'exact_unlabeled_vs_open_ci95': gap_ci(ro, ru, np.random.default_rng(7)),
        'exact_unlabeled_lt_open_p': one_sided(ru, ro),
        'exact_unlabeled_gt_open_p': one_sided(ro, ru),
        'exact_unlabeled_vs_exact_open_gap': float(1 - ru.mean() / Ro_exact.mean()),
        'fraction_of_oracle_advantage_kept_exact': float((ro.mean() - ru.mean()) / (ro.mean() - rf.mean())),
        'tag0_gt_open_p': one_sided(ro, rt0),
        'recovery': {str(r): float((rt0.mean() - R[f'field|tag{r}'][:, 8].mean()) / (rt0.mean() - rf.mean())) for r in TAGS},
    }
    st, K8 = out['stats'], out['K8']
    seq = [K8[f'field|tag{r}'] for r in TAGS] + [K8['field|oracle']]
    out['gates'] = {
        'A1_label_free_for_self_driven': out['replay_max_abs_err_self'] <= 1e-12 and out['naive_minus_oracle_self_max'] == 0.0,
        'A2_replicates_round1_open': abs(K8['open|oracle'] - round1_open()) <= 1e-12,
        'A3_threshold_only_adaptation': K8['self|oracle'] <= .85 * K8['open|oracle'] and st['self_vs_open_p'] < 1e-3,
        'A4_unlabeled_no_better_than_open': out['exact']['8']['R_unlabeled'] >= K8['open|oracle'],
        'A5_noise_reader_loses_to_open': K8['field|tag0.0'] > K8['open|oracle'] and st['tag0_gt_open_p'] < 1e-3,
        'A6_weak_tag_recovers_little': all(seq[i + 1] < seq[i] for i in range(len(seq) - 1)) and st['recovery']['0.05'] < 0.25,
    }
    out['gates'] = {n: bool(v) for n, v in out['gates'].items()}
    per_world = {n: c[:, 8].tolist() for n, c in R.items()}
    per_world['field|exact_unlabeled'] = ru.tolist()
    per_world['field|exact_labeled'] = exact[8]['labeled'].tolist()
    per_world['open|exact_labeled'] = Ro_exact.tolist()
    return out, per_world


# ----------------------------------------------------------------- part B
def part_B(held):
    res = {}
    C = r4.corr_line()
    m0, nz = r4.worlds_B(held, C)
    arms = ('open', 'private', 'field_r1', 'field_r2', 'broadcast', 'shuffled')
    for soma in ('spiking', 'graded'):
        res[f'metric|{soma}'] = {a: r4.run_population(a, soma, C, m0, nz)[0] for a in arms}
        print('  B metric', soma, {a: round(float(v[:, 8].mean()), 5) for a, v in res[f'metric|{soma}'].items()}, flush=True)
    Cp = r4.corr_line(r4.permutation())
    mp, nzp = r4.worlds_B(held, Cp)
    res['permuted|spiking'] = {a: r4.run_population(a, 'spiking', Cp, mp, nzp)[0]
                               for a in ('open', 'private', 'field_r1', 'addressed', 'broadcast')}
    print('  B permuted', {a: round(float(v[:, 8].mean()), 5) for a, v in res['permuted|spiking'].items()}, flush=True)
    M = res['metric|spiking']
    Pm = res['permuted|spiking']
    g = {n: v[:, 8] for n, v in M.items()}
    gp = {n: v[:, 8] for n, v in Pm.items()}
    graded = res['metric|graded']
    graded_spread = max(float(np.max(np.abs(graded[a] - graded['open']))) for a in graded)
    out = {
        'C_neighbour_correlation': float(C[0, 1]), 'permutation': r4.permutation().tolist(),
        'sets': {a: r4.visible_sets(a, C if a != 'addressed' else Cp)
                 for a in ('private', 'field_r1', 'field_r2', 'broadcast', 'shuffled', 'addressed')},
        'permuted_field_r1_mean_partner_corr': float(np.mean([Cp[i, j] for i, s in enumerate(r4.visible_sets('field_r1')) for j in s if j != i])),
        'permuted_addressed_mean_partner_corr': float(np.mean([Cp[i, j] for i, s in enumerate(r4.visible_sets('addressed', Cp)) for j in s if j != i])),
        'curves': {w: summarise(v) for w, v in res.items()},
        'K8': {w: {a: float(c[:, 8].mean()) for a, c in v.items()} for w, v in res.items()},
        'graded_max_abs_R_diff_across_arms': graded_spread,
        'stats': {
            'private_vs_open_gap': float(1 - g['private'].mean() / g['open'].mean()), 'private_vs_open_p': one_sided(g['private'], g['open']),
            'field1_vs_private_gap': float(1 - g['field_r1'].mean() / g['private'].mean()),
            'field1_vs_private_ci95': gap_ci(g['private'], g['field_r1'], np.random.default_rng(7)),
            'field1_vs_private_p': one_sided(g['field_r1'], g['private']),
            'field1_vs_shuffled_gap': float(1 - g['field_r1'].mean() / g['shuffled'].mean()),
            'field1_vs_shuffled_p': one_sided(g['field_r1'], g['shuffled']),
            'shuffled_vs_private_gap': float(1 - g['shuffled'].mean() / g['private'].mean()),
            'broadcast_vs_private_gap': float(1 - g['broadcast'].mean() / g['private'].mean()),
            'field1_fraction_of_broadcast_gain': float((g['private'].mean() - g['field_r1'].mean()) / (g['private'].mean() - g['broadcast'].mean())),
            'field2_fraction_of_broadcast_gain': float((g['private'].mean() - g['field_r2'].mean()) / (g['private'].mean() - g['broadcast'].mean())),
            'permuted_addressed_vs_field1_gap': float(1 - gp['addressed'].mean() / gp['field_r1'].mean()),
            'permuted_addressed_vs_field1_p': one_sided(gp['addressed'], gp['field_r1']),
            'permuted_field1_vs_private_gap': float(1 - gp['field_r1'].mean() / gp['private'].mean()),
            'permuted_addressed_vs_private_gap': float(1 - gp['addressed'].mean() / gp['private'].mean()),
        },
        'wires': {'field_r1': 14, 'broadcast': 56, 'shuffled': 14, 'addressed': 14},
    }
    st = out['stats']
    out['gates'] = {
        'B0_graded_identical_and_addressed_is_field': graded_spread <= 1e-9 and r4.visible_sets('addressed', C) == r4.visible_sets('field_r1'),
        'B1_private_beats_open': g['private'].mean() <= .85 * g['open'].mean() and st['private_vs_open_p'] < 1e-3,
        'B2_field_beats_private': g['field_r1'].mean() <= .90 * g['private'].mean() and st['field1_vs_private_p'] < 1e-3,
        'B3_geometry_matters': st['field1_vs_shuffled_p'] < 1e-3,
        'B4_cheap': st['field1_fraction_of_broadcast_gain'] >= 0.5,
        'B5_conditional_on_metric_world': gp['addressed'].mean() <= .90 * gp['field_r1'].mean() and st['permuted_addressed_vs_field1_p'] < 1e-3,
    }
    out['gates'] = {n: bool(v) for n, v in out['gates'].items()}
    per_world = {f'{w}|{a}': c[:, 8].tolist() for w, v in res.items() for a, c in v.items()}
    return out, per_world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--particles', type=int, default=PARTICLES)
    ap.add_argument('--output', default='results/protocol4.json')
    ap.add_argument('--details', default='results/protocol4_per_world.json')
    args = ap.parse_args()
    train, held = (range(3000, 3032), range(1000, 1032)) if args.smoke else (TRAIN, HELDOUT)
    t0 = time.time()
    A, pwA = part_A(train, held, args.particles)
    print('A K8', {n: round(v, 4) for n, v in A['K8'].items()}, flush=True)
    print('A exact', A['exact'], flush=True)
    B, pwB = part_B(held)
    res = {'config': {'heldout': [held.start, held.stop - 1], 'train': [train.start, train.stop - 1],
                      'particles': args.particles, 'smoke': args.smoke, 'sigma': r4.SIGMA, 'K': r4.K},
           'A': A, 'B': B}
    if not args.smoke:
        for name, ok in {**A['gates'], **B['gates']}.items():
            print(name, 'PASS' if ok else 'FAIL')
    print(f'total {time.time() - t0:.0f}s')
    with open(args.output, 'w') as f:
        json.dump(res, f, indent=1)
    with open(args.details, 'w') as f:
        json.dump({**pwA, **pwB}, f)


if __name__ == '__main__':
    main()
