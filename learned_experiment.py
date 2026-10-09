"""Train all protocol-3 arms, evaluate once on held-out worlds, check gates L1-L4."""
import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

import learned as L

INITS = (0, 1, 2)
ROUND1_BAYES_OPEN = 0.1216   # round 1 spiking open-loop, K = 8 (frozen in protocol 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='results/protocol3.json')
    ap.add_argument('--weights', default='results/protocol3_weights.npz')
    args = ap.parse_args()
    runs, weights = {}, {}
    for arm in L.ARMS:
        for init in INITS:
            p, val, step = L.train(arm, init)
            ev = L.evaluate(p, arm)
            key = f'{arm}|{init}'
            runs[key] = {'val_loss': val, 'best_step': step,
                         'R': float(ev['R'].mean()),
                         'R_bayes_rescored': float(ev['R_bayes_rescored'].mean()),
                         'median_abs_z_t2plus': float(np.median(np.abs(ev['z'][:, 1:]))),
                         'spike_fraction_positive': (float(np.mean(ev['answers'] > 0))
                                                     if arm.startswith('spiking') else None),
                         'per_world_R': ev['R'].tolist()}
            for name, v in p.items():
                weights[f'{key}|{name}'] = v
            print(f"{key:22s} best step {step:4d} val {val:.4f}  R {runs[key]['R']:.4f}  "
                  f"R(bayes re-scored) {runs[key]['R_bayes_rescored']:.4f}  "
                  f"median|z| {runs[key]['median_abs_z_t2plus']:.3f}", flush=True)

    per_init, l1, l2, l3, l4 = {}, [], [], [], []
    for init in INITS:
        sa, so = runs[f'spiking-adaptive|{init}'], runs[f'spiking-open|{init}']
        ga, go = runs[f'graded-adaptive|{init}'], runs[f'graded-open|{init}']
        diff = np.array(so['per_world_R']) - np.array(sa['per_world_R'])
        p = float(wilcoxon(diff, alternative='greater').pvalue)
        rec = {'spiking_gap': 1 - sa['R'] / so['R'], 'spiking_wilcoxon_p': p,
               'z_ratio': sa['median_abs_z_t2plus'] / so['median_abs_z_t2plus'],
               'graded_gap': 1 - ga['R'] / go['R']}
        per_init[init] = rec
        l1.append(sa['R'] <= 0.85 * so['R'] and p < 1e-3)
        l2.append(rec['z_ratio'] <= 0.5)
        l3.append(sa['R'] < ROUND1_BAYES_OPEN)
        l4.append(abs(rec['graded_gap']) <= 0.15)
        print(f"init {init}: spiking gap {rec['spiking_gap']:+.3f} (p {p:.1e})  z ratio {rec['z_ratio']:.3f}  "
              f"graded gap {rec['graded_gap']:+.3f}", flush=True)
    gates = {'L1_learned_receiver_converses': sum(l1) >= 2,
             'L2_threshold_centring_learned': sum(l2) >= 2,
             'L3_beats_programmed_fixed_schedule': sum(l3) >= 2,
             'L4_graded_control_no_gain': sum(l4) >= 2}
    gates_detail = {'L1': l1, 'L2': l2, 'L3': l3, 'L4': l4}
    ref = {'bayes_adaptive_median_abs_z': L.bayes_policy_z('adaptive'),
           'bayes_open_median_abs_z': L.bayes_policy_z('open')}
    print('reference median|z|: Bayesian adaptive', round(ref['bayes_adaptive_median_abs_z'], 3),
          'Bayesian open-loop', round(ref['bayes_open_median_abs_z'], 3))
    for g, ok in gates.items():
        print(g, 'PASS' if ok else 'FAIL', gates_detail[g[:2]])
    with open(args.output, 'w') as f:
        json.dump({'runs': runs, 'per_init': per_init, 'gates': gates,
                   'gates_per_init': gates_detail, 'reference': ref}, f, indent=1)
    np.savez_compressed(args.weights, **weights)


if __name__ == '__main__':
    main()
