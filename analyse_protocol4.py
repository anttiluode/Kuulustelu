"""Post-hoc diagnostics for protocol 4 (added after the held-out run; not gates).

1. Part A: how much does one of S's bits say about the value it was asked about,
   to a reader who does not know the threshold? Correlation of s_t with g_t^T m0.
2. Part B: how well centred are the thresholds under each arm? Median |z| with
   z = (g^T m0 + b) / prior std of the read, over all cells and reads.
"""
import json

import numpy as np

import kuulustelu as k
import round4 as r4
from experiment import HELDOUT


def main():
    G, _ = r4.schedule()
    S0 = k.prior_cov('traces')
    sd = np.sqrt(np.einsum('ti,ij,tj->t', G, S0, G))
    m0, ns, nn = r4.worlds_A(HELDOUT)
    nb = r4.neighbour_bits(m0, nn)
    out = {'A_bit_value_correlation': {}, 'A_threshold_offset_median_abs_z': {}}
    for mode in ('open', 'self', 'field'):
        s, b = r4.run_sender(mode, m0, ns, nb)
        v = m0 @ G.T
        out['A_bit_value_correlation'][mode] = [float(np.corrcoef(s[:, t], v[:, t])[0, 1]) for t in range(r4.K)]
        out['A_threshold_offset_median_abs_z'][mode] = float(np.median(np.abs((v + b) / sd[None])))
    C = r4.corr_line()
    mB, nz = r4.worlds_B(HELDOUT, C)
    out['B_median_abs_z'] = {}
    for arm in ('open', 'private', 'field_r1', 'broadcast'):
        _, bias = r4.run_population(arm, 'spiking', C, mB, nz)
        v = np.einsum('wpk,tk->wtp', mB.reshape(len(mB), r4.P, r4.N_PER), G)
        z = (v + bias) / sd[None, :, None]
        out['B_median_abs_z'][arm] = {'all_reads': float(np.median(np.abs(z))),
                                      'by_read': np.median(np.abs(z), axis=(0, 2)).tolist()}
    print(json.dumps(out, indent=1))
    with open('results/protocol4_diagnostics.json', 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
