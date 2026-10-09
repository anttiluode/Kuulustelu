"""Gate C0 for round 5."""
import itertools
import unittest

import numpy as np
from scipy.special import log_ndtr

import kuulustelu as k
import round4 as r4
import round5 as r5


class TestPopulation(unittest.TestCase):
    def test_population_tree_equals_bruteforce(self):
        upto, M = 3, 2
        m0, noise, nn = r5.worlds_C(range(1000, 1006), M)
        nb = r4.neighbour_bits(m0, nn)
        s, _ = r5.run_population('field', m0, noise, nb)
        G, B = r4.schedule()
        x = np.random.default_rng(1).multivariate_normal(np.zeros(12), k.prior_cov(), size=40)
        w = 2
        brute = []
        for n in itertools.product([1.0, -1.0], repeat=upto):
            ll = np.zeros(len(x))
            for t in range(upto):
                ll += log_ndtr(n[t] * (x @ G[t] + B[t]) / r4.SIGMA)
            for i in range(M):
                mu, S = np.zeros((1, 12)), k.prior_cov()[None]
                for t in range(upto):
                    mu, S = r4.probit_batch(mu, S, G[t], B[t], np.array([n[t]]), r4.SIGMA)
                    bt = -(mu[0] @ G[t])
                    ll += log_ndtr(s[w, i, t] * (x @ G[t] + bt) / r4.SIGMA)
                    mu, S = r4.probit_batch(mu, S, G[t], np.array([bt]), s[w, i, t:t + 1], r4.SIGMA)
            brute.append(ll)
        brute = np.logaddexp.reduce(np.array(brute), axis=0)
        trees = r5.trees_pop(s, upto=upto)
        got = r5.tree_loglik_pop(x, s[w], [[tr[t][w] for t in range(upto)] for tr in trees], upto=upto)
        np.testing.assert_allclose(got, brute, atol=1e-10)

    def test_m1_population_equals_single_sender(self):
        m0, noise, nn = r5.worlds_C(range(1000, 1008), 1)
        nb = r4.neighbour_bits(m0, nn)
        s, b = r5.run_population('field', m0, noise, nb)
        R1 = r5.adf_decode_pop(s, b, r4.SIGMA, m0)[0]
        R2 = r4.adf_decode(s[:, 0], b[:, 0], r4.SIGMA, m0)[0][:, r4.K]
        np.testing.assert_allclose(R1, R2, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
