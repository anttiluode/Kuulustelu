"""Gate A0 / B0 correctness tests for round 4."""
import itertools
import unittest

import numpy as np
from scipy.special import log_ndtr, ndtr

import kuulustelu as k
import round4 as r4


class TestPartA(unittest.TestCase):
    def setUp(self):
        self.seeds = range(1000, 1012)
        self.m0, self.ns, self.nn = r4.worlds_A(self.seeds)
        self.nbits = r4.neighbour_bits(self.m0, self.nn)

    def test_batched_probit_matches_round1(self):
        S0 = k.prior_cov('traces')
        g = k.gate_catalog()[15]
        mu1, S1 = k.probit_update(np.zeros(12), S0, g, 0.1, -1.0, 0.07)
        mu2, S2 = r4.probit_batch(np.zeros((1, 12)), S0[None], g, np.array([0.1]),
                                  np.array([-1.0]), 0.07)
        np.testing.assert_allclose(mu2[0], mu1, atol=1e-14)
        np.testing.assert_allclose(S2[0], S1, atol=1e-14)

    def test_probit_with_inflated_noise_vs_importance(self):
        # sigma_eff is just a larger probit noise: check moments against IS
        S0 = k.prior_cov('traces')
        g = k.gate_catalog()[22]
        x = np.random.default_rng(0).multivariate_normal(np.zeros(12), S0, size=1_000_000)
        for s, sig in ((1.0, 0.2), (-1.0, 0.5)):
            w = ndtr(s * (x @ g + 0.05) / sig)
            w /= w.sum()
            m_is = w @ x
            mu, S = r4.probit_batch(np.zeros((1, 12)), S0[None], g, np.array([0.05]),
                                    np.array([s]), sig)
            sd = np.sqrt(np.diag(S0))
            self.assertLess(np.max(np.abs(mu[0] - m_is) / sd), 0.02)

    def test_replay_recovers_self_thresholds(self):
        s, b = r4.run_sender('self', self.m0, self.ns, self.nbits)
        np.testing.assert_allclose(r4.replay_self(s), b, atol=1e-12)

    def test_open_sender_uses_round1_biases(self):
        _, B = r4.schedule()
        _, b = r4.run_sender('open', self.m0, self.ns, self.nbits)
        np.testing.assert_array_equal(b, np.repeat(B[None], len(self.m0), 0))

    def test_tree_matches_sender_on_realised_branch(self):
        s, b = r4.run_sender('field', self.m0, self.ns, self.nbits)
        tree = r4.threshold_tree(s)
        for w in range(len(self.m0)):
            idx = 0
            for t in range(r4.K):
                idx = 2 * idx + (0 if self.nbits[w, t] > 0 else 1)
                self.assertAlmostEqual(tree[t][w, idx], b[w, t], places=12)

    def test_tree_loglik_equals_bruteforce(self):
        upto = 3
        G, B = r4.schedule()
        s, _ = r4.run_sender('field', self.m0, self.ns, self.nbits)
        tree = r4.threshold_tree(s, upto=upto)
        x = np.random.default_rng(1).multivariate_normal(np.zeros(12), k.prior_cov(), size=50)
        w = 0
        brute = []
        for n in itertools.product([1.0, -1.0], repeat=upto):
            # thresholds by replaying the sender along this branch
            mu, S = np.zeros((1, 12)), k.prior_cov()[None]
            ll = np.zeros(len(x))
            for t in range(upto):
                mu, S = r4.probit_batch(mu, S, G[t], B[t], np.array([n[t]]), r4.SIGMA)
                bt = -(mu[0] @ G[t])
                ll += log_ndtr(n[t] * (x @ G[t] + B[t]) / r4.SIGMA)
                ll += log_ndtr(s[w, t] * (x @ G[t] + bt) / r4.SIGMA)
                mu, S = r4.probit_batch(mu, S, G[t], np.array([bt]), s[w, t:t + 1], r4.SIGMA)
            brute.append(ll)
        brute = np.logaddexp.reduce(np.array(brute), axis=0)
        got = r4.tree_loglik(x, s[w], [tr[w] for tr in tree], upto=upto)
        np.testing.assert_allclose(got, brute, atol=1e-10)

    def test_unlabeled_equals_labeled_when_threshold_is_self_driven(self):
        # with c = 0 the threshold does not depend on n, so marginalising n is a no-op
        s, b = r4.run_sender('self', self.m0, self.ns, self.nbits)
        G, B = r4.schedule()
        x = np.random.default_rng(2).multivariate_normal(np.zeros(12), k.prior_cov(), size=40)
        flat = [np.repeat(b[0, t], 2 ** (t + 1)) for t in range(r4.K)]
        np.testing.assert_allclose(r4.tree_loglik(x, s[0], flat),
                                   r4.labeled_loglik(x, s[0], b[0]), atol=1e-9)


class TestPartB(unittest.TestCase):
    def test_private_belief_equals_single_cell(self):
        C = r4.corr_line()
        m0, noise = r4.worlds_B(range(1000, 1004), C)
        G, _ = r4.schedule()
        S0 = k.prior_cov('traces')
        Sig = r4.joint_prior(C)
        i = 3
        mu, S = np.zeros((4, 96)), np.repeat(Sig[None], 4, 0)
        mu1, S1 = np.zeros((4, 12)), np.repeat(S0[None], 4, 0)
        rng = np.random.default_rng(5)
        for t in range(r4.K):
            b, s = rng.normal(size=4) * .3, np.sign(rng.normal(size=4))
            mu, S = r4.probit_batch(mu, S, r4.gate_vec(i, G[t]), b, s, r4.SIGMA)
            mu1, S1 = r4.probit_batch(mu1, S1, G[t], b, s, r4.SIGMA)
        np.testing.assert_allclose(mu[:, i * 12:(i + 1) * 12], mu1, atol=1e-10)

    def test_addressed_equals_field_in_metric_world(self):
        self.assertEqual(r4.visible_sets('addressed', r4.corr_line()), r4.visible_sets('field_r1'))

    def test_shuffled_partner_counts_and_distance(self):
        sh, f1 = r4.visible_sets('shuffled'), r4.visible_sets('field_r1')
        for i in range(r4.P):
            self.assertEqual(len(sh[i]), len(f1[i]))
            self.assertTrue(all(abs(i - j) >= 3 for j in sh[i] if j != i))

    def test_world_covariance(self):
        C = r4.corr_line()
        m0, _ = r4.worlds_B(range(5000, 9000), C)
        emp = np.cov(m0.T)
        np.testing.assert_allclose(emp, r4.joint_prior(C), atol=0.03)


if __name__ == '__main__':
    unittest.main()
