import unittest

import numpy as np
from scipy.special import ndtr

import kuulustelu as k


def random_spd(n, seed):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T / n + 0.1 * np.eye(n)


class TestReceiver(unittest.TestCase):
    def test_kalman_equals_batch_posterior(self):
        n, sigma, eta = 5, 0.2, 0.5
        rng = np.random.default_rng(1)
        S0 = random_spd(n, 2)
        m0 = rng.multivariate_normal(np.zeros(n), S0)
        mu, S, M = np.zeros(n), S0.copy(), np.eye(n)
        H, ys, bs = [], [], []
        for t in range(7):
            g = rng.normal(size=n); g /= np.linalg.norm(g)
            gt = M.T @ g
            b = rng.normal()
            y = gt @ m0 + b + sigma * rng.normal()
            mu, S = k.kalman_update(mu, S, gt, b, y, sigma)
            H.append(gt); ys.append(y - b)
            M = M - eta * np.outer(g, g @ M)
        H, ys = np.array(H), np.array(ys)
        P = np.linalg.inv(np.linalg.inv(S0) + H.T @ H / sigma ** 2)
        np.testing.assert_allclose(S, P, atol=1e-10)
        np.testing.assert_allclose(mu, P @ H.T @ ys / sigma ** 2, atol=1e-10)

    def _probit_vs_importance(self, n, seed):
        rng = np.random.default_rng(seed)
        S0 = random_spd(n, seed + 1)
        mu0 = rng.normal(size=n) * 0.3
        g = rng.normal(size=n); g /= np.linalg.norm(g)
        b, sigma = -0.2, 0.3
        x = rng.multivariate_normal(mu0, S0, size=2_000_000)
        for s in (1.0, -1.0):
            w = ndtr(s * (x @ g + b) / sigma)
            w /= w.sum()
            m_is = w @ x
            C_is = (x - m_is).T @ ((x - m_is) * w[:, None])
            mu, S = k.probit_update(mu0, S0, g, b, s, sigma)
            sd = np.sqrt(np.diag(S0))
            self.assertLess(np.max(np.abs(mu - m_is) / sd), 0.02)
            self.assertLess(np.max(np.abs(np.diag(S) - np.diag(C_is)) / np.diag(C_is)), 0.02)

    def test_probit_1d(self):
        self._probit_vs_importance(1, 10)

    def test_probit_3d(self):
        self._probit_vs_importance(3, 20)

    def test_effective_gate_matches_written_memory(self):
        rng = np.random.default_rng(3)
        n, eta = 12, 0.7
        m0 = rng.normal(size=n)
        m, M = m0.copy(), np.eye(n)
        G = k.gate_catalog()
        for t in range(10):
            g = G[rng.integers(len(G))]
            self.assertAlmostEqual(g @ m, (M.T @ g) @ m0, places=12)
            m = m - eta * g * (g @ m)
            M = M - eta * np.outer(g, g @ M)

    def test_expected_reduction_matches_explicit_average(self):
        n, sigma = 4, 0.25
        S0 = random_spd(n, 5)
        mu0 = np.random.default_rng(6).normal(size=n) * 0.4
        G = np.random.default_rng(7).normal(size=(9, n))
        G /= np.linalg.norm(G, axis=1, keepdims=True)
        bias = np.linspace(-1, 1, 9)
        red = k.expected_reduction(mu0[None], S0[None], G[None], bias, sigma, 'spiking')[0]
        for a in range(9):
            v = G[a] @ S0 @ G[a] + sigma ** 2
            p_plus = ndtr((G[a] @ mu0 + bias[a]) / np.sqrt(v))
            tr = 0.0
            for s, p in ((1.0, p_plus), (-1.0, 1 - p_plus)):
                tr += p * np.trace(k.probit_update(mu0, S0, G[a], bias[a], s, sigma)[1])
            self.assertAlmostEqual(red[a], np.trace(S0) - tr, places=10)
        red_g = k.expected_reduction(mu0[None], S0[None], G[None], bias, sigma, 'graded')[0]
        for a in range(9):
            Sg = S0 @ G[a]
            self.assertAlmostEqual(red_g[a], Sg @ Sg / (G[a] @ Sg + sigma ** 2), places=12)


class TestMixedPrior(unittest.TestCase):
    def test_mix_one_reproduces_traces_exactly(self):
        np.testing.assert_array_equal(k.prior_cov('mix:1.0'), k.prior_cov('traces'))
        a, _, _ = k.worlds(range(2000, 2010), 'mix:1.0', 4)
        b, _, _ = k.worlds(range(2000, 2010), 'traces', 4)
        np.testing.assert_array_equal(a, b)

    def test_mix_worlds_have_the_stated_covariance(self):
        for alpha in (0.0, 0.5):
            m0, _, _ = k.worlds(range(20000), f'mix:{alpha}', 1)
            S = k.prior_cov(f'mix:{alpha}')
            emp = m0.T @ m0 / len(m0)
            self.assertLess(np.max(np.abs(emp - S)) / np.max(np.abs(S)), 0.05)

    def test_noise_prefix_does_not_depend_on_budget(self):
        _, n12, _ = k.worlds(range(2000, 2004), 'iso', 12)
        _, n36, _ = k.worlds(range(2000, 2004), 'iso', 36)
        np.testing.assert_array_equal(n12, n36[:, :12])


class TestTieBreak(unittest.TestCase):
    def test_near_ties_resolve_to_lowest_index(self):
        s = np.array([[1.0, 1.0 + 1e-15, 0.5], [0.2, 0.9, 0.9 * (1 + 1e-14)]])
        self.assertEqual(k.pick(s).tolist(), [0, 1])
        self.assertEqual(k.pick(np.array([0.1, 0.3, 0.2])).tolist(), [1])


class TestTheoremSmoke(unittest.TestCase):
    def test_graded_adaptive_equals_open_loop(self):
        for eta in (0.0, 1.0):
            sched = k.run('open', 'graded', eta, .03, 'traces', range(3000, 3016), 6,
                          train=True)['schedule']
            a = k.run('adaptive', 'graded', eta, .03, 'traces', range(1000, 1008), 6)
            o = k.run('open', 'graded', eta, .03, 'traces', range(1000, 1008), 6, schedule=sched)
            self.assertTrue(np.array_equal(a['actions'], o['actions']))
            np.testing.assert_allclose(a['R'], o['R'], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
