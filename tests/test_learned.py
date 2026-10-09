import unittest

import numpy as np
from autograd import grad

import kuulustelu as k
import learned as L


class TestSpike(unittest.TestCase):
    def test_forward_is_exactly_pm1(self):
        v = np.array([-2.0, -1e-9, 0.0, 1e-9, 3.0])
        np.testing.assert_array_equal(L.hard_spike(v), [-1, -1, -1, 1, 1])

    def test_backward_is_surrogate(self):
        v = np.linspace(-0.3, 0.3, 7)
        g = grad(lambda x: np.sum(L.hard_spike(x) * np.arange(1, 8)))(v)
        expect = np.arange(1, 8) * (1 - np.tanh(v / L.TAU) ** 2) / L.TAU
        np.testing.assert_allclose(g, expect, rtol=1e-12)


class TestArms(unittest.TestCase):
    def _data(self):
        m0, noise, _ = k.worlds(range(1000, 1016), 'traces', L.K)
        return m0, noise

    def test_open_loop_actions_ignore_answers(self):
        m0, noise = self._data()
        p = L.init_params('spiking-open', 0)
        _, g1, b1, s1 = L.forward(p, m0, noise, 'spiking-open', record=True)
        perm = np.random.default_rng(0).permutation(len(m0))
        _, g2, b2, s2 = L.forward(p, m0[perm], noise[perm], 'spiking-open', record=True)
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(b1, b2)
        self.assertFalse(np.array_equal(s1, s2))

    def test_adaptive_actions_do_depend_on_answers(self):
        m0, noise = self._data()
        p = L.init_params('spiking-adaptive', 0)
        _, g, b, _ = L.forward(p, m0, noise, 'spiking-adaptive', record=True)
        np.testing.assert_allclose(g[:, 0], g[0, 0][None].repeat(len(m0), 0))  # read 1: no answers yet
        self.assertGreater(np.std(b[:, 1]), 0)

    def test_heldout_worlds_match_round1(self):
        a = k.worlds(range(2000, 2512), 'traces', L.K)
        b = k.worlds(range(2000, 2512), 'traces', 12)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1][:, :L.K])

    def test_replay_z_is_zero_for_centred_threshold(self):
        m0, noise = self._data()
        g = np.tile(np.eye(12)[0], (len(m0), L.K, 1))
        b = np.zeros((len(m0), L.K))
        s = np.where(m0[:, :1] + 0 * b > 0, 1.0, -1.0)
        _, z = L.bayes_replay(g, b, s, m0, True)
        np.testing.assert_allclose(z[:, 0], 0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
