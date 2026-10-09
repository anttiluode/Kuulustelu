"""Round 5 (PROTOCOL_5.md): can a far reader recover the label from a population of senders?"""
import numpy as np
from scipy.special import log_ndtr, logsumexp

import kuulustelu as k
import round4 as r4


def worlds_C(seeds, M):
    m0, noise0, noise_n = r4.worlds_A(seeds)
    noise = [noise0] + [np.array([np.random.default_rng([int(s), 20 + i]).normal(size=r4.K)
                                  for s in seeds]) for i in range(1, M)]
    return m0, noise, noise_n


def run_population(mode, m0, noise, n_bits):
    """M independent senders sharing the field (mode 'field') or not ('self'). Returns (W,M,K) bits, biases."""
    out = [r4.run_sender(mode, m0, nz, n_bits) for nz in noise]
    return np.stack([o[0] for o in out], 1), np.stack([o[1] for o in out], 1)


def adf_decode_pop(s, b, sig, m0, upto=r4.K):
    """Reader ADF over all senders' bits. s, b (W,M,K); sig scalar or (W,M,K). Returns R (W,), mu, S."""
    G, _ = r4.schedule()
    W, M, _ = s.shape
    S0 = k.prior_cov('traces')
    mu, S = np.zeros((W, 12)), np.repeat(S0[None], W, 0)
    sig = np.broadcast_to(np.asarray(sig, dtype=float), s.shape)
    for t in range(upto):
        for i in range(M):
            mu, S = r4.probit_batch(mu, S, G[t], b[:, i, t], s[:, i, t], sig[:, i, t])
    return r4.R_single(mu, m0), mu, S


def tag0_thresholds(s, m_t, V_t):
    bt = np.stack([r4.replay_self(s[:, i]) + m_t[None] for i in range(s.shape[1])], 1)
    sig = np.broadcast_to(np.sqrt(r4.SIGMA ** 2 + V_t)[None, None], s.shape)
    return bt, sig


def trees_pop(s, upto=r4.K):
    """Per sender, thresholds along every neighbour sequence: list over senders of r4.threshold_tree."""
    return [r4.threshold_tree(s[:, i], upto=upto) for i in range(s.shape[1])]


def tree_loglik_pop(x, s_w, trees_w, upto=r4.K):
    """log P(all senders' bits | m = x), marginalised over ONE shared neighbour sequence.

    s_w (M,K) bits of one world; trees_w[i][t] (2^(t+1),) thresholds of sender i.
    """
    G, B = r4.schedule()
    a = x @ G[:upto].T
    L = np.zeros((len(x), 1))
    for t in range(upto):
        L = np.repeat(L, 2, axis=1)
        nb = np.tile([1.0, -1.0], L.shape[1] // 2)
        L = L + log_ndtr(nb[None, :] * (a[:, t:t + 1] + B[t]) / r4.SIGMA)
        for i in range(len(s_w)):
            L = L + log_ndtr(s_w[i, t] * (a[:, t:t + 1] + trees_w[i][t][None, :]) / r4.SIGMA)
    return logsumexp(L, axis=1)
