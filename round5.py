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


# ---------------------------------------------------------------- mixture proposal (deviation, see RESULTS_5.md)
def branch_beliefs(s, trees):
    """ADF reader belief per world and per neighbour sequence (W, 256, ...), with log evidence.

    Sequence j: bit (K-1-t) of j is 0 for n_t = +1. Prefix at step t = j >> (K-1-t).
    """
    G, B = r4.schedule()
    K = r4.K
    W, M, _ = s.shape
    J = 2 ** K
    j = np.arange(J)
    S0 = k.prior_cov('traces')
    mu, S = np.zeros((W, J, 12)), np.broadcast_to(S0, (W, J, 12, 12)).copy()
    logZ = np.zeros((W, J))

    def upd(mu, S, g, b, y):
        Sg = S @ g
        v = Sg @ g + r4.SIGMA ** 2
        z = y * (mu @ g + b) / np.sqrt(v)
        return log_ndtr(z), *r4.probit_batch(mu, S, g, b, y, r4.SIGMA)

    for t in range(K):
        n_t = np.where((j >> (K - 1 - t)) & 1, -1.0, 1.0)[None, :]
        lz, mu, S = upd(mu, S, G[t], B[t], n_t)
        logZ += lz
        pref = j >> (K - 1 - t)
        for i in range(M):
            b = trees[i][t][:, pref]
            lz, mu, S = upd(mu, S, G[t], b, s[:, i, t][:, None])
            logZ += lz
    return mu, S, logZ


def importance_mean_mixture(seed, mus, Ss, logZ, loglik_fn, n_particles, inflate=2.0, tag=9):
    """IS with proposal = 0.9 * sum_j pi_j N(mu_j, inflate*S_j) + 0.1 * prior, pi_j ~ ADF evidence."""
    S0 = k.prior_cov('traces')
    n = S0.shape[0]
    jit = 1e-8 * np.trace(S0) / n * np.eye(n)
    pi = np.exp(logZ - logZ.max())
    pi = 0.9 * pi / pi.sum() + 0.1 / len(pi)
    L = np.linalg.cholesky(inflate * Ss + jit)
    L0 = np.linalg.cholesky(S0 + jit)
    r = np.random.default_rng([int(seed), tag])
    from_prior = r.random(n_particles) < 0.1
    comp = r.choice(len(pi), size=n_particles, p=pi)
    z = r.normal(size=(n_particles, n))
    x = np.where(from_prior[:, None], z @ L0.T, mus[comp] + np.einsum('pij,pj->pi', L[comp], z))
    Linv = np.linalg.inv(L)
    d = np.einsum('jab,pjb->pja', Linv, x[:, None, :] - mus[None])
    logdet = np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
    lcomp = -0.5 * np.sum(d * d, 2) - logdet[None] - 0.5 * n * np.log(2 * np.pi)
    lmix = np.log(0.9) + np.log(np.maximum(pi, 1e-300))[None] + lcomp
    lp = r4._mvn_logpdf(x, np.zeros(n), L0)
    from scipy.special import logsumexp as lse
    lq = np.logaddexp(lse(lmix, axis=1), np.log(0.1) + lp)
    lw = loglik_fn(x) + lp - lq
    w = np.exp(lw - lw.max())
    w /= w.sum()
    return w @ x, 1.0 / np.sum(w ** 2)
