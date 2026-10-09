"""Round 4 (PROTOCOL_4.md): the question label (part A) and a shared slow field (part B).

Every cell reads through round 1's frozen gate schedule; only thresholds adapt.
Beliefs are Gaussian, updated by probit ADF (spiking) or Kalman (graded), batched
over worlds.
"""
import numpy as np
from scipy.special import log_ndtr, logsumexp

import kuulustelu as k

SIGMA = 0.03
K = 8
# round 1's frozen open-loop schedule (results/summary.json, traces|spiking|eta=0|sigma=0.03)
SCHEDULE = (325, 469, 328, 472, 198, 467, 477, 29)
_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def schedule():
    """Gates (K, 12) and open-loop biases (K,) of round 1's frozen schedule."""
    AG, AB, _, _ = k.actions(k.prior_cov('traces'))
    idx = np.array(SCHEDULE)
    return AG[idx], AB[idx]


# ------------------------------------------------------------ batched updates
def probit_batch(mu, S, g, b, s, sigma):
    """ADF probit update. mu (...,n), S (...,n,n), g (...,n) or (n,), b, s, sigma broadcast to (...)."""
    g = np.broadcast_to(g, mu.shape)
    Sg = np.einsum('...ij,...j->...i', S, g)
    v = np.einsum('...i,...i->...', g, Sg) + np.asarray(sigma) ** 2
    sv = np.sqrt(v)
    z = s * (np.einsum('...i,...i->...', g, mu) + b) / sv
    r = np.exp(-0.5 * z * z - _LOG_SQRT_2PI - log_ndtr(z))
    mu = mu + (s * r / sv)[..., None] * Sg
    S = S - (r * (z + r) / v)[..., None, None] * Sg[..., :, None] * Sg[..., None, :]
    return mu, S


def kalman_batch(mu, S, g, b, y, sigma):
    g = np.broadcast_to(g, mu.shape)
    Sg = np.einsum('...ij,...j->...i', S, g)
    v = np.einsum('...i,...i->...', g, Sg) + sigma ** 2
    mu = mu + ((y - np.einsum('...i,...i->...', g, mu) - b) / v)[..., None] * Sg
    S = S - (1 / v)[..., None, None] * Sg[..., :, None] * Sg[..., None, :]
    return mu, S


def R_single(mu, m0):
    """Round 1's metric for 12-dim memories, per world."""
    Q = k.questions()
    S0 = k.prior_cov('traces')
    denom = np.einsum('jn,nk,jk->', Q, S0, Q)
    return np.sum(((mu - m0) @ Q.T) ** 2, axis=-1) / denom


# ================================================================= PART A
def worlds_A(seeds):
    m0, noise_s, _ = k.worlds(seeds, 'traces', K)   # round 1's streams [s,1], [s,2]
    noise_n = np.array([np.random.default_rng([int(s), 5]).normal(size=K) for s in seeds])
    return m0, noise_s, noise_n


def neighbour_bits(m0, noise_n):
    G, B = schedule()
    val = m0 @ G.T + B[None, :] + SIGMA * noise_n
    return np.where(val > 0, 1.0, -1.0)


def run_sender(mode, m0, noise_s, n_bits):
    """mode: 'open' | 'self' (c=0) | 'field' (c=1). Returns S's bits and thresholds (W,K)."""
    G, B = schedule()
    W, n = m0.shape
    S0 = k.prior_cov('traces')
    mu, S = np.zeros((W, n)), np.repeat(S0[None], W, axis=0)
    bits, bias = np.zeros((W, K)), np.zeros((W, K))
    for t in range(K):
        if mode == 'field':
            mu, S = probit_batch(mu, S, G[t], B[t], n_bits[:, t], SIGMA)
        b = np.full(W, B[t]) if mode == 'open' else -(mu @ G[t])
        val = m0 @ G[t] + b + SIGMA * noise_s[:, t]
        s = np.where(val > 0, 1.0, -1.0)
        if mode != 'open':
            mu, S = probit_batch(mu, S, G[t], b, s, SIGMA)
        bits[:, t], bias[:, t] = s, b
    return bits, bias


def replay_self(s_bits):
    """What a c=0 sender would have set, given only its own bits (reader-side replay)."""
    G, _ = schedule()
    W = s_bits.shape[0]
    S0 = k.prior_cov('traces')
    mu, S = np.zeros((W, S0.shape[0])), np.repeat(S0[None], W, axis=0)
    bhat = np.zeros((W, K))
    for t in range(K):
        b = -(mu @ G[t])
        mu, S = probit_batch(mu, S, G[t], b, s_bits[:, t], SIGMA)
        bhat[:, t] = b
    return bhat


def adf_decode(s_bits, bias, sigma_eff, m0):
    """Reader ADF with given thresholds and per-step effective noise. Returns R curve (W,K+1), final belief."""
    G, _ = schedule()
    W, n = m0.shape
    S0 = k.prior_cov('traces')
    mu, S = np.zeros((W, n)), np.repeat(S0[None], W, axis=0)
    L = s_bits.shape[1]
    sig = np.broadcast_to(np.asarray(sigma_eff, dtype=float), (W, K))
    R = np.zeros((W, L + 1))
    R[:, 0] = R_single(mu, m0)
    for t in range(L):
        mu, S = probit_batch(mu, S, G[t], bias[:, t], s_bits[:, t], sig[:, t])
        R[:, t + 1] = R_single(mu, m0)
    return R, mu, S


def threshold_tree(s_bits, upto=K):
    """Sender (c=1) thresholds along every neighbour bit sequence.

    Returns list over t of arrays (W, 2^(t+1)); prefix index p at level t has
    children 2p (n=+1) and 2p+1 (n=-1).
    """
    G, B = schedule()
    W = s_bits.shape[0]
    S0 = k.prior_cov('traces')
    n = S0.shape[0]
    mu, S = np.zeros((W, 1, n)), np.repeat(S0[None, None], W, axis=0)
    out = []
    for t in range(upto):
        mu, S = np.repeat(mu, 2, axis=1), np.repeat(S, 2, axis=1)
        nb = np.tile([1.0, -1.0], mu.shape[1] // 2)[None, :]
        mu, S = probit_batch(mu, S, G[t], B[t], nb, SIGMA)
        b = -(mu @ G[t])
        out.append(b)
        mu, S = probit_batch(mu, S, G[t], b, s_bits[:, t][:, None], SIGMA)
    return out


def tree_loglik(x, s_bits_w, tree_w, upto=K):
    """log P(s_{1:upto} | m=x) marginalised over neighbour bits, for one world.

    x (Np, n) particles; tree_w list of (2^(t+1),) thresholds.
    """
    G, B = schedule()
    a = x @ G[:upto].T
    L = np.zeros((len(x), 1))
    for t in range(upto):
        L = np.repeat(L, 2, axis=1)
        nb = np.tile([1.0, -1.0], L.shape[1] // 2)
        L = L + log_ndtr(nb[None, :] * (a[:, t:t + 1] + B[t]) / SIGMA)
        L = L + log_ndtr(s_bits_w[t] * (a[:, t:t + 1] + tree_w[t][None, :]) / SIGMA)
    return logsumexp(L, axis=1)


def labeled_loglik(x, s_bits_w, bias_w, upto=K):
    G, _ = schedule()
    a = x @ G[:upto].T
    return log_ndtr(s_bits_w[None, :upto] * (a + bias_w[None, :upto]) / SIGMA).sum(1)


def _mvn_logpdf(x, mean, chol):
    z = np.linalg.solve(chol, (x - mean).T).T
    return -0.5 * np.sum(z * z, 1) - np.log(np.diag(chol)).sum() - 0.5 * len(mean) * np.log(2 * np.pi)


def importance_mean(seed, centre_mu, centre_S, loglik_fn, n_particles, tag=8):
    """Posterior mean under the trace prior and a likelihood, by importance sampling.

    Proposal = 1/2 N(centre_mu, 4 centre_S) + 1/2 prior. Returns mean, ESS.
    """
    S0 = k.prior_cov('traces')
    n = S0.shape[0]
    jit = 1e-10 * np.trace(S0) / n * np.eye(n)
    L0 = np.linalg.cholesky(S0 + jit)
    L1 = np.linalg.cholesky(4 * centre_S + jit)
    r = np.random.default_rng([int(seed), tag])
    pick = r.random(n_particles) < 0.5
    z = r.normal(size=(n_particles, n))
    x = np.where(pick[:, None], centre_mu + z @ L1.T, z @ L0.T)
    lp = _mvn_logpdf(x, np.zeros(n), L0)
    lq = np.logaddexp(np.log(.5) + _mvn_logpdf(x, centre_mu, L1), np.log(.5) + lp)
    lw = loglik_fn(x) + lp - lq
    w = np.exp(lw - lw.max())
    w /= w.sum()
    return w @ x, 1.0 / np.sum(w ** 2)


# ================================================================= PART B
P = 8
ELL = 2.0
N_PER = 12


def corr_line(perm=None):
    d = np.abs(np.subtract.outer(np.arange(P), np.arange(P)))
    C = np.exp(-d / ELL)
    if perm is not None:
        C = C[np.ix_(perm, perm)]
    return C


def permutation():
    return np.random.default_rng(4041).permutation(P)


def joint_prior(C):
    return np.kron(C, k.prior_cov('traces'))


def worlds_B(seeds, C):
    F = k.trace_filter()
    L = np.linalg.cholesky(C)
    m0, noise = [], []
    for s in seeds:
        u = np.random.default_rng([int(s), 7]).normal(size=(P, k.HISTORY))
        H = L @ u
        m0.append((F @ H.T).T.ravel())
        noise.append(np.array([np.random.default_rng([int(s), 10 + i]).normal(size=K)
                               for i in range(P)]).T)   # (K, P)
    return np.array(m0), np.array(noise)


def visible_sets(arm, C=None):
    d = np.abs(np.subtract.outer(np.arange(P), np.arange(P)))
    if arm == 'private':
        return [[i] for i in range(P)]
    if arm.startswith('field'):
        r = int(arm.split('r')[1])
        return [sorted(np.flatnonzero(d[i] <= r).tolist()) for i in range(P)]
    if arm == 'broadcast':
        return [list(range(P)) for _ in range(P)]
    base = visible_sets('field_r1')
    if arm == 'shuffled':
        rng = np.random.default_rng(4040)
        out = []
        for i in range(P):
            far = np.flatnonzero(d[i] >= 3)
            part = rng.choice(far, len(base[i]) - 1, replace=False)
            out.append(sorted([i] + part.tolist()))
        return out
    if arm == 'addressed':
        out = []
        for i in range(P):
            c = C[i].copy()
            c[i] = -np.inf
            order = np.argsort(-c, kind='stable')
            out.append(sorted([i] + order[:len(base[i]) - 1].tolist()))
        return out
    raise ValueError(arm)


def gate_vec(i, g):
    v = np.zeros(P * N_PER)
    v[i * N_PER:(i + 1) * N_PER] = g
    return v


def run_population(arm, soma, C, m0, noise, chunk=128):
    """Run one threshold arm. Returns pooled R curve per world (W, K+1) and the biases used."""
    G, B = schedule()
    Sig = joint_prior(C)
    W, D = m0.shape
    S0 = k.prior_cov('traces')
    Q = k.questions()
    denom = P * np.einsum('jn,nk,jk->', Q, S0, Q)
    R = np.zeros((W, K + 1))
    biases = np.zeros((W, K, P))
    sets = None if arm == 'open' else visible_sets(arm, C)
    for c0 in range(0, W, chunk):
        sl = slice(c0, min(W, c0 + chunk))
        m, nz = m0[sl], noise[sl]
        w = m.shape[0]
        if sets is not None:
            mu = np.zeros((w, P, D))
            S = np.repeat(np.repeat(Sig[None, None], w, axis=0), P, axis=1)
        dmu, dS = np.zeros((w, D)), np.repeat(Sig[None], w, axis=0)   # decoder
        err = (dmu.reshape(w, P, N_PER) - m.reshape(w, P, N_PER)) @ Q.T
        R[sl, 0] = np.sum(err ** 2, axis=(1, 2)) / denom
        for t in range(K):
            if sets is None:
                b = np.full((w, P), B[t])
            else:
                b = -np.einsum('wpk,k->wp', mu.reshape(w, P, P, N_PER)[:, np.arange(P), np.arange(P)], G[t])
            val = m.reshape(w, P, N_PER) @ G[t] + b + SIGMA * nz[:, t]
            y = val if soma == 'graded' else np.where(val > 0, 1.0, -1.0)
            biases[sl, t] = b
            if sets is not None:
                for i in range(P):
                    for j in sets[i]:
                        gv = gate_vec(j, G[t])
                        if soma == 'graded':
                            mu[:, i], S[:, i] = kalman_batch(mu[:, i], S[:, i], gv, b[:, j], y[:, j], SIGMA)
                        else:
                            mu[:, i], S[:, i] = probit_batch(mu[:, i], S[:, i], gv, b[:, j], y[:, j], SIGMA)
            for j in range(P):
                gv = gate_vec(j, G[t])
                if soma == 'graded':
                    dmu, dS = kalman_batch(dmu, dS, gv, b[:, j], y[:, j], SIGMA)
                else:
                    dmu, dS = probit_batch(dmu, dS, gv, b[:, j], y[:, j], SIGMA)
            err = (dmu.reshape(w, P, N_PER) - m.reshape(w, P, N_PER)) @ Q.T
            R[sl, t + 1] = np.sum(err ** 2, axis=(1, 2)) / denom
    return R, biases
