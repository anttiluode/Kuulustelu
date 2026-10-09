"""Kuulustelu: graded vs spiking soma under adaptive and open-loop questioning.

A branch memory m0 (12 leaky traces of a 32-step history) is read through gates.
Each read is an action (gate g, bias b). The memory may be written by the read:
m_{t+1} = (I - eta g g^T) m_t. The receiver keeps a Gaussian belief over m0.

graded soma : y = g~^T m0 + b + sigma*eps              (exact Kalman update)
spiking soma: y = 1[g~^T m0 + b + sigma*eps > 0]       (probit ADF update)

where g~ = M_t^T g is the effective gate on the ORIGINAL memory.
"""
import numpy as np
from scipy.special import log_ndtr, ndtr

BRANCHES = 12
HISTORY = 32
N_BIAS = 21
BIAS_SPAN = 2.5
_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


# ---------------------------------------------------------------- model
def decays():
    return np.linspace(.35, .98, BRANCHES)


def trace_filter():
    lam = decays()
    powers = HISTORY - 1 - np.arange(HISTORY)
    return (1 - lam[:, None]) * lam[:, None] ** powers[None, :]


def prior_cov(prior='traces'):
    F = trace_filter()
    S = F @ F.T
    if prior == 'traces':
        return S
    if prior == 'iso':
        return np.eye(BRANCHES) * np.trace(S) / BRANCHES
    raise ValueError('prior must be traces or iso')


def gate_catalog():
    """The 24 unit gates of LensLuotainTarget (12 single-branch, 12 sparse)."""
    rng = np.random.default_rng(20261009)
    gates = [row for row in np.eye(BRANCHES)]
    for _ in range(12):
        mask = np.zeros(BRANCHES)
        count = int(rng.integers(2, 7))
        mask[rng.choice(BRANCHES, count, replace=False)] = 1 / np.sqrt(count)
        gates.append(mask)
    return np.array(gates)


def actions(S0):
    """All (gate, bias) pairs. Bias grid scaled by the prior std of each gate."""
    G = gate_catalog()
    sg = np.sqrt(np.einsum('ij,jk,ik->i', G, S0, G))
    grid = np.linspace(-BIAS_SPAN, BIAS_SPAN, N_BIAS)
    gid = np.repeat(np.arange(len(G)), N_BIAS)
    bias = (sg[:, None] * grid[None, :]).ravel()
    bidx = np.tile(np.arange(N_BIAS), len(G))
    return G[gid], bias, gid, bidx


def questions():
    q = np.random.default_rng(20271009).normal(size=(32, BRANCHES))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def worlds(seeds, prior, K):
    """Hidden memories and per-step noise; identical across policies (paired)."""
    S0 = prior_cov(prior)
    F = trace_filter()
    m0, noise, rnd = [], [], []
    for s in seeds:
        r = np.random.default_rng([int(s), 1])
        if prior == 'traces':
            m0.append(F @ r.normal(size=HISTORY))
        else:
            m0.append(np.sqrt(S0[0, 0]) * r.normal(size=BRANCHES))
        noise.append(np.random.default_rng([int(s), 2]).normal(size=K))
        rnd.append(np.random.default_rng([int(s), 3]))
    return np.array(m0), np.array(noise), rnd


# ---------------------------------------------------------------- receiver
def _ratio(z):
    """phi(z)/Phi(z), stable for very negative z."""
    return np.exp(-0.5 * z * z - _LOG_SQRT_2PI - log_ndtr(z))


def kalman_update(mu, S, gt, b, y, sigma):
    Sg = S @ gt
    v = gt @ Sg + sigma ** 2
    mu = mu + Sg * (y - gt @ mu - b) / v
    S = S - np.outer(Sg, Sg) / v
    return mu, S


def probit_update(mu, S, gt, b, s, sigma):
    """ADF / moment-matched update for spike s in {+1, -1}."""
    Sg = S @ gt
    v = gt @ Sg + sigma ** 2
    sv = np.sqrt(v)
    z = s * (gt @ mu + b) / sv
    r = _ratio(z)
    mu = mu + s * Sg * r / sv
    S = S - np.outer(Sg, Sg) * r * (z + r) / v
    return mu, S


def expected_reduction(mu, S, Gt, bias, sigma, soma):
    """Expected drop in trace(S) for every action, batched over worlds.

    mu (W,n), S (W,n,n), Gt (W,A,n) effective gates, bias (A,). Returns (W,A).
    """
    SG = np.einsum('wan,wnk->wak', Gt, S)
    v = np.einsum('wak,wak->wa', SG, Gt) + sigma ** 2
    q = np.einsum('wak,wak->wa', SG, SG)
    if soma == 'graded':
        return q / v
    sv = np.sqrt(v)
    z = (np.einsum('wan,wn->wa', Gt, mu) + bias[None, :]) / sv
    # sum over s=+-1 of P(s) r_s (z_s + r_s), with P(s) r_s = phi(z_s)
    phi = np.exp(-0.5 * z * z - _LOG_SQRT_2PI)
    term = phi * (z + _ratio(z)) + phi * (-z + _ratio(-z))
    return q / v * term


# ---------------------------------------------------------------- policies
TIE_RTOL = 1e-10


def pick(scores):
    """Argmax with deterministic tie-breaking (lowest index within TIE_RTOL).

    Exact ties are common (isotropic prior, zeroed branches under eta=1); without
    a tolerance, last-bit rounding would decide them differently for the
    per-world adaptive score and the averaged open-loop score.
    """
    scores = np.atleast_2d(scores)
    best = np.max(scores, axis=1, keepdims=True)
    tol = TIE_RTOL * np.maximum(np.abs(best), 1e-300)
    return np.argmax(scores >= best - tol, axis=1)


def run(policy, soma, eta, sigma, prior, seeds, K, schedule=None, train=False):
    """Run one policy on a batch of worlds. Returns per-world R curve and actions.

    policy: 'adaptive' | 'open' | 'random' | 'adaptive_b0'
    For policy 'open', pass the fixed schedule (list of action indices).
    With train=True the batch is used to BUILD the open-loop schedule greedily.
    """
    S0 = prior_cov(prior)
    AG, AB, gid, bidx = actions(S0)
    m0, noise, rnd = worlds(seeds, prior, K)
    W, n = m0.shape
    Q = questions()
    denom = np.einsum('jn,nk,jk->', Q, S0, Q)
    mu = np.zeros((W, n))
    S = np.repeat(S0[None], W, axis=0)
    M = np.repeat(np.eye(n)[None], W, axis=0)
    R = np.zeros((W, K + 1))
    Rpred = np.zeros((W, K + 1))
    chosen = np.zeros((W, K), dtype=int)
    built = []

    def score_R():
        err = (mu - m0) @ Q.T
        return np.sum(err ** 2, axis=1) / denom

    def predicted_R():
        return np.einsum('jn,wnk,jk->w', Q, S, Q) / denom

    R[:, 0], Rpred[:, 0] = score_R(), predicted_R()
    for t in range(K):
        Gt = np.einsum('an,wnk->wak', AG, M)  # rows g^T M_t = (M_t^T g)^T
        if train or policy in ('adaptive', 'adaptive_b0'):
            red = expected_reduction(mu, S, Gt, AB, sigma, soma)
        if train:
            a = np.full(W, int(pick(red.mean(axis=0))[0]))
            built.append(int(a[0]))
        elif policy == 'adaptive':
            a = pick(red)
        elif policy == 'adaptive_b0':
            red = np.where(bidx[None, :] == N_BIAS // 2, red, -np.inf)
            a = pick(red)
        elif policy == 'open':
            a = np.full(W, int(schedule[t]))
        elif policy == 'random':
            a = np.array([int(r.integers(len(AB))) for r in rnd])
        else:
            raise ValueError(policy)
        chosen[:, t] = a
        for w in range(W):
            gt = Gt[w, a[w]]
            b = AB[a[w]]
            val = gt @ m0[w] + b + sigma * noise[w, t]
            if soma == 'graded':
                mu[w], S[w] = kalman_update(mu[w], S[w], gt, b, val, sigma)
            else:
                s = 1.0 if val > 0 else -1.0
                mu[w], S[w] = probit_update(mu[w], S[w], gt, b, s, sigma)
            g = AG[a[w]]
            M[w] = M[w] - eta * np.outer(g, g @ M[w])
        R[:, t + 1], Rpred[:, t + 1] = score_R(), predicted_R()
    return {'R': R, 'R_predicted': Rpred, 'actions': chosen, 'schedule': built,
            'gate': gid[chosen], 'bias_index': bidx[chosen]}
