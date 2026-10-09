"""Protocol 3: learned receivers (adaptive vs open-loop) on spiking and graded somas.

Pure NumPy + autograd. The spike is exactly +-1 in the forward pass; its
backward pass uses the straight-through surrogate d/dv tanh(v / TAU).
"""
import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.extend import primitive, defvjp

import kuulustelu as k

K = 8
SIGMA = 0.03
HIDDEN = 64
TAU = 0.1
LR = 3e-3
BATCH = 256
STEPS = 4000
VAL_EVERY = 200
ARMS = ('spiking-adaptive', 'spiking-open', 'graded-adaptive', 'graded-open')


# ------------------------------------------------------------ spike with surrogate
@primitive
def hard_spike(v):
    return np.where(v > 0, 1.0, -1.0)


def _spike_vjp(ans, v):
    def vjp(g):
        t = np.tanh(v / TAU)
        return g * (1 - t * t) / TAU
    return vjp


defvjp(hard_spike, _spike_vjp)


# ------------------------------------------------------------ model
def init_params(arm, init):
    rng = np.random.default_rng([31, init, ARMS.index(arm)])
    n, H = k.BRANCHES, HIDDEN
    p = {
        'Wh': rng.normal(size=(H, H)) * 0.9 / np.sqrt(H),
        'Wx': rng.normal(size=(H, n + 2)) / np.sqrt(n + 2),
        'c': np.zeros(H),
        'D1': rng.normal(size=(H, H)) / np.sqrt(H),
        'd1': np.zeros(H),
        'D2': rng.normal(size=(n, H)) * 0.1 / np.sqrt(H),
        'd2': np.zeros(n),
    }
    if arm.endswith('adaptive'):
        p['A'] = rng.normal(size=(n + 1, H)) * 0.1 / np.sqrt(H)
        p['a'] = np.concatenate([rng.normal(size=n), [0.0]])
    else:
        p['G'] = rng.normal(size=(K, n))
        p['B'] = np.zeros(K)
    return p


def forward(p, m0, noise, arm, record=False):
    """m0 (W,n), noise (W,K). Returns estimate (W,n) and optionally the reads."""
    W = m0.shape[0]
    h = anp.zeros((W, HIDDEN))
    spiking = arm.startswith('spiking')
    gates, biases, answers = [], [], []
    for t in range(K):
        if arm.endswith('adaptive'):
            raw = anp.dot(h, p['A'].T) + p['a']
            g = raw[:, :k.BRANCHES]
            b = raw[:, k.BRANCHES]
        else:
            g = anp.tile(p['G'][t], (W, 1))
            b = anp.ones(W) * p['B'][t]
        g = g / anp.sqrt(anp.sum(g * g, axis=1, keepdims=True))
        v = anp.sum(g * m0, axis=1) + b + SIGMA * noise[:, t]
        s = hard_spike(v) if spiking else v
        x = anp.concatenate([g, b[:, None], s[:, None]], axis=1)
        h = anp.tanh(anp.dot(h, p['Wh'].T) + anp.dot(x, p['Wx'].T) + p['c'])
        if record:
            gates.append(g); biases.append(b); answers.append(s)
    est = anp.dot(anp.tanh(anp.dot(h, p['D1'].T) + p['d1']), p['D2'].T) + p['d2']
    if record:
        return est, np.stack(gates, 1), np.stack(biases, 1), np.stack(answers, 1)
    return est


def loss(p, m0, noise, arm, trS):
    est = forward(p, m0, noise, arm)
    return anp.mean(anp.sum((est - m0) ** 2, axis=1)) / trS


# ------------------------------------------------------------ data
def training_batch(rng, F):
    h = rng.normal(size=(BATCH, k.HISTORY))
    return h @ F.T, rng.normal(size=(BATCH, K))


def train(arm, init, log=None):
    F = k.trace_filter()
    trS = float(np.trace(k.prior_cov('traces')))
    p = init_params(arm, init)
    mom = {kk: np.zeros_like(v) for kk, v in p.items()}
    var = {kk: np.zeros_like(v) for kk, v in p.items()}
    vm0, vnoise, _ = k.worlds(range(4000, 4512), 'traces', K)
    rng = np.random.default_rng(7000 + 10 * init + ARMS.index(arm))
    dloss = grad(loss)
    best = (np.inf, None, -1)
    for step in range(1, STEPS + 1):
        m0, noise = training_batch(rng, F)
        g = dloss(p, m0, noise, arm, trS)
        for kk in p:
            mom[kk] = 0.9 * mom[kk] + 0.1 * g[kk]
            var[kk] = 0.999 * var[kk] + 0.001 * g[kk] ** 2
            mh = mom[kk] / (1 - 0.9 ** step)
            vh = var[kk] / (1 - 0.999 ** step)
            p[kk] = p[kk] - LR * mh / (np.sqrt(vh) + 1e-8)
        if step % VAL_EVERY == 0:
            vl = float(loss(p, vm0, vnoise, arm, trS))
            if vl < best[0]:
                best = (vl, {kk: v.copy() for kk, v in p.items()}, step)
            if log:
                log(f'{arm} init {init} step {step} val {vl:.4f}')
    return best[1], best[0], best[2]


# ------------------------------------------------------------ evaluation
def R_of(est, m0):
    Q = k.questions()
    S0 = k.prior_cov('traces')
    denom = np.einsum('jn,nk,jk->', Q, S0, Q)
    return np.sum(((est - m0) @ Q.T) ** 2, axis=1) / denom


def bayes_replay(gates, biases, answers, m0, spiking):
    """Re-score a receiver's own reads with round 1's Bayesian receiver; also z."""
    S0 = k.prior_cov('traces')
    W = m0.shape[0]
    mu_all = np.zeros_like(m0)
    z = np.zeros((W, K))
    for w in range(W):
        mu, S = np.zeros(k.BRANCHES), S0.copy()
        for t in range(K):
            g, b, s = gates[w, t], biases[w, t], answers[w, t]
            z[w, t] = (g @ mu + b) / np.sqrt(g @ S @ g + SIGMA ** 2)
            if spiking:
                mu, S = k.probit_update(mu, S, g, b, s, SIGMA)
            else:
                mu, S = k.kalman_update(mu, S, g, b, s, SIGMA)
        mu_all[w] = mu
    return R_of(mu_all, m0), z


def evaluate(p, arm):
    m0, noise, _ = k.worlds(range(2000, 2512), 'traces', K)
    est, g, b, s = forward(p, m0, noise, arm, record=True)
    R = R_of(np.asarray(est), m0)
    R_bayes, z = bayes_replay(g, b, s, m0, arm.startswith('spiking'))
    return {'R': R, 'R_bayes_rescored': R_bayes, 'z': z, 'gates': g, 'biases': b, 'answers': s}


def bayes_policy_z(policy):
    """Median |z| (t >= 2) of round 1's programmed policies, for reference."""
    from experiment import TRAIN
    held = range(2000, 2512)
    sched = None
    if policy == 'open':
        sched = k.run('open', 'spiking', 0.0, SIGMA, 'traces', TRAIN, K, train=True)['schedule']
    r = k.run(policy, 'spiking', 0.0, SIGMA, 'traces', held, K, schedule=sched)
    _, z = bayes_replay(r['asked'], r['bias'], r['heard'], r['m0'], True)
    return float(np.median(np.abs(z[:, 1:])))
