# Kuulustelu — frozen protocol

Written and committed **before** any pilot or held-out run. Thresholds, seeds and
configurations below are not changed after results are seen; any deviation is
recorded in `RESULTS.md`.

## Question

In [LensLuotainTarget](https://github.com/anttiluode/LensLuotainTarget) a receiver
that remembers previous answers chose better next gates than one that did not, by
22/512. Why so small, and when would it be large?

**Claim under test.** The receiver's memory of answers can only matter when the
posterior *shape* depends on the answers. With a graded (linear) soma and a
Gaussian memory it never does, so the best question sequence can be fixed in
advance. With a spiking (threshold) soma it does, so the conversation matters.

## Theorem being demonstrated (G1)

State m₀ ~ N(0, Σ₀). Reads y_t = g̃_tᵀ m₀ + b_t + σε_t with effective gate
g̃_t = M_tᵀ g_t, where the memory after reads is m_t = M_t m₀ and
M_{t+1} = (I − η g_t g_tᵀ) M_t. All of this is linear, so the posterior over m₀
is Gaussian and its covariance follows the Kalman recursion

  Σ' = Σ − Σg̃ g̃ᵀΣ / (g̃ᵀΣg̃ + σ²),

which does not contain y. Any objective that is a function of the posterior
covariance therefore has an open-loop optimum, for every η including η = 1.

## Model (shared by every arm)

- 12 leaky branch traces, decays `linspace(0.35, 0.98, 12)`, history length 32,
  m₀ = F h with h ~ N(0, I₃₂). Σ₀ = F Fᵀ. Measured property of this prior: the
  top eigendirection carries 94.2% of the trace, so the primary task is close to
  one-dimensional. This favours bisection and is stated, not hidden.
- Gate catalogue: the same 24 unit gates as LensLuotainTarget (12 single-branch,
  12 random sparse, seed 20261009).
- Bias grid per gate: b ∈ s_g · linspace(−2.5, 2.5, 21), s_g = sqrt(gᵀΣ₀g).
  504 actions (gate, bias). The same catalogue is offered to every policy.
- **Graded soma:** y = g̃ᵀm₀ + b + σε. (b is a known offset; it carries nothing.)
- **Spiking soma:** y = 1[g̃ᵀm₀ + b + σε > 0]. One bit per read. b is the
  excitability offset (tonic depolarisation / inhibition) chosen with the gate.
- Back-action η ∈ {0 (primary), 1 (destructive), 0.25 (exploratory)}.
- Sensor noise σ ∈ {0.03 (primary, as in LensLuotainTarget), 0.1, 0.3}.
  For scale, prior std of a gated read ranges 0.086–1.251 (median 0.593).

## Receiver (identical estimator for every policy)

Gaussian belief (μ, Σ) over m₀.
- Graded: exact Kalman update.
- Spiking: assumed-density filtering with the probit likelihood
  Φ(s(g̃ᵀm + b)/σ) — exact first and second moments of a one-step update from a
  Gaussian, moment-matched thereafter (Opper 1998; Minka's EP). This is an
  approximation after the first read; it is the same approximation for every
  policy.

Forecast = posterior mean μ.

## Objective and metric

Objective minimised by every policy: posterior trace(Σ) — expected squared error
on random isotropic questions about m₀.

Metric on held-out worlds: 32 fixed unit question vectors q_j (seed 20271009,
Gaussian then normalised), never seen by any policy.

  R = Σ_j (q_jᵀ(μ − m₀))² / Σ_j q_jᵀΣ₀q_j   per world, averaged over worlds.

R = 1 means nothing learned; R = 0 means perfect.

## Policies

1. **Adaptive** — greedy: each step choose the action minimising the expected
   posterior trace, expectation over the answer predicted by the current belief.
   Uses previous answers (through μ, Σ) to choose.
2. **Open-loop** — a single fixed action sequence, chosen greedily on 512
   *training* worlds (seeds 3000–3511): each step pick the action minimising the
   mean over training worlds of expected posterior trace, then advance each
   training world with its own simulated answer. Applied unchanged to every
   held-out world. Same receiver, same catalogue, answers integrated for
   estimation but never used for choosing.
3. **Random** — uniformly random action each step.

Budget K = 1…12 reads, recorded at every step; **primary K = 8**.
Held-out worlds: seeds 2000–2511 (512). Smoke/pilot: seeds 1000–1031, used only
to check that the code runs.

## Gates (frozen)

- **G0 correctness** — unit tests pass: Kalman update equals the batch Gaussian
  posterior to 1e-10; probit ADF one-step mean and variance match importance
  sampling within 2% (1-D and 3-D); effective-gate bookkeeping matches direct
  simulation of the written memory.
- **G1 theorem** — graded soma, σ = 0.03, η ∈ {0, 1}: adaptive and open-loop
  choose the identical action sequence in 512/512 held-out worlds and
  |R_adaptive − R_open| ≤ 1e-9 at every K.
- **G2 the spike makes the conversation matter (KILL condition)** — spiking,
  η = 0, σ = 0.03, K = 8: R_adaptive ≤ 0.85 · R_open AND one-sided paired
  Wilcoxon p < 0.001 over the 512 held-out worlds. If this fails the idea is dead.
- **G3 sanity** — same configuration: R_adaptive ≤ 0.85 · R_random.
- **G4 noise shrinks the gap** — gap fraction 1 − R_adaptive/R_open at K = 8,
  spiking, η = 0, is smaller at σ = 0.3 than at σ = 0.03.
- **G5 destruction shrinks the gap** — spiking, σ = 0.03, K = 8: gap fraction at
  η = 1 is smaller than at η = 0. Reason: bisection needs to ask the same
  direction repeatedly at different thresholds; a destructive read removes the
  direction it asked.

## Exploratory (reported, not gated)

- η = 0.25 for continuity with LensLuotainTarget.
- **Isotropic prior** arm: m₀ ~ N(0, (tr Σ₀/12) I₁₂), same trace, twelve equal
  directions. Prediction: spiking gap smaller than with the real prior (8 bits
  must spread over 12 directions) but positive.
- Graded vs spiking absolute R (how much a one-bit soma costs).
- Bias ablation: adaptive gate choice with bias fixed at b = 0 (threshold at
  prior mean), to see whether the gain comes from adapting the threshold.

## What this does not test

No learning: policies are Bayesian and programmed. No spike waveform, no
biological threshold dynamics, no dictionary-free *learned* gate (that is Sol's
experiment in LensLuotainTarget). The receiver knows the prior and the forward
model. ADF is approximate for the spiking soma.
