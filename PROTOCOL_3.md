# Kuulustelu — protocol 3: does a *learned* receiver find the conversation?

Written and committed before any training run. Thresholds, seeds, architecture
and training budget are frozen.

## Why

Rounds 1–2 used a programmed Bayesian receiver. Its threshold-centring follows
from the expected-variance rule it was given, so it is a consequence of that
rule, not a finding. The question here: **trained only to minimise prediction
error, with no Bayesian machinery, does a receiver learn to use earlier spikes to
place later thresholds — and does that earn the same advantage?**

## Shared model

- Memory, prior, held-out worlds and noise: exactly as round 1 (trace prior,
  12 branches, η = 0, σ = 0.03, K = 8). Held-out seeds 2000–2511 with round 1's
  noise arrays, so every learned arm is paired with every round-1 arm.
- **Action space differs from rounds 1–2:** the gate is any unit vector in R¹²
  (not the 24-gate catalogue) and the bias is any real number.
- **Receiver:** recurrent state hₜ ∈ R⁶⁴, h₀ = 0,
  hₜ₊₁ = tanh(W_h hₜ + W_x xₜ + c), xₜ = [gₜ, bₜ, sₜ] (the issued action and the
  answer heard).
- **Decoder:** m̂ = D₂ tanh(D₁ h_K + d₁) + d₂, hidden width 64.
- **Adaptive arm:** [gₜ raw, bₜ] = A hₜ + a; gₜ = raw/‖raw‖. The next question is
  a function of everything heard so far.
- **Open-loop arm:** gₜ, bₜ are free parameters per step (8 × 13), learned but not
  conditioned on answers. Same recurrent state and decoder, so it uses the
  answers for estimation only.
- **Spiking soma:** forward sₜ = sign(gₜᵀm₀ + bₜ + σε), exactly ±1. Backward:
  straight-through surrogate derivative of tanh(v/τ), τ = 0.1.
- **Graded soma:** sₜ = gₜᵀm₀ + bₜ + σε (no surrogate needed).
- Loss: ‖m̂ − m₀‖² / tr Σ₀ (isotropic; the 32 test questions are never used in
  training).

## Training

Adam, learning rate 3e-3, batch 256 fresh prior worlds per step, 4000 steps.
Worlds are drawn from a per-run training stream (seed 7000 + 10·init + arm).
Validation on seeds 4000–4511 every 200 steps; the checkpoint with the best
validation loss is kept. Three independent initialisations (init 0, 1, 2) per
arm. Four arms: {spiking, graded} × {adaptive, open-loop}. Held-out worlds are
evaluated once, after training, on the kept checkpoint.

## Diagnostic: standardised threshold position

For each read t ≥ 2 (the first read has nothing to adapt to), replay the
receiver's own issued gates, biases and spikes through the probit ADF posterior
of round 1 and compute

  zₜ = (gₜᵀμₜ + bₜ) / √(gₜᵀΣₜgₜ + σ²),

where (μₜ, Σₜ) is the posterior just before read t. |z| ≈ 0 means the threshold
sits at the current best guess (bisection). A fixed schedule cannot do that
across worlds, because its thresholds don't depend on what was heard.

## Gates (frozen)

- **L0 correctness** — tests: forward spike is exactly ±1; backward equals the
  surrogate derivative; open-loop actions are unchanged when the answers are
  permuted; held-out m₀ and noise are bit-identical to round 1's.
- **L1 the learned receiver converses (KILL condition)** — spiking, K = 8:
  R_adaptive ≤ 0.85 · R_open AND one-sided paired Wilcoxon p < 0.001, in at least
  2 of 3 inits (init i adaptive vs init i open).
- **L2 threshold-centring is learned** — spiking: median |zₜ| (t ≥ 2) of the
  adaptive arm ≤ 0.5 × that of the open-loop arm, in at least 2 of 3 inits.
- **L3 beats the programmed fixed schedule** — spiking learned adaptive
  R < 0.1216 (round 1's Bayesian open-loop) in at least 2 of 3 inits.
- **L4 graded control** — graded: |1 − R_adaptive/R_open| ≤ 0.15 in at least 2 of
  3 inits. A learned receiver should find nothing to gain from answers when they
  are graded.

## Reported, not gated

- Each learned arm's own reads, re-scored with the Bayesian ADF posterior. This
  separates the quality of the questions from the quality of the learned decoder.
- Median |z| of round 1's programmed Bayesian adaptive and open-loop policies,
  for reference.
- Comparison with round 1's Bayesian adaptive (0.0381).

## Limits stated in advance

Small recurrent networks; one surrogate width; one training budget. A failure of
L1 would mean *this* learner did not find the conversation, not that no learner
can. The continuous action space means the learned open-loop arm is not the same
object as round 1's catalogue schedule.
