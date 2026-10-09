# Kuulustelu — protocol 2: why did the isotropic prior give exactly zero?

Written and committed before any run of the experiments below. Same rules as
[PROTOCOL.md](PROTOCOL.md): thresholds, seeds and configurations are frozen.

## Why this exists

Round 1's exploratory isotropic arm gave a spiking gap of exactly zero. The
explanation in [RESULTS.md](RESULTS.md) — "the conversation needs coupling
between questions" — was written **after** seeing that result. It is a post-hoc
story until it predicts something it was not fitted to. This protocol turns it
into two predictions that could fail.

The story, stated precisely: with a spiking soma, an earlier answer helps choose
the next question only if it changes where the next threshold should sit. That
happens in two ways:

- **(a) revisiting** — asking a direction that has already been asked, whose
  posterior is no longer centred at zero;
- **(b) coupling** — asking a direction correlated with one already asked, whose
  posterior mean has moved because of that correlation.

With an isotropic prior, twelve uncorrelated directions and K ≤ 12 reads, the
best greedy question is always a fresh axis thresholded at zero, so neither (a)
nor (b) is ever used, and adaptive and open-loop must coincide.

## Experiment A — revisiting (budget beyond dimension)

Isotropic prior (as in round 1), spiking soma, η = 0, σ = 0.03, budget K = 1…36.
Same 504 actions, same receiver, same policies (adaptive, open-loop built on
training seeds 3000–3511, random). Held-out seeds 2000–2511.

- **A1 (consistency):** adaptive and open-loop choose identical sequences in
  512/512 worlds for every K ≤ 12. (Already seen in round 1 for K ≤ 12 with
  K_max = 12; re-checked here because the open-loop schedule is rebuilt with
  K_max = 36. Not counted as new evidence.)
- **A2 (the prediction):** at K = 24, gap fraction 1 − R_adaptive/R_open ≥ 0.10
  AND one-sided paired Wilcoxon p < 0.001.
- **A3:** the first K at which the adaptive and open-loop mean R differ by more
  than 1e-9 is in 13…16 (the gap opens once fresh axes run out, not later).

If A2 fails, story (a) is wrong.

## Experiment B — coupling (sweep from isotropic to branch traces)

Prior Σ(α) = α Σ_traces + (1 − α) c I, c = tr(Σ_traces)/12, so every prior has
the same total variance. Worlds: m₀ = √α F h + √(1−α) √c z, with h from the
same random stream as round 1's trace worlds (so α = 1 reproduces round 1
exactly) and z from a separate stream.

α ∈ {0, 0.25, 0.5, 0.75, 0.9, 1.0}. Effective dimension (participation ratio
(tr Σ)²/tr Σ²), a property of the prior and not a result: 12.00, 7.47, 3.51,
1.86, 1.36, 1.12.

Spiking soma, η = 0, σ = 0.03, K = 8 primary.

- **B1 (replication):** α = 1 reproduces round 1's numbers (R_adaptive 0.0381,
  R_open 0.1216) to 1e-12.
- **B2 (the prediction):** gap fraction at K = 8 is non-decreasing in α across
  the six values, allowing at most one inversion of size ≤ 0.03.
- **B3:** gap fraction at α = 0.5 ≥ 0.10 with Wilcoxon p < 0.001 — coupling
  well short of rank one already makes the conversation matter.

If B2 or B3 fails, story (b) is wrong or incomplete.

## Also reported (not gated)

- **Reads saved** (post hoc on round-1 curves, and on every configuration here):
  the number of reads the adaptive policy needs to match the open-loop error at
  K = 8, by linear interpolation of log R between integer K. This is the
  communication-budget version of the gap.
- Gap vs participation ratio for experiment B.

## What this does not test

Same limits as round 1: programmed greedy Bayesian policies, known prior,
probit threshold soma, per-read bias.
