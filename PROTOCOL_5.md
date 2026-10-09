# Kuulustelu — protocol 5: can a far reader recover the label from the population?

Written and committed before any run. Same rules as earlier protocols.

## Why this exists

Round 4: when a cell's threshold is moved by a shared field the reader does not
see, the best possible reader of that one cell does worse than a fixed schedule
(0.541 vs 0.122). After round 4 (Oct 9 2026) I named three exits:

1. the threshold depends only on the cell's own spikes (replayable — round 4 A1);
2. the reader shares the context (e.g. a rhythm both sides lock to);
3. **population reading**: projections leave a patch as a bundle; the field
   shifts every cell in it the same way, so a reader of many cells can infer it.

This protocol tests exit 3 by experiment and exit 2 by derivation.

## Part C — population reader (experiment)

Round 4 part A, with M senders instead of one. All senders read the same memory
m₀ with the same gates. All share the same field, which delivers neighbour N's
answers. Each sender centres its threshold on a belief built from its own past
bits and N's bits up to t. Senders differ only in their read noise: sender 0
uses round 1's stream `[seed, 2]`, so M = 1 is round 4 exactly; sender i ≥ 1
uses `[seed, 20 + i]`.

M ∈ {1, 2, 4, 8}. K = 8, σ = 0.03, η = 0. **Held-out seeds 2000–2255 (256
worlds, to fit the compute budget); training seeds 3000–3511** for the per-step
threshold-shift statistics, as in round 4.

Arms, each at every M:
- **field, oracle** — reader knows every sender's thresholds (ADF).
- **field, tag(0)** — reader replays each sender as if self-driven and treats the
  remaining shift as noise (round 4's tag(0), per sender).
- **field, exact unlabeled** — Bayes reader marginalising all 2⁸ neighbour
  sequences jointly over the M senders (the shift is common to them, so the
  sum is over one shared n). Importance sampling, 4096 particles per world
  (stream `[seed, 8]`), proposal ½ N(μ_tag0, 4Σ_tag0) + ½ prior. ESS reported.
- **self-driven** — senders centre on their own bits only; any reader replays
  exactly (ADF with true thresholds).

### Gates (K = 8)

- **C0 correctness** — unit test: the population tree likelihood equals a
  brute-force sum over neighbour sequences for M = 2, K = 3.
- **C1 replication** — M = 1 field-oracle and field-tag(0) per-world R equal
  round 4's on worlds 2000–2255 to 1e-12.
- **C2 the population recovers the label (KILL for exit 3)** — the penalty
  P(M) = R_exact_unlabeled(M) / R_oracle(M) decreases at every step of M, and
  P(8) ≤ 0.5 · P(1).
- **C3 the field becomes worth having for a far reader** — at M = 8 the exact
  unlabeled reader of the field-driven population beats the self-driven
  population of the same size: R lower, one-sided paired Wilcoxon p < 0.001.

Reported, not gated: field-oracle vs self-driven at each M (the local value of
the field); tag(0) at each M; ESS.

## Part D — shared rhythm as the label (derivation, not run)

Let the field-driven shift be purely rhythmic, δ_t = A cos φ_t, φ uniform. The
reader observes φ̂ = φ + Δ, with Δ independent and symmetric, and
phase-locking value PLV = E[cos Δ]. Then E[δ | φ̂] = A · PLV · cos φ̂, which
explains a fraction PLV² of Var δ = A²/2.

So **a rhythm shared with phase-locking value PLV is a tag with R² = PLV²**.
Round 4's measured tag curve then gives the answer for a shift of the same size,
without a new run. This is recorded as a derivation. It covers only the purely
rhythmic part of a shift; informative shifts, as in part A, are not rhythmic.

## What this does not test

Same as round 4. Senders share gates and memory, so a population of fixed or
self-driven senders is partly redundant; that is why the comparison in C3 is
made at equal M. 256 worlds instead of 512.
