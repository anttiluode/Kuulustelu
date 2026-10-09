# Kuulustelu — protocol 5 outcome ledger

Protocol frozen in [PROTOCOL_5.md](PROTOCOL_5.md) (commit `663b108`) before any
run; code and tests committed before the held-out run. Held-out seeds 2000–2255
(256 worlds). K = 8, σ = 0.03, η = 0.

**1 of 3 gates passed. The kill condition for "the population recovers the
label" (C2) failed.**

## Gates

| Gate | Outcome | Numbers |
|---|---|---|
| C0 correctness | PASS | population tree likelihood = brute force over neighbour sequences (M = 2, K = 3); M = 1 population decoder = round 4's single-sender decoder |
| C1 M = 1 replicates round 4 | PASS | field-oracle and tag(0) per-world R identical to round 4 on worlds 2000–2255 (≤ 1e-12) |
| **C2 the population recovers the label (KILL for exit 3)** | **FAIL** | penalty P(M) = R_unlabeled / R_oracle: 12.7, 12.2, 12.2, 10.3 for M = 1, 2, 4, 8. Not monotone (12.17 → 12.18), and P(8)/P(1) = 0.81 against a required ≤ 0.5 |
| C3 at M = 8 the field-driven population beats the self-driven one for a far reader | FAIL | 0.387 vs 0.057 |

## Results (mixture-proposal estimator, see deviation)

| M senders | field, oracle | field, exact unlabeled | penalty | field, tag(0) | self-driven (any reader) | ESS median / min |
|---|---|---|---|---|---|---|
| 1 | 0.0428 | 0.544 | 12.7 | 0.789 | 0.0494 | 1114 / 11 |
| 2 | 0.0388 | 0.472 | 12.2 | 0.812 | 0.0504 | 1204 / 5 |
| 4 | 0.0371 | 0.452 | 12.2 | 0.843 | 0.0514 | 1200 / 3 |
| 8 | 0.0377 | 0.387 | 10.3 | 0.885 | 0.0571 | 962 / 1 |

## What this says

**A population of identical senders does not reveal the shared shift.** Eight
senders reading the same memory through the same gates leave the best possible
unlabeled reader 10× worse than one that knows the thresholds. A self-driven
population of any size is decoded as well as its oracle.

The reason is structural, and it was visible in the design only in hindsight.
Each field-driven bit says roughly "m₀ is on this side of a threshold that
depends on n". A wrong n paired with a correspondingly shifted m₀ explains the
bits almost as well. Senders that read the same thing through the same gates
all add the same kind of constraint, so they cannot break that trade-off. The
common shift is close to a gauge freedom between the memory and the thresholds.

The protocol flagged that these senders are redundant. It did not foresee that
redundancy would make the shared shift unidentifiable. So the honest scope of
the kill is: **population reading does not recover the label when the senders
are homogeneous.** Whether heterogeneous senders (different gates, different
memories, one shared shift) break the degeneracy is untested. That is the
version Gemini's "population-averaged projection codes" would actually need.

Part D (derivation, not run): a purely rhythmic shift shared with phase-locking
value PLV acts as a tag with R² = PLV². Round 4's curve then says the reader
needs R² above about 0.75, i.e. PLV above about 0.87, before field-driven
thresholds beat a fixed schedule for that reader. The 0.75 interpolates between
measured tags at 0.7 (0.171, worse than fixed) and 1.0 (0.038).

## Deviation — the exact reader's estimator was replaced after the first run

The pre-registered importance sampler (proposal centred on the tag(0) ADF
posterior, 4096 particles) **failed its own diagnostic**: median ESS fell
51 → 24 → 11 → 5 as M grew (minimum ESS 1.0 at M = 8). Its M ≥ 4 numbers were
therefore not measurements. They are kept in
`results/protocol5_v1_tag0_proposal.json`.

The re-run uses a proposal that matches the posterior's structure:
- a mixture over all 256 neighbour sequences;
- each component is the ADF posterior given that sequence, weighted by its ADF
  evidence, with covariance ×2;
- plus 10% prior;
- 8192 particles.

Median ESS is now 960–1200 at every M. A few worlds still have low ESS
(minimum 1.1 at M = 8), so individual-world values there carry Monte Carlo error.

The two estimators agree where the old one was valid (M = 1: 0.544 vs 0.548).
At large M the old one overstated the error (M = 8: 0.451 vs 0.387). Neither
version changes a gate verdict.

## Also noticed, not explained

The self-driven population gets slightly *worse* with more senders
(0.049 → 0.057), and so does tag(0) (0.79 → 0.88). More conditionally
independent bits cannot hurt an exact reader, so this is most likely
accumulated ADF approximation error. The ADF oracle in the penalty's denominator
may carry the same error. An exact-posterior oracle at large M was not run
within budget.
