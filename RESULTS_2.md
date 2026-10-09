# Kuulustelu — protocol 2 outcome ledger

Protocol frozen in [PROTOCOL_2.md](PROTOCOL_2.md) (commit `cac4bfe`) before any
run. Held-out seeds 2000–2511 (512 worlds), open-loop schedules built on training
seeds 3000–3511. Spiking soma, η = 0, σ = 0.03 throughout.

Round 1 found that with an isotropic prior the spiking gap was exactly zero, and
explained it after the fact: answers only steer questions by **(a) revisiting** a
direction already asked or **(b) coupling** to a correlated one. This protocol
tested both halves of that explanation.

![Revisiting and coupling](results/revisit_and_coupling.svg)

## Gates

| Gate | Outcome | Numbers |
|---|---|---|
| A1 identical sequences while fresh axes remain | PASS | 512/512 worlds identical for every K ≤ 12 (consistency, not new evidence) |
| A2 revisiting opens a gap (K = 24) | PASS | R 0.150 adaptive vs 0.260 open-loop: 42.3% lower, 95% CI [39.9, 44.5]%, Wilcoxon p = 5e-77 |
| A3 gap opens at K = 13–16 | PASS | opens at **K = 13**, the first read after all twelve axes have been asked once. At K = 13 sequences differ in 512/512 worlds |
| B1 α = 1 replicates round 1 | PASS | 0.0381 / 0.1216 to 1e-12 |
| B2 gap non-decreasing in coupling | PASS | strictly increasing, no inversion (table below) |
| B3 gap at α = 0.5 ≥ 10% | PASS | 15.2% [11.8, 18.6], p = 1.5e-11 |

## Coupling sweep, K = 8

| α | effective dimension | adaptive R | open-loop R | gap [95% CI] | Wilcoxon p | reads adaptive needs to match open-loop's 8 |
|---|---|---|---|---|---|---|
| 0 | 12.00 | 0.5618 | 0.5618 | 0.0% | 1.0 | 8.0 |
| 0.25 | 7.47 | 0.4941 | 0.5505 | 10.2% [7.3, 13.2] | 1.2e-08 | 6.5 |
| 0.5 | 3.51 | 0.3942 | 0.4651 | 15.2% [11.8, 18.6] | 1.5e-11 | 5.6 |
| 0.75 | 1.86 | 0.2225 | 0.3223 | 30.9% [27.7, 34.0] | 5.8e-39 | 4.0 |
| 0.9 | 1.36 | 0.1276 | 0.2063 | 38.1% [35.3, 40.9] | 2.9e-49 | 3.6 |
| 1.0 | 1.12 | 0.0381 | 0.1216 | 68.7% [64.1, 73.5] | 6.7e-76 | 3.4 |

Effective dimension = participation ratio (tr Σ)²/tr Σ² of the prior.

## What survived

1. **The post-hoc explanation now has predictive support.** Both of its
   routes were tested on predictions it had not been fitted to, and both held:
   - the isotropic gap appears exactly when re-asking becomes necessary;
   - any coupling at all opens a gap, which grows steadily with coupling.
2. **The isotropic zero was a budget effect, not a property of isotropy.** With
   enough reads to revisit, an isotropic memory gives a 42% gap at K = 24.
3. **The steepest rise is between α = 0.9 and 1.0** (38% → 69%), where the prior
   collapses towards one direction. Part of round 1's headline number is
   therefore the near-scalar structure of the trace prior, as Sol pointed out.
   The conversation still matters well away from that regime (15% at effective
   dimension 3.5).

## Reads saved — and a correction to round 1's G5 reading

Measured as reads, not as error at a fixed budget: how many reads does the
adaptive policy need to match the open-loop error at K = 8? This is computed by
interpolating log R between integer K. It is post hoc on round 1 and was
announced as "reported, not gated" in protocol 2.

| configuration (trace prior, spiking) | gap fraction at K = 8 | reads to match open-loop's 8 |
|---|---|---|
| η = 0, σ = 0.03 | 68.7% | 3.4 |
| η = 0, σ = 0.1 | 64.1% | 3.3 |
| η = 0, σ = 0.3 | 52.2% | 3.0 |
| η = 1, σ = 0.03 | 65.2% | 3.2 |
| η = 1, σ = 0.1 | 56.6% | 2.7 |
| η = 1, σ = 0.3 | 24.5% | 3.8 |
| η = 0.25, σ = 0.03 | 62.1% | 4.0 |
| η = 0.25, σ = 0.1 | 60.8% | 3.1 |
| η = 0.25, σ = 0.3 | 46.3% | 2.6 |

On this measure the conversation saves a steady **2–3× in reads (spikes)**
across every noise level and every write strength. Round 1's reading of G5,
"destructive reads halve the gap at high noise", depends on the metric. In error
at a fixed budget the gap shrinks (52% → 25%); in reads saved it does not
(3.0 → 3.8 of 8). The fixed schedule's curve flattens at high noise, so a
smaller error ratio at K = 8 still corresponds to a large saving in reads. G5's
pass stands as frozen. The stronger interpretation is withdrawn.

## Limits

As in round 1: programmed greedy Bayesian policies, a known prior, a probit
threshold soma, a per-read bias. The α interpolation mixes the trace prior with
isotropic noise. It is one path between the two endpoints, not the only one.
The α = 0 mixed worlds use a different random stream from round 1's iso arm
(same distribution); their R (0.5618) differs from round 1's iso number (0.5645)
for that reason.

## Reproduce

```bash
python -m unittest discover -s tests -t . -v
python experiment2.py      # ~45 s, writes results/protocol2.json
python make_figure2.py
```
