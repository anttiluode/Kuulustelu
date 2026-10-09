# Kuulustelu — outcome ledger

Protocol frozen in [PROTOCOL.md](PROTOCOL.md) (commit `e8c9f8c`) before any run.
Held-out worlds: seeds 2000–2511 (512). Open-loop schedules built on training
worlds 3000–3511. Primary budget K = 8 reads. R = error on 32 unseen questions,
normalised so 1 = knows nothing.

## Gates

| Gate | Outcome | Numbers |
|---|---|---|
| G0 correctness | PASS | 7 tests: Kalman = batch posterior to 1e-10; probit ADF one-step moments vs 2M-sample importance sampling within 2% (1-D, 3-D); effective-gate bookkeeping = simulated written memory; expected-reduction formula = explicit outcome average; tie-break |
| G1 graded: open-loop is optimal | PASS | identical action sequence in 512/512 worlds at η = 0 and η = 1; max \|R_adaptive − R_open\| over all worlds and all K = 0 |
| G2 the spike makes the conversation matter (kill condition) | PASS | R 0.0381 adaptive vs 0.1216 open-loop: 68.7% lower, 95% CI [64.1, 73.5]%, one-sided paired Wilcoxon p = 6.7e-76 |
| G3 beats random | PASS | 0.0381 vs 0.1998 (80.9% lower) |
| G4 noise shrinks the gap | PASS | gap 68.7% at σ = 0.03 → 52.2% [47.8, 56.2] at σ = 0.3 |
| G5 destructive reads shrink the gap | PASS, **weakly** | 68.7% at η = 0 → 65.2% [60.5, 69.6] at η = 1. The confidence intervals overlap; this pass is not decisive on its own |

## Full held-out table, K = 8

| prior | η | σ | graded adaptive = open-loop | spiking adaptive | spiking open-loop | spiking random | spiking adaptive, threshold fixed at 0 | gap vs open-loop [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| traces | 0 | 0.03 | 0.0016 | 0.0381 | 0.1216 | 0.1998 | 0.3308 | 0.687 [0.641, 0.735] | 6.7e-76 |
| traces | 0 | 0.1 | 0.0092 | 0.0475 | 0.1323 | 0.2128 | 0.2911 | 0.641 [0.601, 0.679] | 1.2e-54 |
| traces | 0 | 0.3 | 0.0417 | 0.0849 | 0.1777 | 0.2647 | 0.2609 | 0.522 [0.478, 0.562] | 2.5e-30 |
| traces | 1 | 0.03 | 0.0018 | 0.0443 | 0.1273 | 0.3132 | 0.2328 | 0.652 [0.605, 0.696] | 1.9e-47 |
| traces | 1 | 0.1 | 0.0119 | 0.0764 | 0.1762 | 0.3407 | 0.2429 | 0.566 [0.524, 0.606] | 3.0e-23 |
| traces | 1 | 0.3 | 0.0639 | 0.1975 | 0.2614 | 0.3900 | 0.2553 | 0.245 [0.179, 0.306] | 7.8e-07 |
| traces | 0.25 | 0.03 | 0.0017 | 0.0396 | 0.1045 | 0.2448 | 0.2915 | 0.621 [0.575, 0.671] | 1.7e-66 |
| traces | 0.25 | 0.1 | 0.0114 | 0.0561 | 0.1432 | 0.2654 | 0.2604 | 0.608 [0.572, 0.643] | 1.8e-39 |
| traces | 0.25 | 0.3 | 0.0526 | 0.1159 | 0.2158 | 0.3167 | 0.2555 | 0.463 [0.415, 0.506] | 1.5e-16 |
| iso | any | 0.03 | 0.3113 | 0.5645 | 0.5645 | 0.80–0.82 | 0.5645 | 0.000 | 1.0 |
| iso | any | 0.1 | 0.3389 | 0.5845 | 0.5845 | 0.81–0.82 | 0.5845 | 0.000 | 1.0 |
| iso | any | 0.3 | 0.5103 | 0.6938 | 0.6938 | 0.85–0.86 | 0.6938 | 0.000 | 1.0 |

Graded adaptive and graded open-loop chose the identical sequence in **every one
of the 18 graded configurations** (both priors, all η, all σ), not only the two G1
requires.

## What survived

1. **With a graded soma, remembering answers is worthless.** This is the Kalman
   covariance recursion, now observed: 18/18 configurations, 512/512 worlds each,
   identical questions. It holds with destructive reads too. Order matters there,
   but the best order is still computable in advance.

2. **With a spiking soma, remembering answers cuts error by about two-thirds.**
   68.7% at the primary setting. The receiver is the same, the reads are the
   same, the catalogue is the same. The only difference is whether the choice of
   the next question looks at what was already heard.

3. **The gain is entirely in the threshold.** Let the adaptive policy choose the
   gate but pin the threshold at the prior mean, and it becomes *worse* than the
   fixed schedule (0.331 vs 0.122) and stalls near 0.33. The open-loop schedule
   survives because it spreads a fixed ladder of thresholds (bias indices
   10, 7, 13, 10, 9, 5, 15, 8 of 21). The adaptive policy's thresholds cluster at
   the current posterior mean. That is probabilistic bisection (Horstein 1963),
   arriving on its own from the expected-variance rule.

4. **The conversation needs coupling between questions.** Exploratory prediction
   **failed**: with an isotropic prior the spiking gap is exactly zero, identical
   sequences in every world. With twelve uncorrelated directions and at most
   twelve reads, the best next question is always a fresh direction thresholded
   at its mean, whatever came before. Adaptivity pays only when an answer about
   one direction moves the right threshold for the next, through correlation or
   by asking the same direction again. Real branch traces are strongly coupled
   because they share one history (one direction carries 94.2% of the variance),
   and that is where the gap lives.

5. **Noise and destruction interact.** At low noise, destructive reads barely
   matter (gap 68.7% → 65.2%); the memory is redundant, so the same direction can
   be reached through other gates. At high noise, re-asking is how noise gets
   averaged, and a read that erases what it read removes that option: the gap
   falls from 52.2% (η = 0) to 24.5% (η = 1).

6. **Cost of one bit.** At K = 8 the graded soma reaches R = 0.0016. The spiking
   soma reaches 0.0381 with conversation and 0.1216 without — 24× and 76× worse.

## Estimator check (post hoc, added after the held-out run — not a gate)

The spiking receiver uses an approximate (ADF) posterior. On held-out worlds it
is overconfident: it predicts R = 0.0191 for the adaptive policy (actual 0.0381)
and 0.0916 for open-loop (actual 0.1216). To check that the gap is about the
information collected, not the estimator, `check_estimator.py` re-scores the same
reads on 128 held-out worlds with the exact posterior mean by importance sampling
(400,000 prior samples; minimum effective sample size 1,075):

| | ADF receiver | exact posterior |
|---|---|---|
| adaptive | 0.0546 | 0.0346 |
| open-loop | 0.1522 | 0.1303 |
| random | 0.1923 | 0.1821 |
| gap vs open-loop | 64.1% | **73.4%** |

The approximation was costing the adaptive policy more than the fixed schedule.
With the exact posterior the gap is larger, not smaller.

## Deviations from the protocol, all recorded

- **Pilot fix (before held-out):** exact ties between actions (common with the
  isotropic prior and with η = 1) were broken by last-bit rounding, differently
  for per-world adaptive scores and averaged open-loop scores. This produced
  spurious sequence differences in the graded soma. Fixed with a deterministic
  tie-break (relative tolerance 1e-10, lowest index), applied to every policy.
  Commit `567a9a3`. No gate, threshold, seed or configuration changed.
- **Added before held-out:** the receiver's own predicted R, for the calibration
  check above.
- **Added after held-out:** recording of the issued effective gates and answers,
  needed by `check_estimator.py`. It does not change any computation; the
  held-out receipt regenerates identically.

## Limits

- Policies are programmed and Bayesian, not learned. The receiver knows the prior
  and the forward model exactly. Both are greedy (one step ahead), so neither is
  the true optimum of its class.
- The spiking soma is a probit threshold on a linear sum, not a spike waveform
  or membrane dynamics. The per-question bias is an assumption: that the
  receiver (or the circuit around the cell) can set excitability for each read.
- The primary prior is close to one-dimensional, which favours bisection. The
  isotropic arm shows what happens without that structure: nothing.
- Equal number of reads, not equal compute: the adaptive policy scores 504
  actions per step per world; the open-loop policy does that once, offline.
- Nothing here is a new theorem. Linear-Gaussian design being non-adaptive,
  probit ADF, and probabilistic bisection are all established.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
python experiment.py                 # ~3 min, writes results/summary.json and per_world_K8.json
python check_estimator.py            # ~1 min, writes results/estimator_check.json
python make_figure.py
```
