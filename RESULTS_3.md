# Kuulustelu — protocol 3 outcome ledger: learned receivers

Protocol frozen in [PROTOCOL_3.md](PROTOCOL_3.md) (commit `ac8c3f6`) before any
training run. Trace prior, η = 0, σ = 0.03, K = 8. Held-out seeds 2000–2511 with
round 1's exact worlds and noise. Three initialisations per arm, 4000 steps each,
checkpoint chosen on validation seeds 4000–4511. The receivers are recurrent
networks trained only to minimise prediction error: no prior, no posterior, no
threshold rule. They can ask any unit-vector gate and any bias.

![Learned receivers](results/learned_receivers.svg)

## Gates

| Gate | Outcome | Per init (0, 1, 2) |
|---|---|---|
| L0 correctness | PASS | 6 tests: spike exactly ±1 forward; surrogate backward; open-loop actions unchanged when answers are permuted; adaptive actions depend on answers; held-out worlds bit-identical to round 1; z = 0 for a centred threshold |
| L1 learned receiver converses (kill) | **PASS, 3/3** | gap vs learned open-loop 52.7%, 78.4%, 76.6%; Wilcoxon p 7e-34, 1e-71, 3e-60 |
| L2 threshold-centring is learned | **PASS, 2/3** | median \|z\| ratio adaptive/open 0.570 (fails), 0.141, 0.282 |
| L3 beats programmed fixed schedule | PASS, 3/3 | 0.0456, 0.0196, 0.0256 vs 0.1216 — *weaker than it looks, see below* |
| L4 graded control: no gain | **FAIL, 1/3** | gap −11.1%, +16.6%, +16.1% against a ±15% band |

## Numbers, K = 8

| arm | init 0 | init 1 | init 2 |
|---|---|---|---|
| spiking adaptive, R | 0.0456 | 0.0196 | 0.0256 |
| spiking open-loop, R | 0.0965 | 0.0907 | 0.1091 |
| spiking adaptive, reads re-scored by round 1's Bayesian receiver | 0.0500 | 0.0249 | 0.0324 |
| spiking open-loop, re-scored | 0.1117 | 0.0978 | 0.1220 |
| spiking adaptive, median \|z\| (reads 2–8) | 0.565 | 0.176 | 0.350 |
| spiking open-loop, median \|z\| | 0.991 | 1.253 | 1.242 |
| graded adaptive, R | 0.0017 | 0.0014 | 0.0012 |
| graded open-loop, R | 0.0015 | 0.0016 | 0.0014 |

Reference, round 1's programmed receivers: adaptive R 0.0381, median |z| 0.101;
fixed schedule R 0.1216, median |z| 2.555.

## What survived

1. **A receiver trained only to predict learns to converse.** With a spiking
   soma, letting its questions depend on what it heard cuts error by 53–78%
   against the same network whose questions are fixed. All three inits pass.

2. **It learns the same mechanism, unprompted.** Post-hoc diagnostics below
   show that the learned adaptive receivers barely change *which* branches they
   ask: gate directions stay within |cos| 0.94–0.998 of their average across
   worlds. What moves with the answers is the *threshold*, which sits much
   closer to the current best guess (median |z| 0.18–0.57) than the open-loop
   learner's (0.99–1.25). Round 1's control found this by construction: "it is
   the threshold, not the gate". Here a learner with no threshold rule arrived
   at it from the loss alone. Two of three inits clear the frozen centring gate;
   init 0 centres less (ratio 0.57 against the 0.5 bar) and also performs worst.

3. **Its questions are better than the programmed greedy ones.** Two of three
   learned adaptive receivers beat round 1's programmed adaptive receiver
   (0.0196 and 0.0256 vs 0.0381). That is not just a better decoder. Re-scored
   by round 1's own Bayesian receiver, their reads still give 0.0249 and 0.0324.
   This comparison is **not controlled**: the learned receivers have continuous
   gates and are trained over the whole 8-read horizon, while the programmed one
   is greedy and limited to the 24-gate catalogue. Which of those accounts for
   the difference is not separated here.

## L3 is weaker than it looks

The learned **open-loop** arm also beats the programmed fixed schedule
(0.091–0.109 vs 0.1216), again through continuous gates and horizon training.
So passing L3 says as much about the action space as about conversation. The
controlled comparison is L1, learned against learned.

## L4 failed — and the post-hoc diagnosis

The frozen band was ±15% and two of three inits landed at +16%. The question is
whether the graded adaptive receivers were steering questions by answers, which
would contradict round 1's equation, or something else.
`analyse_protocol3.py` (added after the held-out run) checks this:

- **They learned a fixed schedule.** Graded adaptive gate directions stay within
  |cos| ≥ 0.985 of their average in every world. Their bias varies, but with a
  graded soma the bias is a known offset and carries nothing.
- **Every learned graded receiver is beaten by the best fixed schedule.** The
  best fixed continuous-gate schedule (greedy on the covariance recursion, no
  answers involved) has expected R = 0.00075. The six learned graded arms,
  re-scored by the Bayesian receiver, land at 0.00091–0.00134.

So the learned graded receivers (adaptive or not) are all suboptimal fixed
schedules that optimised to different points. The ±15% band was too tight for
learning variance at an error of ~0.001. Nothing here contradicts round 1's
equation, but **the gate failed as frozen and stays failed.**

## Limits

- Small recurrent networks (64 hidden units), one surrogate width (τ = 0.1), one
  training budget, three inits per arm.
- The learned receivers are trained on the true prior, so "no Bayesian
  machinery" means no explicit posterior, not no knowledge of the world they
  live in.
- The comparison with round 1's programmed receivers mixes three differences:
  continuous gates, horizon training, and a learned decoder.
- The spiking soma is still a probit threshold on a linear sum.

## Reproduce

```bash
python -m pip install -r requirements.txt     # numpy, scipy, matplotlib, autograd
python -m unittest discover -s tests -t . -v
python learned_experiment.py                  # ~12 min on 2 CPUs; writes results/protocol3.json + weights
python analyse_protocol3.py                   # post-hoc diagnostics
python make_figure3.py
```
