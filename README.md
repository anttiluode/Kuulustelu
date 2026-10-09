# Kuulustelu — the spike is what makes the conversation matter

*Kuulustelu* is Finnish for interrogation, from *kuulla*, to hear.

A neuron's branches hold a trace of its past. A receiver can only hear that past
through the soma, one read at a time, and it gets to choose each read: which
branches to listen to (the gate) and where to set the threshold (the bias). The
receiver remembers what it has already heard. **When does that memory help it
choose the next question?**

In [LensLuotainTarget](https://github.com/anttiluode/LensLuotainTarget) it helped
by 22 trials out of 512. Here is why that was small, and when it becomes large.

![Error vs number of questions, graded vs spiking soma](results/graded_vs_spiking.svg)

**Graded soma** (the answer is a number): remembering answers is worth exactly
nothing. Across 18 configurations, including reads that erase what they read, the
adaptive receiver asked the *identical* questions as a schedule fixed in advance,
in every one of 512 held-out worlds.

**Spiking soma** (the answer is one bit): remembering answers cuts the error on
unseen questions by **68.7%** at 8 reads (0.038 vs 0.122; 95% CI 64–74%;
paired Wilcoxon p = 7e-76). Scored with the exact posterior instead of the
approximate receiver, the gap is **73%**.

All five pre-registered gates passed. One exploratory prediction failed, and the
failure is informative (below). Full ledger: [RESULTS.md](RESULTS.md). Frozen
protocol: [PROTOCOL.md](PROTOCOL.md).

## Why

The memory is linear: m₀ = F h, twelve leaky traces of a 32-step history. Every
read is y = g̃ᵀm₀ + b + σε through an effective gate g̃ = M_tᵀg, where M_t
records how earlier reads wrote the memory: M_{t+1} = (I − η g gᵀ) M_t.

With a graded soma everything stays Gaussian, and the posterior covariance obeys

  Σ' = Σ − Σg̃ g̃ᵀΣ / (g̃ᵀΣg̃ + σ²)

with no y in it. How uncertain you will be after any sequence of questions does
not depend on the answers, so the best sequence can be computed before the first
answer arrives. That holds for any η, including η = 1.

A threshold breaks this. A spike carries only which side of the threshold the
sum fell on, so how much the next bit can tell you depends on where the threshold
sits relative to what you now believe. The expected-variance rule finds that by
itself: it puts each new threshold at the current posterior mean. That is
probabilistic bisection (Horstein 1963), arrived at without being programmed in.

## What the controls say

- **It is the threshold, not the gate.** Let the adaptive receiver choose the gate
  but pin the threshold at the prior mean, and it does *worse* than the fixed
  schedule (0.331 vs 0.122). It stalls, because asking the same question at the
  same threshold returns the same bit.
- **The conversation needs coupled questions.** *Prediction failed.* With an
  isotropic prior (twelve uncorrelated directions) the gap is exactly zero; the
  best next question is always a fresh direction thresholded at zero, whatever
  was heard. Answers only steer future questions when one answer moves the right
  threshold for another, through correlation or by revisiting a direction. Branch
  traces of one shared history are strongly coupled: one direction holds 94% of
  the variance.
- **Noise shrinks the gap** (69% → 52% from σ = 0.03 to 0.3).
- **Destructive reads hurt most when noise forces you to ask again.** At low noise
  erasing reads barely change the gap (69% → 65%, overlapping intervals). At high
  noise they halve it (52% → 25%): re-asking is how noise is averaged, and a read
  that erases what it read removes that option.
- **One bit is expensive.** Graded reaches 0.0016 at 8 reads; spiking reaches 0.038
  with conversation and 0.122 without.

## Where this sits in the line

- **[LensLuotainTarget](https://github.com/anttiluode/LensLuotainTarget)** — gates on
  a branch memory, receiver-guided selection, small gain. Its gain came from a
  discrete eight-history prior, which is not Gaussian. This repo isolates the other
  route to a large gain: a non-linear read.
- **[Varjoluotain](https://github.com/anttiluode/Varjoluotain)** — a known occluder
  changes what a wall can reveal. Here the gate and the threshold are the occluder,
  and the receiver moves it.
- **The original PerceptionLab `ecg.json` accident** — its reads were never linear:
  `int()` quantisation, an aperture that went blind, max-normalisation. Nothing here
  models that loop. It is the reason to expect that a real cell's questions are
  threshold-shaped, not graded.
- **[Sihti](https://github.com/anttiluode/Sihti)** — a read that writes,
  m' = m − η g (gᵀm), removes the residue η g (gᵀm), so the memory telescopes
  into what was asked plus a core that was never asked.

**For the learned-gate experiment in LensLuotainTarget:** with a graded soma and
Gaussian-like histories, a learned adaptive gate should tie a jointly trained fixed
schedule. That would follow from the equation above, not from a failure of
learning. A spiking soma is where a learned gate has room to win.

## Limits

Programmed, greedy, Bayesian policies with a known prior and forward model;
nothing is learned. The spiking soma is a probit threshold on a linear sum, not a
spike waveform. The per-read bias assumes the cell's excitability can be set for
each question. The primary prior is close to one-dimensional, which favours
bisection. Reads are equal in number, not in compute. Nothing here is a new
theorem: non-adaptive optimal design for linear-Gaussian models, probit
assumed-density filtering (Opper 1998; Minka 2001) and probabilistic bisection
(Horstein 1963; Jedynak, Frazier & Sznitman 2012) are established. What is here is
a clean, pre-registered demonstration of which property of a soma makes the
receiver's memory worth having.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
python experiment.py          # ~3 min, all 36 configurations, gates G1–G5
python check_estimator.py     # ~1 min, exact-posterior re-scoring
python make_figure.py
```

NumPy, SciPy, Matplotlib. Results regenerate byte-identically.
