# Kuulustelu — protocol 4: who knows the question, and does a shared slow field help ask it?

Written and committed before any run of the experiments below. Same rules as
[PROTOCOL.md](PROTOCOL.md): thresholds, seeds and configurations are frozen;
deviations go in `RESULTS_4.md`. Smoke seeds 1000–1031 are used only to check
that code runs.

## Why this exists

Rounds 1–3 say the value of remembering answers is entirely in *moving the
threshold* to where the receiver's belief sits, and that it is zero for a graded
soma. Two claims made in conversation after round 3 (Oct 9 2026) follow from
that and have not been tested:

1. **A one-bit answer is only worth something to a reader who knows the
   question.** If a cell's threshold moves with a slow state the reader does not
   share, the reader sees the spike but not the threshold, and the adaptation
   looks like threshold noise. A waveform that weakly reports that state
   (Martin-Burgos et al., bioRxiv 2026.09.15.751814: in-vivo waveform → LFP
   amplitude, cross-validated R² ≈ 0.03–0.08) would partly label the question.
2. **A shared slow field earns its place by carrying neighbours' answers into
   each cell's threshold** — a shared posterior — and does so only to the extent
   that the world's correlations are metric.

One correction is recorded **before** running, because it changes what part A
can show. If a cell's threshold depends only on its own past spikes, a reader
who sees those spikes and knows the cell's rule can replay the rule and recover
every threshold exactly (this is how sigma-delta decoding works). The label is
then free. The label problem exists only for the part of the threshold driven
by something the reader did not see — here, a neighbour's spikes delivered
through the shared field. Part A is built around that distinction.

## Common setting

Round 1 model: trace prior Σ₀ = F Fᵀ, 12 branches, spiking soma
y = 1[gᵀm₀ + b + σε > 0], σ = 0.03, η = 0, primary budget K = 8. Held-out
seeds 2000–2511 (512 worlds), training seeds 3000–3511. Metric R as in round 1
(32 unseen unit questions, normalised by the prior; 1 = knows nothing).

**Gate schedule.** Every cell in this protocol reads through the eight gates of
round 1's frozen open-loop schedule (actions 325, 469, 328, 472, 198, 467, 477,
29 = gates 15, 22, 15, 22, 9, 22, 22, 1). Gates are anatomy and are known to
every reader. Only the threshold (bias) is adapted. Round 3 found learned
receivers do this on their own.

**Centring rule.** An adapting cell sets b_t = −g_tᵀμ, the threshold at the
mean of its current belief along the gate (probabilistic bisection), belief
updated by probit ADF as in round 1. The **open** cell uses round 1's frozen
biases.

## Part A — the question label

Two cells read the same memory m₀ with the same gates. At each step t the
**neighbour N** reads first, with round 1's open-loop action and its own noise
(stream `[seed, 5]`). Then the **sender S** reads (round 1's noise stream
`[seed, 2]`).

Sender variants:
- **S-open** — round 1's fixed biases.
- **S-self (c = 0)** — centring on a belief built from S's own past bits only.
- **S-field (c = 1)** — centring on a belief built from S's own past bits **and**
  N's bits up to and including step t (the shared field delivers N's answers
  without loss — the best case for the field, so the worst case for the label).

A distant **reader** sees only S's bits, knows the gates, the prior, N's
questions (not N's answers) and S's rule. Readers differ only in what they know
about S's thresholds:

- **oracle** — knows every b_t. ADF decode.
- **naive** — replays S's rule from S's bits alone (as if c = 0) and treats the
  replayed b̂_t as exact.
- **tag(ρ²)** — receives a waveform tag τ_t = δ_t + ν_t, where
  δ_t = b_t − b̂_t is the field-driven part of the threshold and ν_t is Gaussian
  noise sized so that the squared correlation of τ_t with δ_t is ρ²
  (noise stream `[seed, 6]`). With δ's per-step mean m_t and variance V_t
  measured on the 512 training worlds, the reader uses
  b̃_t = b̂_t + m_t + ρ²(τ_t − m_t) and decodes with
  σ_eff² = σ² + V_t(1 − ρ²) (the remaining threshold uncertainty treated as
  noise). ρ² ∈ {0, 0.05, 0.3, 0.7}. ρ² = 0 is the "threshold as noise" reader.
- **exact unlabeled** — the Bayes reader. It marginalises over all 2⁸ possible
  neighbour bit sequences n:
  P(s | m) = Σₙ Πₜ Φ(nₜ(g_tᵀm + b^N_t)/σ) · Φ(sₜ(g_tᵀm + b_t(n_{1:t}, s_{<t}))/σ),
  with b_t(·) from replaying S's rule along every branch. Posterior mean by
  importance sampling: 8192 particles per world (stream `[seed, 8]`), proposal
  = ½ N(μ_tag0, 4Σ_tag0) + ½ prior. Median and minimum effective sample size are
  reported. If the median ESS is below 50 the run is repeated once at 32768
  particles and both are reported.
- **exact labeled** — the same importance sampler with the true b_t (checks
  the ADF oracle; not a gate).

The oracle uses b_t only as a likelihood parameter. A cleverer reader could also
read the *value* of b_t as a message about N's bits; that is not exploited here.

### Gates (K = 8)

- **A0 correctness** — unit tests: probit ADF with σ_eff matches importance
  sampling; the tree likelihood equals a brute-force sum over neighbour
  sequences on a small case; the exact-unlabeled reader equals the exact-labeled
  reader when c = 0.
- **A1 the label is free for self-driven thresholds** (consistency) — c = 0:
  replayed thresholds equal the sender's to 1e-12 in 512/512 worlds, and the
  naive reader's R equals the oracle's exactly.
- **A2 replication** — S-open with the oracle reader reproduces round 1's
  open-loop R (0.1216) to 1e-12.
- **A3 threshold-only adaptation carries the gain** — S-self, oracle:
  R ≤ 0.85 · R(S-open), one-sided paired Wilcoxon p < 0.001.
- **A4 the claim from conversation (KILL for it)** — S-field, exact unlabeled
  reader: mean R ≥ mean R(S-open). That is: to a reader who does not share the
  field, field-driven adaptation is worth no more than the fixed schedule.
  If this fails, the claim "unlabeled adaptation loses to a fixed schedule" is
  wrong as stated; the fraction of the oracle's advantage the exact reader keeps,
  (R_open − R_exact)/(R_open − R_oracle), is reported either way.
- **A5 the noise reader loses** — S-field, tag(0): R > R(S-open), one-sided
  Wilcoxon p < 0.001.
- **A6 a weak tag recovers little** — S-field: R(tag ρ²) decreases through
  ρ² = 0, 0.05, 0.3, 0.7 and the oracle; and the recovery at ρ² = 0.05,
  (R_tag0 − R_tag0.05)/(R_tag0 − R_oracle), is below 0.25.

Reported, not gated: oracle at c = 1 vs c = 0 (does a better-informed asker help
a reader who does not share its information?); the naive reader at c = 1; V_t.

## Part B — a shared slow field among cells

**World.** P = 8 cells on a line. Cell i holds its own memory m₀ⁱ (round 1's
trace prior). Memories are correlated across cells by distance:
Σ = C ⊗ Σ₀, C_ij = exp(−|i − j| / 2) (neighbour correlation 0.61, smallest
eigenvalue 0.25). Worlds: u ~ N(0, I) of shape (8, 32) from stream `[seed, 7]`,
H = L_C u, m₀ⁱ = F Hᵢ. Cell i's read noise: stream `[seed, 10 + i]`.

Every cell reads its own memory with the 8 scheduled gates. At step t all cells
read simultaneously; thresholds use beliefs built from bits up to t − 1.

**Threshold policies.** Each cell keeps a Gaussian belief over the joint 96-dim
memory (prior Σ) and updates it by probit ADF with the bits of the cells in its
*visible set* V_i (its own included):

- **open** — round 1's fixed biases, no belief.
- **private** — V_i = {i}.
- **field r = 1** — V_i = {j : |i − j| ≤ 1}. Metric, no wires. 14 directed
  links.
- **field r = 2** — |i − j| ≤ 2 (reported, not gated).
- **broadcast** — V_i = all cells. 56 directed links.
- **shuffled** — V_i = {i} plus the same number of partners as field r = 1,
  drawn (seed 4040) from cells at distance ≥ 3. Same count, wrong geometry.

Threshold b_{i,t} = −(eᵢ ⊗ g_t)ᵀ μ⁽ⁱ⁾.

"Field" here is Bayesian sharing of neighbours' bits within a radius. A real
ionic or ephaptic field carries a weak, blurred, leaky sum of them, so this is
an **upper bound** on what a metric slow field of that range could carry. The
electric potential itself is quasi-static and holds no history; the physical
candidate is the slow ionic/pump state.

**Decoder (same for every arm).** Joint ADF over all 64 bits with their true
biases, as in round 1 (answers are always used for estimation; arms differ only
in how thresholds were chosen). R pooled over cells:
Σᵢ Σⱼ (q_jᵀ(μᵢ − m₀ⁱ))² / Σᵢ Σⱼ q_jᵀΣ₀q_j per world.

**Permuted world.** Same, but C^π_ij = C_{π(i)π(j)} for a fixed random
permutation π (seed 4041): correlations no longer follow position. Arms: open,
private, field r = 1 (still metric, now pointed at the wrong cells),
**addressed** (each cell's partners = the cells it is most correlated with under
C^π, same count as its field r = 1 set — the wired attacker), broadcast.

In the metric world the addressed sets equal the field r = 1 sets by
construction (correlation falls with distance), so there the field can only win
on wiring. This is stated now, not discovered later.

### Gates (K = 8)

- **B0 consistency** — graded soma: every arm gives identical R (≤ 1e-9),
  since thresholds carry nothing; metric world: addressed sets equal field
  r = 1 sets.
- **B1 private centring beats open** — R ≤ 0.85 · R(open), p < 0.001.
- **B2 the shared field helps (KILL for claim 2)** — field r = 1:
  R ≤ 0.90 · R(private), p < 0.001.
- **B3 geometry matters** — field r = 1 < shuffled, one-sided paired Wilcoxon
  p < 0.001.
- **B4 cheap** — field r = 1 recovers at least half of the private→broadcast
  gain: (R_private − R_field1)/(R_private − R_broadcast) ≥ 0.5.
- **B5 the field's value is conditional on metric correlations** — permuted
  world: addressed ≤ 0.90 · field r = 1, p < 0.001.

## What this does not test

Programmed Bayesian rules with a known prior, as in rounds 1–2. No membrane,
pump or ion dynamics: "slow state" is a belief, "field" is lossless sharing
within a radius. The waveform tag is synthetic Gaussian with a chosen R², not a
model of spike shape. One neighbour in part A; one coupling strength. Readers
are compared at equal reads, not equal compute.
