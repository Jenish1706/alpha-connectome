# Research notes: toward Global WP 0.685

Champion: T2.2d, **WP 0.677236** on the full validation set. Latency is 36.2 µs
as measured on this host and 53.4 µs rescaled to the 54 µs reference host,
where the ceiling is 60 µs. The recipe:

- two fused GRU blocks of width 128, no residual, warm-started from the
  starter pack;
- one fine-tuning epoch on 3,872 training sequences;
- hybrid loss, 0.8 × weighted Pearson + 0.2 × MSE;
- output 2·tanh(z/8).

The zero-initialized residual was tested (T3.1a) and rejected: the model left
the skip path unused.

## 1. What 20 logged runs already tell us

These are the priors that rank everything below. Details are in
`experiments.md`.

- **The inputs are anonymised by per-column rank-Gaussian transforms.** Values
  sit near N(0, 1), saturate at ±5.199 (the normal quantile at 1 − 1e-7), and
  volume-like columns are negative about half the time. Price and volume
  arithmetic (VWAP, spreads, depth ratios) is not recoverable, so the
  microstructure formulas do not mean what they mean on raw books. That is
  consistent with Cycle 1: OFI, volume imbalance and VWAP spread each gained
  at most +0.00007.
- **Linear features are redundant by construction.** The first GRU computes
  W_ih·x over all 112 raw columns, so any fixed linear combination (a
  depth-weighted imbalance numerator, a mid-price difference) is already
  learnable. Only nonlinear transforms (products, ratios, saturations) can add
  information.
- **Optimisation and calibration were the large levers.** Fine-tuning added
  +0.036, the hybrid loss +0.015, and the output temperature +0.009. The
  temperature gain behaves like "start the output small": for large τ,
  2·tanh(z/τ) is about 2z/τ.
- **Training noise is large relative to feature-level effects.** Three seeds
  of one recipe gave a WP sd of 0.00069, hence the 0.0016 acceptance margin.
- **The model saw only 37% of the training set.** One epoch over 3,872
  sequences produced all of E01's gain, and the holdout curve was still
  rising at the end. More data is the most direct way to extend that.

## 2. Literature synthesis, mapped to this data

**Order-flow imbalance.** Cont, Kukanov & Stoikov (2014) show that net order
flow at the best quotes explains contemporaneous price changes almost
linearly, with slope inversely proportional to depth. Xu, Gould & Howison
(2019) extend this to multi-level OFI, where deeper levels add explanatory
power with decaying weights. Both are defined on event-level quote changes,
queue sizes and prices. Here, level identity is scrambled ("the index is an
identifier, not a depth level") and every column is rank-transformed. OFI can
only enter as learned functions of the transformed columns, and an explicit
dp·dv product is a bilinear interaction the GRU can partly form across steps
through its multiplicative gates. That is the most plausible reason T1.1 was
null. What survives is the idea of letting the model form input-input
products in one step: a learned low-rank bilinear layer, (x·A) ⊙ (x·B), of
which OFI is one fixed instance.

**Cross-asset lead-lag.** Price discovery between related instruments shows
lead-lag at sub-second horizons. Hayashi & Yoshida (2005) and Huth & Abergel
(2014) document it on asynchronous tick data. The model already receives
both instruments' full rows every step, and a GRU can learn lagged
cross-dependencies. A hand-made i0 − i1 spread is linear in the inputs,
hence redundant (see T1.3).

**Weight averaging.** Polyak–Ruppert averaging, and SWA (Izmailov et al.,
2018), average iterates late in training. That lands in flatter regions and
reliably improves generalisation when gradients are noisy, which is the case
here: financial targets, and a 250-step TBPTT window per update. Expected
gains are small but cheap, typically a few thousandths of a correlation.
They are larger with more steps to average over. A 0.999 EMA has a horizon of
about 1,000 steps, so it needs the longer full-data schedule of about 6,400
steps; on the 1,200-step recipe it would just act like an earlier
checkpoint.

**Loss calibration.** The hybrid loss works because the Pearson term aligns
with the metric while MSE anchors level and scale; pure Pearson let outputs
drift beyond the metric's clip (T2.1a). Raising α trades anchoring for
alignment. With the tanh clamp now bounding outputs, a higher α may be
tolerable, but T2.1a says the direction is risky. Mean-centering the Pearson
term per window, rather than with running global statistics (Welford), also
leaves the between-window level unconstrained. That is the mechanism of the
T2.1a failure, and the MSE term covers it today.

**Ensembles.** Averaging independently trained models is the most reliable
generalisation gain in noisy forecasting competitions. It works through
decorrelated errors, so structural diversity helps more than reseeding. The
constraint here is inference latency, not accuracy.

## 3. Latency model for batch-1 ONNX Runtime

Measured on this host (fastest pinned runs, starter pack ≈ 31-34 µs):

- **Dispatch.** Every node costs roughly 0.3-1 µs of dispatch regardless of
  size. The LayerNorm GRU exported to about 120 nodes per layer, and the LRU
  to 109, and both landed near 60 µs raw. A fused GRU is one node.
- **Weight traffic.** A batch-1 GRU step is a matrix-vector product, bound by
  reading its weights. Width 128 over two layers reads about 760 KB per step;
  width 96 reads about 460 KB and was about 2 µs faster pinned; width 64
  reads about 230 KB and was about 6 µs faster.
- **Budget.** The rescaled ceiling of 60 µs means at most 11% slower than the
  starter pack solution run in the same session, about 36-37 µs pinned here.
  The champion sits at a ratio of 0.99, so the headroom is about 12%, roughly
  4 µs.

Implications: a feature block must be a handful of nodes (MatMul, Mul,
Concat), and an ensemble must keep total GRU weight bytes near one width-128
model. For example, widths 96 + 64 read about 690 KB against 760 KB.

## 4. Constraints discovered

- **`datasets/train.parquet` is not on disk and cannot be.** It is 29.2 GB,
  and free space is 7.7 GB after the 4,000-sequence sample (11.1 GB),
  `valid.parquet` (5.6 GB) and the CUDA-built PyTorch (5.5 GB; the CPU wheel
  index is blocked by the network policy). Full-data training therefore
  streams from the archive URL.
- **The archive server honours byte ranges over HTTP/1.1** (206), not over
  HTTP/2, which is curl's default. A stream idle for 60 s is dropped, while
  30 s pauses and a slow trickle, 1 MB every 0.5 s, both survive. A streaming
  reader must never stop reading while training catches up.
- **This host is faster than the one the ceiling was set on.** The starter
  pack measures about 33 µs here against 54 µs there, so the rescaled check
  binds.
- **The public leaderboard is scored on the hidden test set.** Our 0.677 is
  on the public validation set, which also chose every champion, so it
  carries selection bias. It is not directly comparable with a leaderboard
  0.6646.

## 5. Ranked hypotheses

Priors are rough, from the evidence above. Cost is wall time per run on this
host.

| Rank | Axis | Hypothesis | Prior ΔWP | Latency | Cost per run |
|---|---|---|---|---|---|
| 1 | A1 | Full data (10,479 fit sequences), 2 epochs, cosine 2e-4 → 1e-5 | +0.005 to +0.015 | none | ≈ 110 min |
| 2 | A3a | EMA 0.999 of the weights over the last 30% of steps | +0.001 to +0.004 | none | same as the recipe |
| 3 | A4 | Two-model average (GRU-96 + a second, different member) | +0.002 to +0.008 | +0 to +5 µs | 2 trainings + export |
| 4 | A3b | Hybrid α 0.85 / 0.90 / 0.95 | −0.005 to +0.003 | none | 3 runs |
| 5 | A2 | Fused interaction projection, learned (x·A)⊙(x·B), OFI-initialised | 0 to +0.002 | +1.5 to 2 µs, 4 nodes | 1 run |

Order of execution: A1 first, as requested and highest prior. Then A3a on
the new recipe, then A3b, A2 and A4. Each candidate is judged by
`scripts/run_experiment.py` against the champion: Δ ≥ 0.0016, bootstrap
P ≥ 0.95, and both latency checks.

Once A1 changes the recipe, every later candidate inherits it, so each costs
about 110 minutes. A wall-clock budget, not the hypothesis list, is the
binding resource.

## 6. Design of the full-data stream (A1)

- Sequences 0-3,999 are read from the local byte-exact sample, and 3,872-3,999
  stay a diagnostic holdout. Sequences 4,000-10,606 come from the archive.
- A reader thread runs curl with `--http1.1`, inflates the gzip stream in
  Python, parses the tar headers to find `train.parquet`, and cuts each row
  group's bytes out of the stream using the offsets in
  `datasets/train.parquet.footer`. Each row group is decoded in memory with
  that footer as metadata, and nothing is written to disk.
- The reader keeps a bounded in-RAM buffer of compressed row groups, about
  2.5 GB. When the buffer is full it trickles instead of pausing, so the
  connection never idles long enough to be dropped.
- Every 256 MB it snapshots the decompressor (`zlib.decompressobj.copy()`).
  After a network error it resumes with a range request from the last
  snapshot, instead of restarting the 34 GB download.
- Epoch order: 1,024 local sequences first, which covers the time the reader
  needs to skip the first 11 GB. Then the remote sequences in file order,
  then the remaining local ones; the dropped remainder is random local
  sequences. Row groups are sequences whose ids are shuffled relative to
  time, so file order carries no temporal leak.
