# Experiments

Research loop toward Global WP 0.665+ on the validation set, starting from the
starter pack GRU baseline (0.617052). Each iteration changes one thing relative
to the current champion, then runs `python scripts/run_experiment.py`.

## Protocol

- **Data.** `datasets/train_head.parquet` holds the first 4,000 of the 10,607
  training sequences, byte-identical to the original file. The first 3,872 are
  used for fitting. The last 128 are a holdout, scored during training as a
  diagnostic only: the baseline was most likely fitted on those sequences, so
  its holdout score is in-sample. The 1,873 validation sequences are used only
  for scoring.
- **Recipe.** Models warm-start from the starter pack baseline weights (two
  GRU blocks of width 128 and a linear head). New input features get zero input
  weights, so a warm-started model begins exactly at the baseline. Training
  uses truncated BPTT with 256 sequences side by side and 250-step windows,
  on a fixed schedule, and keeps the final weights.
- **Scoring.** Global WP over the full validation set, computed like the
  official scorer (`need_prediction AND is_scored`). WP over every required
  row is recorded as well, since the FAQ warns against overfitting the public
  mask.
- **Serving parity.** Engineered features are computed inside the model and
  therefore inside the exported ONNX graph; the solution only feeds raw rows.
  Each run replays the first four validation sequences row by row through the
  exported package, in an isolated interpreter started in the package
  directory, and its predictions must match the batched ones within 1e-4.
- **Latency.** The mean callback time from `scripts/benchmark_latency.py` over
  two validation sequences. The 60 µs ceiling was set when the starter pack
  solution measured 54 µs on the host of the time. This session runs on a
  faster host, where that solution measures about 35 µs, so two checks apply:
  - as measured, the median of three unpinned runs must be at most 60 µs;
  - rescaled to the 54 µs host, the candidate must be at most 60 µs. The
    rescaled figure is 54 µs times the ratio of the fastest of nine pinned
    candidate runs to the fastest of nine interleaved starter pack runs. That
    is the same as being at most 11% slower than the starter pack solution.
    Interference only ever adds time, and the fastest candidate run is stable
    to about ±0.5 µs, while medians drifted by ±2.4 µs between runs.

  The rescaled check is the binding one here. It keeps the ceiling's original
  margin under the scorer's budget: 60 minutes for 39.4M test rows, about
  91 µs a row on the scorer's own hardware.
- **Acceptance.** A candidate is accepted when all of these hold:
  - both latency checks pass,
  - WP beats the champion's WP by at least 0.0016,
  - a paired bootstrap over validation sequences, with 2,000 resamples, gives
    P(candidate > champion) of at least 0.95.

  The margin covers training noise, which the bootstrap cannot see. Three seeds
  of the E01 recipe scored 0.653042, 0.653276 and 0.651986: a WP sd of 0.00069.
  Two single runs therefore differ by noise alone with an sd of about 0.00097,
  and 1.645 of those, the one-sided 95% bound, is 0.0016. The bootstrap alone
  called seed 1 better than seed 2 with P = 1.000.
- **First champion.** Only the untrained starter pack baseline can become the
  first champion. Every later candidate is judged against the champion.
- **Git.** On acceptance: this log entry is written first, then
  `git commit -am "feat: <change> (WP: <score>, <us> us)"`. On rejection or
  failure: `git checkout -- . && git clean -fd`, then this log entry is written
  and committed on its own, so rejected attempts are never lost.
  `configs/champion.json` is tracked, so a revert restores it to the last
  accepted champion. Run artifacts live in the ignored `runs/` directory, and
  `runs/champion/` holds the champion's package and validation statistics.
  The runner refuses to start if the two disagree.

Each candidate is trained once, with seed 0 like the champion, so the margin
above stands in for repeated seeds.

## Log

Latency is shown as raw (unpinned median on this host) / rescaled (to the 54 µs
host); both must be at most 60 µs. "All rows" is WP over every required
validation row, ignoring the public mask.

| ID | Test | Change | WP | All rows | Δ vs champion | P(better) | Latency µs raw / rescaled | Outcome |
|---|---|---|---|---|---|---|---|---|
| E00 | Baseline | Starter pack GRU weights, untrained | 0.617052 | 0.440156 | — | — | 30.0 / 51.9 | Accepted: first champion |
| E01 | Control | Fine-tune the baseline 1 epoch: MSE, lr 2e-4 cosine, 3,872 sequences | 0.653042 | 0.473540 | +0.035990 | 1.000 | 31.6 / 48.2 | Accepted |
| T1.1a | 1.1 OFI | Add sum(dp·dv) for i0 and i1 | 0.653104 | 0.473613 | +0.000062 | 1.000 | 37.5 / 57.5 | Voided: see notes; rerun as T1.1 |
| CAL1 | Calibration | E01 recipe, seed 1 | 0.653276 | 0.473229 | +0.000234 | 0.765 | 33.3 / 51.3 | Calibration only |
| CAL2 | Calibration | E01 recipe, seed 2 | 0.651986 | 0.471799 | −0.001056 | 0.000 | 32.7 / 53.0 | Calibration only |
| T1.1 | 1.1 OFI | Add sum(dp·dv) over the 4 trade slots, i0 and i1 (2 inputs) | 0.653113 | 0.473619 | +0.000071 | 1.000 | 33.9 / 55.2 | Rejected: below the 0.0016 margin |
| T1.2 | 1.2 Volume imbalance | Add (Σ bid v − Σ ask v) / (\|Σ bid\| + \|Σ ask\|), i0 and i1 (2 inputs) | 0.653030 | 0.473479 | −0.000012 | 0.089 | 38.2 / 59.9 | Rejected: no gain, at the latency ceiling |
| T1.3 | 1.3 VWAP mid spread | Add mid(i0) − mid(i1), mid = mean of bid and ask VWAPs (1 input) | 0.653062 | 0.473547 | +0.000020 | 1.000 | 43.9 / 68.7 | Rejected: over the latency ceiling, no gain |
| T2.1a | 2.1 Loss | Pure weighted Pearson loss (1 − WP per training window) instead of MSE | 0.626199 | 0.452891 | −0.026844 | 0.000 | 33.6 / 50.2 | Rejected: much worse |
| T2.1b | 2.1 Loss | Hybrid loss: 0.8 × weighted Pearson + 0.2 × MSE | **0.667873** | 0.497689 | +0.014831 | 1.000 | 32.1 / 51.2 | **Accepted** |
| T2.2a | 2.2 Tanh clamp | Output 2·tanh(z/τ) with τ = 2 (unit slope at 0) | **0.669708** | 0.498827 | +0.001835 | 1.000 | 31.5 / 55.7 | **Accepted** (just above the margin) |
| T2.2b | 2.2 Tanh clamp | τ = 1 instead of 2 (slope 2 at 0, earlier saturation) | 0.665566 | 0.492818 | −0.004142 | 0.000 | 37.4 / 51.2 | Rejected: worse |
| T2.2c | 2.2 Tanh clamp | τ = 4 instead of 2 (slope 0.5 at 0, nearly linear in range) | **0.673977** | 0.503062 | +0.004269 | 1.000 | 34.1 / 54.1 | **Accepted** |
| T2.2d | 2.2 Tanh clamp | τ = 8 instead of 4 (extends the sweep past its edge) | **0.677236** | 0.505950 | +0.003259 | 1.000 | 36.2 / 53.4 | **Accepted** |
| T2.2e | 2.2 Tanh clamp | τ = 16 instead of 8 | 0.678451 | 0.506879 | +0.001215 | 1.000 | 35.1 / 53.4 | Rejected: inside the noise margin |

## Notes

**E01 (control).** One epoch of fine-tuning, 1,200 steps, lifted validation WP
by 0.036, and WP over all required rows rose from 0.440 to 0.474. The holdout
WP (all required rows) went 0.392 → 0.397 → 0.408 → 0.410 → 0.409 over the
epoch, so it was flattening at the end. The baseline shows no in-sample
advantage on the training sample, so it was likely undertrained rather than
fitted to these sequences. Every later change is made on top of this recipe.

**T1.1a (voided).** The first version of the rule accepted OFI for a gain of
+0.000062 (bootstrap sd 0.000019, P = 1.000). That gain is a tenth of the
seed-to-seed noise measured afterwards, and it cost about 4 µs of latency
(rescaled 57.5 µs, most of the headroom). The rule had no margin for training
noise, so it would also have accepted a pure reseed. The result was reverted
before it was committed, the judge was corrected, and 1.1 is rerun below.

**Calibration.** CAL1 and CAL2 rerun the E01 recipe with seeds 1 and 2 through
`scripts/run_experiment.py --calibrate`, which never accepts. They set the
0.0016 margin. They also showed that latency medians drift between runs, while
the fastest pinned runs are stable, so the rescaled check now uses fastest
runs over nine pairs. The feature layer lost a redundant Concat and a global
Clip, so each feature costs only its own ops.

**T1.1.** Hypothesis: an explicit dp·dv product gives the GRU an order-flow
signal it cannot form in one linear step. On this data it adds almost nothing.
WP rose by 0.000071, consistent across validation sequences (P = 1.000) but a
tenth of the seed noise, and all-rows WP moved by +0.00008. The holdout curve
tracked the control to the fourth decimal. It cost about 1.5 µs pinned, so it
was rejected and reverted. The trade columns are rank-transformed, so their
product is not the financial dp·dv, which likely explains the null result.

**T1.2.** Hypothesis: book-side volume imbalance is a classic short-horizon
predictor of price moves. Here it is neutral: WP moved by −0.000012 and
all-rows WP by −0.00006. The imbalance is a fixed linear combination of inputs
the first GRU layer already sees, divided by a normalizer, so the model gains
little it could not already form. It cost about 5 µs pinned (a matmul plus
slices, abs, add and divide), which alone brought the rescaled latency to the
ceiling. The denominator is |Σ bid| + |Σ ask| rather than the literal
Σ v_total, because the rank-transformed volumes are signed and their sum
crosses zero (min |Σ v_total| 4.6e-5 over four validation sequences).

**T1.3.** Hypothesis: the cross-instrument mid-price spread carries lead-lag
information between i0 and i1. The gain is +0.000020, a thirtieth of the seed
noise, and it cost about 9.5 µs pinned: 44-column gathers, a sigmoid, reshapes,
reductions and divisions, 68.7 µs rescaled. Each level is weighted by
sigmoid(1.702 v), about the volume's percentile, because the rank-transformed
volumes are signed and cannot weight a VWAP directly. The price-like columns
are also rank-transformed per column, so a difference of "mids" across
instruments is not a price spread, which likely explains the null result.

**Cycle 1 summary.** None of the three engineered features helped: the gains
were +0.00007, −0.00001 and +0.00002, against a 0.0016 noise margin. The
anonymising rank transforms remove the price and volume arithmetic these
features rely on, and the GRU already sees every raw column. Each feature also
costs 1.5-9.5 µs per call, out of about 11 µs of rescaled headroom.

**T2.1a.** Hypothesis: optimizing the metric directly beats MSE. The holdout
WP matched the control (0.4094 vs 0.4092), but validation WP fell by 0.027.
The loss ignores the level and scale of the outputs within each window, so
both drifted. On six validation sequences the predictions had sd 1.74 (MSE
model: 0.33), a mean near +1.3, and 30% of rows beyond ±2, where the metric
clips them. The ranking was fine; the calibration was lost. This is the
failure the hybrid's MSE term and the 2.2 tanh clamp are meant to contain.

**T2.1b.** Hypothesis: the Pearson term aligns training with the metric, and a
small MSE term keeps the outputs calibrated, which pure Pearson lost. WP rose
by 0.0148, nine times the noise margin, to 0.667873, past the 0.665 target.
All-rows WP rose from 0.474 to 0.498, so the gain is not an artefact of the
public mask. The holdout WP climbed to 0.433, against the control's 0.409, from
the first evaluation onward. Latency is unchanged, since only the loss changed.

**T2.2a.** Hypothesis: softly bounding outputs to the metric's [−2, 2] tames the
rare large predictions, which the metric clips anyway, while keeping slope 1 near
zero. WP rose by 0.0018, just above the 0.0016 margin. Three signals agree:
all-rows WP +0.0011, holdout +0.0017, and the clamp alone lifted the untrained
baseline's holdout from 0.392 to 0.402. The hybrid model's outputs have
sd 0.40 and none reach ±2, so the gain comes from compressing the tails, not
from clamping. The clamp adds three small ops; the rescaled latency moved from
51.2 to 55.7 µs, partly from noise in the reference runs.

**T2.2b.** A sharper clamp, 2·tanh(z), saturates for |z| above about 0.5, which
compresses much of the model's range (sd about 0.4). It also distorts the warm
start: the untrained holdout WP fell to 0.369, against 0.402 at τ = 2. Training
recovered only partly (holdout 0.4265 vs 0.4344), and validation WP dropped by
0.0041.

**T2.2c.** At τ = 4 the clamp is nearly linear across the model's range, with
slope 0.5 at zero, so it mostly halves the warm start's output scale. The
untrained holdout WP rose to 0.425, against 0.402 at τ = 2 and 0.392 with no
clamp. So the baseline's raw outputs overshoot on these sequences and the
metric clips them. After training, validation WP rose by 0.0043 over τ = 2 and
all-rows WP by 0.0042. Over the sweep, WP rises monotonically with τ (1: 0.6656,
2: 0.6697, 4: 0.6740), so the best value may lie beyond the grid; τ = 8 is
tested next to find out.

**T2.2d.** WP keeps rising with τ: +0.0033 from τ = 4 to 8, after +0.0043 from
2 to 4. For large τ, 2·tanh(z/τ) is about 2z/τ over the model's range, so this
is mainly a smaller output scale at the warm start rather than clamping. That
starting scale seems to shape how the hybrid loss fine-tunes the network. The
untrained holdout WP at τ = 8 was 0.433, and the trained holdout reached 0.445.
τ = 16 is next, to find where the curve flattens.

**T2.2e.** τ = 16 gained only +0.0012, inside the 0.0016 margin, and the
holdout ended slightly below τ = 8 (0.4442 vs 0.4447). The sweep has flattened,
so τ = 8 stays.

**Cycle 2 summary.** The loss was the big lever. The hybrid loss added +0.0148.
Pure Pearson lost 0.027, because without an MSE anchor the output scale and
level drifted and 30% of predictions ended beyond the metric's clip. The tanh
temperature added another +0.0074 over τ = 2 → 4 → 8, where the curve
flattens:

| τ | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| WP | 0.6656 | 0.6697 | 0.6740 | 0.6772 | 0.6785 |

For large τ the clamp mostly shrinks the warm start's output scale, so the
gain is better read as "start the output small" than as clamping.
