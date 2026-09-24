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
    rescaled figure is 54 µs times the median ratio over five alternating
    candidate and starter pack pairs pinned to one core, which is the same as
    at most 11% slower than the starter pack solution.

  The rescaled check is the binding one here. It keeps the ceiling's original
  margin under the scorer's budget: 60 minutes for 39.4M test rows, about
  91 µs a row on the scorer's own hardware.
- **Acceptance.** A candidate is accepted when all of these hold:
  - both latency checks pass,
  - WP beats the champion's WP,
  - a paired bootstrap over validation sequences, with 2,000 resamples, gives
    P(candidate > champion) of at least 0.95.
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

The bootstrap measures evaluation noise over validation sequences only. It
does not measure seed-to-seed training variance, since each candidate is
trained once.

## Log

| ID | Test | Change | WP | Δ vs champion | P(better) | Latency µs (pinned) | Outcome |
|---|---|---|---|---|---|---|---|
