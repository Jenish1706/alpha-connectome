# Alpha Connectome

Research and submission tooling for the Alpha Connectome ML competition.

## Local starter pack

The datasets and model artifacts stay out of Git. Stream the starter pack into the repository layout:

```bash
curl -fL https://files.wundernn.io/wnn_connectome_starterpack.tar.gz \
  | python scripts/fetch_starterpack.py
```

This writes the Parquet files to `datasets/` and the rest of the pack (docs, `METRIC.md`, the reference scorer `utils.py`, `baseline/`) to `wnn_connectome_starterpack/`. Both paths are gitignored. A plain `tar -xz -C .` of the same stream puts the files under `wnn_connectome_starterpack/datasets/` instead, where the tests do not look.

The archive is about 34 GB, and `datasets/train.parquet` alone is 29.2 GB. On a smaller disk, such as a Claude Code cloud session, keep a training sample instead:

```bash
curl -fL https://files.wundernn.io/wnn_connectome_starterpack.tar.gz \
  | python scripts/fetch_starterpack.py --train-sample 512
```

This keeps the first 512 training sequences as `datasets/train_head.parquet`, plus the complete training footer as `datasets/train.parquet.footer`. The footer is enough to validate the layout of all 10,607 training sequences.

## Code map

- `src/data/schema.py`: column order and types, footer-level and row-level contract checks, sequence readers, and the `DataPoint` callback type.
- `src/utils/metric.py`: Global Weighted Pearson, matching the starter pack scorer, plus a mergeable streaming accumulator.
- `src/data/streamer.py`: PyTorch TBPTT chunk streamer over row groups, sharded across `DataLoader` workers.
- `scripts/benchmark_latency.py`: replays rows through a `solution.py`, enforces the callback contract, times it against the 60-minute budget, and optionally scores it.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Tests that need the starter pack skip when it is missing. Row-level checks cover the first 4 sequences of each file; `FULL_DATA_CHECKS=1 pytest -q` checks every sequence. `ALPHA_CONNECTOME_DATA` points the tests at another datasets directory.

```bash
python scripts/benchmark_latency.py --data datasets/valid.parquet --sequences 2 --score
python scripts/benchmark_latency.py --solution wnn_connectome_starterpack/baseline/solution.py \
  --data datasets/valid.parquet --sequences 0 --score
```

The scorer requires a root-level `solution.py` in the final submission archive. The working submission scaffold is under `submission/`.
