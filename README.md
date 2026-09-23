# Alpha Connectome

Research and submission tooling for the Alpha Connectome ML competition.

## Local starter pack

The competition datasets and model artifacts are intentionally excluded from Git. Download them locally when needed:

```bash
mkdir -p data
curl -L https://files.wundernn.io/wnn_connectome_starterpack.tar.gz \
  | tar -xz -C data
```

This creates `data/wnn_connectome_starterpack/`; use its `datasets/` and `baseline/` directories with the scripts in this repository.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/benchmark_latency.py --help
```

The scorer requires a root-level `solution.py` in the final submission archive. The working submission scaffold is under `submission/`.
