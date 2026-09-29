#!/usr/bin/env python3
"""Compare two runs' validation statistics: WP of each, the delta, a paired bootstrap.

    python scripts/compare_runs.py runs/<candidate> runs/<reference>

For screening runs judged against a reference other than the champion, such as
a change to the short recipe against the short-recipe champion T2.2d. It
decides nothing: scripts/run_experiment.py stays the only way to accept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import MIN_DELTA, paired_bootstrap, wp_from_stats  # noqa: E402


def compare(candidate: Path, reference: Path) -> dict:
    ours, theirs = (np.load(run / "val_stats.npy") for run in (candidate, reference))
    if ours.shape != theirs.shape:
        raise ValueError(f"runs scored different sequences: {ours.shape} vs {theirs.shape}")
    wp, ref = (float(wp_from_stats(stats.sum(0))) for stats in (ours, theirs))
    p_better, sd = paired_bootstrap(ours, theirs)
    return {"wp": wp, "reference_wp": ref, "delta": wp - ref, "delta_sd": sd, "p_better": p_better}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidate", type=Path)
    ap.add_argument("reference", type=Path)
    args = ap.parse_args(argv)
    result = compare(args.candidate, args.reference)
    clears = result["delta"] >= MIN_DELTA and result["p_better"] >= 0.95
    print(f"{args.candidate.name}: WP {result['wp']:.6f}\n{args.reference.name}: WP {result['reference_wp']:.6f}\n"
          f"delta {result['delta']:+.6f} (sd {result['delta_sd']:.6f}), P(better) {result['p_better']:.3f}: "
          f"{'clears' if clears else 'does not clear'} the {MIN_DELTA} margin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
