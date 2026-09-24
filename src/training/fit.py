"""Warm-started or from-scratch training with truncated BPTT, and batched evaluation.

Training keeps the final weights of a fixed schedule. The holdout (the last
sequences of the training sample) is scored along the way as a diagnostic only:
the starter pack baseline was most likely fitted on those very sequences, so a
holdout score favours staying close to it and would bias checkpoint choice.
Whether a run helps is decided on the validation set by scripts/run_experiment.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from src.training.data import NEED, iter_batches
from src.training.losses import LOSSES
from src.utils.metric import METRIC_CLIP, WPAccumulator


@dataclass
class TrainConfig:
    epochs: float = 0.0
    batch_sequences: int = 256
    chunk: int = 250
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 20
    loss: str = "mse"
    fit_sequences: int = 3872  # sample row groups 0 .. fit_sequences-1 are fitted
    holdout: int = 128  # the next ``holdout`` row groups are scored as a diagnostic
    eval_every: int = 4  # holdout evaluations every N sequence batches


@dataclass
class History:
    records: list[dict] = field(default_factory=list)


def split_groups(path: Path, fit: int, holdout: int) -> tuple[list[int], list[int]]:
    """Fixed row-group indices: 0..fit-1 are fitted, the next ``holdout`` are the diagnostic."""
    total = pq.ParquetFile(path).metadata.num_row_groups
    if fit < 1 or holdout < 1 or fit + holdout > total:
        raise ValueError(f"{fit} fitting + {holdout} holdout sequences do not fit in {total}")
    return list(range(fit)), list(range(fit, fit + holdout))


@torch.no_grad()
def predict(model, path: Path, groups: list[int], *, batch_sequences: int = 256,
            chunk: int = 1000, on_batch=None) -> WPAccumulator:
    """Run ``model`` over whole sequences; score need (AND is_scored when present) rows.

    ``on_batch(batch, predictions)`` sees each batch's (B, 20000, 2) predictions.
    """
    model.eval()
    acc = WPAccumulator()
    batches = [groups[i:i + batch_sequences] for i in range(0, len(groups), batch_sequences)]
    for batch in iter_batches(path, batches):
        n = len(batch.groups)
        state = model.initial_state(n)
        out = np.empty((n, batch.features.shape[1], 2), dtype=np.float32)
        for start in range(0, batch.features.shape[1], chunk):
            prediction, state = model(torch.from_numpy(batch.features[:, start:start + chunk]), state)
            out[:, start:start + chunk] = prediction.numpy()
        need = np.broadcast_to(NEED, (n, NEED.size))
        acc.update(batch.targets.reshape(-1, 2), out.reshape(-1, 2), need.reshape(-1),
                   None if batch.is_scored is None else batch.is_scored.reshape(-1))
        if on_batch is not None:
            on_batch(batch, out)
    return acc


def sequence_stats(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-sequence weighted raw moments (B, 2, 6) for paired bootstrap of Global WP."""
    y = np.clip(targets, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    p = np.clip(predictions, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    w = np.abs(y) * mask[..., None]
    return np.stack([w.sum(1), (w * y).sum(1), (w * p).sum(1), (w * y * y).sum(1),
                     (w * p * p).sum(1), (w * y * p).sum(1)], axis=-1)


def train(model, path: Path, config: TrainConfig, *, seed: int,
          log=print) -> tuple[torch.nn.Module, History]:
    """Fit on the leading sample sequences with a fixed schedule; return the final weights."""
    if config.chunk <= NEED.argmax():
        raise ValueError(f"chunk {config.chunk} leaves the first window without scored rows")
    fit_groups, holdout = split_groups(path, config.fit_sequences, config.holdout)
    rng = np.random.default_rng(seed)
    loss_fn = LOSSES[config.loss]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    per_epoch = len(fit_groups) // config.batch_sequences  # full batches only
    if per_epoch == 0:
        raise ValueError(f"{len(fit_groups)} fitting sequences cannot fill a batch of "
                         f"{config.batch_sequences}")
    total_batches = max(1, round(config.epochs * per_epoch))
    steps_per_batch = math.ceil(NEED.size / config.chunk)
    total_steps = total_batches * steps_per_batch

    def lr_at(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    history = History()

    def evaluate(step: int, batches_done: int, loss: float | None) -> None:
        result = predict(model, path, holdout).result()
        history.records.append({"step": step, "batches": batches_done, "train_loss": loss,
                                "holdout_wp": result["weighted_pearson"], "t0": result["t0"],
                                "t1": result["t1"], "time": time.time()})
        log(f"  step {step:5d} batch {batches_done:3d}: holdout WP {result['weighted_pearson']:.5f}")

    evaluate(0, 0, None)
    order = []
    while len(order) < total_batches:
        shuffled = rng.permutation(fit_groups).tolist()
        size = config.batch_sequences
        order += [shuffled[i * size:(i + 1) * size] for i in range(per_epoch)]
    order = order[:total_batches]
    step = 0
    started = time.time()
    need_t = torch.from_numpy(NEED)
    for done, batch in enumerate(iter_batches(path, order), start=1):
        model.train()
        n = len(batch.groups)
        state = model.initial_state(n)
        for start in range(0, NEED.size, config.chunk):
            stop = start + config.chunk
            x = torch.from_numpy(batch.features[:, start:stop])
            y = torch.from_numpy(batch.targets[:, start:stop])
            mask = need_t[start:stop].expand(n, -1)
            prediction, state = model(x, state)
            loss = loss_fn(prediction, y, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss became {loss.item()} at step {step}")
            state = [s.detach() for s in state]
            step += 1
        rate = step * config.chunk * config.batch_sequences / (time.time() - started)
        log(f"  batch {done}/{total_batches} loss {loss.item():.4f} ({rate / 1e3:.0f}k rows/s)")
        if done % config.eval_every == 0 or done == total_batches:
            evaluate(step, done, loss.item())
    return model, history
