"""Closed-form estimators: DiM, MoP and the random control. See implementation_plan.md (T3)."""
import time

import torch
from torch import Tensor

from steer.config import Fold, SteerVec, gen_seed
from steer.data import with_suffix
from steer.model import read_acts, sync


def _mean_last(model, tok, texts: list[str], layer: int) -> Tensor:
    """Mean block-`layer` activation at the last token of each text + READ_SUFFIX."""
    return read_acts(model, tok, with_suffix(texts), layer, "last").mean(0)


def _timed(model, fn) -> tuple[Tensor, float]:
    device = next(model.parameters()).device
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return out, time.perf_counter() - t0


def dim(model, tok, fold: Fold, layer: int) -> SteerVec:
    """Difference of means, mu_p - mu_n, read at the suffix token."""
    v, secs = _timed(model, lambda: _mean_last(model, tok, [p.pos for p in fold.train], layer)
                     - _mean_last(model, tok, [p.neg for p in fold.train], layer))
    cost = {"fwd": 2 * len(fold.train), "bwd": 0, "seconds": secs, "lr": None, "steps": None}
    return SteerVec("dim", layer, fold.idx, v, cost)


def generic_mean(model, tok, short_wiki_texts: list[str], layer: int) -> Tensor:
    """mu_generic, read at the same suffix token as mu_p so position/suffix features cancel."""
    return _mean_last(model, tok, short_wiki_texts, layer)


def mop(model, tok, fold: Fold, layer: int, mu_generic: Tensor) -> SteerVec:
    """Mean of positives, mu_p - mu_generic. mu_generic's cost is not charged (task-independent)."""
    v, secs = _timed(model, lambda: _mean_last(model, tok, [p.pos for p in fold.train], layer)
                     - mu_generic.cpu())
    cost = {"fwd": len(fold.train), "bwd": 0, "seconds": secs, "lr": None, "steps": None}
    return SteerVec("mop", layer, fold.idx, v, cost)


def random_vec(d: int, layer: int, fold: Fold | int) -> SteerVec:
    """Gaussian direction; only its direction is used."""
    k = fold.idx if isinstance(fold, Fold) else fold
    g = torch.Generator().manual_seed(gen_seed(k, 0) + layer)
    cost = {"fwd": 0, "bwd": 0, "seconds": 0.0, "lr": None, "steps": None}
    return SteerVec("random", layer, k, torch.randn(d, generator=g), cost)
