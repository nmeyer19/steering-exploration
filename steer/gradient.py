"""Gradient estimators: Eq. 5, DPO-grad and the norm-matched retrain. See implementation_plan.md (T4)."""
import json
import math
import os
import time
import zlib

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from steer.config import (BATCH, BETA, EVAL_EVERY, GRADIENT_METHODS, LR_GRID, MAX_STEPS, PATIENCE,
                          RESULTS_DIR, SEED, Fold, Pair, SteerVec)
from steer.model import Steerer, encode, seq_logprobs, sync

LOG_DIR = f"{RESULTS_DIR}/logs"


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _lp(model, tok, steerer, v, texts: list[str]) -> Tensor:
    return seq_logprobs(model, encode(tok, texts, "right", _device(model)), steerer, v)


def _dpo(lp_p, lp_n, ref_p, ref_n) -> Tensor:
    return -F.logsigmoid(BETA * ((lp_p - ref_p) - (lp_n - ref_n))).mean()


class RefCache(dict):
    """Sentence -> unsteered seq_logprob. `lookup(pairs)` gives (ref_p, ref_n) on the model's device."""

    @classmethod
    @torch.no_grad()
    def build(cls, model, tok, pairs: list[Pair], batch_size: int = 64) -> "RefCache":
        ref = cls()
        ref.device = _device(model)
        texts = sorted({t for p in pairs for t in (p.pos, p.neg)})
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            ref.update(zip(chunk, _lp(model, tok, None, None, chunk).tolist()))
        return ref

    def lookup(self, pairs: list[Pair]) -> tuple[Tensor, Tensor]:
        return tuple(torch.tensor([self[getattr(p, s)] for p in pairs], device=self.device)
                     for s in ("pos", "neg"))


def eq5_loss(model, tok, steerer, v, pairs: list[Pair]) -> Tensor:
    """Mean NLL of the target-side sentences (all tokens after BOS). Negatives are never used."""
    return -_lp(model, tok, steerer, v, [p.pos for p in pairs]).mean()


def dpo_loss(model, tok, steerer, v, pairs: list[Pair], ref: RefCache) -> Tensor:
    lp_p = _lp(model, tok, steerer, v, [p.pos for p in pairs])
    lp_n = _lp(model, tok, steerer, v, [p.neg for p in pairs])
    return _dpo(lp_p, lp_n, *ref.lookup(pairs))


def _batch_seed(fold: int, layer: int, tag: str) -> int:
    """Deterministic across processes (unlike hash() on str)."""
    return zlib.crc32(f"{SEED}/{fold}/{layer}/{tag}".encode())


@torch.no_grad()
def _evaluate(model, tok, steerer, v, method, val: list[Pair], ref: RefCache) -> dict:
    """Val loss plus mean delta log p of each half vs. unsteered (the latter is logging only)."""
    lp_p = _lp(model, tok, steerer, v, [p.pos for p in val])
    lp_n = _lp(model, tok, steerer, v, [p.neg for p in val])
    ref_p, ref_n = ref.lookup(val)
    loss = -lp_p.mean() if method == "eq5" else _dpo(lp_p, lp_n, ref_p, ref_n)
    return {"val_loss": loss.item(), "dlogp_pos": (lp_p - ref_p).mean().item(),
            "dlogp_neg": (lp_n - ref_n).mean().item()}


def _train_one(model, tok, steerer, fold, layer, r_l, method, tag, lr, fixed_norm, init, max_steps,
               ref, cost, log) -> dict:
    """One learning rate. Returns the best-val-loss candidate (step 0 is never one)."""
    device = _device(model)
    d = model.config.hidden_size
    u = nn.Parameter(init.to(device, torch.float32) / r_l if init is not None
                     else torch.zeros(d, device=device))
    opt = torch.optim.Adam([u], lr=lr)
    g = torch.Generator().manual_seed(_batch_seed(fold.idx, layer, tag))
    per_seq = 1 if method == "eq5" else 2  # sequences per pair in the loss

    def record(step, train_loss):
        ev = _evaluate(model, tok, steerer, r_l * u, method, fold.val, ref)
        row = {"lr": lr, "step": step, "train_loss": train_loss, **ev, "u_norm": u.norm().item()}
        log.write(json.dumps(row) + "\n")
        log.flush()
        return ev["val_loss"]

    record(0, None)  # logging only: not charged, not a candidate
    best = {"val_loss": math.inf, "lr": lr, "step": None, "u": None}
    bad, window = 0, []
    for step in range(1, max_steps + 1):
        idx = torch.randperm(len(fold.train), generator=g)[:BATCH].tolist()
        pairs = [fold.train[i] for i in idx]
        v = r_l * u
        loss = eq5_loss(model, tok, steerer, v, pairs) if method == "eq5" \
            else dpo_loss(model, tok, steerer, v, pairs, ref)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if fixed_norm is not None:
            with torch.no_grad():
                u.mul_(fixed_norm / u.norm())
        cost["fwd"] += per_seq * len(pairs)
        cost["bwd"] += per_seq * len(pairs)
        cost["steps_run"] += 1
        window.append(loss.item())

        if step % EVAL_EVERY == 0 or step == max_steps:
            val_loss = record(step, sum(window) / len(window))
            cost["fwd"] += per_seq * len(fold.val)
            window = []
            if val_loss < best["val_loss"]:
                best.update(val_loss=val_loss, step=step, u=u.detach().clone())
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    return best


def train_vec(model, tok, fold: Fold, layer: int, r_l: float, method: str, fixed_norm: float | None = None,
              init: Tensor | None = None, lr: float | None = None, max_steps: int = MAX_STEPS,
              log_dir: str = LOG_DIR) -> SteerVec:
    """Adam on u (v = r_l * u) with early stopping on val loss; best LR by best val loss.

    fixed_norm (= ||v|| / r_l) projects u back to that norm after each step and requires `init`.
    Cost counts every LR tried; DPO is also charged for its reference passes.
    """
    assert method in GRADIENT_METHODS, method
    if fixed_norm is not None and init is None:
        raise ValueError("fixed_norm needs a nonzero init")
    nm = "_nm" if fixed_norm is not None else ""
    tag = method + nm
    device = _device(model)
    os.makedirs(log_dir, exist_ok=True)

    sync(device)
    t0 = time.perf_counter()
    cost = {"fwd": 0, "bwd": 0, "steps_run": 0}
    # DPO needs unsteered log-probs of train+val; Eq. 5 only needs val's for logging (not charged).
    ref = RefCache.build(model, tok, fold.train + fold.val if method == "dpo" else fold.val)
    if method == "dpo":
        cost["fwd"] += len(ref)

    lrs = [lr] if lr is not None else LR_GRID
    best = None
    with open(f"{log_dir}/{method}_{fold.idx}_{layer}{nm}.jsonl", "w") as log, Steerer(model, layer) as st:
        for lr_ in lrs:
            cand = _train_one(model, tok, st, fold, layer, r_l, method, tag, lr_, fixed_norm, init,
                              max_steps, ref, cost, log)
            if cand["u"] is not None and (best is None or cand["val_loss"] < best["val_loss"]):
                best = cand
    sync(device)
    cost["seconds"] = time.perf_counter() - t0

    if best is None or best["u"].norm() < 1e-6:
        raise RuntimeError(f"{tag} fold {fold.idx} layer {layer}: best u is ~0; refusing to return it")
    cost.update(lr=best["lr"], steps=best["step"], best_val_loss=best["val_loss"], lrs=lrs)
    return SteerVec(tag, layer, fold.idx, (r_l * best["u"]).cpu(), cost)


def norm_matched(model, tok, fold: Fold, layer: int, r_l: float, base: SteerVec, alpha_star: float,
                 max_steps: int = MAX_STEPS, log_dir: str = LOG_DIR) -> SteerVec:
    """Retrain at ||v|| = alpha_star * r_l, warm-started from base's direction, at base's LR.

    The returned cost is the total (base + retrain); the retrain alone is under cost["own"].
    """
    method = base.method.removesuffix("_nm")
    nm = train_vec(model, tok, fold, layer, r_l, method, fixed_norm=alpha_star,
                   init=base.v / base.v.norm() * alpha_star * r_l, lr=base.cost["lr"],
                   max_steps=max_steps, log_dir=log_dir)
    own = nm.cost
    nm.cost = {**own, "own": own, **{k: base.cost[k] + own[k] for k in ("fwd", "bwd", "seconds")}}
    return nm
