"""T3/T4: closed-form and gradient estimators on the real fold 0, with short training."""
import json
import math

import pytest
import torch

from steer.closed_form import dim, generic_mean, mop, random_vec
from steer.config import HEADLINE_LAYER, TARGET
from steer.data import make_folds
from steer.gradient import RefCache, _evaluate, norm_matched, train_vec
from steer.model import Steerer, load_model, residual_norm

LAYER, STEPS = HEADLINE_LAYER, 40


@pytest.fixture(scope="module")
def setup():
    model, tok = load_model()
    fold = make_folds(TARGET)[0]
    # r_l from the fold's own sentences keeps the test offline; the real run uses WikiText.
    r_l = residual_norm(model, tok, [p.pos for p in fold.train] + [p.neg for p in fold.train], LAYER)
    return model, tok, fold, r_l


@pytest.fixture(scope="module")
def log_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("logs")


@pytest.fixture(scope="module")
def trained(setup, log_dir):
    model, tok, fold, r_l = setup
    return {m: train_vec(model, tok, fold, LAYER, r_l, m, max_steps=STEPS, log_dir=str(log_dir))
            for m in ("eq5", "dpo")}


def val_metrics(setup, method, v):
    model, tok, fold, _ = setup
    ref = RefCache.build(model, tok, fold.val)
    with Steerer(model, LAYER) as st:
        return _evaluate(model, tok, st, None if v is None else v.to(ref.device), method, fold.val, ref)


def _ok(vec, d):
    assert vec.v.shape == (d,) and vec.v.device.type == "cpu"
    assert torch.isfinite(vec.v).all() and vec.v.norm() > 0


def test_closed_form(setup):
    model, tok, fold, _ = setup
    d = model.config.hidden_size
    generic = ["The river runs north through the valley.", "In 1912 the town built a new station.",
               "The committee met twice that year.", "Its population was about four thousand."]
    mu_g = generic_mean(model, tok, generic, LAYER)  # offline stand-in for wiki_short_texts
    for vec in (dim(model, tok, fold, LAYER), mop(model, tok, fold, LAYER, mu_g), random_vec(d, LAYER, fold)):
        _ok(vec, d)
        assert vec.fold == fold.idx and vec.layer == LAYER
    assert torch.equal(random_vec(d, LAYER, fold).v, random_vec(d, LAYER, fold.idx).v)


def test_eq5(setup, trained):
    vec = trained["eq5"]
    _ok(vec, setup[0].config.hidden_size)
    before, after = val_metrics(setup, "eq5", None), val_metrics(setup, "eq5", vec.v)
    print(f"\neq5: val loss {before['val_loss']:.3f} -> {after['val_loss']:.3f}, "
          f"dlogp_pos {after['dlogp_pos']:+.3f}, dlogp_neg {after['dlogp_neg']:+.3f}, "
          f"alpha {vec.v.norm() / setup[3]:.3f}, cost {vec.cost}")
    assert after["val_loss"] < before["val_loss"]
    assert after["dlogp_pos"] > 0
    assert vec.cost["bwd"] > 0 and vec.cost["lr"] in (3e-4, 1e-3, 3e-3) and vec.cost["steps"] > 0


def test_dpo(setup, trained):
    vec = trained["dpo"]
    _ok(vec, setup[0].config.hidden_size)
    after = val_metrics(setup, "dpo", vec.v)
    margin = after["dlogp_pos"] - after["dlogp_neg"]
    print(f"\ndpo: val loss {math.log(2):.4f} -> {after['val_loss']:.4f}, "
          f"dlogp_pos {after['dlogp_pos']:+.3f}, dlogp_neg {after['dlogp_neg']:+.3f}, margin {margin:+.3f}, "
          f"alpha {vec.v.norm() / setup[3]:.3f}, cost {vec.cost}")
    assert after["val_loss"] < math.log(2)  # DPO loss at u = 0
    assert margin > 0
    # DPO is charged for reference passes on top of training: more fwd than bwd + val evals alone.
    assert vec.cost["fwd"] > vec.cost["bwd"]


def test_logs(setup, trained, log_dir):
    fold = setup[2]
    for m in ("eq5", "dpo"):
        rows = [json.loads(line) for line in open(log_dir / f"{m}_{fold.idx}_{LAYER}.jsonl")]
        assert {r["lr"] for r in rows} == set(trained[m].cost["lrs"])
        assert all({"train_loss", "val_loss", "u_norm", "dlogp_pos", "dlogp_neg"} <= r.keys() for r in rows)
        assert all(r["u_norm"] == 0 and abs(r["dlogp_pos"]) < 1e-3 for r in rows if r["step"] == 0)


def test_norm_matched(setup, trained, log_dir):
    model, tok, fold, r_l = setup
    base = trained["eq5"]
    nm = norm_matched(model, tok, fold, LAYER, r_l, base, alpha_star=0.1, max_steps=20, log_dir=str(log_dir))
    assert nm.method == "eq5_nm"
    assert abs(nm.v.norm().item() / r_l - 0.1) < 1e-4
    assert nm.cost["fwd"] == base.cost["fwd"] + nm.cost["own"]["fwd"]
    assert (log_dir / f"eq5_{fold.idx}_{LAYER}_nm.jsonl").exists()
