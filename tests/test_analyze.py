"""T7 on a synthetic grid with known linear curves: kl = c * alpha, effect = e * alpha, etc."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from steer import analyze
from steer.config import ALPHAS, KL_BUDGET

C = {"dim": 1.0, "mop": 0.8, "eq5": 2.0, "dpo": 0.5, "random": 0.1}  # random: max KL 0.08 < budget
E = {"dim": 0.4, "mop": 0.1, "eq5": 0.2, "dpo": 0.3, "random": 0.01}
FOLDS, LAYERS, BASE_P = [0, 1, 2], [15, 8], 0.3
PROMPTS = ["I went to the museum, and", "So here's the thing about the park:"]


def metrics(m, alpha, fold):
    e = E[m] * (1 + 0.1 * fold)
    return {"kl": C[m] * alpha, "p_target": BASE_P + e * alpha, "distinct2": 0.9 - 0.5 * alpha,
            "leak": 0.05 * alpha, "dlogp_pos": 3 * alpha, "dlogp_neg": 2 * alpha, "margin": alpha,
            "n_gen": 6, "n_empty": 0}


@pytest.fixture
def results(tmp_path):
    rows, gens, vdir = [], [], tmp_path / "vectors"
    vdir.mkdir()
    zero = {k: 0.0 for k in ["kl", "dlogp_pos", "dlogp_neg", "margin", "leak"]}
    for f in FOLDS:
        rows.append({"fold": f, "layer": -1, "method": "none", "alpha": 0.0, "kind": "baseline",
                     "p_target": BASE_P, "distinct2": 0.9, "n_gen": 6, "n_empty": 0} | zero)
        for layer in LAYERS:
            for m in C:
                for a in ALPHAS[1:]:
                    rows.append({"fold": f, "layer": layer, "method": m, "alpha": a, "kind": "sweep"} | metrics(m, a, f))
                torch.save({"v": torch.randn(16), "cost": {"fwd": 10, "bwd": 5, "seconds": 1.0},
                            "method": m, "layer": layer, "fold": f}, vdir / f"{f}_{layer}_{m}.pt")
            for m in ["eq5", "dpo"]:
                rows.append({"fold": f, "layer": layer, "method": m, "alpha": 0.3, "kind": "native"} | metrics(m, 0.3, f))
                a = KL_BUDGET / C[m]
                rows.append({"fold": f, "layer": layer, "method": m, "alpha": a, "kind": "norm_matched"}
                            | metrics(m, a, f) | {"p_target": BASE_P + 0.5})
                torch.save({"v": torch.randn(16), "cost": {"fwd": 11, "bwd": 6, "seconds": 1.5, "own": {"fwd": 1}},
                            "method": m + "_nm", "layer": layer, "fold": f}, vdir / f"{f}_{layer}_{m}_nm.pt")
    for r in rows:
        gens.append({k: r[k] for k in ["fold", "layer", "method", "alpha", "kind"]}
                    | {"prompts": [p for p in PROMPTS for _ in range(3)],
                       "continuations": [f" {r['method']} sample {i} pizza" for i in range(6)]})
    for name, data in [("grid.jsonl", rows), ("generations.jsonl", gens)]:
        with open(tmp_path / name, "w") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in data)
    json.dump({"leak_words": {"0": ["pizza", "soup"]}}, open(tmp_path / "meta.json", "w"))
    return tmp_path


def test_load_grid_prepends_baseline(results):
    df = analyze.load_grid(results)
    zero = df[(df.kind == "sweep") & (df.alpha == 0)]
    assert len(zero) == len(FOLDS) * len(LAYERS) * len(C)
    assert (zero.kl == 0).all() and (zero.effect == 0).all() and (zero.margin == 0).all()
    assert np.allclose(df[df.kind == "sweep"].effect, df[df.kind == "sweep"].p_target - BASE_P)


def test_at_budget_recovers_linear_values(results):
    with pytest.warns(UserWarning, match="never reaches"):
        at = analyze.at_budget(analyze.load_grid(results))
    for _, r in at.iterrows():
        if r.method == "random":
            assert not r.reached and np.isnan(r.alpha) and np.isnan(r.effect)
            continue
        a = KL_BUDGET / C[r.method]
        assert r.reached and r.alpha == pytest.approx(a)
        assert r.effect == pytest.approx(E[r.method] * (1 + 0.1 * r.fold) * a)
        assert r.dlogp_pos == pytest.approx(3 * a) and r.margin == pytest.approx(a)
        assert r.distinct2 == pytest.approx(0.9 - 0.5 * a) and not r.degenerate


def test_alpha_at_budget_and_running_max():
    rows = pd.DataFrame({"alpha": [0, 0.1, 0.2, 0.3], "kl": [0, 0.08, 0.05, 0.2], "kind": "sweep"})
    with pytest.warns(UserWarning, match="non-monotone"):
        a = analyze.alpha_at_budget(rows)
    assert a == pytest.approx(0.2 + 0.1 * (0.02 / 0.12))  # interpolated on the running max [0, .08, .08, .2]
    assert analyze.alpha_at_budget(rows.assign(kl=[0, 0.01, 0.02, 0.03])) is None
    assert analyze.alpha_at_budget(rows.assign(kl=[0, 0.05, 0.1, 0.3])) == pytest.approx(0.2)  # exact hit


def test_summary_nm_and_costs(results):
    df = analyze.load_grid(results)
    with pytest.warns(UserWarning):
        s = analyze.summary(df, costs=analyze.load_costs(results))
    nm = s[(s.method == "eq5_nm") & (s.layer == 15)].iloc[0]
    assert nm.effect_mean == pytest.approx(0.5) and nm.fwd_mean == 11  # stored as base + retrain; not added again
    assert s[s.method == "random"].n_unreached.eq(len(FOLDS)).all()


def test_outputs_written(results):
    df = analyze.load_grid(results)
    with pytest.warns(UserWarning):
        analyze.figure_1(df, out=str(results / "fig1.pdf"))
        t = analyze.table_1(df, out=str(results / "table1.tex"))
    for name in ["fig1.pdf", "table1.tex", "summary.csv"]:
        assert (results / name).stat().st_size > 0
    tex = (results / "table1.tex").read_text()
    assert t.shape == (7, 14) and "$\\pm$" in tex
    assert "\\begin{tabular" in tex and "\\begin{table" not in tex.replace("\\begin{tabular", "")  # bare tabular
    cos = analyze.cosines(results)
    assert (results / "cosines.csv").exists() and cos.cos_mean.between(-1, 1).all()
    text = analyze.examples(results)
    assert "## dim" in text and "training-topic noun" not in text


def test_report_written(results):
    from steer import report
    out = report.build(results)
    html = open(out).read()
    assert "Verdicts" in html and "data:image/png;base64" in html
    for f in ["curves", "p1_dissociation", "p2_2x2", "cost", "cosines"]:
        assert (results / "figs" / f"{f}.png").exists()
    rows = report.verdicts(analyze._with_nm(analyze.load_grid(results)), 15)
    by = {r["check"]: r for r in rows}
    # synthetic effects at the budget: dim 0.4*0.1/1 = .04 > eq5 0.2*0.1/2 = .01 in every fold
    assert by["DiM steers more than Eq. 5 (rescaled)"]["status"] == "holds"
    assert by["DiM steers more than random"]["status"] == "n/a"  # random never reaches the budget


def test_nm_sweep_replaces_single_point(results):
    """When a norm-matched direction was swept, its budget row is interpolated at KL = budget, not the raw point."""
    rows = [{"fold": f, "layer": 15, "method": "eq5_nm", "alpha": a, "kind": "sweep"} | metrics("eq5", a, f)
            for f in FOLDS for a in ALPHAS[1:]]
    with open(results / "grid.jsonl", "a") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    with pytest.warns(UserWarning):
        pts = analyze._with_nm(analyze.load_grid(results))
    nm15 = pts[(pts.method == "eq5_nm") & (pts.layer == 15)]
    assert len(nm15) == len(FOLDS) and nm15.reached.all() and np.allclose(nm15.kl, KL_BUDGET)
    nm8 = pts[(pts.method == "eq5_nm") & (pts.layer == 8)]  # no sweep there: falls back to the raw point
    assert len(nm8) == len(FOLDS) and not nm8.reached.any()
