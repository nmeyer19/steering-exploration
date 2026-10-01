"""Interpolation at the KL budget, Figure 1, Table 1, cosines and examples. See implementation_plan.md (T7).

Pure pandas / numpy / matplotlib (+ torch.load for the saved vectors); never loads the model.
"""
import argparse
import itertools
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from steer.config import DISTINCT2_MIN, HEADLINE_LAYER, KL_BUDGET, METHODS  # noqa: E402

METRICS = ["kl", "p_target", "effect", "distinct2", "leak", "dlogp_pos", "dlogp_neg", "margin"]
ZERO_AT_BASELINE = ["kl", "effect", "dlogp_pos", "dlogp_neg", "margin"]
TABLE_METHODS = ["dim", "mop", "eq5", "eq5_nm", "dpo", "dpo_nm", "random"]
NM_METHODS = ["eq5_nm", "dpo_nm"]
COLORS = {  # validated categorical slots (CVD-safe, adjacent pairs); random is a neutral control
    "dim": "#2a78d6", "eq5": "#eb6834", "dpo": "#1baf7a", "mop": "#eda100", "random": "#8a8984",
}
COLORS.update({"eq5_nm": COLORS["eq5"], "dpo_nm": COLORS["dpo"]})
LABELS = {"dim": "DiM", "mop": "MoP", "eq5": "Eq. 5", "eq5_nm": "Eq. 5 (NM)", "dpo": "DPO-grad",
          "dpo_nm": "DPO-grad (NM)", "random": "random"}
BASELINE_LAYER = -1  # the unsteered baseline row is layer-independent; stored with layer = -1
GROUP = ["fold", "layer", "method"]


def _read_jsonl(path) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_grid(results_dir) -> pd.DataFrame:
    """grid.jsonl + `effect`, with the fold's baseline prepended as alpha = 0 to every sweep curve."""
    df = pd.DataFrame(_read_jsonl(os.path.join(results_dir, "grid.jsonl")))
    df = df.drop_duplicates(["fold", "layer", "method", "alpha", "kind"], keep="last")
    base = df[df.kind == "baseline"].set_index("fold")
    df["effect"] = df.p_target - df.fold.map(base.p_target)
    zeros = []
    for (fold, layer, method), rows in df[df.kind == "sweep"].groupby(GROUP):
        if (rows.alpha == 0).any():
            continue
        row = base.loc[fold].to_dict() | {"fold": fold, "layer": layer, "method": method, "alpha": 0.0, "kind": "sweep"}
        zeros.append(row | {m: 0.0 for m in ZERO_AT_BASELINE})
    df = pd.concat([df, pd.DataFrame(zeros)], ignore_index=True) if zeros else df
    return df.sort_values(GROUP + ["kind", "alpha"]).reset_index(drop=True)


def _interp(rows: pd.DataFrame, budget: float, label="") -> dict | None:
    """Linear interpolation of alpha and every metric at KL = budget on a monotone (running max) curve."""
    rows = rows.sort_values("alpha")
    kl = rows.kl.to_numpy(float)
    mono = np.maximum.accumulate(kl)
    if not np.allclose(mono, kl):
        warnings.warn(f"non-monotone KL {label}: {np.round(kl, 4).tolist()} -> running max")
    hit = np.flatnonzero(mono >= budget)
    if len(hit) == 0 or (hit[0] == 0 and mono[0] != budget):  # never reached, or no point below it
        return None
    i = hit[0]
    if mono[i] == budget:
        t, j = 0.0, i
    else:
        j, t = i - 1, (budget - mono[i - 1]) / (mono[i] - mono[i - 1])
    cols = ["alpha"] + [m for m in METRICS if m in rows and m != "kl"]
    x0, x1 = rows[cols].iloc[j].to_numpy(float), rows[cols].iloc[i].to_numpy(float)
    return dict(zip(cols, x0 + t * (x1 - x0))) | {"kl": budget, "monotone": bool(np.allclose(mono, kl))}


def alpha_at_budget(rows, budget=KL_BUDGET) -> float | None:
    """The alpha at which a sweep curve (including its alpha = 0 point) reaches KL = budget, else None."""
    rows = pd.DataFrame(rows)
    rows = rows[rows.kind == "sweep"] if "kind" in rows else rows
    out = _interp(rows, budget)
    return None if out is None else float(out["alpha"])


def at_budget(df, budget=KL_BUDGET) -> pd.DataFrame:
    """One row per (fold, layer, method): every metric at KL = budget. NaN + reached=False if never reached."""
    out = []
    for key, rows in df[df.kind == "sweep"].groupby(GROUP):
        r = _interp(rows, budget, label=str(key))
        if r is None:
            warnings.warn(f"{key} never reaches KL={budget} (max {rows.kl.max():.4f}); NaN, not extrapolated")
            r = {c: np.nan for c in ["alpha"] + METRICS} | {"monotone": True}
        out.append(dict(zip(GROUP, key)) | r | {"reached": r["kl"] == budget})
    out = pd.DataFrame(out)
    out["degenerate"] = out.distinct2 < DISTINCT2_MIN
    for _, r in out[out.degenerate].iterrows():
        warnings.warn(f"degenerate at budget: fold {r.fold} layer {r.layer} {r.method} distinct2={r.distinct2:.3f}")
    return out


def load_costs(results_dir) -> pd.DataFrame:
    """Cost per saved vector (fwd, bwd, seconds). *_nm vectors already store base + retrain cost."""
    vdir, rows = os.path.join(results_dir, "vectors"), []
    for name in sorted(os.listdir(vdir)) if os.path.isdir(vdir) else []:
        d = torch.load(os.path.join(vdir, name), weights_only=False)
        c = d.get("cost") or {}
        rows.append({"fold": d["fold"], "layer": d["layer"], "method": d["method"],
                     **{k: float(c.get(k) or 0) for k in ["fwd", "bwd", "seconds"]}})
    return pd.DataFrame(rows, columns=GROUP + ["fwd", "bwd", "seconds"])


def _with_nm(df, budget=KL_BUDGET) -> pd.DataFrame:
    """At-budget rows per (fold, layer, method), with eq5_nm / dpo_nm as their own methods.

    The norm-matched direction is swept like every other method (run.py) and interpolated at the budget: a fixed
    norm does not fix KL. Only where no such sweep exists does the single retrained point stand in, at its own KL.
    """
    at = at_budget(df, budget)
    nm = df[df.kind == "norm_matched"].copy()
    nm["method"] = [m if m.endswith("_nm") else m + "_nm" for m in nm.method]
    swept = set(map(tuple, at[GROUP].to_numpy().tolist()))
    nm = nm[[(f, l, m) not in swept for f, l, m in nm[GROUP].to_numpy().tolist()]]
    nm["reached"], nm["degenerate"] = False, nm.distinct2 < DISTINCT2_MIN
    return pd.concat([at, nm], ignore_index=True)


def summary(df, budget=KL_BUDGET, costs=None) -> pd.DataFrame:
    """Mean and std across folds per (layer, method), at the budget; optional cost columns."""
    pts = _with_nm(df, budget)
    if costs is not None and len(costs):
        pts = pts.merge(costs, on=GROUP, how="left")
    cols = [c for c in ["alpha"] + METRICS + ["fwd", "bwd", "seconds"] if c in pts]
    g = pts.groupby(["layer", "method"])
    s = g[cols].agg(["mean", "std"])
    s.columns = [f"{c}_{a}" for c, a in s.columns]
    s["n_folds"] = g.effect.count()
    s["n_unreached"] = g.reached.apply(lambda x: int((~x.astype(bool)).sum()))
    s["n_degenerate"] = g.degenerate.apply(lambda x: int(x.astype(bool).sum()))
    return s.reset_index()


def _curve(rows):
    """Sweep curve as (kl, effect), KL made monotone, ties dropped so np.interp is well defined."""
    rows = rows.sort_values("alpha")
    kl, eff = np.maximum.accumulate(rows.kl.to_numpy(float)), rows.effect.to_numpy(float)
    keep = np.r_[True, np.diff(kl) > 0]
    return kl[keep], eff[keep]


def figure_1(df, layer=HEADLINE_LAYER, out="results/fig1.pdf", budget=KL_BUDGET):
    """Effect vs KL, mean +- std band across folds, native (o) and norm-matched (s) markers, budget line."""
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    plot_curves(ax, df, layer, budget)
    ax.legend(frameon=False, fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=200)  # for the Markdown draft
    plt.close(fig)
    return out


def plot_curves(ax, df, layer, budget=KL_BUDGET, fontsize=5):
    """Draw the effect-vs-KL curves for one layer onto `ax` (no legend; callers add one)."""
    d = df[df.layer == layer]
    pts = d[d.kind == "native"].groupby(["method", "kind"])[["kl", "effect"]].agg(["mean", "std"])
    xmax = 3 * budget  # zoom on the budget; native points beyond it are drawn at the edge as ">" with their KL
    for m in [m for m in METHODS + NM_METHODS if m in set(d[d.kind == "sweep"].method)]:
        curves = [_curve(r) for _, r in d[(d.method == m) & (d.kind == "sweep")].groupby("fold")]
        if not curves:
            continue
        grid = np.linspace(0, min(min(kl[-1] for kl, _ in curves), xmax), 50)
        ys = np.array([np.interp(grid, k, e) for k, e in curves])
        mu, sd = ys.mean(0), ys.std(0, ddof=1) if len(ys) > 1 else np.zeros_like(grid)
        ax.plot(grid, mu, color=COLORS[m], lw=1.2, ls="--" if m in NM_METHODS else "-", label=LABELS[m])
        ax.fill_between(grid, mu - sd, mu + sd, color=COLORS[m], alpha=0.10 if m in NM_METHODS else 0.18, lw=0)
    n_edge = 0  # stagger labels of off-axis native points so they don't overlap
    for (m, kind), r in pts.iterrows():
        filled, x, y = kind == "norm_matched", r[("kl", "mean")], r[("effect", "mean")]
        style = {"color": COLORS.get(m.removesuffix("_nm"), "k"), "mfc": None if filled else "white", "mew": 0.8,
                 "ms": 3.5, "lw": 0.6}
        if x > xmax:
            ax.errorbar(xmax, y, yerr=np.nan_to_num(r[("effect", "std")]), fmt=">", clip_on=False, **style)
            ax.annotate(f"{LABELS.get(m, m)} native, KL={x:.2g}", (xmax, y), xytext=(-5, -8 * n_edge),
                        textcoords="offset points", ha="right",
                        va="center", fontsize=fontsize, color="#52514e")
            n_edge += 1
        else:
            ax.errorbar(x, y, xerr=np.nan_to_num(r[("kl", "std")]), yerr=np.nan_to_num(r[("effect", "std")]),
                        fmt="s" if filled else "o", **style)
    ax.axvline(budget, color="k", ls="--", lw=0.7)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("KL on WikiText (nats/token)")
    ax.set_ylabel(r"$\Delta$ P(target)")
    ax.plot([], [], "o", color="k", mfc="white", ms=3.5, label="native")
    ax.spines[["top", "right"]].set_visible(False)


TABLE_COLS = [  # (column, header, format); cost columns show the mean only
    ("effect", r"$\Delta$P", "{:.3f}"), ("dlogp_pos", r"$\Delta\log p(w_p)$", "{:.2f}"),
    ("dlogp_neg", r"$\Delta\log p(w_n)$", "{:.2f}"), ("margin", "margin", "{:.2f}"),
    ("fwd", "fwd", "{:.0f}"), ("bwd", "bwd", "{:.0f}"), ("seconds", "sec", "{:.1f}"),
]
COST_COLS = {"fwd", "bwd", "seconds"}


def table_1(df, out="results/table1.tex", results_dir=None, budget=KL_BUDGET) -> pd.DataFrame:
    """Table 1 (LaTeX, mean +- std across folds) and summary.csv, next to `out`."""
    results_dir = results_dir or os.path.dirname(out) or "."
    s = summary(df, budget, costs=load_costs(results_dir))
    s.to_csv(os.path.join(os.path.dirname(out) or ".", "summary.csv"), index=False)
    layers = sorted(s.layer.unique(), key=lambda l: (l != HEADLINE_LAYER, l))
    cells = {}
    for layer, (col, head, fmt) in itertools.product(layers, TABLE_COLS):
        vals = []
        for m in TABLE_METHODS:
            r = s[(s.layer == layer) & (s.method == m)]
            mu = r[f"{col}_mean"].iloc[0] if len(r) and f"{col}_mean" in r else np.nan
            sd = r[f"{col}_std"].iloc[0] if len(r) and f"{col}_std" in r else np.nan
            if np.isnan(mu):
                vals.append("--")
            elif col in COST_COLS or np.isnan(sd):
                vals.append(fmt.format(mu))
            else:
                vals.append(f"{fmt.format(mu)} $\\pm$ {fmt.format(sd)}")
        cells[(f"block {layer}", head)] = vals
    t = pd.DataFrame(cells, index=[m.replace("_", r"\_") for m in TABLE_METHODS])
    t.to_latex(out, escape=False, column_format="l" + "c" * t.shape[1], multicolumn_format="c")
    return t


def cosines(results_dir) -> pd.DataFrame:
    """Pairwise cosine similarity among the saved vectors per (fold, layer), averaged across folds."""
    vdir, vecs = os.path.join(results_dir, "vectors"), {}
    for name in sorted(os.listdir(vdir)):
        d = torch.load(os.path.join(vdir, name), weights_only=False)
        vecs.setdefault((d["fold"], d["layer"]), {})[d["method"]] = d["v"].float().flatten()
    rows = []
    for (fold, layer), vs in vecs.items():
        for a, b in itertools.combinations(sorted(vs), 2):
            cos = torch.nn.functional.cosine_similarity(vs[a], vs[b], dim=0).item()
            rows.append({"fold": fold, "layer": layer, "a": a, "b": b, "cos": cos})
    out = (pd.DataFrame(rows).groupby(["layer", "a", "b"]).cos.agg(["mean", "std", "count"])
           .reset_index().rename(columns={"mean": "cos_mean", "std": "cos_std", "count": "n_folds"}))
    out.to_csv(os.path.join(results_dir, "cosines.csv"), index=False)
    return out


def examples(results_dir, k=3, fold=0, layer=HEADLINE_LAYER, budget=KL_BUDGET, n_prompts=2) -> str:
    """examples.md: each method's continuations of 2 fixed prompts at the alpha whose KL is nearest the budget."""
    df = load_grid(results_dir)
    gens = {(g["fold"], g["layer"], g["method"], round(g["alpha"], 6), g["kind"]): g
            for g in _read_jsonl(os.path.join(results_dir, "generations.jsonl"))}

    d = df[(df.fold == fold) & (df.layer == layer)]
    picks = [("none (unsteered)", df[(df.fold == fold) & (df.kind == "baseline")])]
    for m in [m for m in METHODS + NM_METHODS if m in set(d.method)]:
        sw = d[(d.method == m) & (d.kind == "sweep") & (d.alpha > 0)]
        if len(sw):
            picks.append((m, sw.loc[[(sw.kl - budget).abs().idxmin()]]))
    lines = [f"# Examples: fold {fold}, block {layer}, alpha with KL nearest {budget}\n"]
    prompts = None
    for label, r in picks:
        if r.empty:
            continue
        r = r.iloc[0]
        key = (int(r.fold), int(r.layer), r.method, round(float(r.alpha), 6), r.kind)
        g = gens.get(key)
        if g is None:
            continue
        if prompts is None:
            prompts = list(dict.fromkeys(g["prompts"]))[:n_prompts]
        lines.append(f"## {label} (alpha={r.alpha:.3f}, KL={r.kl:.3f}, P(target)={r.p_target:.3f})\n")
        for p in prompts:
            conts = [c for q, c in zip(g["prompts"], g["continuations"]) if q == p][:k]
            lines += [f"**{p}**\n"] + [f"- {c.strip()!r}" for c in conts] + [""]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results_dir, "examples.md"), "w") as f:
        f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    df = load_grid(a.results)
    layer = HEADLINE_LAYER if HEADLINE_LAYER in set(df.layer) else int(df[df.layer >= 0].layer.iloc[0])
    pd.set_option("display.width", 200, "display.max_columns", 30)
    s = summary(df, costs=load_costs(a.results))
    print(s[["layer", "method", "alpha_mean", "effect_mean", "effect_std", "margin_mean",
             "n_folds", "n_unreached", "n_degenerate"]].round(4).to_string(index=False))
    figure_1(df, layer=layer, out=os.path.join(a.results, "fig1.pdf"))
    table_1(df, out=os.path.join(a.results, "table1.tex"))
    cosines(a.results)
    examples(a.results, layer=layer)
    from steer import report  # imported here: report builds on this module
    report.build(a.results)
    print(f"wrote fig1.pdf, table1.tex, summary.csv, cosines.csv, examples.md, report.html, figs/ to {a.results}/")


if __name__ == "__main__":
    main()
