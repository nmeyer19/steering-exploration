"""One-page HTML report: automatic verdicts for P1 / P2 plus the figures that show why.

Run with `python -m steer.report [--results results]`; writes results/report.html and results/figs/*.png.
Pure pandas / numpy / matplotlib on top of analyze.py; never loads the model.
"""
import argparse
import base64
import html
import io
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from steer import analyze  # noqa: E402
from steer.analyze import COLORS, LABELS  # noqa: E402
from steer.config import HEADLINE_LAYER, KL_BUDGET  # noqa: E402

ORDER = ["dim", "mop", "eq5", "eq5_nm", "dpo", "dpo_nm", "random"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#e8e7e3", "#2a78d6"])  # two hues, gray midpoint
STYLE = {"font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
         "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titleweight": "bold"}


# ---------------------------------------------------------------- data

def load(results_dir, budget=KL_BUDGET):
    """Everything the report needs, with analyze.py's warnings captured instead of printed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = analyze.load_grid(results_dir)
        costs = analyze.load_costs(results_dir)
        pts = analyze._with_nm(df, budget).merge(costs, on=analyze.GROUP, how="left")
        summ = analyze.summary(df, budget, costs=costs)
    meta_path = os.path.join(results_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return df, pts, summ, meta, [str(w.message) for w in caught]


def _lrs(results_dir):
    """Chosen learning rate per (fold, layer, method), from the saved vectors."""
    vdir, out = os.path.join(results_dir, "vectors"), {}
    for name in sorted(os.listdir(vdir)) if os.path.isdir(vdir) else []:
        d = torch.load(os.path.join(vdir, name), weights_only=False)
        out[(d["fold"], d["layer"], d["method"])] = (d.get("cost") or {}).get("lr")
    return out


def _per_fold(pts, layer, col):
    """method -> Series over folds of `col` at the budget."""
    d = pts[pts.layer == layer]
    return {m: d[d.method == m].set_index("fold")[col] for m in ORDER if (d.method == m).any()}


# ---------------------------------------------------------------- verdicts

def _check(label, a, b, op, names):
    """Compare two per-fold Series fold by fold; returns a verdict row."""
    if a is None or b is None:
        return {"check": label, "status": "n/a", "detail": "method missing"}
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if j.empty:
        return {"check": label, "status": "n/a", "detail": "budget not reached"}
    k, n = int(op(j.a, j.b).sum()), len(j)
    status = "holds" if k == n else ("mixed" if k > n / 2 else "fails")
    return {"check": label, "status": status, "k": k, "n": n,
            "detail": f"{names[0]} {j.a.mean():+.3f} vs {names[1]} {j.b.mean():+.3f} (mean over folds)"}


def verdicts(pts, layer):
    """The P1 / P2 predictions from research_plan.md §3, each checked fold by fold."""
    e, lp, mg = (_per_fold(pts, layer, c) for c in ["effect", "dlogp_pos", "margin"])
    gt, ge, lt = (lambda a, b: a > b), (lambda a, b: a >= b), (lambda a, b: a < b)
    have = all(m in e for m in ["mop", "eq5", "dim"])
    mop = ((e["mop"] - e["eq5"]).abs(), (e["mop"] - e["dim"]).abs()) if have else (None, None)
    L = LABELS
    rows = [
        ("P1", "DiM steers more than Eq. 5 (rescaled)", e.get("dim"), e.get("eq5"), gt, (L["dim"], L["eq5"])),
        ("P1", "DiM steers more than Eq. 5 (norm-matched)", e.get("dim"), e.get("eq5_nm"), gt,
         (L["dim"], L["eq5_nm"])),
        ("P1", "Eq. 5 raises log p(w_p) more than DiM", lp.get("eq5"), lp.get("dim"), gt, (L["eq5"], L["dim"])),
        ("P1", "Eq. 5 has a smaller margin than DiM", mg.get("eq5"), mg.get("dim"), lt, (L["eq5"], L["dim"])),
        ("P2", "DPO-grad steers at least as well as DiM", e.get("dpo"), e.get("dim"), ge, (L["dpo"], L["dim"])),
        ("P2", "DPO-grad (NM) steers at least as well as DiM", e.get("dpo_nm"), e.get("dim"), ge,
         (L["dpo_nm"], L["dim"])),
        ("P2", "MoP is closer to Eq. 5 than to DiM", *mop, lt, ("|MoP − Eq. 5|", "|MoP − DiM|")),
        ("sanity", "DiM steers more than random", e.get("dim"), e.get("random"), gt, (L["dim"], L["random"])),
    ]
    return [{"prediction": pred} | _check(label, a, b, op, names) for pred, label, a, b, op, names in rows]


def health(df, pts, warns):
    """Things that would make the numbers untrustworthy, in plain words."""
    notes = []
    un = pts[pts.reached == False]  # noqa: E712
    for _, r in un.iterrows():
        notes.append(f"{LABELS.get(r.method, r.method)} at block {r.layer}, fold {r.fold} never reached KL = "
                     f"{KL_BUDGET}; its budget numbers are missing (extend ALPHAS).")
    for _, r in pts[pts.degenerate == True].iterrows():  # noqa: E712
        notes.append(f"{LABELS.get(r.method, r.method)} at block {r.layer}, fold {r.fold} is degenerate at the "
                     f"budget (distinct-2 = {r.distinct2:.2f}); read its continuations before trusting it.")
    for w in warns:
        if "not monotone" in w or "monotone" in w.lower():
            notes.append(f"KL was not monotone in alpha somewhere: {w}")
    dpo = pts[pts.method.isin(["dpo", "dpo_nm"])]
    if len(dpo) and (dpo.dlogp_pos < 0).mean() > 0.5:
        notes.append("DPO-grad lowers log p of the target sentences too (Δlog p(w_p) < 0 in most folds): it meets "
                     "its margin mainly by pushing the negatives down. That's the DPO confound in research_plan §2.4, "
                     "so a weak DPO-grad result is ambiguous rather than decisive for P2.")
    return notes


# ---------------------------------------------------------------- figures

def _png(fig, path):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return base64.b64encode(buf.getvalue()).decode()


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _bars(ax, pts, layer, col, methods, hatch_nm=True):
    """One bar per method (mean over folds), fold values as dots, NM variants hatched."""
    d = pts[pts.layer == layer]
    xs, labels = [], []
    for i, m in enumerate(methods):
        v = d[d.method == m][col].dropna()
        if v.empty:
            continue
        ax.bar(i, v.mean(), width=0.72, color=COLORS[m], edgecolor="white", linewidth=1.5,
               hatch="////" if (hatch_nm and m.endswith("_nm")) else None)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=9, color=INK, zorder=3, lw=0)
        xs.append(i)
        labels.append(LABELS[m])
    ax.set_xticks(xs, labels, rotation=30, ha="right")
    ax.axhline(0, color=INK2, lw=0.6)
    _clean(ax)


def fig_curves(df, layers, budget, path):
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 3.2), sharey=True, squeeze=False)
    for ax, layer in zip(axes[0], layers):
        analyze.plot_curves(ax, df, layer, budget, fontsize=6)
        ax.set_title(f"Block {layer}" + (" (headline)" if layer == HEADLINE_LAYER else ""))
        _clean(ax)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 1.08))
    return _png(fig, path)


def fig_p1(pts, layer, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    _bars(axes[0], pts, layer, "effect", ORDER)
    axes[0].set_title("Steering effect at the budget")
    axes[0].set_ylabel("Δ P(target) vs. unsteered")

    ax, d = axes[1], pts[pts.layer == layer]
    methods = [m for m in ORDER if (d.method == m).any() and m != "random"]
    for i, m in enumerate(methods):
        r = d[d.method == m]
        ax.bar(i - 0.19, r.dlogp_pos.mean(), 0.36, color=COLORS[m], edgecolor="white", lw=1.5)
        ax.bar(i + 0.19, r.dlogp_neg.mean(), 0.36, color=COLORS[m], alpha=0.45, edgecolor="white", lw=1.5,
               hatch="....")
    ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=30, ha="right")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.legend(handles=[Patch(facecolor=INK2, label="target sentence w_p"),
                       Patch(facecolor=INK2, alpha=0.45, hatch="....", edgecolor="white", label="its twin w_n")],
              frameon=False, fontsize=7, loc="best")
    ax.set_title("What steering does to held-out pairs")
    ax.set_ylabel("Δ log p (nats per sentence)")
    _clean(ax)

    fig.suptitle(f"P1, the dissociation (block {layer}, all at KL = {KL_BUDGET})", y=1.02, fontweight="bold")
    fig.tight_layout()
    return _png(fig, path)


def fig_2x2(pts, layers, path):
    fig, axes = plt.subplots(1, len(layers), figsize=(3.9 * len(layers), 3.1), squeeze=False)
    cells = [[("dim", None), ("dpo", "dpo_nm")], [("mop", None), ("eq5", "eq5_nm")]]
    allv = pts[pts.method.isin(ORDER)].groupby(["layer", "method"]).effect.mean()
    vmax = max(float(np.nanmax(np.abs(allv.values))) if len(allv) else 0.1, 1e-3)
    for ax, layer in zip(axes[0], layers):
        d = pts[pts.layer == layer].groupby("method").effect.agg(["mean", "std"])
        grid = np.array([[d["mean"].get(m, np.nan) for m, _ in row] for row in cells])
        ax.imshow(grid, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
        for i, row in enumerate(cells):
            for j, (m, nm) in enumerate(row):
                mu, sd = d["mean"].get(m, np.nan), d["std"].get(m, np.nan)
                txt = f"{LABELS[m]}\n{mu:+.3f}" + (f" ± {sd:.3f}" if not np.isnan(sd) else "")
                if nm and nm in d.index:
                    txt += f"\nNM: {d['mean'][nm]:+.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks([0, 1], ["closed-form", "optimized"])
        ax.set_yticks([0, 1], ["contrastive", "positives only"])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Block {layer}")
    fig.suptitle("P2: steering effect at the budget, by estimator type", y=1.03, fontweight="bold")
    fig.tight_layout()
    return _png(fig, path)


def fig_cost(pts, layer, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    d = pts[pts.layer == layer].copy()
    d["passes"] = d.fwd.fillna(0) + d.bwd.fillna(0)
    g = d.groupby("method").agg(passes=("passes", "mean"), eff=("effect", "mean"), sd=("effect", "std"))
    for m in [m for m in ORDER if m in g.index and m != "random"]:
        r = g.loc[m]
        ax.errorbar(max(r.passes, 1), r.eff, yerr=np.nan_to_num(r.sd), fmt="s" if m.endswith("_nm") else "o",
                    color=COLORS[m], ms=7, mec="white", mew=1.5, lw=1)
        ax.annotate(LABELS[m], (max(r.passes, 1), r.eff), xytext=(7, 4), textcoords="offset points",
                    fontsize=7, color=INK)
    if "random" in g.index:
        ax.axhline(g.loc["random"].eff, color=COLORS["random"], ls="--", lw=0.8)
        ax.text(0.01, g.loc["random"].eff, " random direction", transform=ax.get_yaxis_transform(),
                va="bottom", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("sequences processed to estimate v (forward + backward, incl. tuning)")
    ax.set_ylabel("Δ P(target) at the budget")
    ax.set_title(f"Effect vs. estimation cost (block {layer})")
    _clean(ax)
    return _png(fig, path)


def fig_training(results_dir, layer, lrs, path):
    ldir = os.path.join(results_dir, "logs")
    methods = [m for m in ["eq5", "dpo"] if os.path.isdir(ldir)
               and any(f.startswith(f"{m}_") and f.endswith(f"_{layer}.jsonl") for f in os.listdir(ldir))]
    if not methods:
        return None
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 3.2), sharey=False, squeeze=False)
    for ax, m in zip(axes[0], methods):
        for f in sorted(os.listdir(ldir)):
            parts = f.removesuffix(".jsonl").split("_")
            if parts[0] != m or len(parts) != 3 or int(parts[2]) != layer:
                continue
            fold = int(parts[1])
            rows = pd.DataFrame(analyze._read_jsonl(os.path.join(ldir, f)))
            lr = lrs.get((fold, layer, m))
            if lr is not None and "lr" in rows:
                rows = rows[np.isclose(rows.lr, lr)]
            ax.plot(rows.step, rows.dlogp_pos, color=COLORS[m], lw=1.4, alpha=0.9,
                    label="w_p (target)" if fold == 0 else None)
            ax.plot(rows.step, rows.dlogp_neg, color=COLORS[m], lw=1.4, ls="--", alpha=0.9,
                    label="w_n (twin)" if fold == 0 else None)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{LABELS[m]} during training (block {layer}, one line per fold)")
        ax.set_xlabel("training step (chosen learning rate)")
        ax.set_ylabel("val Δ log p vs. unsteered")
        ax.legend(frameon=False, fontsize=7)
        _clean(ax)
    fig.tight_layout()
    return _png(fig, path)


def fig_cosines(cos, layers, path):
    if cos is None or cos.empty:
        return None
    fig, axes = plt.subplots(1, len(layers), figsize=(4.4 * len(layers), 3.8), squeeze=False)
    for ax, layer in zip(axes[0], layers):
        c = cos[cos.layer == layer]
        ms = [m for m in ORDER if m in set(c.a) | set(c.b)]
        mat = np.eye(len(ms))
        for _, r in c.iterrows():
            i, j = ms.index(r.a), ms.index(r.b)
            mat[i, j] = mat[j, i] = r.cos_mean
        ax.imshow(mat, cmap=DIVERGING, vmin=-1, vmax=1)
        for i in range(len(ms)):
            for j in range(len(ms)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(ms)), [LABELS[m] for m in ms], rotation=35, ha="right")
        ax.set_yticks(range(len(ms)), [LABELS[m] for m in ms])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Block {layer}")
    fig.suptitle("Cosine similarity between the estimated vectors (mean over folds)", y=1.02, fontweight="bold")
    fig.tight_layout()
    return _png(fig, path)


# ---------------------------------------------------------------- HTML

CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--ok:#0b7a3b;--mid:#9a6700;--bad:#b42318}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px}
.sub{color:var(--ink2);margin:0 0 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.s{font-weight:600;white-space:nowrap}.holds{color:var(--ok)}.mixed{color:var(--mid)}.fails{color:var(--bad)}.na{color:var(--ink2)}
img{max-width:100%;height:auto;display:block;margin:8px 0}.read{color:var(--ink2);font-size:14px;margin:4px 0 0}
.warn{border-left:4px solid var(--mid)}.warn li{margin:4px 0}.scroll{overflow-x:auto}pre{white-space:pre-wrap;font-size:13px}
details summary{cursor:pointer;font-weight:600}
"""
ICON = {"holds": "✔ holds", "mixed": "≈ mixed", "fails": "✘ fails", "n/a": "– n/a"}


def _verdict_table(rows):
    body = []
    for r in rows:
        folds = "" if r["status"] == "n/a" else f" ({r['k']}/{r['n']} folds)"
        cls = r["status"].replace("/", "")
        body.append(f"<tr><td>{r['prediction']}</td><td>{html.escape(r['check'])}</td>"
                    f"<td class='s {cls}'>{ICON[r['status']]}{folds}</td><td>{html.escape(r['detail'])}</td></tr>")
    return ("<div class='scroll'><table><tr><th>Prediction</th><th>Check</th><th>Result</th><th>Numbers</th></tr>"
            f"{''.join(body)}</table></div>")


def _summary_table(summ, layer):
    s = summ[summ.layer == layer].set_index("method")
    cols = [("effect", "Δ P(target)", 3), ("dlogp_pos", "Δ log p(w_p)", 2), ("dlogp_neg", "Δ log p(w_n)", 2),
            ("margin", "margin", 2), ("distinct2", "distinct-2", 2), ("alpha", "α at budget", 3)]
    head = "".join(f"<th>{h}</th>" for _, h, _ in cols) + "<th>fwd</th><th>bwd</th><th>seconds</th>"
    rows = []
    for m in [m for m in ORDER if m in s.index]:
        r, tds = s.loc[m], []
        for c, _, p in cols:
            mu, sd = r.get(f"{c}_mean", np.nan), r.get(f"{c}_std", np.nan)
            tds.append("–" if pd.isna(mu) else f"{mu:.{p}f}" + ("" if pd.isna(sd) else f" ± {sd:.{p}f}"))
        for c in ["fwd", "bwd", "seconds"]:
            v = r.get(f"{c}_mean", np.nan)
            tds.append("–" if pd.isna(v) else (f"{v:.1f}" if c == "seconds" else f"{v:,.0f}"))
        rows.append(f"<tr><td>{LABELS[m]}</td>" + "".join(f"<td class='num'>{t}</td>" for t in tds) + "</tr>")
    return f"<div class='scroll'><table><tr><th>Method</th>{head}</tr>{''.join(rows)}</table></div>"


def _fig(b64, alt, reading):
    if b64 is None:
        return ""
    return (f"<div class='card'><img alt='{html.escape(alt)}' src='data:image/png;base64,{b64}'>"
            f"<p class='read'><b>How to read it.</b> {reading}</p></div>")


def build(results_dir="results", budget=KL_BUDGET):
    df, pts, summ, meta, warns = load(results_dir, budget)
    layers = sorted({int(x) for x in df.layer if x >= 0}, key=lambda l: (l != HEADLINE_LAYER, l))
    head = layers[0]
    figs = os.path.join(results_dir, "figs")
    os.makedirs(figs, exist_ok=True)
    cos = analyze.cosines(results_dir) if os.path.isdir(os.path.join(results_dir, "vectors")) else None
    base = df[df.kind == "baseline"]

    img = {
        "curves": fig_curves(df, layers, budget, os.path.join(figs, "curves.png")),
        "p1": fig_p1(pts, head, os.path.join(figs, "p1_dissociation.png")),
        "p2": fig_2x2(pts, layers, os.path.join(figs, "p2_2x2.png")),
        "cost": fig_cost(pts, head, os.path.join(figs, "cost.png")),
        "train": fig_training(results_dir, head, _lrs(results_dir), os.path.join(figs, "training.png")),
        "cos": fig_cosines(cos, layers, os.path.join(figs, "cosines.png")),
    }
    ex_path = os.path.join(results_dir, "examples.md")
    if not os.path.exists(ex_path):
        analyze.examples(results_dir, layer=head, budget=budget)
    examples = open(ex_path).read() if os.path.exists(ex_path) else ""

    notes = health(df, pts, warns)
    folds = sorted(df.fold.unique())
    target = meta.get("TARGET", "?")
    unsteered = f"{base.p_target.mean():.3f}" if len(base) else "?"
    by_layer = "".join(
        f"<h3>Block {layer}{' (headline)' if layer == HEADLINE_LAYER else ''}</h3>{_verdict_table(verdicts(pts, layer))}"
        for layer in layers)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Steering Results</title>
<style>{CSS}</style></head><body><main>
<h1>Steering-vector results</h1>
<p class="sub">{html.escape(str(meta.get('MODEL_ID', '')))} · steering toward <b>{html.escape(str(target))}</b> ·
folds {', '.join(map(str, folds))} · blocks {', '.join(map(str, layers))} · budget KL = {budget} nats/token ·
unsteered P(target) = {unsteered}</p>

<div class="card {'warn' if notes else ''}"><b>{'Check these before reading the results' if notes
    else 'No health problems detected'}</b>{'<ul>' + ''.join(f'<li>{html.escape(n)}</li>' for n in notes) + '</ul>'
    if notes else '<p class="read">Every curve reached the budget and nothing was flagged as degenerate.</p>'}</div>

<h2>Verdicts</h2>
<div class="card">
<p class="read">Each check compares two methods at KL = {budget}, fold by fold. <b>Holds</b> means it's true in every
fold, <b>mixed</b> in most, <b>fails</b> in at most half. With 3 folds these are consistency checks, not significance
tests; read them next to the figures. The claim is supported if the P1 and P2 checks mostly hold at the headline
block.</p>
{by_layer}</div>

<h2>1. Effect vs. side effect</h2>
{_fig(img['curves'], 'Effect vs KL curves', 'Higher is better at any given KL. Compare methods along the dashed '
      'line (the budget). Bands are ±1 std across folds. Dashed curves are the norm-matched (NM) retrains: the '
      'gradient methods retrained at the norm where their first curve hit the budget, then swept like the rest. '
      'Hollow circles are the unconstrained gradient vectors at the norm they trained to, usually far off to the '
      'right. A curve that rises steeply buys a lot of behavior per unit of damage.')}

<h2>2. P1: the dissociation</h2>
{_fig(img['p1'], 'P1 dissociation', 'P1 predicts: in the left panel DiM is above Eq. 5; in the middle panel Eq. 5 '
      'has the tallest solid bar (it raises the target sentences most) but its hatched bar is nearly as tall (it '
      'raises their twins too, so it has learned shared content, not sentiment). Dots are individual folds. '
      'Hatched bars with diagonal lines are the norm-matched (NM) retrains.')}

<h2>3. P2: which ingredient matters</h2>
{_fig(img['p2'], 'P2 two-by-two', 'Blue is more steering toward the target, orange is steering the wrong way. '
      'If contrast is the ingredient, the top row (contrastive) is blue and the bottom row is pale or orange. If '
      'closed-form estimation is the ingredient, the left column is blue instead. NM gives the norm-matched '
      'retrain of the optimized methods, also read off at KL = 0.1.')}

<h2>4. Cost</h2>
{_fig(img['cost'], 'Effect vs cost', 'Up and to the left is better: more steering for less computation. The x-axis '
      'is logarithmic; closed-form methods need one forward pass per training sentence and no backward passes.')}

<h2>5. How the gradient methods trained</h2>
{_fig(img['train'], 'Training dynamics', 'Solid lines are the target sentences, dashed lines their minimal-pair '
      'twins. For Eq. 5, dashed lines rising alongside solid ones means it is learning what the pair shares. For '
      'DPO-grad, a solid line falling below zero means it is meeting its objective by suppressing the twins '
      'rather than promoting the target.')}

<h2>6. Do the vectors point the same way?</h2>
{_fig(img['cos'], 'Cosine similarity', 'Blue cells are similar directions, orange are opposed, gray is unrelated. '
      'The claim predicts DPO-grad is closer to DiM than to Eq. 5. The random direction should be near 0 with '
      'everything.')}

<h2>All numbers at the budget</h2>
<div class="card">{''.join(f'<h3>Block {layer}</h3>' + _summary_table(summ, layer) for layer in layers)}
<p class="read">Mean ± std across folds, interpolated at KL = {budget}. NM rows are the norm-matched retrains,
swept and interpolated at the budget like every other method; their costs include the original training.</p></div>

<h2>Example continuations</h2>
<div class="card"><details><summary>Show examples (fold 0, headline block)</summary>
<pre>{html.escape(examples)}</pre></details></div>
</main></body></html>"""
    out = os.path.join(results_dir, "report.html")
    with open(out, "w") as f:
        f.write(doc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    print(f"wrote {build(a.results)}")


if __name__ == "__main__":
    main()
