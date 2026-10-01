"""The evaluation grid (T6). `python -m steer.run [--folds 0 1 2] [--layers 15 8] [--smoke] [--results DIR]`.

Resumable: a point is keyed on (fold, layer, method, alpha, kind) and skipped if already in grid.jsonl;
training is skipped if the vector file exists. --smoke overrides are passed down as arguments, not by
mutating config.
"""
import argparse
import json
import os
import platform
import random
import time

import numpy as np
import torch
import transformers
from tqdm import tqdm

from steer import config
from steer.analyze import alpha_at_budget
from steer.closed_form import dim, generic_mean, mop, random_vec
from steer.config import ALPHAS, GRADIENT_METHODS, HEADLINE_LAYER, LAYERS, METHODS, N_FOLDS, SEED, SteerVec, gen_seed
from steer.data import leak_words, make_folds, wiki_short_texts, wiki_texts
from steer.gradient import norm_matched, train_vec
from steer.metrics import SentimentClassifier, evaluate_point, pair_logprobs
from steer.model import Steerer, load_model, residual_norm

BASELINE_LAYER = -1  # the unsteered baseline is layer-independent


def point_key(r: dict) -> tuple:
    """Sweep/baseline points are identified by alpha; native and norm-matched ones (one per method) are not,
    since their alpha = ||v|| / r_l carries float noise from r_l."""
    alpha = round(float(r["alpha"]), 6) if r["kind"] in ("sweep", "baseline") else None
    return (int(r["fold"]), int(r["layer"]), r["method"], alpha, r["kind"])


class Grid:
    """grid.jsonl / generations.jsonl, appended and flushed after every point."""

    def __init__(self, out: str):
        self.path, self.gen_path = os.path.join(out, "grid.jsonl"), os.path.join(out, "generations.jsonl")
        self.rows = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.rows = [json.loads(line) for line in f if line.strip()]
        self.done = {point_key(r) for r in self.rows}

    def has(self, **key) -> bool:
        return point_key(key) in self.done

    def add(self, row: dict, prompts: list[str], gens: list[str]):
        with open(self.gen_path, "a") as f:
            key = {k: row[k] for k in ("fold", "layer", "method", "alpha", "kind")}
            f.write(json.dumps(key | {"prompts": [p for p in prompts for _ in range(len(gens) // len(prompts))],
                                      "continuations": gens}) + "\n")
            f.flush()
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        self.rows.append(row)
        self.done.add(point_key(row))


def vec_path(out, fold, layer, method) -> str:
    return os.path.join(out, "vectors", f"{fold}_{layer}_{method}.pt")


def get_vec(out, fold, layer, method, make) -> SteerVec:
    """Load the saved vector if it exists, else build it with make() and save it."""
    path = vec_path(out, fold, layer, method)
    if os.path.exists(path):
        d = torch.load(path, weights_only=False)
        return SteerVec(d["method"], d["layer"], d["fold"], d["v"], d["cost"])
    vec = make()
    torch.save({"v": vec.v.detach().cpu(), "cost": vec.cost, "method": vec.method, "layer": vec.layer,
                "fold": vec.fold}, path)
    return vec


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+")
    ap.add_argument("--layers", type=int, nargs="+")
    ap.add_argument("--smoke", action="store_true", help="1 fold, headline layer, 2 alphas, tiny sets")
    ap.add_argument("--results")
    a = ap.parse_args(argv)
    s = a.smoke
    a.folds = a.folds or ([0] if s else list(range(N_FOLDS)))
    a.layers = a.layers or ([HEADLINE_LAYER] if s else LAYERS)
    a.results = a.results or ("results_smoke" if s else config.RESULTS_DIR)
    a.alphas = [0.1, 0.4] if s else ALPHAS[1:]
    a.n_prompts, a.n_wiki, a.n_wiki_short = (4, 8, 8) if s else (None, config.N_WIKI, config.N_WIKI_SHORT)
    a.max_steps = 40 if s else config.MAX_STEPS
    return a


def main(argv=None):
    a = parse_args(argv)
    out = a.results
    os.makedirs(os.path.join(out, "vectors"), exist_ok=True)
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)
    device = config.get_device()
    t_start = time.time()

    model, tok = load_model(device)
    clf = SentimentClassifier(device)
    folds = make_folds(config.TARGET)
    wiki, wiki_short = wiki_texts(tok, n=a.n_wiki), wiki_short_texts(tok, n=a.n_wiki_short)
    r_l = {l: residual_norm(model, tok, wiki, l) for l in a.layers}
    mu_generic = {l: generic_mean(model, tok, wiki_short, l) for l in a.layers}
    d = model.config.hidden_size

    meta = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    meta |= {"r_l": r_l, "device": device, "folds": a.folds, "layers": a.layers, "alphas": a.alphas,
             "smoke": a.smoke, "n_prompts": a.n_prompts, "n_wiki": a.n_wiki, "n_wiki_short": a.n_wiki_short,
             "max_steps": a.max_steps, "leak_words": {f.idx: leak_words(f.train_topics) for f in folds},
             "versions": {"python": platform.python_version(), "torch": torch.__version__,
                          "transformers": transformers.__version__}}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    grid = Grid(out)
    per_layer = len(METHODS) * len(a.alphas) + 2 * len(GRADIENT_METHODS) + len(GRADIENT_METHODS) * len(a.alphas)
    bar = tqdm(total=len(a.folds) * (1 + len(a.layers) * per_layer), desc="grid")
    train_kw = {"max_steps": a.max_steps, "log_dir": os.path.join(out, "logs")}

    for fi in a.folds:
        fold = folds[fi]
        prompts = [p for _, p in fold.prompts][: a.n_prompts]

        def evaluate(st, v_applied, seed, **key):
            if grid.has(**key):
                return bar.update()
            m, gens = evaluate_point(model, tok, clf, st, v_applied, fold, prompts, wiki, base, seed)
            grid.add(key | m | {"seed": seed}, prompts, gens)
            bar.set_postfix_str(f"{key['layer']} {key['method']} {key['kind']} a={key['alpha']:.3f} kl={m['kl']:.3f}")
            bar.update()

        with Steerer(model, a.layers[0]) as st:
            base = {s: pair_logprobs(model, tok, st, None, [getattr(p, s) for p in fold.test]) for s in ("pos", "neg")}
            evaluate(st, None, gen_seed(fi, 0), fold=fi, layer=BASELINE_LAYER, method="none", alpha=0.0,
                     kind="baseline")
        baseline = next(r for r in grid.rows if point_key(r) == (fi, BASELINE_LAYER, "none", 0.0, "baseline"))

        for layer in a.layers:
            rl = r_l[layer]
            makers = {
                "dim": lambda: dim(model, tok, fold, layer),
                "mop": lambda: mop(model, tok, fold, layer, mu_generic[layer]),
                "random": lambda: random_vec(d, layer, fold),
                **{g: (lambda g=g: train_vec(model, tok, fold, layer, rl, g, **train_kw)) for g in GRADIENT_METHODS},
            }
            vecs = {m: get_vec(out, fi, layer, m, makers[m]) for m in METHODS}

            with Steerer(model, layer) as st:
                for m, vec in vecs.items():
                    v_hat = vec.v / vec.v.norm()
                    for alpha in a.alphas:
                        evaluate(st, (alpha * rl * v_hat).to(device), gen_seed(fi, ALPHAS.index(alpha)),
                                 fold=fi, layer=layer, method=m, alpha=alpha, kind="sweep")
                    if m in GRADIENT_METHODS:
                        evaluate(st, vec.v.to(device), gen_seed(fi, 99), fold=fi, layer=layer, method=m,
                                 alpha=(vec.v.norm() / rl).item(), kind="native")

                # stage 2: retrain each gradient method at the alpha where its own sweep reaches the budget
                for m in GRADIENT_METHODS:
                    sweep = [r for r in grid.rows if (r["fold"], r["layer"], r["method"], r["kind"]) == (fi, layer, m, "sweep")]
                    alpha_star = alpha_at_budget([baseline | {"layer": layer, "method": m, "kind": "sweep"}] + sweep)
                    if alpha_star is None:
                        tqdm.write(f"fold {fi} layer {layer} {m}: KL budget not reached; no norm-matched point")
                        bar.update()
                        continue
                    nm = get_vec(out, fi, layer, m + "_nm",
                                 lambda: norm_matched(model, tok, fold, layer, rl, vecs[m], alpha_star, **train_kw))
                    evaluate(st, nm.v.to(device), gen_seed(fi, 98), fold=fi, layer=layer, method=m,
                             alpha=(nm.v.norm() / rl).item(), kind="norm_matched")
                    # a fixed norm does not fix KL (the retrained direction does more damage per unit norm),
                    # so sweep the retrained direction too and let analyze interpolate it at the budget
                    nm_hat = nm.v / nm.v.norm()
                    for alpha in a.alphas:
                        evaluate(st, (alpha * rl * nm_hat).to(device), gen_seed(fi, ALPHAS.index(alpha)),
                                 fold=fi, layer=layer, method=m + "_nm", alpha=alpha, kind="sweep")
    bar.close()
    print(f"done in {time.time() - t_start:.0f}s; {len(grid.rows)} rows in {grid.path}")


if __name__ == "__main__":
    main()
