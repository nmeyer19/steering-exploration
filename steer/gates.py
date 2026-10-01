"""Pre-run gates 7-11 (T6), printed as PASS/FAIL with the numbers. `python -m steer.gates [--fold 0] [--layer 15]`."""
import argparse

import numpy as np

from steer import config
from steer.closed_form import dim, random_vec
from steer.config import HEADLINE_LAYER, KL_BUDGET, N_FOLDS, gen_seed
from steer.data import make_folds, wiki_texts
from steer.metrics import SentimentClassifier, generate, kl_on_wiki
from steer.model import Steerer, load_model, residual_norm

ALPHA_02_IDX = config.ALPHAS.index(0.2)  # DiM at alpha = 0.2 uses the grid's seed for that alpha
POSITIVE = ["This was the best meal I have ever had, absolutely wonderful.", "I loved every minute of it.",
            "What a fantastic, delightful experience. Highly recommended!", "The staff were kind and it was great."]
NEGATIVE = ["This was the worst meal I have ever had, absolutely awful.", "I hated every minute of it.",
            "What a terrible, miserable experience. Avoid it!", "The staff were rude and it was horrible."]


def report(n: int, name: str, ok: bool, msg: str):
    print(f"[{'PASS' if ok else 'FAIL'}] gate {n} ({name}): {msg}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    ap.add_argument("--n-wiki", type=int, default=config.N_WIKI)
    a = ap.parse_args(argv)

    device = config.get_device()
    model, tok = load_model(device)
    clf = SentimentClassifier(device)
    folds = make_folds(config.TARGET)
    fold = folds[a.fold]
    wiki = wiki_texts(tok, n=a.n_wiki)
    r_l = residual_norm(model, tok, wiki, a.layer)

    with Steerer(model, a.layer) as st:
        # 7. headroom: baseline P(positive) over every fold's prompts, independent of config.TARGET
        base_gens = {}
        for f in folds[:N_FOLDS]:
            base_gens[f.idx] = generate(model, tok, st, None, [p for _, p in f.prompts], gen_seed(f.idx, 0))
        p_tgt = np.concatenate([clf.p_target(g) for g in base_gens.values()])
        p_pos = float(np.mean(p_tgt if config.TARGET == "pos" else 1 - p_tgt))  # binary softmax
        rec = "neg" if p_pos > 0.5 else "pos"
        report(7, "headroom", rec == config.TARGET,
               f"baseline P(positive) = {p_pos:.3f}; recommended TARGET = {rec!r}, config.TARGET = {config.TARGET!r}")

        # 8. the budget is reachable for DiM and random
        v_dim, v_rand = dim(model, tok, fold, a.layer).v, random_vec(model.config.hidden_size, a.layer, fold).v
        for name, v in [("dim", v_dim), ("random", v_rand)]:
            v_hat = (v / v.norm()).to(device)
            lo, hi = (kl_on_wiki(model, tok, st, al * r_l * v_hat, wiki) for al in (0.025, 0.8))
            report(8, f"budget reachable, {name}", hi >= KL_BUDGET and lo < KL_BUDGET,
                   f"KL(a=0.025) = {lo:.4f} < {KL_BUDGET} and KL(a=0.8) = {hi:.4f} >= {KL_BUDGET}")

        # 9. sign: DiM at alpha = 0.2 raises P(target) on this fold
        prompts = [p for _, p in fold.prompts]
        steered = generate(model, tok, st, (0.2 * r_l * v_dim / v_dim.norm()).to(device), prompts,
                           gen_seed(fold.idx, ALPHA_02_IDX))
        p0, p1 = float(clf.p_target(base_gens[fold.idx]).mean()), float(clf.p_target(steered).mean())
        report(9, "sign", p1 > p0, f"P(target) unsteered {p0:.3f} -> DiM a=0.2 {p1:.3f} (TARGET={config.TARGET!r})")

    # 10. continuations to read by eye
    print("\ngate 10 (read these): 10 continuations, unsteered vs DiM a=0.2")
    idx = np.linspace(0, len(steered) - 1, 10).astype(int)
    for i in idx:
        p = prompts[i // (len(steered) // len(prompts))]
        print(f"  PROMPT  {p!r}\n    none: {base_gens[fold.idx][i]!r}\n    dim : {steered[i]!r}")

    # 11. classifier sanity on obvious sentences
    tgt, opp = (POSITIVE, NEGATIVE) if config.TARGET == "pos" else (NEGATIVE, POSITIVE)
    pt, po = clf.p_target(tgt), clf.p_target(opp)
    report(11, "classifier", bool((pt > 0.5).all() and (po < 0.5).all()),
           f"P(target) on target-side {np.round(pt, 3).tolist()}, on opposite-side {np.round(po, 3).tolist()}")


if __name__ == "__main__":
    main()
