"""T5 tests: pure metric functions, then model-dependent invariants on a tiny workload."""
import numpy as np
import pytest
import torch

from steer import config
from steer.config import SAMPLES_PER_PROMPT
from steer.metrics import distinct2, leak_rate

# ---- pure functions ----


def test_distinct2():
    assert distinct2(["a b c d"]) == 1.0
    assert distinct2(["a b a b"]) == pytest.approx(2 / 3)  # bigrams: ab, ba, ab
    assert distinct2(["A b a B"]) == pytest.approx(2 / 3)  # lowercased
    assert distinct2(["x", "", "a b c d", "a a a"]) == pytest.approx((1.0 + 0.5) / 2)  # short texts skipped
    assert distinct2(["", "one"]) == 0.0


def test_leak_rate():
    words = ["pizza", "car"]
    texts = ["I ate PIZZA.", "Two cars here", "carpet cleaning", "scar tissue", "nothing", "pizzas, cars!"]
    assert leak_rate(texts, words) == pytest.approx(3 / 6)
    assert leak_rate([], words) == 0.0
    assert leak_rate(["the cafe", "a café"], ["cafe"]) == 0.5


# ---- model-dependent ----

WIKI = [
    "The river flows through the valley and into the sea, passing several small towns along the way.",
    "In 1905 the company built a new factory near the railway station.",
    "Robert Boulter is an English film, television and theatre actor.",
    "The album was released in March and reached number three on the national chart.",
]
PROMPTS = ["Yesterday I went to see the movie, and", "So here's the thing about the hotel:"]


@pytest.fixture(scope="module")
def env():
    from steer.metrics import SentimentClassifier
    from steer.model import Steerer, load_model

    model, tok = load_model()
    d = model.config.hidden_size
    v = torch.randn(d, generator=torch.Generator().manual_seed(0))
    with Steerer(model, config.HEADLINE_LAYER) as steerer:
        yield {"model": model, "tok": tok, "steerer": steerer, "clf": SentimentClassifier(model.device),
               "v_hat": (v / v.norm()).to(model.device), "r": 180.0}


def test_generate_length_and_determinism(env):
    from steer.metrics import generate

    args = (env["model"], env["tok"], env["steerer"])
    a = generate(*args, None, PROMPTS, seed=7)
    b = generate(*args, None, PROMPTS, seed=7)
    c = generate(*args, None, PROMPTS, seed=8)
    assert len(a) == len(PROMPTS) * SAMPLES_PER_PROMPT
    assert a == b and a != c
    assert not any(PROMPTS[0] in g for g in a)  # continuations only
    assert env["steerer"].v is None
    # chunking gives the same count and is itself deterministic
    d1 = generate(*args, None, PROMPTS, seed=7, batch_size=2)
    assert len(d1) == len(a) and d1 == generate(*args, None, PROMPTS, seed=7, batch_size=2)


def test_generate_steered_differs(env):
    from steer.metrics import generate

    args = (env["model"], env["tok"], env["steerer"])
    v = 0.8 * env["r"] * env["v_hat"]
    assert generate(*args, v, PROMPTS, seed=7) != generate(*args, None, PROMPTS, seed=7)


def test_classifier(env):
    clf = env["clf"]
    texts = ["This was a wonderful, delightful experience.", "This was awful and I hated every minute."]
    old = config.TARGET
    try:
        config.TARGET = "pos"
        p_pos = clf.p_target(texts)
        config.TARGET = "neg"
        p_neg = clf.p_target(texts)
    finally:
        config.TARGET = old
    assert p_pos[0] > 0.9 and p_pos[1] < 0.1
    np.testing.assert_allclose(p_pos + p_neg, 1.0, atol=1e-5)
    assert clf.p_target([""]).shape == (1,)


def test_kl(env):
    from steer.metrics import kl_on_wiki

    args = (env["model"], env["tok"], env["steerer"])
    assert kl_on_wiki(*args, torch.zeros_like(env["v_hat"]), WIKI) == pytest.approx(0.0, abs=1e-6)
    lo = kl_on_wiki(*args, 0.05 * env["r"] * env["v_hat"], WIKI)
    hi = kl_on_wiki(*args, 0.4 * env["r"] * env["v_hat"], WIKI)
    assert 0 < lo < hi
    # batch weighting: result independent of batch size
    assert kl_on_wiki(*args, 0.4 * env["r"] * env["v_hat"], WIKI, batch_size=3) == pytest.approx(hi, rel=1e-4)
    assert env["steerer"].v is None


def test_evaluate_point(env):
    from steer.data import make_folds
    from steer.metrics import evaluate_point, pair_diagnostics, pair_logprobs

    m, tok, st = env["model"], env["tok"], env["steerer"]
    fold = make_folds(config.TARGET)[0]
    fold.test = fold.test[:8]
    base = {k: pair_logprobs(m, tok, st, None, [getattr(p, k) for p in fold.test]) for k in ("pos", "neg")}

    zero = pair_diagnostics(m, tok, st, torch.zeros_like(env["v_hat"]), fold.test, base)
    assert all(abs(x) < 1e-4 for x in zero.values())

    res, gens = evaluate_point(m, tok, env["clf"], st, None, fold, PROMPTS, WIKI, base, seed=0)
    assert res["kl"] == res["dlogp_pos"] == res["dlogp_neg"] == res["margin"] == 0
    assert res["n_gen"] == len(gens) == len(PROMPTS) * SAMPLES_PER_PROMPT
    assert set(res) == {"kl", "p_target", "distinct2", "leak", "dlogp_pos", "dlogp_neg", "margin", "n_gen", "n_empty"}

    v = 0.2 * env["r"] * env["v_hat"]
    res2, _ = evaluate_point(m, tok, env["clf"], st, v, fold, PROMPTS, WIKI, base, seed=0)
    assert res2["kl"] > 0
    assert res2["margin"] == pytest.approx(res2["dlogp_pos"] - res2["dlogp_neg"], abs=1e-4)
    print("\nbaseline", res, "\nrandom a=0.2", res2)
