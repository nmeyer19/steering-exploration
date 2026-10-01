"""Gates 1-7 for steer/model.py. See implementation_plan.md (T1)."""
import pytest
import torch

from steer.model import Steerer, encode, load_model, read_acts, residual_norm, seq_logprobs, steer_mask

LAYER = 15
TEXTS = ["The movie was surprisingly good.", "I hated every minute of that long and boring dinner."]


@pytest.fixture(scope="module")
def mt():
    return load_model()


def _dev(model):
    return next(model.parameters()).device


def _rand_v(model, scale=50.0):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(model.config.hidden_size, generator=g) * scale).to(_dev(model))


def test_steer_mask_left_and_right():
    am = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 1]])
    assert steer_mask(am).tolist() == [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]


def test_encode(mt):
    model, tok = mt
    b = encode(tok, TEXTS, "left", _dev(model))
    first = (b["attention_mask"].cumsum(1) == 1) & b["attention_mask"].bool()
    assert (b["input_ids"][first] == tok.bos_token_id).all()
    assert b["attention_mask"][:, -1].all()  # left padding: every row ends with a real token


def test_1_zero_vector_is_noop(mt):
    model, tok = mt
    b = encode(tok, TEXTS, device=_dev(model))
    with torch.no_grad():
        base = model(**b).logits
        with Steerer(model, LAYER) as s:
            s.set(torch.zeros(model.config.hidden_size, device=_dev(model)), steer_mask(b["attention_mask"]))
            steered = model(**b).logits
    assert torch.equal(base, steered)


def _capture_inputs(model, b, v=None, mask=None):
    """Inputs to layers[LAYER] and layers[LAYER + 1] (pre-hooks), optionally steering at LAYER."""
    got = {}
    hooks = [model.model.layers[i].register_forward_pre_hook(
        lambda m, args, kw, i=i: got.__setitem__(i, (args[0] if args else kw["hidden_states"]).clone()),
        with_kwargs=True) for i in (LAYER, LAYER + 1)]
    try:
        with torch.no_grad(), Steerer(model, LAYER) as s:
            s.set(v, mask)
            model(**b)
    finally:
        for h in hooks:
            h.remove()
    return got[LAYER], got[LAYER + 1]


def test_2_right_layer(mt):
    model, tok = mt
    b = encode(tok, TEXTS, device=_dev(model))
    m, v = steer_mask(b["attention_mask"]), _rand_v(model)
    in_l0, in_n0 = _capture_inputs(model, b)
    in_l1, in_n1 = _capture_inputs(model, b, v, m)
    assert torch.equal(in_l0, in_l1)
    torch.testing.assert_close(in_n1 - in_n0, m[..., None] * v, rtol=0, atol=1e-4)


@pytest.mark.parametrize("side", ["right", "left"])
def test_3_bos_untouched(mt, side):
    model, tok = mt
    b = encode(tok, TEXTS, side, _dev(model))
    am = b["attention_mask"]
    first = (am.cumsum(1) == 1) & am.bool()
    _, out0 = _capture_inputs(model, b)
    _, out1 = _capture_inputs(model, b, _rand_v(model), steer_mask(am))
    assert torch.equal(out0[first], out1[first])
    assert not torch.allclose(out0[am.bool() & ~first], out1[am.bool() & ~first])


def test_4_decode_steps_steered(mt):
    model, tok = mt
    b = encode(tok, ["I think that", "The weather today is"], "left", _dev(model))
    kw = dict(do_sample=False, max_new_tokens=12, pad_token_id=tok.pad_token_id)
    calls = []
    with torch.no_grad(), Steerer(model, LAYER) as s:
        base = model.generate(**b, **kw)
        h = model.model.layers[LAYER].register_forward_hook(lambda m, i, o: calls.append(
            (o[0] if isinstance(o, tuple) else o).shape[1]))
        s.set(_rand_v(model, 200.0), steer_mask(b["attention_mask"]))
        steered = model.generate(**b, **kw)
        h.remove()
    T = b["input_ids"].shape[1]
    assert calls[0] == T and all(c == 1 for c in calls[1:])  # prefill, then KV-cached decode steps
    assert not torch.equal(base[:, T:], steered[:, T:])


def test_5_logprobs_match_hf_loss(mt):
    model, tok = mt
    b = encode(tok, TEXTS[:1], device=_dev(model))
    ids, T = b["input_ids"], b["input_ids"].shape[1]
    with torch.no_grad():
        lp = seq_logprobs(model, b)
        ref = -model(ids, labels=ids).loss * (T - 1)
    assert abs(lp.item() - ref.item()) < 1e-4


def test_6_gradients_flow(mt):
    model, tok = mt
    r = residual_norm(model, tok, TEXTS, LAYER)
    u = torch.nn.Parameter(torch.zeros(model.config.hidden_size, device=_dev(model)))
    b = encode(tok, TEXTS, device=_dev(model))
    with Steerer(model, LAYER) as s:
        (-seq_logprobs(model, b, s, r * u).mean()).backward()
    assert u.grad is not None and u.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("steer", [False, True])
def test_7_padding_invariance(mt, steer):
    model, tok = mt
    short, long_ = TEXTS
    v = _rand_v(model) if steer else None
    with torch.no_grad(), Steerer(model, LAYER) as s:
        alone = seq_logprobs(model, encode(tok, [short], device=_dev(model)), s, v)
        padded = seq_logprobs(model, encode(tok, [short, long_], device=_dev(model)), s, v)
    assert abs(alone[0].item() - padded[0].item()) < 1e-4


def test_read_acts_and_residual_norm(mt):
    model, tok = mt
    last = read_acts(model, tok, TEXTS, LAYER, "last", batch_size=1)
    allb = read_acts(model, tok, TEXTS, LAYER, "all_but_first")
    n_real = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in TEXTS)
    assert last.shape == (2, model.config.hidden_size) and last.device.type == "cpu"
    assert allb.shape == (n_real, model.config.hidden_size)
    r = residual_norm(model, tok, TEXTS, LAYER)
    assert 10 < r < 2000  # BOS (~20,000) is excluded
