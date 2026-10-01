"""Generation, sentiment classifier, KL on WikiText, pair diagnostics. See implementation_plan.md (T5).

Every metric takes the vector as applied (alpha * r_l * v_hat), or None for the unsteered baseline.
The steerer is expected to be entered (hook registered) by the caller.
"""
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from steer import config
from steer.config import CLF_ID, GEN_TOKENS, SAMPLES_PER_PROMPT, TEMPERATURE
from steer.data import leak_words
from steer.model import encode, seq_logprobs, steer_mask


@torch.no_grad()
def generate(model, tok, steerer, v_applied, prompts: list[str], seed: int, batch_size=96) -> list[str]:
    """Sampled continuations, aligned with prompts repeated SAMPLES_PER_PROMPT times each."""
    expanded = [p for p in prompts for _ in range(SAMPLES_PER_PROMPT)]
    out = []
    torch.manual_seed(seed)
    try:
        for i in range(0, len(expanded), batch_size):
            batch = encode(tok, expanded[i : i + batch_size], side="left", device=model.device)
            steerer.set(v_applied, steer_mask(batch["attention_mask"]))
            ids = model.generate(
                batch["input_ids"], attention_mask=batch["attention_mask"], do_sample=True,
                temperature=TEMPERATURE, top_p=1.0, top_k=0, max_new_tokens=GEN_TOKENS,
                pad_token_id=tok.pad_token_id,
            )
            new = ids[:, batch["input_ids"].shape[1] :]
            out += tok.batch_decode(new, skip_special_tokens=True)
    finally:
        steerer.set(None)
    return out


class SentimentClassifier:
    def __init__(self, device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(CLF_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(CLF_ID).to(device).eval()

    @torch.no_grad()
    def p_target(self, texts: list[str], batch_size=64) -> np.ndarray:
        """P(TARGET sentiment) per text; label 0 = NEGATIVE, 1 = POSITIVE. Reads config.TARGET at call time."""
        col = 1 if config.TARGET == "pos" else 0
        probs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i : i + batch_size], truncation=True, max_length=128,
                           padding=True, return_tensors="pt").to(self.device)
            probs.append(self.model(**enc).logits.softmax(-1)[:, col].float().cpu())
        return torch.cat(probs).numpy() if probs else np.zeros(0)


@torch.no_grad()
def kl_on_wiki(model, tok, steerer, v_applied, wiki_texts, batch_size=50) -> float:
    """Per-token KL(steered || base), averaged over every real position except BOS."""
    total, count = 0.0, 0.0
    for i in range(0, len(wiki_texts), batch_size):
        batch = encode(tok, wiki_texts[i : i + batch_size], device=model.device)
        mask = steer_mask(batch["attention_mask"])
        base = model(**batch).logits.log_softmax(-1)
        steerer.set(v_applied, mask)
        try:
            steered = model(**batch).logits.log_softmax(-1)
        finally:
            steerer.set(None)
        kl = (steered.exp() * (steered - base)).sum(-1)
        del base, steered
        total += (kl * mask).sum().item()
        count += mask.sum().item()
    return total / count


@torch.no_grad()
def pair_logprobs(model, tok, steerer, v_applied, texts: list[str], batch_size=64) -> torch.Tensor:
    """(N,) sequence log-probs on CPU, steered if v_applied is not None."""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = encode(tok, texts[i : i + batch_size], device=model.device)
        if v_applied is None:
            out.append(seq_logprobs(model, batch).cpu())
        else:
            out.append(seq_logprobs(model, batch, steerer, v_applied).cpu())
    return torch.cat(out)


def pair_diagnostics(model, tok, steerer, v_applied, test_pairs, base) -> dict:
    """Mean steered-minus-unsteered log-probs of the pos and neg test sentences. base: {"pos", "neg"}."""
    d_pos = pair_logprobs(model, tok, steerer, v_applied, [p.pos for p in test_pairs]) - base["pos"]
    d_neg = pair_logprobs(model, tok, steerer, v_applied, [p.neg for p in test_pairs]) - base["neg"]
    return {"dlogp_pos": d_pos.mean().item(), "dlogp_neg": d_neg.mean().item(),
            "margin": (d_pos - d_neg).mean().item()}


def distinct2(texts) -> float:
    """Mean over texts (>= 2 tokens) of unique / total bigrams, on lowercased whitespace tokens."""
    scores = []
    for t in texts:
        w = t.lower().split()
        if len(w) >= 2:
            bigrams = list(zip(w, w[1:]))
            scores.append(len(set(bigrams)) / len(bigrams))
    return float(np.mean(scores)) if scores else 0.0


def leak_rate(texts, words) -> float:
    """Fraction of texts mentioning any of words as a whole word (optional trailing s), case-insensitive."""
    if not texts:
        return 0.0
    pat = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")s?\b", re.IGNORECASE)
    return sum(bool(pat.search(t)) for t in texts) / len(texts)


def evaluate_point(model, tok, clf, steerer, v_applied, fold, prompts, wiki, base, seed) -> tuple[dict, list[str]]:
    gens = generate(model, tok, steerer, v_applied, prompts, seed)
    if v_applied is None:
        kl, diag = 0.0, {"dlogp_pos": 0.0, "dlogp_neg": 0.0, "margin": 0.0}
    else:
        kl = kl_on_wiki(model, tok, steerer, v_applied, wiki)
        diag = pair_diagnostics(model, tok, steerer, v_applied, fold.test, base)
    metrics = {
        "kl": kl,
        "p_target": float(clf.p_target(gens).mean()),
        "distinct2": distinct2(gens),
        "leak": leak_rate(gens, leak_words(fold.train_topics)),
        **diag,
        "n_gen": len(gens),
        "n_empty": sum(not g.strip() for g in gens),
    }
    return metrics, gens
