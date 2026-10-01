"""Model loading, the steering hook, sequence log-probs and activation reads. See implementation_plan.md (T1)."""
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from steer.config import MODEL_ID, get_device


def load_model(device=None):
    """fp32 SmolLM2 in eval mode with frozen weights, plus its tokenizer (pad = eos)."""
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
    model.requires_grad_(False)
    return model, tok


def sync(device) -> None:
    """Wait for queued kernels so wall-clock timings are honest."""
    device = str(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def encode(tok, texts: list[str], side: str = "right", device=None) -> dict[str, Tensor]:
    """BOS + tokens (no other specials), padded on `side`; attention_mask built by hand."""
    assert side in ("right", "left")
    seqs = [[tok.bos_token_id] + tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        sl = slice(0, len(s)) if side == "right" else slice(T - len(s), T)
        ids[i, sl] = torch.tensor(s)
        mask[i, sl] = 1
    device = device or get_device()
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}


def steer_mask(attention_mask: Tensor) -> Tensor:
    """(B, T) float: 1 at real tokens except each row's first (BOS); 0 at pads and BOS."""
    return ((attention_mask.cumsum(1) > 1) & attention_mask.bool()).float()


class Steerer:
    """Context manager that adds `v` (masked) to the output of model.model.layers[layer]."""

    def __init__(self, model, layer: int):
        self.block = model.model.layers[layer]
        self.v = self.mask = self.handle = None

    def set(self, v: Tensor | None, mask: Tensor | None = None) -> None:
        """Kept live (not copied) so gradients reach v. mask=None steers every position."""
        self.v, self.mask = v, mask

    def __enter__(self) -> "Steerer":
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a) -> None:
        self.handle.remove()
        self.handle = None
        self.set(None)

    def _hook(self, module, inputs, output):
        if self.v is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if self.mask is not None and h.shape[1] == self.mask.shape[1]:
            h = h + self.mask[..., None].to(h.dtype) * self.v  # full-sequence forward / prefill
        else:
            h = h + self.v  # decode step: new tokens are always steered
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


def seq_logprobs(model, batch, steerer=None, v=None) -> Tensor:
    """(B,) sum of log p(token_t | tokens_<t) over every real token after BOS. No no_grad here."""
    ids, mask = batch["input_ids"], batch["attention_mask"]
    steered = steerer is not None and v is not None
    if steered:
        steerer.set(v, steer_mask(mask))
    try:
        logits = model(input_ids=ids, attention_mask=mask).logits
    finally:
        if steered:
            steerer.set(None)
    lp = F.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (lp * steer_mask(mask)[:, 1:]).sum(1)  # == attention_mask[:, 1:] under right padding


@torch.no_grad()
def read_acts(model, tok, texts, layer, position: str, batch_size=64) -> Tensor:
    """Block-`layer` outputs on CPU: "last" -> (N, d); "all_but_first" -> (total_real_tokens - N, d)."""
    assert position in ("last", "all_but_first")
    device = next(model.parameters()).device
    store, out = {}, []

    def hook(module, inputs, output):
        store["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        for i in range(0, len(texts), batch_size):
            b = encode(tok, texts[i:i + batch_size], "right", device)
            model(**b)
            h, m = store["h"], b["attention_mask"]
            if position == "last":
                out.append(h[torch.arange(len(h), device=h.device), m.sum(1) - 1].cpu())
            else:
                out.append(h[steer_mask(m).bool()].cpu())
    finally:
        handle.remove()
    return torch.cat(out)


def residual_norm(model, tok, wiki_texts, layer) -> float:
    """r_l: mean ||h|| at block `layer` over all non-BOS tokens of `wiki_texts`."""
    return read_acts(model, tok, wiki_texts, layer, "all_but_first").norm(dim=-1).mean().item()
