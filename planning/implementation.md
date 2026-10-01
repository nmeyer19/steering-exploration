# Implementation plan

This is the code needed to run §5 of [research_plan.md](research_plan.md). The work is split into tasks that can be built in parallel once the shared contracts in T0 exist. Each task lists its files, the exact functions to write, the gotchas, and a check that tells you it's done.

**Verified facts about the environment** (checked, not assumed):
- transformers 5.x. `LlamaDecoderLayer` returns a plain **tensor**, not a tuple; hooks must handle both. Model loading takes `dtype=`, not `torch_dtype=`.
- SmolLM2-135M: 30 layers, d = 576. **BOS = EOS = `<|endoftext|>` = id 0**, and there is **no pad token**. The tokenizer does **not** add BOS by itself.
- Since pad will be set to EOS, pad = BOS = id 0. **Never locate BOS or padding by token id**; always use `attention_mask`.
- At layer 15, the BOS position has a residual norm around 20,000, against about 180 at other positions. Excluding BOS from steering *and* from every norm or mean is essential.
- `output_hidden_states[ℓ + 1]` equals the output of `model.model.layers[ℓ]`.

---

## Layout

```
estimating-steering-vectors/
  requirements.txt
  steer/
    __init__.py
    config.py        # T0: constants + shared dataclasses
    model.py         # T1: loading, steering hook, log-probs, activations
    data.py          # T2: pairs, prompts, folds, WikiText
    closed_form.py   # T3: DiM, MoP, random
    gradient.py      # T4: Eq. 5, DPO-grad, norm-matched retrain
    metrics.py       # T5: generation, classifier, KL, diagnostics
    run.py           # T6: the grid
    gates.py         # T6: pre-run gates 7-11
    analyze.py       # T7: interpolation, figure, table, examples
  tests/
    test_model.py    # T1: gates 1-6
    test_data.py     # T2
    test_metrics.py  # T5
    test_estimators.py  # T3, T4
    test_analyze.py  # T7, on a synthetic grid
  writeup/
    main.tex         # T9
  results/           # created by run.py
```

Everything runs in fp32. Device is chosen automatically: CUDA if available, then MPS, then CPU.

---

## Dependency graph

```
T0 config + contracts
 ├─ T1 model.py ─────┬─ T3 closed_form.py ─┐
 ├─ T2 data.py ──────┼─ T4 gradient.py ────┼─ T6 run.py + gates.py ─ T7 analyze.py ─ T9 writeup
 └─ T5 metrics.py ───┘   (all use T1)      │
                                           └─ must pass gates 1-11 before the full run
```

Once T0 exists, every other task can be *written* in parallel against the signatures below, because they live in separate files. They can only be *run* together once T1 works.

---

## T0: Config and shared contracts

**File:** `steer/config.py`

```python
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
CLF_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"   # label 0 = NEGATIVE, 1 = POSITIVE
WIKI_ID, WIKI_CONFIG = "Salesforce/wikitext", "wikitext-2-raw-v1"

LAYERS = [15, 8]            # 0-indexed model.model.layers[i]; 15 is the headline
HEADLINE_LAYER = 15
ALPHAS = [0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]   # α = ‖v_applied‖ / r_l
KL_BUDGET = 0.1

TARGET = "neg"              # set after the headroom gate
READ_SUFFIX = " Overall, it was"

N_FOLDS, TEST_TOPICS_PER_FOLD, VAL_TOPICS = 3, 4, 2
N_TRAIN, N_VAL, N_TEST_PAIRS = 64, 32, 64

BATCH = 16
LR_GRID = [3e-4, 1e-3, 3e-3]   # Adam on u, where v = r_l * u
MAX_STEPS, EVAL_EVERY, PATIENCE = 400, 20, 5
BETA = 0.1

PROMPTS_PER_TOPIC, SAMPLES_PER_PROMPT = 8, 3
GEN_TOKENS, TEMPERATURE = 30, 0.7

N_WIKI, WIKI_LEN = 200, 64     # KL set and r_l / μ_generic set
N_WIKI_SHORT, WIKI_SHORT_LEN = 200, 20   # MoP's generic baseline (short chunks + READ_SUFFIX)
DISTINCT2_MIN = 0.5
SEED = 0
RESULTS_DIR = "results"

METHODS = ["dim", "mop", "eq5", "dpo", "random"]
GRADIENT_METHODS = ["eq5", "dpo"]

def get_device() -> str: ...   # "cuda" > "mps" > "cpu"
def gen_seed(fold: int, alpha_idx: int) -> int:  return SEED * 10_000 + fold * 100 + alpha_idx
```

**Why this learning-rate grid.** Adam moves every coordinate by about the learning rate per step. So ‖Δu‖ ≈ lr · √576 ≈ 24 · lr per step, and ‖Δv‖/r_ℓ is the same. At lr = 1e-2 that's a quarter of the residual norm per step, which is far too large. The grid {3e-4, 1e-3, 3e-3} gives 0.7–7% of r_ℓ per step.

**Why this α grid.** The endpoints are cheap insurance. Gradient vectors may hit KL = 0.1 at small α, and the random direction may need a large one. Interpolating at the budget needs points on both sides of it.

**Shared dataclasses** (in `config.py`):

```python
@dataclass
class Pair:
    topic: str
    noun: str
    pos: str   # the TARGET-side sentence (D_p)
    neg: str   # its minimal-pair twin (D_n)

@dataclass
class Fold:
    idx: int
    train_topics: list[str]; val_topics: list[str]; test_topics: list[str]
    train: list[Pair]; val: list[Pair]; test: list[Pair]
    prompts: list[tuple[str, str]]   # (topic, neutral prompt), test topics only

@dataclass
class SteerVec:
    method: str          # one of METHODS, or "eq5_nm" / "dpo_nm" for norm-matched retrains
    layer: int
    fold: int
    v: torch.Tensor      # (d,), raw, un-normalized, on CPU
    cost: dict           # {"fwd": int, "bwd": int, "seconds": float, "lr": float|None, "steps": int|None}
```

`pos` always means the *target* side. If `TARGET == "neg"`, `pos` holds the negative-sentiment sentence. Only the classifier (T5) and the pair builder (T2) read `TARGET`.

**`requirements.txt`:** `torch`, `transformers>=4.45`, `datasets`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `pytest`.

---

## T1: Model, steering hook, log-probs

**Files:** `steer/model.py`, `tests/test_model.py`

```python
def load_model(device=None) -> tuple[model, tok]
```
- `AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)`, `.eval()`, `requires_grad_(False)`.
- `tok.pad_token = tok.eos_token` if it is None.

```python
def encode(tok, texts: list[str], side: str = "right", device=None) -> dict[str, Tensor]
```
- Tokenize each text with `add_special_tokens=False`. Prepend `tok.bos_token_id` to every sequence. Pad on `side` with `tok.pad_token_id`, and build `attention_mask` by hand.
- Every sequence therefore has BOS as its first *real* token. That token is the attention sink and is never steered.
- Returns `{"input_ids", "attention_mask"}` on the device.

```python
def steer_mask(attention_mask: Tensor) -> Tensor   # (B, T), float
```
- 1 at every real token except the first real token of each row; 0 at pads and at that first token.
- Must work with **left** padding, where the first real token isn't at index 0: `((attention_mask.cumsum(1) > 1) & attention_mask.bool()).float()`.

```python
class Steerer:
    def __init__(self, model, layer: int)
    def set(self, v: Tensor | None, mask: Tensor | None = None)
    def __enter__(self) -> "Steerer"   # registers the forward hook on model.model.layers[layer]
    def __exit__(self, *a)             # removes it and clears v and mask
```

The hook:

```python
def _hook(self, module, inputs, output):
    if self.v is None:
        return output
    h = output[0] if isinstance(output, tuple) else output
    if self.mask is not None and h.shape[1] == self.mask.shape[1]:
        h = h + self.mask[..., None].to(h.dtype) * self.v       # full-sequence forward / prefill
    else:
        h = h + self.v                                          # decode step: new tokens always steered
    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
```

- `v` is kept as a live tensor, not copied, so gradients flow to an `nn.Parameter` for T4.
- If `set` is given a v but no mask, steer every position. Callers always pass a mask; this is only a fallback.

```python
def seq_logprobs(model, batch, steerer=None, v=None) -> Tensor   # (B,)
```
- The sum of log p(token_t | tokens_<t) over every real token after BOS.
- Shift logits and targets by one, mask by `attention_mask[:, 1:]`, and use `log_softmax` then `gather`.
- If `steerer` and `v` are given, call `steerer.set(v, steer_mask(batch["attention_mask"]))` before the forward pass, and `steerer.set(None)` after it. Otherwise run unsteered. It must not wrap anything in `no_grad`; callers decide that.

```python
def read_acts(model, tok, texts, layer, position: str, batch_size=64) -> Tensor
```
- `position="last"` → `(N, d)`: the activation at each sequence's last real token (right padding, index `attention_mask.sum(1) - 1`).
- `position="all_but_first"` → `(total_tokens, d)`: every real token except BOS.
- Read with a temporary forward hook on `model.model.layers[layer]` that stores `h.detach()`, under `torch.no_grad()`. Returns on CPU.

```python
def residual_norm(model, tok, wiki_texts, layer) -> float
```
- r_ℓ: the mean ‖h‖ over `read_acts(wiki_texts, layer, "all_but_first")`.

**Tests (gates 1–6), `tests/test_model.py`:**
1. **No-op.** With v = 0 (a zero tensor, not None), steered logits equal unsteered logits exactly.
2. **Right layer.** With a forward *pre*-hook capturing the input to `layers[ℓ + 1]`: steering at ℓ changes that input by exactly `mask · v`. The input to `layers[ℓ]` is unchanged.
3. **BOS untouched.** The block-ℓ output at the first real token is identical with and without steering, for both right and left padding.
4. **Decode steps are steered.** Greedy generation with a large v differs from the unsteered generation.
5. **Log-probs match.** For one unpadded sequence, `seq_logprobs` equals `-model(ids, labels=ids).loss * (T - 1)` to within 1e-4.
6. **Gradients flow.** A loss built from `seq_logprobs` with `v = r * u` (`u` an `nn.Parameter`) gives a nonzero `u.grad`, and no model parameter has a gradient.
7. **Padding invariance.** `seq_logprobs` for a sentence is the same (to 1e-4) whether it's batched alone or right-padded next to a longer one, with and without steering.

Use a module-scoped pytest fixture so the model loads once.

---

## T2: Data

**Files:** `steer/data.py`, `tests/test_data.py`

### Constants

```python
TOPICS = {   # 12 topics, 4 nouns each. The nouns double as the leakage word lists (T5).
  "food":    ["pizza", "pasta", "burger", "soup"],
  "movie":   ["movie", "film", "sequel", "documentary"],
  "hotel":   ["hotel", "suite", "resort", "motel"],
  "concert": ["concert", "band", "orchestra", "festival"],
  "book":    ["book", "novel", "memoir", "anthology"],
  "museum":  ["museum", "exhibit", "gallery", "sculpture"],
  "car":     ["car", "truck", "sedan", "minivan"],
  "phone":   ["phone", "laptop", "tablet", "headphones"],
  "park":    ["park", "trail", "beach", "campsite"],
  "flight":  ["flight", "airline", "airport", "layover"],
  "coffee":  ["coffee", "latte", "cafe", "espresso"],
  "game":    ["videogame", "boardgame", "puzzle", "chess"],
}

ADJ_TEMPLATES  = ["The {noun} was {w}.", "I thought the {noun} was {w}.", "Honestly, the {noun} was {w}."]
VERB_TEMPLATES = ["I {w} the {noun}.", "We really {w} the {noun}."]

# (positive-sentiment word, negative-sentiment word)
ADJ_PAIRS  = [("good","bad"), ("great","terrible"), ("wonderful","awful"),
              ("excellent","horrible"), ("amazing","disappointing"), ("fantastic","dreadful")]
VERB_PAIRS = [("loved","hated"), ("liked","disliked"), ("adored","despised"), ("enjoyed","loathed")]

PROMPT_TEMPLATES = [   # noun-agnostic, so every topic × noun combination reads naturally
  "My friend told me about the {noun}, so I",
  "So here's the thing about the {noun}:",
  "We decided to check out the {noun}, and",
  "I finally got around to the {noun}, and",
  "Last weekend I checked out the {noun} for the first time. It",
  "Here is what happened with the {noun}:",
  "A few days ago I had my first experience with the {noun}, and",
  "Everyone keeps asking me about the {noun}, so",
]
```

### Functions

```python
def all_pairs(target: str) -> list[Pair]
```
- Cross every topic × noun × template × matching word pair: 12 × 4 × (3 × 6 + 2 × 4) = 1,248 pairs.
- Orient each by `target`: if `target == "neg"`, `Pair.pos` is the negative-word sentence.

```python
def make_folds(target: str, seed: int = SEED) -> list[Fold]
```
- Shuffle the 12 topics with `random.Random(seed)`. Fold k's test topics are `[4k, 4k+4)`.
- The other 8 are shuffled with `seed + k`. The first 2 are val, the other 6 are train.
- Sample `N_TRAIN` train pairs, `N_VAL` val pairs and `N_TEST_PAIRS` test pairs from those topics' pairs. Stratify round-robin over topics so each topic contributes about equally.
- **Prompts:** for each test topic, 2 nouns × 4 templates, sampled with the fold's seed. That's 8 prompts per topic and 32 per fold.
- Fully deterministic: the same folds for every method, run and process.

```python
def wiki_texts(tok, n=N_WIKI, length=WIKI_LEN) -> list[str]
```
- `load_dataset(WIKI_ID, WIKI_CONFIG, split="validation")`. Join the non-empty lines with "\n", tokenize without special tokens, take the first `n` consecutive chunks of `length - 1` tokens (leaving room for BOS), and decode each back to a string.
- Used for KL, r_ℓ and nothing else.

```python
def wiki_short_texts(tok, n=N_WIKI_SHORT, length=WIKI_SHORT_LEN) -> list[str]
```
- The same procedure with short chunks, starting *after* the chunks `wiki_texts` uses so the two sets don't overlap. Used only for MoP's generic baseline (T3).

```python
def with_suffix(texts: list[str]) -> list[str]   # t + READ_SUFFIX
def leak_words(topics: list[str]) -> list[str]   # all nouns of those topics
```

**Tests:**
- 3 folds, with disjoint train/val/test topics within each fold, and every topic a test topic exactly once across folds.
- The right sizes for train, val, test and prompts.
- Prompts come only from test topics, and train pairs only from train topics.
- Every pair differs in exactly one whitespace token, and `pos` contains the TARGET-side word.
- Determinism: two calls give identical folds.

The WikiText functions need network access; mark their test `@pytest.mark.slow`.

---

## T3: Closed-form estimators

**File:** `steer/closed_form.py` (tests in `tests/test_estimators.py`)

```python
def dim(model, tok, fold, layer) -> SteerVec
```
- `read_acts(with_suffix([p.pos ...]), layer, "last").mean(0) − read_acts(with_suffix([p.neg ...]), layer, "last").mean(0)`, over `fold.train`.
- Cost: `fwd = 2 * N_TRAIN`, `bwd = 0`, plus wall-clock time.

```python
def generic_mean(model, tok, short_wiki_texts, layer) -> Tensor
```
- `read_acts(with_suffix(short_wiki_texts), layer, "last").mean(0)`.
- **This is read at the same suffix token as μ_p, on purpose.** An average over all WikiText positions would leave the "was"-token and suffix features in μ_p − μ_generic. MoP would then be handicapped by position artifacts, not by the lack of contrast, which would confound P2.
- Computed once per layer by T6 and passed in. Its cost isn't charged to MoP, because it doesn't depend on the task data.

```python
def mop(model, tok, fold, layer, mu_generic) -> SteerVec
```
- `read_acts(with_suffix([p.pos ...]), layer, "last").mean(0) − mu_generic`.
- Cost: `fwd = N_TRAIN`.

```python
def random_vec(d, layer, fold) -> SteerVec
```
- Gaussian from `torch.Generator().manual_seed(gen_seed(fold, 0) + layer)`. Only its direction is used. Cost is zero.

**Timing:** call `torch.cuda.synchronize()` (or `torch.mps.synchronize()`) before reading `time.perf_counter()`. The same applies in T4; put the helper in `model.py` as `sync(device)`.

---

## T4: Gradient estimators

**File:** `steer/gradient.py` (tests in `tests/test_estimators.py`)

### Shared training loop

```python
def train_vec(model, tok, fold, layer, r_l, method, fixed_norm=None, init=None, lr=None) -> SteerVec
```
`method` is `"eq5"` or `"dpo"`. The learning rates tried are `[lr]` if one is given, otherwise `LR_GRID`. For each:
1. `u = nn.Parameter(init.clone() / r_l if init is not None else torch.zeros(d))`. v = r_l · u inside the forward pass, so one learning-rate grid works at both layers.
2. `Adam([u], lr=lr)`. Each step samples a batch of `BATCH` pairs from `fold.train` with a generator seeded by `(SEED, fold, layer, method)`.
3. If `fixed_norm` is set, after each step project `u.data *= fixed_norm / u.norm()`. This is the norm-matched retrain (below).
4. Every `EVAL_EVERY` steps, compute the loss on all of `fold.val` under `torch.no_grad()`. Keep a copy of the u with the best val loss. Stop after `PATIENCE` evaluations without improvement, or at `MAX_STEPS`.
5. **Never return u = 0.** Step 0 isn't a candidate for "best"; the first candidate is after `EVAL_EVERY` steps. If the best u still has norm < 1e-6, raise an error rather than return a vector that can't be normalized.

Across learning rates, keep the one with the lowest best-val-loss, and return `v = r_l · u_best` on CPU.

**Cost** counts *every* learning rate tried, because tuning is part of the method's cost:
- `fwd`: sequences processed in all training and validation forward passes, plus reference passes for DPO.
- `bwd`: sequences processed in backward passes.
- `seconds`, and the chosen `lr` and `steps`.

**Logs:** per evaluation, record train loss, val loss, ‖u‖, and on val the mean Δ log p(w_p) and Δ log p(w_n) relative to unsteered. Save to `results/logs/{method}_{fold}_{layer}[_nm].jsonl`. The Δ log p values show whether DPO met its margin by pushing the negatives down.

### Losses

```python
def eq5_loss(model, tok, steerer, v, pairs) -> Tensor
    return -seq_logprobs(model, encode(tok, [p.pos for p in pairs]), steerer, v).mean()
```
The loss is on all tokens of w after BOS, per the PDF. Negatives are never used.

```python
def dpo_loss(model, tok, steerer, v, pairs, ref) -> Tensor
    lp_p = seq_logprobs(model, encode(tok, [p.pos for p in pairs]), steerer, v)
    lp_n = seq_logprobs(model, encode(tok, [p.neg for p in pairs]), steerer, v)
    ref_p, ref_n = ref.lookup(pairs)
    return -F.logsigmoid(BETA * ((lp_p - ref_p) - (lp_n - ref_n))).mean()
```
- **`ref`:** a dict from sentence to its unsteered `seq_logprobs`. It is filled once per fold for the train and val sentences before training; charge those forward passes to DPO.
- DPO's loss can keep falling just by growing ‖v‖. Early stopping on val loss and the matched-KL comparison deal with that.

### Norm-matched retrain

A rescaled gradient vector is evaluated off the norm it was trained at. Its *direction* was optimized for a different regime, and rescaling doesn't fix that. (The native-norm point doesn't either: it lies exactly on the rescaled curve, since the direction is the same.)

The fair comparison retrains at the norm where it will be judged. T6 runs this after the grid:

```python
def norm_matched(model, tok, fold, layer, r_l, base: SteerVec, alpha_star: float) -> SteerVec
    return train_vec(..., method=base.method, fixed_norm=alpha_star,
                     init=base.v / base.v.norm() * alpha_star * r_l, lr=base.cost["lr"])
```
- `alpha_star` is the α at which the method's own rescaled curve reaches KL = 0.1 (from T7's `alpha_at_budget`).
- Its method tag is `"eq5_nm"` or `"dpo_nm"`. Its cost adds to the base method's cost.

**Tests, `tests/test_estimators.py`** (on a small fold, `MAX_STEPS = 40`):
- DiM, MoP and random return shape `(d,)` with finite, nonzero norm.
- Eq. 5 and DPO lower val loss relative to the loss at u = 0.
- Eq. 5 raises val Δ log p(w_p); DPO raises the val margin.
- With `fixed_norm`, the returned vector has ‖v‖/r_ℓ equal to `fixed_norm` to within 1e-4.

---

## T5: Metrics

**Files:** `steer/metrics.py`, `tests/test_metrics.py`

Every metric takes the vector *as applied*: `v_applied = α · r_ℓ · v̂`, or None for the unsteered baseline.

```python
def generate(model, tok, steerer, v_applied, prompts: list[str], seed: int, batch_size=96) -> list[str]
```
- Repeat each prompt `SAMPLES_PER_PROMPT` times *before* tokenizing, so the mask lines up with the batch.
- `encode(..., side="left")`, then `steerer.set(v_applied, steer_mask(attention_mask))`.
- `torch.manual_seed(seed)` right before `model.generate(input_ids, attention_mask=..., do_sample=True, temperature=TEMPERATURE, top_p=1.0, top_k=0, max_new_tokens=GEN_TOKENS, pad_token_id=tok.pad_token_id)`. Every method gets the same seed at the same α index, so sampling noise is shared across methods.
- Decode only the new tokens, with `skip_special_tokens=True`. Clear the steerer afterward.
- Returns a list aligned with the expanded prompts: `len(prompts) * SAMPLES_PER_PROMPT`.

```python
class SentimentClassifier:
    def __init__(self, device)
    def p_target(self, texts: list[str]) -> np.ndarray
```
- Returns `softmax[:, 1]` if `TARGET == "pos"`, else `softmax[:, 0]`. Truncate at 128 tokens, batch at 64.
- An empty continuation is scored as the string "" (the classifier handles it); count those as a separate `n_empty` field.
- Score **continuations only**, never the prompt.

```python
def kl_on_wiki(model, tok, steerer, v_applied, wiki_texts, batch_size=50) -> float
```
- Per batch: unsteered logits, then steered logits (mask from `steer_mask`).
- `KL = sum_vocab softmax(s) * (log_softmax(s) - log_softmax(b))`, averaged over every real position except BOS, weighted by position count across batches.
- Don't cache the base logits: 200 × 64 × 49k floats is about 2.5 GB. One extra forward pass is cheaper.

```python
def pair_logprobs(model, tok, steerer, v_applied, texts: list[str]) -> Tensor   # (N,), no_grad
def pair_diagnostics(model, tok, steerer, v_applied, test_pairs, base) -> dict
```
- `base` holds the unsteered `pair_logprobs` for the pos and neg test sentences, computed once per fold.
- Returns `dlogp_pos`, `dlogp_neg` and `margin` = dlogp_pos − dlogp_neg: means over the test pairs of steered minus unsteered log-probs.

```python
def distinct2(texts) -> float
def leak_rate(texts, words) -> float
```
- `distinct2`: mean over texts of unique bigrams / total bigrams, on lowercased whitespace tokens. Texts with fewer than 2 tokens are skipped.
- `leak_rate`: the fraction of texts containing any of `words` as a whole word, case-insensitive, with an optional trailing "s". Use a regex `\b(word1|word2|...)s?\b`.
- The caller passes `leak_words(fold.train_topics)`, so test-topic nouns, which are expected to appear, don't count.

```python
def evaluate_point(model, tok, clf, steerer, v_applied, fold, prompts, wiki, base, seed) -> tuple[dict, list[str]]
```
- Returns the metric dict `{kl, p_target, distinct2, leak, dlogp_pos, dlogp_neg, margin, n_gen, n_empty}` and the generations.
- With `v_applied=None`, `kl`, `dlogp_*` and `margin` are exactly 0.

**Tests:** the pure functions on hand-built examples; the v = 0 invariants above; KL increases from α = 0.05 to α = 0.4 for a random direction; and generation output has the right length and is deterministic under the same seed.

---

## T6: The grid and the gates

**Files:** `steer/run.py`, `steer/gates.py`

### `python -m steer.run [--folds 0 1 2] [--layers 15 8] [--smoke]`

```
set seeds; load model, tokenizer, classifier; folds = make_folds(TARGET)
wiki = wiki_texts(tok); wiki_short = wiki_short_texts(tok)
for layer in layers: r_l[layer] = residual_norm(...); mu_generic[layer] = generic_mean(...)
save results/meta.json: config values, r_l per layer, device, versions, TARGET

for fold in folds:
    base = unsteered pair_logprobs for fold.test pos and neg
    prompts = [p for _, p in fold.prompts]
    baseline row: evaluate_point(v_applied=None), method="none", alpha=0, seed=gen_seed(fold, 0)
    for layer in layers:
        vecs = dim, mop, random_vec, train_vec("eq5"), train_vec("dpo")
        save each to results/vectors/{fold}_{layer}_{method}.pt  (v and cost)
        for vec in vecs:
            for i, alpha in enumerate(ALPHAS[1:], start=1):
                v_applied = alpha * r_l * vec.v / vec.v.norm()
                row = evaluate_point(..., seed=gen_seed(fold, i))
                    | {fold, layer, method, alpha, kind: "sweep"}
            if vec.method in GRADIENT_METHODS:
                native point: v_applied = vec.v, alpha = ‖v‖/r_l, kind = "native", seed = gen_seed(fold, 99)

# stage 2, after the sweeps for a (fold, layer) are done:
for each gradient method:
    alpha_star = alpha_at_budget(sweep rows for (fold, layer, method))   # from analyze.py
    if alpha_star is not None:
        nm = norm_matched(...); save it; evaluate at v_applied = nm.v, kind = "norm_matched",
        seed = gen_seed(fold, 98)
```

- Rows are appended to `results/grid.jsonl` and generations to `results/generations.jsonl` as `{key..., prompts, continuations}`. Flush after each point.
- **Resumable.** A point's key is `(fold, layer, method, alpha, kind)`. Skip it if already in `grid.jsonl`; skip training if its vector file exists.
- **`--smoke`:** 1 fold, the headline layer, α ∈ {0.1, 0.4}, 4 prompts, 8 wiki texts, `MAX_STEPS = 40`, and outputs to `results_smoke/`. It must run end to end on CPU or MPS in a few minutes.
- **Size of the full run:** 3 folds × 2 layers × (5 methods × 6 α + 2 native + 2 norm-matched) = 204 points, plus 3 baselines, and 12 + 12 gradient runs. Each point is 96 generations of 30 tokens, a 200-sequence KL pass and 128 test-pair log-probs.

### `python -m steer.gates`

The gates before the full run, printed as PASS/FAIL with the numbers:
- **7. Headroom.** Baseline mean P(positive) over all 3 folds' prompts. Report it and the recommended `TARGET` (steer negative if P(positive) > 0.5). The number goes in the write-up.
- **8. The budget is reachable.** For DiM and random at the headline layer on fold 0: KL at α = 0.8 ≥ 0.1, and KL at α = 0.025 < 0.1. Otherwise extend `ALPHAS`.
- **9. Sign.** DiM at α = 0.2 increases P(target) on fold 0. If it decreases, the pair orientation is flipped somewhere.
- **10. Continuations.** Print 10 of DiM's continuations at α = 0.2, and the unsteered ones, to read by eye before trusting the classifier.
- **11. Classifier sanity.** P(target) on 4 obviously target-side and 4 obviously opposite-side sentences.

---

## T7: Analysis

**Files:** `steer/analyze.py`, `tests/test_analyze.py`. Run with `python -m steer.analyze [--results results]`.

```python
def load_grid(results_dir) -> pd.DataFrame
```
- Adds `effect = p_target − baseline_p_target(fold)`.
- Prepends the baseline row as α = 0 to every (fold, layer, method) sweep curve, with `kl = 0`, `effect = 0` and the diagnostics at 0.

```python
def alpha_at_budget(rows, budget=KL_BUDGET) -> float | None
def at_budget(df, budget=KL_BUDGET) -> pd.DataFrame
```
- Per (fold, layer, method), using `kind == "sweep"` rows only: sort by α. **Enforce monotone KL** by taking a running max and warning when that changes anything. Linearly interpolate α and **every** metric at KL = `budget`.
- If KL never reaches the budget, return None or NaN and flag it. Never extrapolate.
- Flag rows where the interpolated distinct-2 is below `DISTINCT2_MIN`.
- `alpha_at_budget` is imported by `run.py` for the norm-matched stage.
- `summary(df)` adds the norm-matched points as their own methods (`eq5_nm`, `dpo_nm`), since they already sit near the budget, then aggregates the mean and std across folds.

```python
def figure_1(df, layer=HEADLINE_LAYER, out="results/fig1.pdf")
```
- Interpolate each fold's effect-vs-KL curve onto a common grid, `np.linspace(0, max KL reached by every fold, 50)`. Plot the mean with a ±std band, one per method.
- Plot the native and norm-matched points as markers, and add a dashed vertical line at the budget.
- One fixed color per method, sized for ~3.3 in wide.

```python
def table_1(df, out="results/table1.tex")
```
- Rows: dim, mop, eq5, eq5_nm, dpo, dpo_nm, random. Column groups: blocks 15 and 8.
- Columns: effect, Δ log p(w_p), Δ log p(w_n), margin, leak, fwd, bwd, seconds. Cost comes from the saved vectors.
- Formatted as mean ± std via `to_latex`. Also write `results/summary.csv`.

```python
def cosines(results_dir) -> pd.DataFrame   # results/cosines.csv
def examples(results_dir, k=3)             # results/examples.md
```
- `cosines`: pairwise cosine similarity among the vectors per fold and layer, averaged across folds.
- `examples`: for 2 fixed prompts from fold 0 at the headline layer, each method's continuations at the α whose KL is closest to the budget. Also list continuations flagged for leakage.

**Tests:** on a synthetic `grid.jsonl` with known linear curves, `at_budget` recovers the right values; a curve that never reaches the budget gives NaN; and the figure and table are written without errors.

---

## T9: Write-up skeleton

**File:** `writeup/main.tex`: one page, `article` class, 1 in margins, 10pt.

The sections follow research_plan.md §10, with `\input{../results/table1.tex}`, `\includegraphics{../results/fig1.pdf}` and `% TODO` markers where numbers go:
1. The question, and what "better" means.
2. The claim and the 2 × 2.
3. The setup.
4. Figure 1, Table 1 and one example.
5. Results for P1 and P2, open questions, limitations.

Include the bibliography entries from research_plan.md §9 with `% VERIFY` markers on each.

---

## Order of work

- **Parallel build:** T1, T2, T3+T4, T5, T6+T7 and T9, in separate files against the T0 contracts.
- **Integration:** `pytest -x`, then `python -m steer.run --smoke` and `python -m steer.analyze --results results_smoke`.
- **On the GPU:** `python -m steer.gates`; set `TARGET`; `python -m steer.run`; `python -m steer.analyze`.
