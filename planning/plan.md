# Research plan: difference-of-means vs. gradient-based steering vectors

This is a working plan, not the write-up. The final deliverable is ≤1 page of LaTeX.

**Scope.** One falsifiable claim, one experiment built to test it, one small model, one task. Anything that doesn't bear on the claim goes in §8 as an open question.

**What the results can and can't mean.** This is a proof of concept for a *mechanism*, not a general ranking of the estimators.
- **The test is close to in-distribution.** "Held-out" means new topics and a new format: continuing a neutral prompt rather than scoring a complete sentence. But the genre (short consumer reviews), the sentiment vocabulary and the register are the same as the training templates.
- **Out-of-distribution behavior is untested, and could go either way.** A method that transfers better here need not generalize better to different text, and the ranking there could differ. This experiment can't tell us which.
- **The setting is narrow.** One 135M model and one simple, templated concept (sentiment). Whether any of this extends to larger models, natural text or harder behaviors (refusal, personas) is open.
- **Why it's still worth doing.** A small model and controlled data make the mechanism *testable*: every pair differs in exactly one word, so "the contrast cancels shared content" can be checked directly. If the mechanism doesn't show up in this clean setting, it's unlikely to be the explanation elsewhere. If it does, the next step is the natural-data test in §8.

---

## 1. What "better" means

The verdict has to come from a metric that none of the estimators optimizes directly. Otherwise the answer is decided by construction.

**Verdict metric: steering the intended behavior on new inputs, at a matched side-effect budget.**
- **Behavior.** Steer open-ended continuations of sentiment-neutral prompts on *held-out topics*. Score only the continuation with an off-the-shelf SST-2 classifier. Report the change in mean P(target sentiment) from the v = 0 baseline.
- **Side effect.** Per-token KL(p^v ‖ p) on unrelated text (WikiText validation, ~200 × 64 tokens).
- **Headline number.** Sweep the steering scale to trace an effect-vs-KL curve for each method. Read off the effect at KL = 0.1 nats/token.
- **Why matched KL.** Any vector steers more if you make it bigger. The question is which *direction* buys the most behavior per unit of damage.

**Diagnostics (not the verdict).** These explain *why* a method wins or loses. They are computed on held-out pairs at the same scale as the headline.
- **Δ log p(w_p)** and **Δ log p(w_n)**: how much steering raises each half of a pair, relative to v = 0.
- **Margin**: Δ log p(w_p) − Δ log p(w_n), i.e., how much steering prefers the target sentence over its minimal-pair twin.
- *(Topic leakage was also computed but dropped: it couldn't separate the methods from the random direction or the unsteered baseline. See §8.)*

Eq. 5 optimizes something close to Δ log p(w_p), and DPO-grad optimizes something close to the margin. That's why these are diagnostics, not the verdict.

**Cost** is reported alongside, not folded into "better": forward and backward passes, wall-clock time, and the hyperparameters that had to be tuned.

**Guards.**
- Distinct-2 on steered continuations; flag points below a threshold as degenerate.
- Read ~10 continuations per method at the headline setting.
- Perplexity under the unsteered model is not used, because it rewards repetition.

---

## 2. The two methods, and why I expect DiM to transfer better

### 2.1 What each method is actually estimating

**Difference-of-means (DiM, Eq. 4)** describes the geometry: the direction along which the model's representations of positive and negative sentences differ. It never runs the model forward from the intervention, so it has no information about what that direction *does* downstream.

**Gradient-based optimization (Eq. 5)** measures the effect: it finds the vector that, when added, makes the positive sentences most likely. It optimizes through the rest of the network, so it knows what the vector does. But the thing it makes likely is *the positive training strings*, not the behavior we evaluate. It is closer to a tiny fine-tune on D_p than to a direction for "negative sentiment".

### 2.2 Why the minimal-pair contrast should matter

**DiM cancels what the pair shares.** In a minimal pair ("The pizza was good" / "The pizza was bad"), topic, template and syntax are identical. They cancel in μ_p − μ_n. What survives is, roughly, the thing that differs: sentiment.

**Eq. 5 has no such cancellation, and the gradient is dominated by the wrong tokens.** The loss is summed over every token of w. For each token, the gradient on the logits is (p − onehot), which is largest for tokens the model found *hard to predict*.
- In "The pizza was bad", the sentiment word "bad" given "The pizza was" is fairly predictable. Maybe 1 in 10.
- The topic word "pizza" given "The" is maybe 1 in 1000.
- So most of the gradient goes into making *topic and template* tokens likely, not sentiment. The minimal-pair format makes this worse: nearly every token is shared, and only one or two carry the concept.

**Result: I expect a dissociation.** Eq. 5 should win on the objective it optimizes (raising log p of held-out positive sentences) but lose on the behavior we want (sentiment of continuations on new topics). It should also raise log p(w_n) almost as much as log p(w_p), since most of what it learned is shared between them. On new-topic prompts, it should pull continuations toward training topics.

### 2.3 Where DiM is weak, and Eq. 5 might win anyway

The argument isn't one-sided. These are the reasons the prediction could fail.

- **Reading vs. writing.** DiM finds the direction along which the model *represents* "this sentence was negative". It's read at the end of a completed sentence. Nothing guarantees that adding it makes the model *produce* negative text; that depends on whether later layers use the direction when generating. The gradient methods test this directly. This is the strongest argument for Eq. 5, and it matters most at early layers, where much of what's represented is not yet used.
- **Lexical residue.** The pair difference also carries whatever is specific to the particular words: the identity of "hated" vs. "loved", not just the abstract concept. Mitigations are varied sentiment words, and reading at a shared suffix position rather than the sentiment word itself. Neither removes it completely.
- **Magnitude.** DiM gives a direction but no principled scale. ‖μ_p − μ_n‖ isn't calibrated to anything about steering. The gradient methods learn a norm, and the direction they learn is tuned to that norm. The α sweep puts every method on the same scale. The norm-matched retrain (§4.6) makes sure the gradient methods aren't judged at a norm they weren't trained for.
- **Distribution shift, for both methods.** Training data are complete short sentences. Evaluation is continuing a neutral prompt. Both methods are extrapolating; the claim is only that the contrast makes DiM extrapolate better.

### 2.4 What the DPO-grad arm is for

DiM and Eq. 5 differ in two ways at once:
- **Contrastive vs. not:** uses the negatives, or ignores them.
- **Closed-form vs. optimized:** reads off activations, or optimizes through the network.

If DiM wins, either difference could be the reason. Two extra arms fill in a 2 × 2 so the reason can be identified:

|  | **Closed-form** | **Optimized** |
|---|---|---|
| **Contrastive** | DiM: μ_p − μ_n | DPO-grad |
| **Positives only** | MoP: μ_p − μ_generic | Eq. 5 |

- **DPO-grad** is contrastive *and* optimized. In a minimal pair, the tokens before the differing word have identical contexts, so their log p^v terms cancel *exactly* in the DPO difference. The tokens after it cancel approximately. So its gradient concentrates on the sentiment token and what follows, which is the cancellation DiM gets, while still optimizing through the network.
- **Mean-of-positives (MoP)** is closed-form but not contrastive with the negatives. It is μ_p minus the mean activation on generic text (short WikiText chunks), so it's not dominated by the model's average activation. The generic text gets the same "Overall, it was" suffix and is read at the same token as μ_p. Otherwise MoP would carry position and suffix-token features that DiM cancels, and it would lose for a reason unrelated to contrast. It costs no training and is one more curve on the same plot.

**How to read the outcomes:**

| Outcome | What it says |
|---|---|
| DPO-grad ≈ DiM (or better), MoP ≈ Eq. 5 (poor) | **Contrast is the reason.** The claim holds. |
| DPO-grad < DiM, MoP ≈ DiM | Contrast isn't the reason. Something about closed-form estimation matters. The leading candidate: DiM is a difference of real activations, so it stays on the distribution the model normally sees, while optimized vectors can exploit off-distribution directions and overfit training pairs. |
| DPO-grad < DiM, MoP poor | Contrast helps, but the closed form adds something beyond it (as above), or DPO's quirks are hurting it (next point). |
| Eq. 5 ≥ DiM | The dissociation fails. Optimizing through the network outweighs the lack of contrast, so reading vs. writing dominates. |

**DPO-specific confounds.** DPO differs from Eq. 5 in more than using the negatives. It has a reference-model term and a β-controlled sigmoid. It can also meet its margin by *lowering* the negatives rather than raising the positives. So a weak DPO-grad result is ambiguous, not decisive. Logging absolute log-probs (Δ log p of each half) shows which of the two it did.

**Prior evidence.** BiPO (Cao et al., 2024) trains steering vectors with a preference objective. DPO-grad here is the simple version, used as a control, not as a new method.

---

## 3. The claim

**Claim.** On minimal-pair data, contrast is what makes a steering vector transfer to the intended behavior.

**Predictions (each falsifies the claim if it fails):**
- **P1: Dissociation.** At KL = 0.1, DiM has a larger behavioral effect than Eq. 5. Eq. 5 has the larger Δ log p(w_p), but a smaller margin.
- **P2: Contrast carries it.** DPO-grad's behavioral effect ≥ DiM's, and MoP's is closer to Eq. 5's than to DiM's.

---

## 4. Setup

### 4.1 Model
SmolLM2-135M, base model: 30 blocks, d = 576. There is no chat template to confound the results. It's small enough that compute is not the constraint.

### 4.2 Data
- **Training data: sentiment minimal pairs, built from templates.** Topic (~12) × sentence template (~5) × sentiment-word pair (~6, e.g., good/bad, loved/hated, great/terrible). Deduplicate and sample. Varying the words and templates keeps the vector from reducing to a single token swap.
- **Why templates rather than an existing dataset.**
  - The claim is about minimal pairs, so the pairs must differ *only* in sentiment. Standard sentiment sets (SST-2, IMDB) aren't paired, so topic, length and style differ too.
  - The held-out-topic evaluation needs topic labels, which single-domain sets like IMDB don't provide.
  - The cost is artificial, low-diversity text, which is what makes the test close to in-distribution (see Scope). Counterfactually-augmented IMDB (Kaushik et al., 2020) has natural-text minimal pairs, but its reviews are long, its edits span several places, and it's all one topic. It's the right follow-up, not the main test.
- **Splits: topic cross-validation.** 3 folds, each holding out 4 test topics. Within each fold, 2 training topics are held out for validation (early stopping and learning-rate choice). Folds exist because topic-to-topic variance would swamp method differences with a single split.
- **Training-set size.** N = 64 pairs per fold for every method.
- **Evaluation prompts.** ~8 sentiment-neutral prompts per test topic (e.g., "I went to the museum last weekend and").
- **Held-out pairs** for the diagnostics come from the test topics.
- **Direction.** Measure the v = 0 baseline sentiment first. Steer toward the side with more headroom (probably negative), so D_p is that side of each pair.

### 4.3 Layers
- **Headline:** block 15 (mid-depth).
- **Check:** block 8. The reading-vs-writing gap should be largest at early layers, so if Eq. 5 is going to beat DiM anywhere, it's there.

Both are fixed in advance. Choosing layers by a sweep scored on any one method would favor that method.

### 4.4 Conventions (identical across methods, so only the estimator differs)
- **Layer ℓ** is the output of transformer block ℓ. Reading and writing use the same forward hook, which avoids the off-by-one in `output_hidden_states`.
- **Closed-form read position (DiM, MoP).** Every sentence gets the suffix "Overall, it was", and activations are read at its last token. Otherwise the last token is often the sentiment word itself, and μ_p − μ_n becomes mostly a token-embedding difference.
- **Write positions.** v is added at every position except 0 (the attention-sink token), in training and evaluation, for all methods. This departs from Eq. 1's last-position-only intervention; I'll state that in the write-up.

### 4.5 Estimators
1. **DiM (Eq. 4).** v = μ_p − μ_n.
2. **MoP.** v = μ_p − μ_generic, with μ_generic the mean activation over WikiText (excluding position 0).
3. **Eq. 5.** Minimize E_{w∼D_p}[−log p^v(w)] over v, with the loss on all tokens of w. Adam, v initialized at 0, early stopping on validation-topic loss.
4. **DPO-grad.** Minimize −log σ(β[(log p^v(w_p) − log p⁰(w_p)) − (log p^v(w_n) − log p⁰(w_n))]), with p⁰ the unsteered model (log-probs cached) and β ≈ 0.1. Same optimizer and stopping rule as Eq. 5.
5. **Controls.** A random direction, and v = 0.

**Tuning parity.** Gradient hyperparameters are tuned on validation topics only, and that tuning counts toward their cost.

### 4.6 Scale and norm
- Residual norms grow with depth. Apply every vector as α · r_ℓ · v̂, where r_ℓ is the mean ‖h^(ℓ)‖ on WikiText (excluding position 0).
- Sweep α ∈ {0, .025, .05, .1, .2, .4, .8}. The wide range makes sure every method's curve brackets KL = 0.1, so the budget point is interpolated, never extrapolated.
- **Norm-matched retrain.** Rescaling a gradient-trained vector evaluates it off the norm it was trained at. Its *direction* was optimized for a different regime, and rescaling doesn't fix that. (Evaluating it at its native norm doesn't test this either: that point lies exactly on the rescaled curve, since the direction is the same.) So after each gradient method's sweep, find the α* where its curve hits KL = 0.1. Then retrain it with ‖v‖ held at α* · r_ℓ (projected after each step, warm-started from the original direction). Both versions are reported.
- **A fixed norm does not fix KL.** In the first full run, the retrained DPO-grad direction landed at KL 0.3–1.4, not 0.1: it does more damage per unit length. So the retrained direction is swept over α and interpolated at KL = 0.1 like every other method. It's still evaluated close to the norm it was trained at, which is the point of the retrain. If DiM still wins against the norm-matched vectors, the "unfair norm" objection is answered.
- The native-norm point is still plotted, since it shows where the unconstrained optimizer ended up relative to the budget.

---

## 5. Experiment

**E1: The 2 × 2 comparison (tests P1 and P2).**
- **Grid:** {DiM, MoP, Eq. 5, DPO-grad, random} × 2 layers × 6 nonzero α × 3 folds, plus native and norm-matched points for the gradient methods. That's about 200 evaluation points and 24 gradient runs, all small at 135M.
- **Figure 1.** Behavioral effect vs. KL at block 15, one curve per method, with a ±std band across folds.
- **Table 1.** Per method, at KL = 0.1 at block 15 (block 8 in a second column group):
  - Behavioral effect (for Eq. 5 and DPO-grad, both the rescaled and the norm-matched versions).
  - Δ log p(w_p), Δ log p(w_n), margin.
  - Cost: forward/backward passes and wall-clock time.
- **Examples.** 2–3 steered continuations per method at the headline setting. If Eq. 5's continuations of a museum prompt drift toward pizza, that one example makes the argument better than a number does.
- **Free diagnostic.** Pairwise cosine similarity among the four vectors at each layer. The prediction: DPO-grad sits closer to DiM than to Eq. 5.

**Uncertainty.** With three folds, report effect sizes with fold std, not significance tests.

---

## 6. Time budget

Compute is not the constraint at 135M: gradient runs take seconds, and the evaluation grid is roughly 15–20 minutes. Human time is.

| Task | Time |
|---|---|
| Template-built pair dataset + neutral prompts | 20–30 min |
| Hooks + four estimators | 45 min |
| Evaluation: generation, classifier, KL, diagnostics | 45 min |
| Debugging, sanity checks, reading outputs | 30–60 min |
| Figure + table | 20 min |
| 1-page LaTeX write-up | 45–60 min |

The build-and-run portion is about 2.5–3 hours. The first cuts if it runs long: block 8 → headline layer only; MoP (DPO-grad still gives half the answer).

---

## 7. Choices he left open

| His question | My answer |
|---|---|
| Which model? | SmolLM2-135M base |
| Which task? | Sentiment minimal pairs, built from templates |
| Which layers? | Block 15 (headline) and block 8, fixed in advance |
| How long do we train? | Early stopping on validation-topic loss |
| How large is the training set? | 64 pairs |
| What separate eval set? | Held-out topics (3-fold), neutral prompts; WikiText for side effects |
| What does "better" mean? | Steering of the intended behavior on held-out topics at matched KL; likelihood metrics are diagnostics only |

---

## 8. Open questions (for the discussion paragraph, not experiments)
- **Does it hold out of distribution?** The test here shares its genre, vocabulary and register with the training data. On natural-text minimal pairs (counterfactually-edited IMDB), on a larger model, or on a non-sentiment behavior, the ranking could change. This is the most important open question.
- **Does the ranking depend on N?** DiM has nothing to overfit, so it should look best at small N. Does Eq. 5 close the gap as N grows, as topic-specific leakage averages out?
- **Does it depend on depth, beyond two layers?** The reading-vs-writing argument predicts a crossover somewhere.
- **Is minimal-pair data the fair test?** It's the format that most disadvantages Eq. 5. On naturalistic, non-paired reviews, shared content cancels less for DiM too, and the result might reverse.
- **Does Eq. 5 leak training content into new topics?** I measured this as the rate of training-topic nouns in continuations on new topics. It couldn't separate any method from the random direction, because those nouns ("car", "film", "coffee") appear naturally. A better test would be a per-token split: does Eq. 5 raise log p at the positions a pair shares, or at the sentiment word?
- **Would restricting Eq. 5's loss to the sentiment tokens fix it?** If so, what contrast really buys is *knowing which tokens matter*.
- **Is it symmetric?** −v_DiM is the DiM vector for the opposite behavior by construction. Nothing forces −v_Eq5 to steer the other way.

---

## 9. Related work to check and cite
- Rimsky et al. 2024: CAA (DiM).
- Rafailov et al. 2023: DPO.
- Cao et al. 2024, *BiPO*: steering vectors from a preference objective; closest to DPO-grad.
- Im & Li 2025: a unified analysis of steering methods.
- Wu et al. 2025, *AxBench*: a steering benchmark.
- Kaushik et al. 2020: counterfactually-augmented data (natural-text sentiment minimal pairs); the out-of-distribution follow-up.

These are from memory; verify every citation before using it.

---

## 10. Write-up (≤1 page)
- **Definition of "better" and setup:** ~⅓ page.
- **The claim and the 2 × 2:** 3–4 lines, with the prediction-difficulty argument in one sentence.
- **Figure 1** and **Table 1**, plus one example continuation.
- **Discussion:** P1, then P2 and what the 2 × 2 says about mechanism. Then the scope: the test is close to in-distribution, out-of-distribution behavior is untested and could rank differently, and this is a proof of concept on one small model and one narrow concept. Then 2 of the §8 questions, then the remaining limitations (a classifier proxy, one damage measure, three folds).
- **Framing throughout:** a proof of concept for a mechanism, not a verdict on which estimator is better in general.
