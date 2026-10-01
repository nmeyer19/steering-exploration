"""Sentiment minimal pairs, topic folds, neutral prompts and WikiText chunks. See implementation_plan.md (T2)."""
import random

from steer.config import (N_FOLDS, N_TEST_PAIRS, N_TRAIN, N_VAL, N_WIKI, N_WIKI_SHORT, READ_SUFFIX, SEED,
                          TEST_TOPICS_PER_FOLD, VAL_TOPICS, WIKI_CONFIG, WIKI_ID, WIKI_LEN, WIKI_SHORT_LEN,
                          Fold, Pair)

TOPICS = {  # 12 topics, 4 nouns each. The nouns double as the leakage word lists (T5).
    "food": ["pizza", "pasta", "burger", "soup"],
    "movie": ["movie", "film", "sequel", "documentary"],
    "hotel": ["hotel", "suite", "resort", "motel"],
    "concert": ["concert", "band", "orchestra", "festival"],
    "book": ["book", "novel", "memoir", "anthology"],
    "museum": ["museum", "exhibit", "gallery", "sculpture"],
    "car": ["car", "truck", "sedan", "minivan"],
    "phone": ["phone", "laptop", "tablet", "smartwatch"],  # plan had "headphones": plural breaks "the X was"
    "park": ["park", "trail", "beach", "campsite"],
    "flight": ["flight", "airline", "airport", "layover"],
    "coffee": ["coffee", "latte", "cafe", "espresso"],
    "game": ["videogame", "boardgame", "puzzle", "chess"],
}

ADJ_TEMPLATES = ["The {noun} was {w}.", "I thought the {noun} was {w}.", "Honestly, the {noun} was {w}."]
VERB_TEMPLATES = ["I {w} the {noun}.", "We really {w} the {noun}."]

# (positive-sentiment word, negative-sentiment word)
ADJ_PAIRS = [("good", "bad"), ("great", "terrible"), ("wonderful", "awful"),
             ("excellent", "horrible"), ("amazing", "disappointing"), ("fantastic", "dreadful")]
VERB_PAIRS = [("loved", "hated"), ("liked", "disliked"), ("adored", "despised"), ("enjoyed", "loathed")]

PROMPT_TEMPLATES = [  # noun-agnostic: each reads naturally for any topic noun
    "My friend told me about the {noun}, so I",
    "So here's the thing about the {noun}:",
    "We decided to check out the {noun}, and",
    "I finally got around to the {noun}, and",
    "Last weekend I checked out the {noun} for the first time. It",
    "Here is what happened with the {noun}:",
    "A few days ago I had my first experience with the {noun}, and",
    "Everyone keeps asking me about the {noun}, so",
]
PROMPT_NOUNS, PROMPT_TEMPLATES_PER_NOUN = 2, 4


def all_pairs(target: str) -> list[Pair]:
    """Every topic x noun x template x word pair (1,248), with `pos` the `target`-side sentence."""
    assert target in ("pos", "neg")
    out = []
    for topic, nouns in TOPICS.items():
        for noun in nouns:
            for templates, words in ((ADJ_TEMPLATES, ADJ_PAIRS), (VERB_TEMPLATES, VERB_PAIRS)):
                for t in templates:
                    for wp, wn in words:
                        sp, sn = t.format(noun=noun, w=wp), t.format(noun=noun, w=wn)
                        out.append(Pair(topic, noun, *((sp, sn) if target == "pos" else (sn, sp))))
    return out


def _stratified(pairs: list[Pair], topics: list[str], n: int, rng: random.Random) -> list[Pair]:
    """n pairs from `topics`, taken round-robin over topics from a shuffled pool per topic."""
    pools = {t: [p for p in pairs if p.topic == t] for t in topics}
    for t in topics:
        rng.shuffle(pools[t])
    out = [pools[topics[i % len(topics)]][i // len(topics)] for i in range(n)]
    rng.shuffle(out)
    return out


def make_folds(target: str, seed: int = SEED) -> list[Fold]:
    """N_FOLDS topic folds; each topic is a test topic exactly once. Deterministic in `seed`."""
    pairs = all_pairs(target)
    topics = sorted(TOPICS)
    random.Random(seed).shuffle(topics)
    folds = []
    for k in range(N_FOLDS):
        rng = random.Random(seed + k)
        test = topics[TEST_TOPICS_PER_FOLD * k: TEST_TOPICS_PER_FOLD * (k + 1)]
        rest = [t for t in topics if t not in test]
        rng.shuffle(rest)
        val, train = rest[:VAL_TOPICS], rest[VAL_TOPICS:]
        prompts = [(t, tpl.format(noun=noun))
                   for t in test
                   for noun in rng.sample(TOPICS[t], PROMPT_NOUNS)
                   for tpl in rng.sample(PROMPT_TEMPLATES, PROMPT_TEMPLATES_PER_NOUN)]
        folds.append(Fold(k, train, val, test,
                          _stratified(pairs, train, N_TRAIN, rng), _stratified(pairs, val, N_VAL, rng),
                          _stratified(pairs, test, N_TEST_PAIRS, rng), prompts))
    return folds


def _wiki_chunks(tok, n: int, length: int, start: int = 0) -> list[str]:
    """n consecutive chunks of length-1 tokens (room for BOS) from token offset `start`, decoded."""
    from datasets import load_dataset
    ds = load_dataset(WIKI_ID, WIKI_CONFIG, split="validation")
    ids = tok("\n".join(l for l in ds["text"] if l.strip()), add_special_tokens=False)["input_ids"]
    m = length - 1
    assert start + n * m <= len(ids), "not enough WikiText tokens"
    return [tok.decode(ids[start + i * m: start + (i + 1) * m]) for i in range(n)]


def wiki_texts(tok, n: int = N_WIKI, length: int = WIKI_LEN) -> list[str]:
    """KL / r_l set: the first n chunks of WikiText-2 validation."""
    return _wiki_chunks(tok, n, length)


def wiki_short_texts(tok, n: int = N_WIKI_SHORT, length: int = WIKI_SHORT_LEN) -> list[str]:
    """MoP's generic baseline set: short chunks starting after those wiki_texts uses (at its defaults)."""
    return _wiki_chunks(tok, n, length, start=N_WIKI * (WIKI_LEN - 1))


def with_suffix(texts: list[str]) -> list[str]:
    return [t + READ_SUFFIX for t in texts]


def leak_words(topics: list[str]) -> list[str]:
    return [noun for t in topics for noun in TOPICS[t]]
