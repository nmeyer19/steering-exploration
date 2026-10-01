from collections import Counter

import pytest

from steer import config
from steer.data import (ADJ_PAIRS, TOPICS, VERB_PAIRS, all_pairs, leak_words, make_folds, wiki_short_texts,
                        wiki_texts, with_suffix)

POS_WORDS = {p for p, _ in ADJ_PAIRS + VERB_PAIRS}
NEG_WORDS = {n for _, n in ADJ_PAIRS + VERB_PAIRS}


@pytest.fixture(scope="module")
def folds():
    return make_folds(config.TARGET)


@pytest.mark.parametrize("target", ["pos", "neg"])
def test_pairs(target):
    pairs = all_pairs(target)
    assert len(pairs) == 1248
    assert len({(p.pos, p.neg) for p in pairs}) == 1248
    side = NEG_WORDS if target == "neg" else POS_WORDS
    for p in pairs:
        a, b = p.pos.split(), p.neg.split()
        assert len(a) == len(b)
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        assert len(diff) == 1
        assert diff[0][0].rstrip(".") in side
        assert p.noun in TOPICS[p.topic] and p.noun in p.pos


def test_fold_topics(folds):
    assert len(folds) == config.N_FOLDS
    test_counts = Counter(t for f in folds for t in f.test_topics)
    assert set(test_counts) == set(TOPICS) and set(test_counts.values()) == {1}
    for f in folds:
        assert len(f.test_topics) == config.TEST_TOPICS_PER_FOLD
        assert len(f.val_topics) == config.VAL_TOPICS
        assert len(f.train_topics) == len(TOPICS) - config.TEST_TOPICS_PER_FOLD - config.VAL_TOPICS
        assert set(f.train_topics) | set(f.val_topics) | set(f.test_topics) == set(TOPICS)


def test_fold_contents(folds):
    for f in folds:
        assert (len(f.train), len(f.val), len(f.test)) == (config.N_TRAIN, config.N_VAL, config.N_TEST_PAIRS)
        assert len(f.prompts) == config.PROMPTS_PER_TOPIC * config.TEST_TOPICS_PER_FOLD
        for split, topics in ((f.train, f.train_topics), (f.val, f.val_topics), (f.test, f.test_topics)):
            assert {p.topic for p in split} == set(topics)
            counts = Counter(p.topic for p in split).values()
            assert max(counts) - min(counts) <= 1  # round-robin stratification
            assert len({(p.pos, p.neg) for p in split}) == len(split)  # no duplicates
        prompt_topics = Counter(t for t, _ in f.prompts)
        assert set(prompt_topics) == set(f.test_topics)
        assert set(prompt_topics.values()) == {config.PROMPTS_PER_TOPIC}
        assert len(set(f.prompts)) == len(f.prompts)
        for t, s in f.prompts:
            assert any(n in s for n in TOPICS[t])


def test_determinism(folds):
    assert make_folds(config.TARGET) == folds
    assert make_folds(config.TARGET, seed=1) != folds


def test_helpers():
    assert with_suffix(["a", "b"]) == ["a" + config.READ_SUFFIX, "b" + config.READ_SUFFIX]
    assert leak_words(["food", "car"]) == TOPICS["food"] + TOPICS["car"]


@pytest.mark.slow
def test_wiki():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID)
    long, short = wiki_texts(tok), wiki_short_texts(tok)
    assert (len(long), len(short)) == (config.N_WIKI, config.N_WIKI_SHORT)
    assert not set(long) & set(short)
    for texts, length in ((long, config.WIKI_LEN), (short, config.WIKI_SHORT_LEN)):
        lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        assert all(length - 4 <= n <= length for n in lens), (min(lens), max(lens))
    assert wiki_texts(tok, n=3) == long[:3]
