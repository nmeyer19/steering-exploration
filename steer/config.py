"""Every constant and shared type for the experiment. See implementation_plan.md (T0)."""
from dataclasses import dataclass, field

import torch

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
CLF_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"  # label 0 = NEGATIVE, 1 = POSITIVE
WIKI_ID, WIKI_CONFIG = "Salesforce/wikitext", "wikitext-2-raw-v1"

LAYERS = [15, 8]  # 0-indexed model.model.layers[i]; 15 is the headline
HEADLINE_LAYER = 15
ALPHAS = [0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]  # alpha = ||v_applied|| / r_l
KL_BUDGET = 0.1

TARGET = "neg"  # set after the headroom gate (steer.gates)
READ_SUFFIX = " Overall, it was"

N_FOLDS, TEST_TOPICS_PER_FOLD, VAL_TOPICS = 3, 4, 2
N_TRAIN, N_VAL, N_TEST_PAIRS = 64, 32, 64

BATCH = 16
LR_GRID = [3e-4, 1e-3, 3e-3]  # Adam on u, where v = r_l * u
MAX_STEPS, EVAL_EVERY, PATIENCE = 400, 20, 5
BETA = 0.1

PROMPTS_PER_TOPIC, SAMPLES_PER_PROMPT = 8, 3
GEN_TOKENS, TEMPERATURE = 30, 0.7

N_WIKI, WIKI_LEN = 200, 64  # KL set and r_l set
N_WIKI_SHORT, WIKI_SHORT_LEN = 200, 20  # MoP's generic baseline (short chunks + READ_SUFFIX)
DISTINCT2_MIN = 0.5
SEED = 0
RESULTS_DIR = "results"

METHODS = ["dim", "mop", "eq5", "dpo", "random"]
GRADIENT_METHODS = ["eq5", "dpo"]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gen_seed(fold: int, alpha_idx: int) -> int:
    """Generation seed shared by every method at the same (fold, alpha index)."""
    return SEED * 10_000 + fold * 100 + alpha_idx


@dataclass
class Pair:
    topic: str
    noun: str
    pos: str  # the TARGET-side sentence (D_p)
    neg: str  # its minimal-pair twin (D_n)


@dataclass
class Fold:
    idx: int
    train_topics: list[str]
    val_topics: list[str]
    test_topics: list[str]
    train: list[Pair]
    val: list[Pair]
    test: list[Pair]
    prompts: list[tuple[str, str]]  # (topic, neutral prompt), test topics only


@dataclass
class SteerVec:
    method: str  # one of METHODS, or "eq5_nm" / "dpo_nm" for norm-matched retrains
    layer: int
    fold: int
    v: torch.Tensor  # (d,), raw, un-normalized, on CPU
    cost: dict = field(default_factory=dict)  # fwd, bwd, seconds, lr, steps
