import re
from typing import List, Tuple
import numpy as np
import cv2
from typing import List
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

def tokenize_for_overlap(text: str) -> List[str]:
    """
    initial word separation
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text) # turn all numbers and symbols into space
    tokens = [t for t in text.split() if t] # separate by space
    return tokens


def jaccard_overlap(t1: str, t2: str) -> float:
    """
    calculate the Jaccard Overlap for two texts with 
        |W1 ∩ W2| / |W1 ∪ W2|
    measuring how much "repeat" are there in two consecutive frames
    """
    w1 = set(tokenize_for_overlap(t1))
    w2 = set(tokenize_for_overlap(t2))
    if not w1 and not w2:
        return 0.0
    inter = w1 & w2
    union = w1 | w2
    return len(inter) / len(union)


def compute_lexical_redundancy(texts: List[str]) -> Tuple[float, List[float]]:
    """
    Given a list of texts ordered by time (one sentence per frame),
    compute the Jaccard overlap between each pair of adjacent frames:
    overlap[i] = Jaccard(text[i-1], text[i])

    Returns:
        - avg_overlap: the average Jaccard overlap across all adjacent frame pairs (a single float)
        - overlaps: a list of Jaccard overlaps for each adjacent pair, with length = len(texts) - 1
    """
    if len(texts) <= 1:
        return 0.0, []

    overlaps = []
    for i in range(1, len(texts)):
        o = jaccard_overlap(texts[i - 1], texts[i])
        overlaps.append(o)

    avg_overlap = sum(overlaps) / len(overlaps)
    return avg_overlap, overlaps



_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2") # any random embedder will be fine
    return _embedder


# ================== Class-specific Information Density ==================

# Define action vocabularies by category; all keys must be lowercase
CLASS_ACTION_WORDS = {
    "__default__": {
        # Fallback: if a category has no custom definition, use this general action vocabulary
        "move", "moves", "moving", "moved",
        "push", "pushes", "pushing", "pushed",
        "pull", "pulls", "pulling", "pulled",
        "lift", "lifts", "lifting", "lifted",
        "open", "opens", "opening", "opened",
        "close", "closes", "closing", "closed",
    },

    "bending something so that it deforms": {
        "bend", "bends", "bending", "bent",
        "deform", "deforms", "deforming", "deformed",
        "curve", "curves", "curving", "curved",
        "warp", "warps", "warping", "warped",
        "flex", "flexes", "flexing", "flexed",
        "twist", "twists", "twisting", "twisted",
    },

    "closing something": {
        "close", "closes", "closing", "closed",
        "shut", "shuts", "shutting", "shut",
        "seal", "seals", "sealing", "sealed",
    },

    "folding something": {
        "fold", "folds", "folding", "folded",
        "crease", "creases", "creasing", "creased",
        "flatten", "flattens", "flattening", "flattened",
    },

    "pouring something into something": {
        "pour", "pours", "pouring", "poured",
        "flow", "flows", "flowing", "flowed",
        "fill", "fills", "filling", "filled",
        "stream", "streams", "streaming", "streamed",
    },

    "pouring something into something until it overflows": {
        "pour", "pours", "pouring", "poured",
        "flow", "flows", "flowing", "flowed",
        "fill", "fills", "filling", "filled",
        "overflow", "overflows", "overflowing", "overflowed",
        "spill", "spills", "spilling", "spilled",
        "splash", "splashes", "splashing", "splashed",
    },

    "something falling like a rock": {
        "fall", "falls", "falling", "fell", "fallen",
        "drop", "drops", "dropping", "dropped",
        "plummet", "plummets", "plummeting", "plummeted",
    },

    "throwing something": {
        "throw", "throws", "throwing", "threw", "thrown",
        "toss", "tosses", "tossing", "tossed",
        "fling", "flings", "flinging", "flung",
        "hurl", "hurls", "hurling", "hurled",
    },

    "uncovering something": {
        "uncover", "uncovers", "uncovering", "uncovered",
        "reveal", "reveals", "revealing", "revealed",
        "remove", "removes", "removing", "removed",
        "lift", "lifts", "lifting", "lifted",
        "open", "opens", "opening", "opened",
    },
}


def simple_tokenize(text: str) -> List[str]:
    """simple tokenizer to calculate info density"""
    return [t for t in text.lower().split() if t]


# In metrics.py (keep the original CLASS_ACTION_WORDS unchanged; 
# only modify the functions)

def _normalize_class_name(cls: str) -> str:
    if not cls:
        return "__default__"
    return cls.strip().lower()


def information_density_class_aware(text: str, cls: str) -> float:
    """
    Compute information density using both class-specific and default vocabularies:
    info_density = (# of tokens in (class_vocab ∪ default_vocab)) / (# of all tokens)
    """
    tokens = simple_tokenize(text)
    if not tokens:
        return 0.0

    key = _normalize_class_name(cls)
    default_vocab = CLASS_ACTION_WORDS["__default__"]
    class_vocab = CLASS_ACTION_WORDS.get(key, set())

    # Key point: use a union here
    action_vocab = default_vocab | class_vocab

    info_tokens = sum(1 for t in tokens if t in action_vocab)
    return info_tokens / len(tokens)


def average_information_density_class_aware(texts: List[str], cls: str) -> float:
    if not texts:
        return 0.0
    return sum(information_density_class_aware(t, cls) for t in texts) / len(texts)


