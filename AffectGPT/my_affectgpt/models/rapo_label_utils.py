import re
import math
import json
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_RAPO_VOCAB = [
    "neutral",
    "positive",
    "negative",
    "joy",
    "happy",
    "sad",
    "sadness",
    "angry",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
    "love",
    "excited",
    "anxious",
    "frustrated",
    "calm",
]


DEFAULT_ALIAS = {
    "happiness": "happy",
    "joyful": "joy",
    "delight": "joy",
    "delighted": "joy",
    "mad": "angry",
    "annoyed": "angry",
    "furious": "angry",
    "upset": "sad",
    "depressed": "sad",
    "sorrow": "sadness",
    "fearful": "fear",
    "scared": "fear",
    "surprised": "surprise",
    "disgusted": "disgust",
    "frustration": "frustrated",
    "excitement": "excited",
    "positivity": "positive",
    "negativity": "negative",
    "pos": "positive",
    "neg": "negative",
}


def _clean_label_text(text: str) -> str:
    text = text.strip().lower().replace("’", "'")
    # Remove common response templates only (exact phrase-level),
    # but never strip generic substrings like "emotion"/"state" globally.
    template_patterns = [
        r"\bthe character'?s emotional state is\b",
        r"\bthe character'?s sentiment state is\b",
        r"\bthe most likely label is\b",
        r"\bemotional state is\b",
        r"\bsentiment state is\b",
        r"\bemotion labels? are\b",
    ]
    for pattern in template_patterns:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[^a-z0-9,;/\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_token(token: str, alias_map: Dict[str, str]) -> str:
    token = token.strip().lower()
    if not token:
        return ""
    token = alias_map.get(token, token)
    return token


def split_labels(text: str) -> List[str]:
    if text is None:
        return []
    cleaned = _clean_label_text(text)
    if not cleaned:
        return []
    candidates = re.split(r",|;|/| and | or |\|", cleaned)
    labels = [x.strip() for x in candidates if x.strip()]
    return labels


def parse_labels(text: str, alias_map: Optional[Dict[str, str]] = None) -> List[str]:
    alias_map = alias_map or {}
    labels = split_labels(text)
    labels = [normalize_token(x, alias_map) for x in labels]
    labels = [x for x in labels if x]
    return list(dict.fromkeys(labels))


def read_vocab_from_json(vocab_path: str) -> List[str]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "vocab" in data:
            data = data["vocab"]
        else:
            data = list(data.keys())
    if not isinstance(data, list):
        raise ValueError(f"Unsupported vocab format in {vocab_path}")
    vocab = []
    for item in data:
        if not isinstance(item, str):
            continue
        token = item.strip().lower()
        if token:
            vocab.append(token)
    if not vocab:
        raise ValueError(f"Empty vocabulary in {vocab_path}")
    return list(dict.fromkeys(vocab))


def build_vocab(cfg_vocab: Optional[Sequence[str]] = None, vocab_path: str = "") -> List[str]:
    if cfg_vocab:
        vocab = [str(x).strip().lower() for x in cfg_vocab if str(x).strip()]
        if vocab:
            return list(dict.fromkeys(vocab))
    if vocab_path:
        return read_vocab_from_json(vocab_path)
    return list(DEFAULT_RAPO_VOCAB)


def labels_to_multihot(
    labels_batch: Sequence[str],
    vocab_to_idx: Dict[str, int],
    alias_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[List[float]], List[int], List[int]]:
    alias_map = alias_map or {}
    num_labels = len(vocab_to_idx)
    targets = []
    valid_mask = []
    label_counts = []
    for raw_text in labels_batch:
        labels = parse_labels(raw_text, alias_map=alias_map)
        vec = [0.0] * num_labels
        mapped = 0
        for label in labels:
            if label in vocab_to_idx:
                vec[vocab_to_idx[label]] = 1.0
                mapped += 1
        targets.append(vec)
        valid_mask.append(1 if mapped > 0 else 0)
        label_counts.append(mapped)
    return targets, valid_mask, label_counts


def label_count_to_confidence(label_count: int) -> float:
    if label_count <= 0:
        return 0.0
    return 1.0 / math.sqrt(float(label_count))
