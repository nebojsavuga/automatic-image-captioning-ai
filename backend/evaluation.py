import random
from typing import Dict, List, Tuple

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge


def split_images(
    image_names: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("Train and validation ratios must be > 0.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    shuffled = image_names[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_images = shuffled[:train_end]
    val_images = shuffled[train_end:val_end]
    test_images = shuffled[val_end:]
    return train_images, val_images, test_images


def remove_special_tokens(tokens: List[str]) -> List[str]:
    return [token for token in tokens if token not in {"startseq", "endseq"}]


def calculate_metrics(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
) -> Dict[str, object]:
    bleu = corpus_bleu(references, hypotheses) if references else 0.0

    rouge_scores = {"rouge-1": {}, "rouge-2": {}, "rouge-l": {}}
    if references and hypotheses:
        rouge = Rouge()
        hypothesis_texts = [" ".join(tokens) for tokens in hypotheses]
        first_reference_texts = [" ".join(refs[0]) for refs in references]
        rouge_scores = rouge.get_scores(hypothesis_texts, first_reference_texts, avg=True)

    meteor_values = []
    for refs, hyp in zip(references, hypotheses):
        try:
            meteor_values.append(meteor_score(refs, hyp))
        except LookupError:
            meteor_values = []
            break
    meteor = float(np.mean(meteor_values)) if meteor_values else float("nan")

    return {
        "bleu": bleu,
        "rouge": rouge_scores,
        "meteor": meteor,
    }
