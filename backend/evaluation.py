import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge


_NLTK_RESOURCES_READY = False


def ensure_meteor_resources():
    global _NLTK_RESOURCES_READY

    if _NLTK_RESOURCES_READY:
        return True

    resource_checks = [
        ["corpora/wordnet", "corpora/wordnet.zip"],
        ["corpora/omw-1.4", "corpora/omw-1.4.zip"],
    ]
    resource_downloads = ["wordnet", "omw-1.4"]

    try:
        for options in resource_checks:
            found = False
            for resource_path in options:
                try:
                    nltk.data.find(resource_path)
                    found = True
                    break
                except LookupError:
                    continue
            if not found:
                raise LookupError

        wordnet.synsets("dog")
        _NLTK_RESOURCES_READY = True
        return True
    except LookupError:
        pass

    try:
        for resource_name in resource_downloads:
            nltk.download(resource_name, quiet=True)
        wordnet.synsets("dog")
        _NLTK_RESOURCES_READY = True
        return True
    except Exception:
        return False


def split_images(
    image_names: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
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


def remove_special_tokens(tokens: List[str]):
    return [token for token in tokens if token not in {"startseq", "endseq"}]


def calculate_metrics(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
):
    bleu = corpus_bleu(references, hypotheses) if references else 0.0

    rouge_scores = {"rouge-1": {}, "rouge-2": {}, "rouge-l": {}}
    if references and hypotheses:
        rouge = Rouge()
        hypothesis_texts = [" ".join(tokens) for tokens in hypotheses]
        first_reference_texts = [" ".join(refs[0]) for refs in references]
        rouge_scores = rouge.get_scores(hypothesis_texts, first_reference_texts, avg=True)

    meteor_values = []
    meteor_available = ensure_meteor_resources()
    if meteor_available:
        for refs, hyp in zip(references, hypotheses):
            try:
                meteor_values.append(meteor_score(refs, hyp))
            except LookupError:
                meteor_values = []
                meteor_available = False
                break
    meteor = float(np.mean(meteor_values)) if meteor_values else float("nan")

    return {
        "bleu": bleu,
        "rouge": rouge_scores,
        "meteor": meteor,
        "meteor_available": meteor_available,
    }


def calculate_sample_metrics(
    references: List[List[str]],
    hypothesis: List[str],
):
    bleu1 = 0.0
    meteor = float("nan")

    if references and hypothesis:
        smoothing = SmoothingFunction().method1
        bleu1 = sentence_bleu(
            references,
            hypothesis,
            weights=(1.0, 0.0, 0.0, 0.0),
            smoothing_function=smoothing,
        )

        if ensure_meteor_resources():
            try:
                meteor = float(meteor_score(references, hypothesis))
            except LookupError:
                meteor = float("nan")

    return {
        "bleu1": float(bleu1),
        "meteor": meteor,
    }


def export_evaluation_to_csv(
    output_file: str,
    image_names: List[str],
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
    sample_metrics: Optional[List[Dict[str, float]]] = None,
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, image_name in enumerate(image_names):
        refs = references[idx] if idx < len(references) else []
        hyp = hypotheses[idx] if idx < len(hypotheses) else []
        metrics = sample_metrics[idx] if sample_metrics and idx < len(sample_metrics) else {}

        row = {
            "image_name": image_name,
            "predicted_caption": " ".join(hyp),
            "bleu1": metrics.get("bleu1", float("nan")),
            "meteor": metrics.get("meteor", float("nan")),
            "category": "",
            "comment": "",
        }

        for ref_idx in range(5):
            row[f"reference_{ref_idx + 1}"] = " ".join(refs[ref_idx]) if ref_idx < len(refs) else ""

        rows.append(row)

    fieldnames = [
        "image_name",
        "predicted_caption",
        "reference_1",
        "reference_2",
        "reference_3",
        "reference_4",
        "reference_5",
        "bleu1",
        "meteor",
        "category",
        "comment",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
