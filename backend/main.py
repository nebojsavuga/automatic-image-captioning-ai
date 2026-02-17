import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import get_model
from preprocess import build_caption_mapping, get_captions_with_file_names

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTIONS_FILE = BASE_DIR / "images" / "results.csv"
DEFAULT_FEATURES_FILE = BASE_DIR / "images" / "image_features.npz"


def split_images(
    image_names: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    shuffled = image_names[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_images = shuffled[:train_end]
    val_images = shuffled[train_end:val_end]
    test_images = shuffled[val_end:]
    return train_images, val_images, test_images


def build_vocabulary(
    caption_mapping: Dict[str, List[str]], train_images: List[str]
) -> Tuple[Dict[str, int], Dict[int, str], int]:
    words = set()
    for image_name in train_images:
        for caption in caption_mapping[image_name]:
            words.update(caption.split())

    word2idx = {word: idx + 1 for idx, word in enumerate(sorted(words))}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(word2idx) + 1
    return word2idx, idx2word, vocab_size


def get_max_length(caption_mapping: Dict[str, List[str]], train_images: List[str]) -> int:
    max_length = 0
    for image_name in train_images:
        for caption in caption_mapping[image_name]:
            max_length = max(max_length, len(caption.split()))
    return max_length


def create_training_samples(
    image_names: List[str],
    caption_mapping: Dict[str, List[str]],
    features_by_image: Dict[str, np.ndarray],
    word2idx: Dict[str, int],
    max_length: int,
    max_samples: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_inputs = []
    sequence_inputs = []
    targets = []

    for image_name in image_names:
        feature = features_by_image[image_name]
        for caption in caption_mapping[image_name]:
            seq = [word2idx[word] for word in caption.split() if word in word2idx]
            for i in range(1, len(seq)):
                image_inputs.append(feature)
                sequence_inputs.append(seq[:i])
                targets.append(seq[i])
                if max_samples is not None and len(targets) >= max_samples:
                    break
            if max_samples is not None and len(targets) >= max_samples:
                break
        if max_samples is not None and len(targets) >= max_samples:
            break

    sequence_inputs = pad_sequences(sequence_inputs, maxlen=max_length, padding="post")

    return (
        np.array(image_inputs, dtype=np.float32),
        np.array(sequence_inputs, dtype=np.int32),
        np.array(targets, dtype=np.int32),
    )


def remove_special_tokens(tokens: List[str]) -> List[str]:
    return [token for token in tokens if token not in {"startseq", "endseq"}]


def generate_caption(
    feature: np.ndarray,
    model,
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    max_length: int,
) -> str:
    in_tokens = ["startseq"]
    for _ in range(max_length):
        sequence = [word2idx.get(token, 0) for token in in_tokens]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        yhat = model.predict([feature.reshape(1, -1), sequence], verbose=0)
        predicted_idx = int(np.argmax(yhat[0]))
        predicted_word = idx2word.get(predicted_idx)
        if predicted_word is None or predicted_word == "endseq":
            break
        in_tokens.append(predicted_word)
    return " ".join(remove_special_tokens(in_tokens)).strip()


def evaluate_model(
    model,
    test_images: List[str],
    caption_mapping: Dict[str, List[str]],
    features_by_image: Dict[str, np.ndarray],
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    max_length: int,
) -> Dict[str, object]:
    references = []
    hypotheses = []

    for image_name in test_images:
        generated_caption = generate_caption(
            features_by_image[image_name], model, word2idx, idx2word, max_length
        )
        hypothesis_tokens = generated_caption.split()
        if not hypothesis_tokens:
            hypothesis_tokens = ["."]

        reference_tokens = []
        for caption in caption_mapping[image_name]:
            cleaned = remove_special_tokens(caption.split())
            if cleaned:
                reference_tokens.append(cleaned)

        if not reference_tokens:
            continue

        references.append(reference_tokens)
        hypotheses.append(hypothesis_tokens)

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
        "references": references,
        "hypotheses": hypotheses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate captioning model.")
    parser.add_argument("--captions-file", default=str(DEFAULT_CAPTIONS_FILE))
    parser.add_argument("--features-file", default=str(DEFAULT_FEATURES_FILE))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=250000,
        help="Cap training samples for safer local runs.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=50000,
        help="Cap validation samples for safer local runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    captions_df = get_captions_with_file_names(args.captions_file)
    caption_mapping = build_caption_mapping(captions_df)

    features_data = np.load(args.features_file)
    image_names = [str(name) for name in features_data["image_names"]]
    features = features_data["features"].astype(np.float32)

    features_by_image = {image_name: features[i] for i, image_name in enumerate(image_names)}
    image_names = [name for name in image_names if name in caption_mapping]

    if args.max_images is not None:
        image_names = image_names[: args.max_images]

    if len(image_names) < 3:
        raise ValueError("Need at least 3 images with both features and captions.")

    train_images, val_images, test_images = split_images(
        image_names, args.train_ratio, args.val_ratio, args.seed
    )

    if not train_images or not val_images or not test_images:
        raise ValueError(
            "Split produced an empty subset. Adjust ratios or use more images."
        )

    word2idx, idx2word, vocab_size = build_vocabulary(caption_mapping, train_images)
    max_length = get_max_length(caption_mapping, train_images)
    feature_dim = features.shape[1]

    x_img_train, x_seq_train, y_train = create_training_samples(
        train_images,
        caption_mapping,
        features_by_image,
        word2idx,
        max_length,
        max_samples=args.max_train_samples,
    )
    x_img_val, x_seq_val, y_val = create_training_samples(
        val_images,
        caption_mapping,
        features_by_image,
        word2idx,
        max_length,
        max_samples=args.max_val_samples,
    )

    if len(y_train) == 0:
        raise ValueError("No training samples created. Check caption preprocessing.")

    model = get_model(max_length, vocab_size, feature_dim=feature_dim)

    fit_kwargs = {
        "x": [x_img_train, x_seq_train],
        "y": y_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    if len(y_val) > 0:
        fit_kwargs["validation_data"] = ([x_img_val, x_seq_val], y_val)

    model.fit(**fit_kwargs)

    metrics = evaluate_model(
        model,
        test_images,
        caption_mapping,
        features_by_image,
        word2idx,
        idx2word,
        max_length,
    )

    print(f"Images: train={len(train_images)} val={len(val_images)} test={len(test_images)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max caption length: {max_length}")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"ROUGE: {metrics['rouge']}")
    print(f"METEOR: {metrics['meteor']:.4f}")

    print("\nSample predictions:")
    for idx, image_name in enumerate(test_images[:5]):
        generated = " ".join(metrics["hypotheses"][idx]) if idx < len(metrics["hypotheses"]) else ""
        reference = (
            " ".join(metrics["references"][idx][0]) if idx < len(metrics["references"]) else ""
        )
        print(f"- {image_name}")
        print(f"  Predicted: {generated}")
        print(f"  Reference: {reference}")


if __name__ == "__main__":
    main()
