import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence as TypingSequence, Tuple

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

from caption_utils import build_vocabulary, get_max_length
from evaluation import (
    calculate_metrics,
    calculate_sample_metrics,
    export_evaluation_to_csv,
    remove_special_tokens,
    split_images,
)
from image_feature_extraction import load_and_preprocess_image
from model import get_fine_tune_model
from preprocess import (
    build_caption_mapping,
    filter_images_with_captions,
    get_captions_with_file_names,
    list_available_images,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTIONS_FILE = BASE_DIR / "images" / "results.csv"
DEFAULT_IMAGE_DIR = BASE_DIR / "images" / "flickr30k_images"
DEFAULT_EVALUATION_FILE = BASE_DIR / "images" / "evaluation_results_fine_tune.csv"
DEFAULT_MODEL_FILE = BASE_DIR / "images" / "fine_tune_best_model.keras"


def create_caption_records(
    image_names: List[str],
    caption_mapping: Dict[str, List[str]],
    word2idx: Dict[str, int],
    max_length: int,
    max_samples: int = None,
):
    records = []

    for image_name in image_names:
        for caption in caption_mapping[image_name]:
            seq = [word2idx[word] for word in caption.split() if word in word2idx][:max_length]
            if len(seq) < 2:
                continue

            for i in range(1, len(seq)):
                records.append((image_name, seq[:i], seq[i]))
                if max_samples is not None and len(records) >= max_samples:
                    return records

    return records


class CaptionDataGenerator(Sequence):
    def __init__(
        self,
        records: TypingSequence[Tuple[str, List[int], int]],
        image_dir: str,
        max_length: int,
        batch_size: int,
        shuffle: bool = True,
        image_cache_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.records = list(records)
        self.image_dir = Path(image_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_cache_size = max(0, image_cache_size)
        self.image_cache: Dict[str, np.ndarray] = {}
        self.cache_order: List[str] = []
        self.indices = np.arange(len(self.records))
        self.on_epoch_end()

    def __len__(self):
        if not self.records:
            return 0
        return int(np.ceil(len(self.records) / self.batch_size))

    def __getitem__(self, index: int):
        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        batch_indices = self.indices[batch_slice]

        batch_images = []
        batch_sequences = []
        batch_targets = []

        for record_idx in batch_indices:
            image_name, sequence, target = self.records[int(record_idx)]
            batch_images.append(self._load_image(image_name))
            batch_sequences.append(sequence)
            batch_targets.append(target)

        padded_sequences = pad_sequences(batch_sequences, maxlen=self.max_length, padding="post")

        return (
            (
                np.array(batch_images, dtype=np.float32),
                np.array(padded_sequences, dtype=np.int32),
            ),
            np.array(batch_targets, dtype=np.int32),
        )

    def on_epoch_end(self):
        if self.shuffle and len(self.indices) > 0:
            np.random.shuffle(self.indices)

    def _load_image(self, image_name: str):
        cached = self.image_cache.get(image_name)
        if cached is not None:
            return cached

        image_path = self.image_dir / image_name
        image_array = load_and_preprocess_image(str(image_path))[0]

        if self.image_cache_size > 0:
            if len(self.cache_order) >= self.image_cache_size:
                oldest_key = self.cache_order.pop(0)
                self.image_cache.pop(oldest_key, None)
            self.image_cache[image_name] = image_array
            self.cache_order.append(image_name)

        return image_array


def generate_caption(
    image_array: np.ndarray,
    model,
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    max_length: int,
):
    in_tokens = ["startseq"]

    for _ in range(max_length):
        sequence = [word2idx.get(token, 0) for token in in_tokens]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        prediction = model.predict([image_array.reshape(1, 224, 224, 3), sequence], verbose=0)
        predicted_idx = int(np.argmax(prediction[0]))
        predicted_word = idx2word.get(predicted_idx)

        if predicted_word is None or predicted_word == "endseq":
            break

        in_tokens.append(predicted_word)

    return " ".join(remove_special_tokens(in_tokens)).strip()


def evaluate_model(
    model,
    test_images: List[str],
    caption_mapping: Dict[str, List[str]],
    image_dir: str,
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    max_length: int,
):
    references = []
    hypotheses = []
    evaluated_images = []
    sample_metrics = []

    for image_name in test_images:
        image_path = Path(image_dir) / image_name
        image_array = load_and_preprocess_image(str(image_path))[0]
        generated_caption = generate_caption(image_array, model, word2idx, idx2word, max_length)

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
        evaluated_images.append(image_name)
        sample_metrics.append(calculate_sample_metrics(reference_tokens, hypothesis_tokens))

    metric_values = calculate_metrics(references, hypotheses)

    return {
        **metric_values,
        "references": references,
        "hypotheses": hypotheses,
        "evaluated_images": evaluated_images,
        "sample_metrics": sample_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CNN + LSTM image captioning model.")
    parser.add_argument("--captions-file", default=str(DEFAULT_CAPTIONS_FILE))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--evaluation-file", default=str(DEFAULT_EVALUATION_FILE))
    parser.add_argument("--model-file", default=str(DEFAULT_MODEL_FILE))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--cnn-trainable-layers",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--image-cache-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=800000,
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=120000,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    captions_df = get_captions_with_file_names(args.captions_file)
    caption_mapping = build_caption_mapping(captions_df)

    image_names = list_available_images(args.image_dir)
    image_names = filter_images_with_captions(image_names, caption_mapping)

    if args.max_images is not None:
        image_names = image_names[: args.max_images]

    train_images, val_images, test_images = split_images(
        image_names, args.train_ratio, args.val_ratio, args.seed
    )

    word2idx, idx2word, vocab_size = build_vocabulary(caption_mapping, train_images)
    max_length = min(get_max_length(caption_mapping, train_images), 30)

    train_records = create_caption_records(
        train_images,
        caption_mapping,
        word2idx,
        max_length,
        max_samples=args.max_train_samples,
    )
    val_records = create_caption_records(
        val_images,
        caption_mapping,
        word2idx,
        max_length,
        max_samples=args.max_val_samples,
    )

    train_generator = CaptionDataGenerator(
        train_records,
        args.image_dir,
        max_length,
        args.batch_size,
        shuffle=True,
        image_cache_size=args.image_cache_size,
    )

    fit_kwargs = {
        "x": train_generator,
        "epochs": args.epochs,
    }
    callbacks = []

    if val_records:
        val_generator = CaptionDataGenerator(
            val_records,
            args.image_dir,
            max_length,
            args.batch_size,
            shuffle=False,
            image_cache_size=args.image_cache_size,
        )
        fit_kwargs["validation_data"] = val_generator

        model_output_path = Path(args.model_file)
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath=str(model_output_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            )
        )

        if args.patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=args.patience,
                    restore_best_weights=True,
                )
            )

    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    model = get_fine_tune_model(
        max_length,
        vocab_size,
        cnn_trainable_layers=args.cnn_trainable_layers,
        learning_rate=args.learning_rate,
    )

    model.fit(**fit_kwargs)

    metrics = evaluate_model(
        model,
        test_images,
        caption_mapping,
        args.image_dir,
        word2idx,
        idx2word,
        max_length,
    )

    export_evaluation_to_csv(
        args.evaluation_file,
        metrics["evaluated_images"],
        metrics["references"],
        metrics["hypotheses"],
        metrics["sample_metrics"],
    )

    print(f"Images: train={len(train_images)} val={len(val_images)} test={len(test_images)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Max caption length: {max_length}")
    print(f"Training samples: {len(train_records)}")
    print(f"Validation samples: {len(val_records)}")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"ROUGE: {metrics['rouge']}")
    print(f"METEOR: {metrics['meteor']:.4f}")
    print(f"Saved evaluation results to: {args.evaluation_file}")
    if val_records:
        print(f"Saved best model to: {args.model_file}")

    print("\nSample predictions:")
    for idx, image_name in enumerate(metrics["evaluated_images"][:5]):
        generated = (
            " ".join(metrics["hypotheses"][idx]) if idx < len(metrics["hypotheses"]) else ""
        )
        reference = (
            " ".join(metrics["references"][idx][0]) if idx < len(metrics["references"]) else ""
        )
        print(f"- {image_name}")
        print(f"  Predicted: {generated}")
        print(f"  Reference: {reference}")


if __name__ == "__main__":
    main()
