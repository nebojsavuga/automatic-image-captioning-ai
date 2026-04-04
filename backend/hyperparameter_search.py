import argparse
import csv
import itertools
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from caption_utils import build_vocabulary, get_max_length
from evaluation import split_images
from fine_tune import CaptionDataGenerator, create_caption_records
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
DEFAULT_OUTPUT_FILE = BASE_DIR / "images" / "hyperparameter_search_results.csv"


def parse_int_list(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def save_results_to_csv(output_file: str, rows: List[Dict[str, object]]):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial",
        "learning_rate",
        "batch_size",
        "cnn_trainable_layers",
        "epochs",
        "best_val_loss",
        "final_val_loss",
        "train_samples",
        "val_samples",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple hyperparameter search for the fine-tuning model."
    )
    parser.add_argument("--captions-file", default=str(DEFAULT_CAPTIONS_FILE))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--max-train-samples", type=int, default=120000)
    parser.add_argument("--max-val-samples", type=int, default=20000)
    parser.add_argument("--learning-rates", type=parse_float_list, default=[1e-4, 3e-4])
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[8, 16])
    parser.add_argument("--cnn-trainable-layers", type=parse_int_list, default=[10, 20, 30])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--image-cache-size", type=int, default=32)
    parser.add_argument("--max-trials", type=int, default=None)
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

    if len(image_names) < 3:
        raise ValueError("Need at least 3 images for hyperparameter search.")

    train_images, val_images, _ = split_images(
        image_names, args.train_ratio, args.val_ratio, args.seed
    )

    if not train_images or not val_images:
        raise ValueError("Train/validation split is empty. Adjust ratios or use more images.")

    word2idx, _, vocab_size = build_vocabulary(caption_mapping, train_images)
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

    if not train_records or not val_records:
        raise ValueError("No training or validation records created.")

    combinations = list(
        itertools.product(
            args.learning_rates,
            args.batch_sizes,
            args.cnn_trainable_layers,
        )
    )

    if args.max_trials is not None:
        combinations = combinations[: args.max_trials]

    results = []
    best_result = None

    for trial_idx, (learning_rate, batch_size, cnn_layers) in enumerate(combinations, start=1):
        print(
            f"Trial {trial_idx}/{len(combinations)}: "
            f"lr={learning_rate}, batch_size={batch_size}, cnn_layers={cnn_layers}"
        )

        train_generator = CaptionDataGenerator(
            train_records,
            args.image_dir,
            max_length,
            batch_size,
            shuffle=True,
            image_cache_size=args.image_cache_size,
        )
        val_generator = CaptionDataGenerator(
            val_records,
            args.image_dir,
            max_length,
            batch_size,
            shuffle=False,
            image_cache_size=args.image_cache_size,
        )

        model = get_fine_tune_model(
            max_length,
            vocab_size,
            cnn_trainable_layers=cnn_layers,
            learning_rate=learning_rate,
        )

        callbacks = []
        if args.patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=args.patience,
                    restore_best_weights=True,
                )
            )

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        val_losses = history.history.get("val_loss", [])
        if not val_losses:
            raise ValueError("Validation loss was not recorded during training.")

        result = {
            "trial": trial_idx,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "cnn_trainable_layers": cnn_layers,
            "epochs": args.epochs,
            "best_val_loss": float(min(val_losses)),
            "final_val_loss": float(val_losses[-1]),
            "train_samples": len(train_records),
            "val_samples": len(val_records),
        }
        results.append(result)

        save_results_to_csv(args.output_file, results)

        if best_result is None or result["best_val_loss"] < best_result["best_val_loss"]:
            best_result = result

    print(f"\nSaved search results to: {args.output_file}")
    if best_result is not None:
        print("Best parameters:")
        print(f"  learning_rate={best_result['learning_rate']}")
        print(f"  batch_size={best_result['batch_size']}")
        print(f"  cnn_trainable_layers={best_result['cnn_trainable_layers']}")
        print(f"  best_val_loss={best_result['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
