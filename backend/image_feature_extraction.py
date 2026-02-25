import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

from preprocess import (
    build_caption_mapping,
    filter_images_with_captions,
    get_captions_with_file_names,
    list_available_images,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTIONS_FILE = BASE_DIR / "images" / "results.csv"
DEFAULT_IMAGE_DIR = BASE_DIR / "images" / "flickr30k_images"
DEFAULT_OUTPUT_FILE = BASE_DIR / "images" / "image_features.npz"


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)


def extract_image_features(
    image_dir: str, image_names: List[str], batch_size: int
) -> np.ndarray:
    resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    feature_extractor = Model(inputs=resnet.input, outputs=resnet.output)

    all_features = []
    for start_idx in tqdm(range(0, len(image_names), batch_size), desc="Extracting features"):
        batch_names = image_names[start_idx : start_idx + batch_size]
        batch_images = []
        for image_name in batch_names:
            image_path = os.path.join(image_dir, image_name)
            batch_images.append(load_and_preprocess_image(image_path)[0])
        batch_images_array = np.array(batch_images, dtype=np.float32)
        batch_features = feature_extractor.predict(batch_images_array, verbose=0)
        all_features.append(batch_features)

    return np.vstack(all_features).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ResNet50 image features.")
    parser.add_argument("--captions-file", default=str(DEFAULT_CAPTIONS_FILE))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick local testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captions_df = get_captions_with_file_names(args.captions_file)
    caption_mapping = build_caption_mapping(captions_df)

    image_names = list_available_images(args.image_dir)
    image_names = filter_images_with_captions(image_names, caption_mapping)

    if args.max_images is not None:
        image_names = image_names[: args.max_images]

    if not image_names:
        raise ValueError("No images found that match captions file entries.")

    features = extract_image_features(args.image_dir, image_names, args.batch_size)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(
        args.output_file,
        image_names=np.array(image_names),
        features=features,
    )

    print(f"Extracted features for {len(image_names)} images.")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
