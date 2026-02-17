import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTIONS_FILE = BASE_DIR / "images" / "results.csv"
DEFAULT_IMAGE_DIR = BASE_DIR / "images" / "flickr30k_images"
DEFAULT_OUTPUT_FILE = BASE_DIR / "images" / "preprocessed_captions.csv"


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return "startseq endseq"
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "startseq endseq"
    return f"startseq {text} endseq"


def get_captions_with_file_names(path: str) -> pd.DataFrame:
    captions = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=["image", "number", "caption"],
        engine="python",
    )
    captions = captions.dropna(subset=["image", "caption"]).copy()
    captions["image"] = captions["image"].astype(str).str.strip()
    captions["caption"] = captions["caption"].astype(str).str.strip()

    if not captions.empty:
        first_image = captions.iloc[0]["image"].lower()
        first_caption = captions.iloc[0]["caption"].lower()
        if first_image in {"image_name", "image", "filename"} or first_caption in {
            "comment",
            "caption",
        }:
            captions = captions.iloc[1:].copy()

    return captions.reset_index(drop=True)


def normalize_image_name(image_name: str) -> str:
    image_name = image_name.strip()
    if "#" in image_name:
        image_name = image_name.split("#", 1)[0]
    return image_name


def build_caption_mapping(captions_df: pd.DataFrame) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for _, row in captions_df.iterrows():
        image_name = normalize_image_name(str(row["image"]))
        caption = preprocess_text(row["caption"])
        mapping.setdefault(image_name, []).append(caption)
    return mapping


def list_available_images(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)
    )


def filter_images_with_captions(
    image_names: List[str], caption_mapping: Dict[str, List[str]]
) -> List[str]:
    return [name for name in image_names if name in caption_mapping]


def save_preprocessed_captions(caption_mapping: Dict[str, List[str]], output_file: str) -> None:
    rows = []
    for image_name, captions in caption_mapping.items():
        for caption in captions:
            rows.append({"image": image_name, "caption": caption})
    preprocessed_df = pd.DataFrame(rows)
    preprocessed_df.to_csv(output_file, index=False)
    print(f"Saved {len(preprocessed_df)} preprocessed captions to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Flickr captions.")
    parser.add_argument("--captions-file", default=str(DEFAULT_CAPTIONS_FILE))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captions_df = get_captions_with_file_names(args.captions_file)
    caption_mapping = build_caption_mapping(captions_df)

    image_names = list_available_images(args.image_dir)
    matched_images = filter_images_with_captions(image_names, caption_mapping)

    caption_mapping = {image: caption_mapping[image] for image in matched_images}

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_preprocessed_captions(caption_mapping, args.output_file)

    caption_count = sum(len(v) for v in caption_mapping.values())
    print(f"Images with captions: {len(caption_mapping)}")
    print(f"Total captions: {caption_count}")


if __name__ == "__main__":
    main()
