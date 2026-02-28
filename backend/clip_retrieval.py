import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    import open_clip
except ImportError:
    open_clip = None

from evaluation import calculate_metrics, remove_special_tokens, split_images
from preprocess import (
    build_caption_mapping,
    filter_images_with_captions,
    get_captions_with_file_names,
    list_available_images,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTIONS_FILE = BASE_DIR / "images" / "results.csv"
DEFAULT_IMAGE_DIR = BASE_DIR / "images" / "flickr30k_images"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
OPENCLIP_RN50_ALIASES = {"openai/clip-rn50", "clip-rn50", "rn50"}


def to_feature_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        # Fallback for model outputs without explicit pooled feature.
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported output type for feature extraction: {type(output)}")


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text_bank(
    train_images: List[str],
    caption_mapping: Dict[str, List[str]],
    captions_per_image: int,
) -> List[str]:
    text_bank: List[str] = []
    for image_name in train_images:
        captions = caption_mapping[image_name][:captions_per_image]
        for caption in captions:
            cleaned = " ".join(remove_special_tokens(caption.split())).strip()
            if cleaned:
                text_bank.append(cleaned)
    return text_bank


def encode_texts(
    text_bank: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    encoded_batches = []
    for start_idx in range(0, len(text_bank), batch_size):
        batch_texts = text_bank[start_idx : start_idx + batch_size]
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = to_feature_tensor(text_features)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        encoded_batches.append(text_features.cpu().numpy())
    return np.vstack(encoded_batches).astype(np.float32)


def encode_images(
    image_dir: str,
    image_names: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    encoded_batches = []
    for start_idx in range(0, len(image_names), batch_size):
        batch_names = image_names[start_idx : start_idx + batch_size]
        batch_images = []
        for image_name in batch_names:
            image_path = os.path.join(image_dir, image_name)
            with Image.open(image_path) as image:
                batch_images.append(image.convert("RGB"))

        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = to_feature_tensor(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        encoded_batches.append(image_features.cpu().numpy())
    return np.vstack(encoded_batches).astype(np.float32)


def encode_texts_openclip(
    text_bank: List[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    encoded_batches = []
    for start_idx in range(0, len(text_bank), batch_size):
        batch_texts = text_bank[start_idx : start_idx + batch_size]
        tokens = tokenizer(batch_texts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        encoded_batches.append(text_features.cpu().numpy())
    return np.vstack(encoded_batches).astype(np.float32)


def encode_images_openclip(
    image_dir: str,
    image_names: List[str],
    model,
    image_preprocess,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    encoded_batches = []
    for start_idx in range(0, len(image_names), batch_size):
        batch_names = image_names[start_idx : start_idx + batch_size]
        batch_images = []
        for image_name in batch_names:
            image_path = os.path.join(image_dir, image_name)
            with Image.open(image_path) as image:
                batch_images.append(image_preprocess(image.convert("RGB")))

        image_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        encoded_batches.append(image_features.cpu().numpy())
    return np.vstack(encoded_batches).astype(np.float32)


def evaluate_predictions(
    test_images: List[str],
    caption_mapping: Dict[str, List[str]],
    predicted_captions: Dict[str, str],
) -> Dict[str, object]:
    references = []
    hypotheses = []

    for image_name in test_images:
        predicted_caption = predicted_captions.get(image_name, "")
        hypothesis_tokens = normalize_text(predicted_caption).split()
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

    metric_values = calculate_metrics(references, hypotheses)
    return {
        **metric_values,
        "references": references,
        "hypotheses": hypotheses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP retrieval baseline for image captioning."
    )
    parser.add_argument("--captions-file", default=str(DEFAULT_CAPTIONS_FILE))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--captions-per-image", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    captions_df = get_captions_with_file_names(args.captions_file)
    caption_mapping = build_caption_mapping(captions_df)

    image_names = list_available_images(args.image_dir)
    image_names = filter_images_with_captions(image_names, caption_mapping)

    if args.max_images is not None:
        image_names = image_names[: args.max_images]

    if len(image_names) < 3:
        raise ValueError("Need at least 3 images with captions.")
    if args.captions_per_image <= 0:
        raise ValueError("captions_per_image must be >= 1.")

    train_images, _, test_images = split_images(
        image_names, args.train_ratio, args.val_ratio, args.seed
    )

    text_bank = build_text_bank(train_images, caption_mapping, args.captions_per_image)
    if not text_bank:
        raise ValueError("Text bank is empty. Check captions preprocessing.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_openclip_rn50 = args.clip_model.lower() in OPENCLIP_RN50_ALIASES

    if use_openclip_rn50:
        if open_clip is None:
            raise ImportError(
                "RN50 backend requires open_clip_torch. Install with: pip install open_clip_torch"
            )
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            "RN50", pretrained="openai", device=device
        )
        tokenizer = open_clip.get_tokenizer("RN50")
        model.eval()

        text_embeddings = encode_texts_openclip(
            text_bank, model, tokenizer, device, args.batch_size
        )
        image_embeddings = encode_images_openclip(
            args.image_dir,
            test_images,
            model,
            image_preprocess,
            device,
            args.batch_size,
        )
    else:
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        model = CLIPModel.from_pretrained(args.clip_model).to(device)
        model.eval()

        text_embeddings = encode_texts(
            text_bank, model, processor, device, args.batch_size
        )
        image_embeddings = encode_images(
            args.image_dir, test_images, model, processor, device, args.batch_size
        )

    similarities = image_embeddings @ text_embeddings.T
    best_indices = np.argmax(similarities, axis=1)
    predicted_captions = {
        image_name: text_bank[int(best_idx)]
        for image_name, best_idx in zip(test_images, best_indices)
    }

    metrics = evaluate_predictions(test_images, caption_mapping, predicted_captions)

    print(f"CLIP model: {args.clip_model}")
    print(f"Backend: {'open_clip (RN50)' if use_openclip_rn50 else 'transformers'}")
    print(f"Images: train={len(train_images)} test={len(test_images)}")
    print(f"Text bank size: {len(text_bank)}")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"ROUGE: {metrics['rouge']}")
    print(f"METEOR: {metrics['meteor']:.4f}")

    print("\nSample predictions:")
    for idx, image_name in enumerate(test_images[:5]):
        generated = (
            " ".join(metrics["hypotheses"][idx])
            if idx < len(metrics["hypotheses"])
            else ""
        )
        reference = (
            " ".join(metrics["references"][idx][0])
            if idx < len(metrics["references"])
            else ""
        )
        print(f"- {image_name}")
        print(f"  Predicted: {generated}")
        print(f"  Reference: {reference}")


if __name__ == "__main__":
    main()  