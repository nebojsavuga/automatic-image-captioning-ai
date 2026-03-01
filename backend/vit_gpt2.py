import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

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
DEFAULT_MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"


def load_caption_components(model_name: str, device: torch.device):
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load model '{model_name}'. "
            "If model is private/gated, run `hf auth login`."
        ) from exc

    model = model.to(device)
    model.eval()
    return model, image_processor, tokenizer


def generate_caption(
    image: Image.Image,
    model: VisionEncoderDecoderModel,
    image_processor: ViTImageProcessor,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text.strip()


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    parser = argparse.ArgumentParser(description="CLIP+GPT2 captioning evaluation.")
    parser.add_argument("--captions-file", default=str(DEFAULT_CAPTIONS_FILE))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=25)
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

    _, _, test_images = split_images(image_names, args.train_ratio, args.val_ratio, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_processor, tokenizer = load_caption_components(args.model_name, device)

    predicted_captions: Dict[str, str] = {}
    for image_name in test_images:
        image_path = os.path.join(args.image_dir, image_name)
        with Image.open(image_path) as image:
            predicted_captions[image_name] = generate_caption(
                image.convert("RGB"),
                model,
                image_processor,
                tokenizer,
                device,
                args.max_new_tokens,
            )

    metrics = evaluate_predictions(test_images, caption_mapping, predicted_captions)

    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Images: test={len(test_images)}")
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
