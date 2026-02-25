import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from preprocess import build_caption_mapping, get_captions_with_file_names

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTIONS_FILE = BASE_DIR / "images" / "results.csv"
DEFAULT_IMAGE_DIR = BASE_DIR / "images" / "flickr30k_images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "images"

BACKBONES: Dict[str, str] = {
    "vit": "openai/clip-vit-base-patch32",
    "resnet": "openai/clip-rn50",
}


def extract_features(image_dir: Path, image_names: List[str], model_name: str, batch_size: int, device: str) -> np.ndarray:
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    all_features = []
    with torch.no_grad():
        for start in range(0, len(image_names), batch_size):
            batch_names = image_names[start : start + batch_size]
            batch_images = [Image.open(image_dir / name).convert("RGB") for name in batch_names]
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_image_features(pixel_values=inputs["pixel_values"])
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            all_features.append(features.cpu().numpy().astype(np.float32))

    return np.vstack(all_features).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP image features (ViT or ResNet).")
    parser.add_argument("--captions-file", default=str(DEFAULT_CAPTIONS_FILE))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--backbone", choices=["vit", "resnet"], default="vit")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    captions_df = get_captions_with_file_names(args.captions_file)
    caption_mapping = build_caption_mapping(captions_df)

    image_dir = Path(args.image_dir)
    image_names = sorted(
        p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    image_names = [name for name in image_names if name in caption_mapping]

    if args.max_images is not None:
        image_names = image_names[: args.max_images]

    if not image_names:
        raise ValueError("No images matched with captions.")

    model_name = BACKBONES[args.backbone]
    features = extract_features(image_dir, image_names, model_name, args.batch_size, args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"clip_features_{args.backbone}.npz"

    np.savez(
        output_file,
        image_names=np.array(image_names),
        features=features,
        metadata=np.array([
            json.dumps(
                {
                    "backbone": args.backbone,
                    "model_name": model_name,
                    "feature_dim": int(features.shape[1]),
                    "images": int(len(image_names)),
                }
            )
        ], dtype=object),
    )

    print(f"Saved: {output_file}")
    print(f"Images: {len(image_names)}")
    print(f"Feature shape: {features.shape}")


if __name__ == "__main__":
    main()
