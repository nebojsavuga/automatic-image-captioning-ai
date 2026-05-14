# automatic-image-captioning-ai

## Team:
- Nebojša Vuga R2 23/2024
- Bogdan Janošević R2 43/2024

## Problem Definition:
The problem this project addresses is the automatic generation of textual descriptions for images. The goal of the project is to develop a system that uses artificial intelligence techniques to analyze the visual content of an image and automatically generate a description corresponding to that image.

## Course Connection:
The project is linked to the course Data Exploration and Analysis Systems.

## Dataset:
The dataset used for the project is sourced from the following link:
https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset?resource=download.
This dataset consists of 30,000 images from Flickr, where each image includes five different textual descriptions. These descriptions are provided in the form of a single sentence that describes the content of the image. The target feature is the textual description of the image, which consists of a sentence approximately 10–20 words long. The attribute of the dataset is the image itself.

## Methodology:
All images in the dataset will be scaled to standard dimensions (224x224). Additionally, pixel values will be normalized to a range of 0 to 1. The accompanying textual descriptions will be tokenized with the addition of special tokens: Start and End.

CNN and RNN (LSTM):
Convolutional Neural Networks (CNN) will be used to extract visual features from the images. These extracted features will then be passed to a sequential model, such as Long Short-Term Memory (LSTM), to generate the textual description of the image.

## Evaluation:
The dataset will be split into training, validation, and test sets in a 70:20:10 ratio.

For evaluating the predictions, the following metrics will be used:

BLEU (Bilingual Evaluation Understudy):
BLEU is suitable for strictly evaluating the precision of n-grams. It does not consider word order, synonyms, or meaning.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
ROUGE is effective for evaluating the coverage of key words and phrases from reference descriptions (recall). It is particularly useful for assessing whether the description includes all essential elements and reflects coverage better than BLEU.

METEOR (Metric for Evaluation of Translation with Explicit Ordering):
METEOR is best for evaluating semantic similarity and linguistic diversity. It combines precision and recall through harmonic mean. It also accounts for synonyms, resulting in a much more comprehensive evaluation. In addition to accuracy, it assesses linguistic fluency.


# Automatic Image Captioning AI

## Authors
Nebojša Vuga R2 23/2024  
Bogdan Janošević R2 43/2024

## Problem Definition
Automatic generation of textual descriptions for images using AI. The system analyzes visual information and generates a matching description.

## Dataset
Flickr30k: 30,000 images, each with 5 different descriptions. [Dataset link](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset?resource=download)

## Methodology
- Images scaled to 224x224, normalized (0-1)
- Text descriptions tokenized with Start/End tokens
- CNN (ResNet50/VGG16) for feature extraction
- LSTM for sequence generation

## Evaluation
- BLEU: n-gram precision
- ROUGE: recall of key phrases
- METEOR: semantic similarity, language diversity

## How to Run

### 1. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Prepare dataset

Download the [Flickr30k dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset?resource=download) and place files as follows:

```
backend/images/flickr30k_images/   ← all .jpg images
backend/images/results.csv         ← pipe-separated captions file
```

### 3. Preprocess captions

```bash
python backend/preprocess.py
```

| Parameter | Default | Description |
|---|---|---|
| `--captions-file` | `backend/images/results.csv` | Path to the raw captions CSV |
| `--image-dir` | `backend/images/flickr30k_images` | Directory containing the images |
| `--output-file` | `backend/images/preprocessed_captions.csv` | Where to save preprocessed captions |

### 4. Extract image features (ResNet50)

```bash
python backend/image_feature_extraction.py
```

| Parameter | Default | Description |
|---|---|---|
| `--captions-file` | `backend/images/results.csv` | Path to the captions CSV |
| `--image-dir` | `backend/images/flickr30k_images` | Directory containing the images |
| `--output-file` | `backend/images/image_features.npz` | Where to save extracted feature vectors |
| `--batch-size` | `32` | Number of images per batch during extraction |
| `--max-images` | `None` | Cap the number of images (useful for quick local tests) |

### 5. Train the CNN + LSTM model

Uses the pre-extracted features from step 4.

```bash
python backend/main.py
```

| Parameter | Default | Description |
|---|---|---|
| `--captions-file` | `backend/images/results.csv` | Path to the captions CSV |
| `--features-file` | `backend/images/image_features.npz` | Pre-extracted feature file from step 4 |
| `--evaluation-file` | `backend/images/evaluation_results_main.csv` | Where to save per-image evaluation results |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `64` | Training mini-batch size |
| `--train-ratio` | `0.7` | Fraction of images used for training |
| `--val-ratio` | `0.2` | Fraction of images used for validation (rest = test) |
| `--seed` | `3` | Random seed for reproducibility |
| `--max-images` | `None` | Cap total images used |
| `--max-train-samples` | `800000` | Max training word-prediction samples (memory guard) |
| `--max-val-samples` | `120000` | Max validation samples |

---

## Alternative / Additional Scripts

### Fine-tune (end-to-end CNN + LSTM, no pre-extracted features)

Trains ResNet50 + LSTM end-to-end directly from raw images — slower but potentially higher accuracy.

```bash
python backend/fine_tune.py
```

| Parameter | Default | Description |
|---|---|---|
| `--captions-file` | `backend/images/results.csv` | Path to the captions CSV |
| `--image-dir` | `backend/images/flickr30k_images` | Directory containing the images |
| `--evaluation-file` | `backend/images/evaluation_results_fine_tune.csv` | Where to save evaluation results |
| `--model-file` | `backend/images/fine_tune_best_model.keras` | Where to save the best checkpoint |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `16` | Training mini-batch size |
| `--train-ratio` | `0.7` | Fraction of images for training |
| `--val-ratio` | `0.2` | Fraction of images for validation |
| `--seed` | `3` | Random seed |
| `--max-images` | `None` | Cap total images |
| `--learning-rate` | `1e-4` | Adam optimizer learning rate |
| `--cnn-trainable-layers` | `30` | Number of ResNet50 tail layers to unfreeze |
| `--image-cache-size` | `64` | Number of images to cache in memory per epoch |
| `--patience` | `2` | Early-stopping patience (epochs without val improvement) |
| `--max-train-samples` | `800000` | Max training samples |
| `--max-val-samples` | `120000` | Max validation samples |

---

### ViT + GPT-2 (transformer baseline)

Evaluates a pre-trained ViT-GPT2 model from Hugging Face — no training required.

```bash
python backend/vit_gpt2.py
```

| Parameter | Default | Description |
|---|---|---|
| `--captions-file` | `backend/images/results.csv` | Path to the captions CSV |
| `--image-dir` | `backend/images/flickr30k_images` | Directory containing the images |
| `--model-name` | `nlpconnect/vit-gpt2-image-captioning` | Hugging Face model identifier |
| `--train-ratio` | `0.7` | Used only to determine the test split |
| `--val-ratio` | `0.2` | Used only to determine the test split |
| `--seed` | `3` | Random seed |
| `--max-images` | `None` | Cap total images |
| `--max-new-tokens` | `25` | Maximum tokens to generate per caption |
| `--num-beams` | `1` | Beam search width (1 = greedy) |
| `--length-penalty` | `1.0` | Exponential penalty on sequence length for beam search |
| `--no-repeat-ngram-size` | `2` | Suppress repeated n-grams of this size |

---

### CLIP retrieval baseline

Retrieves the nearest training caption for each test image using CLIP embeddings — no training required.

```bash
python backend/clip_retrieval.py
```

| Parameter | Default | Description |
|---|---|---|
| `--captions-file` | `backend/images/results.csv` | Path to the captions CSV |
| `--image-dir` | `backend/images/flickr30k_images` | Directory containing the images |
| `--clip-model` | `openai/clip-vit-base-patch32` | CLIP model name; use `rn50` for the OpenCLIP RN50 variant |
| `--train-ratio` | `0.7` | Used to build the training text bank |
| `--val-ratio` | `0.2` | Used to determine the test split |
| `--seed` | `3` | Random seed |
| `--batch-size` | `32` | Batch size for encoding images/texts |
| `--captions-per-image` | `1` | Number of captions per training image added to the text bank |
| `--max-images` | `None` | Cap total images |

---

### Hyperparameter search (fine-tune grid search)

Runs a grid search over learning rate, batch size, and CNN trainable layers for the fine-tune model.

```bash
python backend/hyperparameter_search.py
```

| Parameter | Default | Description |
|---|---|---|
| `--captions-file` | `backend/images/results.csv` | Path to the captions CSV |
| `--image-dir` | `backend/images/flickr30k_images` | Directory containing the images |
| `--output-file` | `backend/images/hyperparameter_search_results.csv` | Where to save the search results |
| `--train-ratio` | `0.7` | Training split ratio |
| `--val-ratio` | `0.2` | Validation split ratio |
| `--seed` | `3` | Random seed |
| `--max-images` | `1000` | Images to use per trial (keep small for speed) |
| `--max-train-samples` | `120000` | Max training samples per trial |
| `--max-val-samples` | `20000` | Max validation samples per trial |
| `--learning-rates` | `1e-4,3e-4` | Comma-separated list of learning rates to try |
| `--batch-sizes` | `8,16` | Comma-separated list of batch sizes to try |
| `--cnn-trainable-layers` | `10,20,30` | Comma-separated list of CNN layer counts to try |
| `--epochs` | `3` | Epochs per trial |
| `--patience` | `1` | Early-stopping patience per trial |
| `--image-cache-size` | `32` | Image cache size per trial |
| `--max-trials` | `None` | Limit total trials (runs all combinations by default) |

---

## Git note
Do not commit dataset files or generated artifacts (`backend/images/...`) to Git.

## Results
After training, BLEU / ROUGE / METEOR scores are printed to the console and per-image predictions are saved to the evaluation CSV. The Jupyter notebook `notebooks/Results_summary.ipynb` contains a summary of results across all approaches.

## Project for course: Sistemi za istraživanje i analizu podataka

