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

## Usage
1. Install requirements:
	```bash
	pip install -r backend/requirements.txt
	```
2. Prepare dataset in `backend/images/flickr30k_images/` and captions in `backend/images/results.csv`
3. Run preprocessing and feature extraction:
	```bash
	python backend/preprocess.py
	python backend/image_feature_extraction.py
	```
4. Train and evaluate model:
	```bash
	python backend/main.py
	```

## Git note
Do not commit dataset files or generated artifacts (`backend/images/...`) to Git.

## Results
Evaluation metrics and generated captions are shown after training. Example images and captions are visualized.

## Project for course: Sistemi za istraživanje i analizu podataka

