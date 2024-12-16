import re
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array

stop_words = [
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "is",
    "are",
    "was",
    "were",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "of",
    "to",
    "with",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "they",
    "we",
    "you",
    "i",
    "me",
    "my",
    "your",
    "our",
    "their",
    "be",
    "been",
    "being",
    "do",
    "did",
    "does",
    "have",
    "has",
    "having",
    "not",
]


def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return "start " + " ".join(words) + " end"


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


print(preprocess_text('Hello from world .'))