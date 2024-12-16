import re
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd


def preprocess_text(text):
    if not isinstance(text, str):
        return ""  
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = text.replace('\s+', ' ')
    words = text.split()
    return "startseq " + " ".join(words) + " endseq"


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def get_captions_with_file_names(path):
    captions = pd.read_csv(
        path, sep="|", header=None, names=["image", "number", "caption"], skiprows=1
    )
    return captions

captions = get_captions_with_file_names("images/results.csv")
print(captions)
captions["processed_caption"] = captions["caption"].apply(preprocess_text)
print(captions)