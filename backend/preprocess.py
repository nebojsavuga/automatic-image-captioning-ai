import os
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


def preprocess_and_save_images_to_npy(folder_path, output_file):
    preprocessed_images = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)  
            preprocessed_image = preprocess_image(image_path)  
            preprocessed_images.append(preprocessed_image[0])  
    
    preprocessed_images = np.array(preprocessed_images)
    np.save(output_file, preprocessed_images)
    print(f"Sačuvano {len(preprocessed_images)} preprocesiranih slika u {output_file}")
    return preprocessed_images



captions = get_captions_with_file_names("images/results.csv")
captions["caption"] = captions["caption"].apply(preprocess_text)
captions.to_csv("images/preprocessed_captions.csv", index=False)
# folder_path = "images/flickr30k_images"  
# output_file = "images/preprocessed_images.npy"
# preprocessed_images = preprocess_and_save_images_to_npy(folder_path, output_file)
# print(f"Ukupno preprocesiranih slika: {len(preprocessed_images)}")
