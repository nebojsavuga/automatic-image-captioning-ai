from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
import numpy as np

preprocessed_images = np.load("images/preprocessed_images.npy")

resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=resnet.input, outputs=resnet.output)

image_features = feature_extractor.predict(preprocessed_images)
np.save("Images/image_features.npy", image_features)

print(f"Izvučene karakteristike iz {len(preprocessed_images)} slika, dimenzije: {image_features.shape}")
