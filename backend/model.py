from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def get_model(max_length, vocabulary_size, feature_dim=2048, learning_rate=1e-3):
    image_inputs = Input(shape=(feature_dim,), name="image_features")
    image_branch = Dropout(0.5)(image_inputs)
    image_branch = Dense(256, activation="relu")(image_branch)

    sequence_inputs = Input(shape=(max_length,), name="input_sequence")
    sequence_branch = Embedding(vocabulary_size, 256, mask_zero=True)(sequence_inputs)
    sequence_branch = Dropout(0.5)(sequence_branch)
    sequence_branch = LSTM(256)(sequence_branch)

    decoder = add([image_branch, sequence_branch])
    decoder = Dense(256, activation="relu")(decoder)
    outputs = Dense(vocabulary_size, activation="softmax")(decoder)

    model = Model(inputs=[image_inputs, sequence_inputs], outputs=outputs)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
    )
    return model


def get_fine_tune_model(
    max_length,
    vocabulary_size,
    image_shape=(224, 224, 3),
    cnn_trainable_layers=30,
    learning_rate=1e-4,
):
    image_inputs = Input(shape=image_shape, name="image")

    cnn_base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    for layer in cnn_base.layers:
        layer.trainable = False

    if cnn_trainable_layers > 0:
        for layer in cnn_base.layers[-cnn_trainable_layers:]:
            if "batch_normalization" not in layer.name:
                layer.trainable = True

    image_branch = cnn_base(image_inputs)
    image_branch = Dropout(0.5)(image_branch)
    image_branch = Dense(256, activation="relu")(image_branch)

    sequence_inputs = Input(shape=(max_length,), name="input_sequence")
    sequence_branch = Embedding(vocabulary_size, 256, mask_zero=True)(sequence_inputs)
    sequence_branch = Dropout(0.5)(sequence_branch)
    sequence_branch = LSTM(256)(sequence_branch)

    decoder = add([image_branch, sequence_branch])
    decoder = Dense(256, activation="relu")(decoder)
    outputs = Dense(vocabulary_size, activation="softmax")(decoder)

    model = Model(inputs=[image_inputs, sequence_inputs], outputs=outputs)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
    )
    return model
