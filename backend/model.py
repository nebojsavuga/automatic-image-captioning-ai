from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input, add
from tensorflow.keras.models import Model


def get_model(max_length, vocabulary_size, feature_dim=2048):
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
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model
