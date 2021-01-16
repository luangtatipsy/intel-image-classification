from typing import Tuple

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model


def create_cnn_model(image_size: Tuple[int, int], num_classes: int) -> Model:
    width, height = image_size
    input_ = Input(shape=(width, height, 3), name="input")

    x = Conv2D(16, (3, 3), activation="relu", name="block_1")(input_)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(32, (3, 3), activation="relu", name="block_2")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", name="block_3")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    prediction = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=input_, outputs=prediction)
