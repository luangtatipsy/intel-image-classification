from typing import Tuple

from tensorflow.keras.applications.inception_v3 import InceptionV3
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
    input_ = Input(shape=(*image_size, 3), name="input")

    x = Conv2D(64, (3, 3), activation="relu", name="conv_1_1")(input_)
    x = Conv2D(64, (3, 3), activation="relu", name="conv_1_2")(x)
    x = MaxPooling2D((3, 3))(x)

    x = Conv2D(128, (3, 3), activation="relu", name="conv_2_1")(x)
    x = Conv2D(128, (3, 3), activation="relu", name="conv_2_2")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation="relu", name="conv_3_1")(x)
    x = Conv2D(256, (3, 3), activation="relu", name="conv_3_2")(x)
    x = MaxPooling2D((3, 3))(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)

    prediction = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=input_, outputs=prediction)


def create_inception_model(image_size: Tuple[int, int], num_classes: int) -> Model:
    inception_v3 = InceptionV3(
        input_shape=(*image_size, 3), include_top=False, weights="imagenet"
    )
    for layer in inception_v3.layers:
        layer.trainable = False

    output = inception_v3.output
    x = Flatten()(output)
    x = Dense(512, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)

    prediction = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inception_v3.input, outputs=prediction)
