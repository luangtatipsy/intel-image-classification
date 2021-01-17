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


def create_model(image_size: Tuple[int, int], num_classes: int) -> Model:
    inception_v3 = InceptionV3(
        input_shape=(*image_size, 3), include_top=False, weights="imagenet"
    )
    for layer in inception_v3.layers:
        layer.trainable = False

    # output = inception_v3.output
    output = inception_v3.get_layer("mixed9").output

    x = Flatten()(output)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)

    prediction = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inception_v3.input, outputs=prediction)
