import os

import tensorflowjs as tfjs

from models import create_model
from utils import ImageClassifier

IMAGE_SIZE = (150, 150)
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "intel_img_clf_best_weight.h5")
OUTPUT_PATH = os.path.join(MODEL_DIR, "intel_img_clf_best_weight.js")

if __name__ == "__main__":
    model = create_model(
        image_size=IMAGE_SIZE, num_classes=len(ImageClassifier.CLASSES)
    )
    model.load_weights(MODEL_PATH)
    print("Model has been loaded")

    tfjs.converters.save_keras_model(model, OUTPUT_PATH)
    print(f"Model has been converted into {OUTPUT_PATH}")
