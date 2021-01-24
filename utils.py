import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from models import create_model


class ImagePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, image_size=(150, 150)):
        self.image_size = image_size

    def fit(self, X, y=None, **fit_params):
        return self

    def preprocess(self, x):
        x = load_img(x, target_size=self.image_size)
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        return x

    def transform(self, X):
        return [self.preprocess(x) for x in X]


class ImageClassifier(BaseEstimator, ClassifierMixin):
    CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

    def __init__(
        self, model_path="models/intel_img_clf_best_weight.h5", image_size=(150, 150)
    ):
        self.model = create_model(
            image_size=image_size, num_classes=len(ImageClassifier.CLASSES)
        )
        self.model.load_weights(model_path)

    def fit(self, X, y=None, **fit_params):
        return self

    def predict(self, X):
        y_preds = [self.model.predict(x) for x in X]
        y_preds = [
            (ImageClassifier.CLASSES[y_pred.argmax()], y_pred.max())
            for y_pred in y_preds
        ]

        return y_preds

    def transform(self, X):
        return self.predict(X)
