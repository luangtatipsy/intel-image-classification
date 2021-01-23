import os

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import create_model

dataset_dir = "datasets"
model_dir = "models"
log_dir = "logs"

_, test_dir, train_dir = sorted(os.listdir(dataset_dir))

test_dir = os.path.join(dataset_dir, test_dir)
train_dir = os.path.join(dataset_dir, train_dir)

IMAGE_SIZE = (150, 150)

classes = os.listdir(train_dir)
num_classes = len(classes)
num_epochs = 15

data_augment_generator = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    preprocessing_function=preprocess_input,
    fill_mode="nearest",
)

train_data_generator = data_augment_generator.flow_from_directory(
    train_dir, batch_size=32, class_mode="categorical", target_size=(150, 150)
)
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data_generator = data_generator.flow_from_directory(
    test_dir, batch_size=128, class_mode="categorical", target_size=(150, 150)
)

model = create_model(image_size=IMAGE_SIZE, num_classes=num_classes)
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

best_model_path = os.path.join(model_dir, "intel_img_clf_best_weight.h5")
checkpoint_callback = ModelCheckpoint(
    best_model_path,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, verbose=1, factor=0.25, min_lr=0.000003
)


csv_logger = CSVLogger(os.path.join(log_dir, "training_log.csv"))

history = model.fit(
    train_data_generator,
    validation_data=test_data_generator,
    epochs=num_epochs,
    callbacks=[checkpoint_callback, reduce_lr, csv_logger],
    verbose=1,
)
