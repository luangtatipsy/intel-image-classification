import os

from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import create_inception_model

dataset_dir = "datasets"

_, test_folder, train_folder = sorted(os.listdir(dataset_dir))

test_dir = os.path.join(dataset_dir, test_folder)
train_dir = os.path.join(dataset_dir, train_folder)

IMAGE_SIZE = (150, 150)

classes = os.listdir(train_dir)
num_classes = len(classes)
num_epochs = 50

data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3,
    fill_mode="nearest",
)


train_data_generator = data_generator.flow_from_directory(
    directory=train_dir,
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    batch_size=64,
)

test_data_generator = data_generator.flow_from_directory(
    directory=test_dir,
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    batch_size=64,
)

model = create_inception_model(image_size=IMAGE_SIZE, num_classes=num_classes)
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

best_model_path = "inception_best_model.h5"
checkpoint_callback = ModelCheckpoint(
    best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=2, min_delta=0.001, mode="auto"
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=2, min_lr=0.000003
)
csv_logger = CSVLogger("training_log.csv")

history = model.fit(
    train_data_generator,
    steps_per_epoch=train_data_generator.n // train_data_generator.batch_size,
    validation_data=test_data_generator,
    validation_steps=test_data_generator.n // test_data_generator.batch_size,
    epochs=num_epochs,
    callbacks=[checkpoint_callback, early_stopping, reduce_lr, csv_logger],
    verbose=1,
)

model.save("inception_model.h5")
model.save_weights("inception_model.h5")
