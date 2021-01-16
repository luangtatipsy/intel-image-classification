import os

from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import create_cnn_model

dataset_dir = "datasets"

_, test_folder, train_folder = sorted(os.listdir(dataset_dir))

test_dir = os.path.join(dataset_dir, test_folder)
train_dir = os.path.join(dataset_dir, train_folder)

RANDOM_SEED = 7
WIDTH, HEIGHT = (150, 150)

classes = os.listdir(train_dir)
num_classes = len(classes)
num_epochs = 100

data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3,
)


train_data_generator = data_generator.flow_from_directory(
    directory=train_dir,
    target_size=(WIDTH, HEIGHT),
    color_mode="rgb",
    class_mode="categorical",
    subset="training",
    batch_size=64,
    shuffle=False,
    seed=RANDOM_SEED,
)

test_data_generator = data_generator.flow_from_directory(
    directory=test_dir,
    target_size=(WIDTH, HEIGHT),
    color_mode="rgb",
    class_mode="categorical",
    subset="validation",
    batch_size=16,
    shuffle=False,
    seed=RANDOM_SEED,
)

model = create_cnn_model(image_size=(WIDTH, HEIGHT), num_classes=num_classes)
model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=2, min_delta=0.001, mode="auto"
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.001)
csv_logger = CSVLogger("trainning_log.csv")

history = model.fit(
    train_data_generator,
    steps_per_epoch=train_data_generator.n // train_data_generator.batch_size,
    validation_data=test_data_generator,
    validation_steps=test_data_generator.n // test_data_generator.batch_size,
    epochs=num_epochs,
    verbose=1,
)

model.save("cnn.h5")
model.save_weights("cnn_weights.h5")
