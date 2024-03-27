import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.preprocessing.image as image
from conf import (
    CLASS_NAMES,
    SIZE,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    USE_EARLY_STOPPING,
)


def save_model(model, path):
    if os.path.exists(path):
        os.rename(path, f"{path}_bk")

    model.save(path)


def count_file():
    for current_dir, sub_dirs, files_list in os.walk("./resources/animals"):
        print(current_dir)
        print(len(files_list))


def load_by_generator():
    path = "resources/animals10"
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )
    train_generator = train_datagen.flow_from_directory(
        directory=path,
        target_size=(SIZE, SIZE),
        class_mode="categorical",
        subset="training",
    )
    validation_generator = train_datagen.flow_from_directory(
        path,
        target_size=(SIZE, SIZE),
        class_mode="categorical",
        subset="validation",
    )
    return train_generator, validation_generator


def visualize(history):
    plt.xlabel("epochs")
    fig = plt.figure()
    loss = fig.add_subplot(1, 2, 1)
    accuracy = fig.add_subplot(1, 2, 2)

    epochs = range(len(history.history["loss"]))
    loss.plot(epochs, history.history["loss"], label="training_loss")
    loss.plot(epochs, history.history["val_loss"], label="validation_loss", ls="dashed")

    accuracy.plot(epochs, history.history["accuracy"], label="training_acc")
    accuracy.plot(
        epochs, history.history["val_accuracy"], label="validation_acc", ls="dashed"
    )

    loss.legend()
    accuracy.legend()

    fig.tight_layout()
    fig.savefig("./models/result.png")


def train(model, model_path):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    train_data, validation_data = load_by_generator()

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            (keras.callbacks.EarlyStopping(patience=3) if USE_EARLY_STOPPING else None),
        ],
    )
    visualize(history)
    model.save(model_path)


def predict(model_path: str):
    if not os.path.exists(model_path):
        print(f"{model_path} is not exists")
        return

    model = keras.models.load_model(model_path)
    for file in glob.glob("./resources/tests/*.jpg"):
        img = image.load_img(file, target_size=(SIZE, SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        image_tensor = np.vstack([x])
        result = model.predict(image_tensor)
        print(result)
        top = tf.argmax(result, 1)
        print(f"{file} is {CLASS_NAMES[top.numpy()[0]]}")
