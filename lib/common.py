import os
import datetime
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
import keras
import keras.preprocessing.image as image
from lib.conf import (
    CLASS_NAMES,
    SIZE,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    USE_EARLY_STOPPING,
    METRICS,
    MODEL_DIR,
    SEPARATOR,
    VALIDATION_STEP,
)


def save_model(model, path):
    if os.path.exists(path):
        os.rename(path, f"{path}_bk")

    model.save(path)


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

    # keras.utils.image_dataset_from_directory(
    #     f"{path}/cat",
    #     validation_split=0.2,
    #     subset="training",
    #     image_size=(SIZE, SIZE),
    #     batch_size=BATCH_SIZE
    # )
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
    fig = plt.figure()
    fig.suptitle(
        f"epochs:{EPOCHS} batch_size:{BATCH_SIZE} "
        + f"learning_rate:{LEARNING_RATE} metrics:{METRICS}"
    )
    loss = fig.add_subplot(2, 1, 1)
    accuracy = fig.add_subplot(2, 1, 2)

    epochs = range(len(history.history["loss"]))
    loss.plot(epochs, history.history["loss"], label="training_loss")
    loss.plot(epochs, history.history["val_loss"], label="validation_loss", ls="dashed")

    accuracy.plot(epochs, history.history[METRICS], label="training_acc")
    accuracy.plot(
        epochs, history.history[f"val_{METRICS}"], label="validation_acc", ls="dashed"
    )

    loss.legend()
    accuracy.legend()

    fig.tight_layout()
    fig.savefig(f"./models/graph/{datetime.datetime.now()}.png")


def load_json(json_path):
    if os.path.exists(json_path):
        return json.load(open(json_path, "r"))

    return json.load(open("./models/template.json", "r"))


def save_json(json_path, history):
    data = load_json(json_path)
    data["epochs"] = EPOCHS if data["epochs"] is None else data["epochs"] + EPOCHS
    data["metrics"] = METRICS
    data["loss"] = history["loss"][-1]
    data["val_loss"] = history["val_loss"][-1]
    data["measure"] = history[METRICS][-1]
    data["val_measure"] = history[f"val_{METRICS}"][-1]
    data["datetime"] = datetime.datetime.now().strftime("%m/%d %H:%M")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def train(model):
    train_data, validation_data = load_by_generator()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[METRICS],
    )

    print(
        f"epochs:{EPOCHS}\nbatch_size:{BATCH_SIZE}\n"
        + f"learning_rate:{LEARNING_RATE}\nmetrics:{METRICS}\n"
        + f"classes: {train_data.class_indices}"
    )

    history = model.fit(
        train_data,
        validation_data=validation_data,
        validation_steps=VALIDATION_STEP,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=(
            [keras.callbacks.EarlyStopping(patience=3)] if USE_EARLY_STOPPING else []
        ),
    )

    model_name = model.name.replace(SEPARATOR, "/")
    save_model(model, f"{MODEL_DIR}/{model_name}.keras")
    save_json(f"{MODEL_DIR}/{model_name}.json", history.history)
    visualize(history)


def predict(model):
    # 正解率
    # クラスごとの適合率
    # クラスごとの再現率
    test_size = len(glob.glob("./resources/tests/*.jpg"))
    accuracy = 0
    results = []
    for file in glob.glob("./resources/tests/*.jpg"):
        label = file.split("/")[-1].split("_")[0]
        img = image.load_img(file, target_size=(SIZE, SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        image_tensor = np.vstack([x])
        result = model.predict(image_tensor)
        top = tf.argmax(result, 1)
        class_names = CLASS_NAMES[top.numpy()[0]]
        results.append(f"predict:{class_names}/ label:{label}")
        if label == class_names:
            accuracy += 1

    print(f"accuracy: {accuracy / test_size}")
    print("\n".join(results))
