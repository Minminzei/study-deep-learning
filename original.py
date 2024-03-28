import os
from lib.common import train, predict
from lib.conf import SIZE, CLASS_NAMES, MODEL_DIR, SEPARATOR
import keras


def build_model(model_name: str):
    if os.path.exists(f"{MODEL_DIR}/{model_name}.keras"):
        return keras.models.load_model(f"{MODEL_DIR}/{model_name}.keras")

    return keras.models.Sequential(
        [
            keras.layers.Input(shape=(SIZE, SIZE, 3)),
            keras.layers.Conv2D(
                16,
                (3, 3),
                activation=keras.activations.relu,
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(
                32,
                (3, 3),
                activation=keras.activations.relu,
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(
                64,
                (3, 3),
                activation=keras.activations.relu,
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(
                64,
                (3, 3),
                activation=keras.activations.relu,
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(
                64,
                (3, 3),
                activation=keras.activations.relu,
            ),
            keras.layers.Conv2D(
                64,
                (3, 3),
                activation=keras.activations.relu,
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(
                516,
                activation=keras.activations.relu,
            ),
            keras.layers.Dense(
                256,
                activation=keras.activations.relu,
            ),
            keras.layers.Dense(
                len(CLASS_NAMES), activation="softmax", name="classification"
            ),
        ],
        name=model_name.replace("/", SEPARATOR),
    )


class Original:
    model = None
    model_name = "original/fromzero"

    def __init__(self):
        self.model = build_model(self.model_name)

    def summary(self):
        self.model.summary()
        print(self.model.name)

    def train(self):
        train(self.model)

    def predict(self):
        predict(self.model)
