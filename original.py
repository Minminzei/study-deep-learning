import os
from lib.common import train, predict
from lib.conf import SIZE, CLASS_NAMES
import keras

model_dir = "models/original"


def build_model(model_path: str):
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)

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
            keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ],
        name="original_model",
    )


class Original:
    model = None
    model_path = f"{model_dir}/fromzero.keras"

    def __init__(self):
        self.model = build_model(self.model_path)

    def summary(self):
        self.model.summary()

    def train(self):
        train(self.model, self.model_path)

    def predict(self):
        predict(self.model_path)
