import os
from lib import common, conf
import keras

model_dir = "models/original"
model_path = f"{model_dir}/fromzero.keras"


def build_model():
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)

    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(conf.SIZE, conf.SIZE, 3)),
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
            keras.layers.Flatten(),
            keras.layers.Dense(
                516,
                activation=keras.activations.relu,
            ),
            keras.layers.Dense(
                256,
                activation=keras.activations.relu,
            ),
            keras.layers.Dense(len(conf.CLASS_NAMES), activation="softmax"),
        ],
        name="efficientnet_model",
    )
    return model


class EfficientNet:
    model = None

    def __init__(self):
        self.model = build_model()

    def summary(self):
        self.model.summary()

    def train(self):
        common.train(self.model, model_path)

    def transfer_learning(self):
        common.transfer_learning(self.model, model_path)

    def fine_tuning(self):
        common.fine_tuning(self.model, model_path)

    def predict(self, image):
        return common.predict(self.model, image)
