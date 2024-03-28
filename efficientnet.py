import os
from lib import common, conf
import keras

model_dir = "models/efficientnet"


def build_model_train(model_path: str, weights: str = None, trainable: bool = False):
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    if weights is None:
        return keras.applications.EfficientNetB0(
            weights=None, classes=len(conf.CLASS_NAMES)
        )

    model = keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=[conf.SIZE, conf.SIZE, 3],
    )
    model.trainable = trainable
    return keras.Sequential(
        [
            model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(516, activation=keras.activations.relu),
            keras.layers.Dense(
                len(conf.CLASS_NAMES), activation="softmax", name="flower_prob"
            ),
        ]
    )


def build_model(mode: str = "train"):
    if mode == "train":
        model_path = f"{model_dir}/fromzero.keras"
        return model_path, build_model_train(model_path, None, False)

    elif mode == "transfer_learning":
        model_path = f"{model_dir}/transfer_learning.keras"
        return model_path, build_model_train(model_path, "imagenet", False)
    else:
        model_path = f"{model_dir}/fine_tuning.keras"
        return model_path, build_model_train(model_path, "imagenet", True)


class EfficientNet:
    model = None
    model_path = None

    def __init__(self, mode: str = "train"):
        self.model_path, self.model = build_model(mode)

    def summary(self):
        self.model.summary()

    def train(self):
        common.train(self.model, self.model_path)

    def predict(self):
        common.predict(self.model_path)
