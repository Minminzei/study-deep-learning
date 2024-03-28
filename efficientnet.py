import os
from lib.common import train, predict
from lib.conf import SIZE, CLASS_NAMES, MODEL_DIR, SEPARATOR
import keras


def build_model_train(model_name: str, weights: str = None, trainable: bool = False):
    if os.path.exists(f"{MODEL_DIR}/{model_name}.keras"):
        return keras.models.load_model(f"{MODEL_DIR}/{model_name}.keras")

    model = keras.applications.EfficientNetB4(
        weights=weights,
        include_top=False,
        input_shape=[SIZE, SIZE, 3],
    )
    model.trainable = False if weights is None else trainable
    return keras.Sequential(
        [
            model,
            keras.layers.Flatten(),
            keras.layers.Dense(516, activation=keras.activations.relu),
            keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ],
        name=model_name.replace("/", SEPARATOR),
    )


def build_model(mode: str = "train"):
    if mode == "train":
        model_name = "efficientnet/fromzero"
        return model_name, build_model_train(model_name, None, False)
    elif mode == "transfer_learning":
        model_name = "efficientnet/transfer_learning"
        return model_name, build_model_train(model_name, "imagenet", False)
    else:
        model_name = "efficientnet/fine_tuning"
        return model_name, build_model_train(model_name, "imagenet", True)


class EfficientNet:
    model = None
    model_name = None

    def __init__(self, mode: str = "train"):
        self.model_name, self.model = build_model(mode)

    def summary(self):
        self.model.summary()
        print(self.model.name)

    def train(self):
        self.model.summary()
        train(self.model)

    def predict(self):
        predict(self.model)
