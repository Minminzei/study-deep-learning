import os
from lib.common import train, predict
from lib.conf import SIZE, CLASS_NAMES, MODEL_DIR, SEPARATOR
import keras


def build_model(model_name: str):
    if os.path.exists(f"{MODEL_DIR}/{model_name}.keras"):
        return keras.models.load_model(f"{MODEL_DIR}/{model_name}.keras")

    resnet = keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=[SIZE, SIZE, 3],
    )
    resnet.trainable = True
    model = keras.layers.GlobalAveragePooling2D()(resnet.output)
    model = keras.layers.Dense(
        len(CLASS_NAMES), activation=keras.activations.softmax, name="classification"
    )(model)
    return keras.models.Model(
        resnet.input, model, name=model_name.replace("/", SEPARATOR)
    )


class ResNet:
    model = None
    model_name = "resnet/fromzero"

    def __init__(self):
        self.model = build_model(self.model_name)

    def summary(self):
        self.model.summary()
        print(self.model.name)

    def train(self):
        self.model.summary()
        train(self.model)

    def predict(self):
        predict(self.model)
