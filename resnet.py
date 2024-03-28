import os
from lib.common import train, predict
from lib.conf import SIZE, CLASS_NAMES, MODEL_DIR, SEPARATOR
import keras


def build_model_train(model_name: str, weights: str = None, trainable: bool = False):
    if os.path.exists(f"{MODEL_DIR}/{model_name}.keras"):
        return keras.models.load_model(f"{MODEL_DIR}/{model_name}.keras")

    resnet = keras.applications.ResNet50(
        weights=weights,
        include_top=False,
        input_shape=[SIZE, SIZE, 3],
    )
    resnet.trainable = False if weights is None else trainable
    model = keras.layers.GlobalAveragePooling2D()(resnet.output)
    model = keras.layers.Dense(
        len(CLASS_NAMES), activation=keras.activations.softmax, name="classification"
    )(model)
    return keras.models.Model(
        resnet.input, model, name=model_name.replace("/", SEPARATOR)
    )


def build_model(type: str = "fromzero"):
    if type == "fine_tuning":
        model_name = "resnet/fine_tuning"
        return model_name, build_model_train(model_name, "imagenet", True)
    elif type == "transfer_learning":
        model_name = "resnet/transfer_learning"
        return model_name, build_model_train(model_name, "imagenet", False)
    else:
        model_name = "resnet/fromzero"
        return model_name, build_model_train(model_name, None, False)


class ResNet:
    model = None
    model_name = None

    def __init__(self, type: str = "fromzero"):
        self.model_name, self.model = build_model(type)

    def summary(self):
        self.model.summary()
        print(self.model.name)

    def train(self):
        self.model.summary()
        train(self.model)

    def predict(self):
        predict(self.model)
