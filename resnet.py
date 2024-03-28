import os
from lib import common, conf
import keras

model_dir = "models/resnet"


def build_model(model_path: str):
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)

    return keras.applications.ResNet50(
        weights=None,
        classes=len(conf.CLASS_NAMES),
        include_top=False,
        input_shape=[conf.SIZE, conf.SIZE, 3],
    )


class ResNet:
    model = None
    model_path = f"{model_dir}/fromzero.keras"

    def __init__(self):
        self.model = build_model(self.model_path)

    def summary(self):
        self.model.summary()

    def train(self):
        common.train(self.model, self.model_path)

    def predict(self):
        common.predict(self.model_path)
