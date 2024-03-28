import os
import glob
import tensorflow as tf
import numpy as np
import keras
import keras.preprocessing.image as image
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt

size = 150
convert_animals = [
    "cat",
    "dog",
    "horse",
]


def build_model(model_path: str):
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)

    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(size, size, 3)),
            keras.layers.Conv2D(
                16,
                (3, 3),
                activation=keras.activations.relu,
                kernel_regularizer=keras.regularizers.l2(0.001),
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.2),
            keras.layers.Conv2D(
                32,
                (3, 3),
                activation=keras.activations.relu,
                kernel_regularizer=keras.regularizers.l2(0.001),
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(
                64,
                (3, 3),
                activation=keras.activations.relu,
                kernel_regularizer=keras.regularizers.l2(0.001),
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.2),
            keras.layers.Conv2D(
                64,
                (3, 3),
                activation=keras.activations.relu,
                kernel_regularizer=keras.regularizers.l2(0.001),
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(
                64,
                (3, 3),
                activation=keras.activations.relu,
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(
                1024,
                activation=keras.activations.relu,
                kernel_regularizer=keras.regularizers.l2(0.001),
            ),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(
                256,
                activation=keras.activations.relu,
            ),
            keras.layers.Dense(3, activation="softmax"),
        ],
        name="animal_model",
    )
    return model


def pretrained_model(model_path: str):
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)

    base_model = InceptionV3(
        include_top=False, weights=None, input_shape=(size, size, 3)
    )
    base_model.load_weights("./models/inception_v3.h5")
    last_outputs = base_model.get_layer("mixed7").output

    for layer in base_model.layers:
        layer.trainable = False

    x = keras.layers.Flatten()(last_outputs)
    x = keras.layers.Dense(1024, activation=keras.activations.relu)(x)
    x = keras.layers.Dense(10, activation=keras.activations.softmax)(x)
    model = keras.models.Model(base_model.input, x)
    return model


def save_model(model, path):
    if os.path.exists(path):
        os.rename(path, f"{path}_bk")

    model.save(path)


def count_file():
    for current_dir, sub_dirs, files_list in os.walk("./resources/animals"):
        print(current_dir)
        print(len(files_list))


def argument_image(source, label):
    img = tf.image.resize(source, (size, size))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img, label


def load_by_generator():
    path = "resources/animals"
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
        target_size=(size, size),
        class_mode="categorical",
        subset="training",
    )

    validation_generator = train_datagen.flow_from_directory(
        path,
        target_size=(size, size),
        class_mode="categorical",
        subset="validation",
    )
    return train_generator, validation_generator


def summary():
    model = keras.models.load_model("./models/model.keras")
    model.summary()


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


def hyperparameter_tuning():
    print("hyperparameter_tuning")


def train():
    model_path = "./models/plane_model.keras"
    model = build_model(model_path)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()

    train_data, validation_data = load_by_generator()

    history = model.fit(
        train_data,
        epochs=100,
        validation_data=validation_data,
        validation_steps=1,
        batch_size=32,
        # 早期終了。学習が進んできたらコメントアウトを外す
        # callbacks=[keras.callbacks.EarlyStopping(patience=3)],
    )
    save_model(model, model_path)
    visualize(history)


def transfer_learning():
    model_path = "./models/transfer_model.keras"
    model = pretrained_model(model_path)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()
    train_data, validation_data = load_by_generator()()
    history = model.fit(
        train_data,
        epochs=30,
        validation_data=validation_data,
        validation_steps=1,
    )

    save_model(model, model_path)
    visualize(history)


def predict():
    model = keras.models.load_model("./models/plane_model.keras")
    for file in glob.glob("./resources/tests/*.jpg"):
        img = image.load_img(file, target_size=(size, size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        image_tensor = np.vstack([x])
        result = model.predict(image_tensor)
        print(result)
        top = tf.argmax(result, 1)
        print(f"{file} is {convert_animals[top.numpy()[0]]}")


train()
