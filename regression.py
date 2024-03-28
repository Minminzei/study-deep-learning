import keras
import numpy as np
import matplotlib.pyplot as plt

xdata = np.array([0, 1, 3, 5, 6], dtype=float)
ydata = np.array([1, 3, 7, 11, 13], dtype=float)
epochs = 200
input = 10

# modelビルド
model = keras.models.Sequential([keras.layers.Dense(1, input_shape=[1])])

# modelコンパイル
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.MeanSquaredError())

# modelトレーニング
history = model.fit(xdata, ydata, epochs=epochs)

# model推論
output = model.predict(np.array([input], dtype=float))

# 学習した回帰式を取得
weights = model.get_weights()
weight = weights[0][0][0]
bias = weights[1][0]

# 結果を出力
print(f"epochs: {epochs}")
print(f"predict: x={input}, y={output[0][0]}")
print(f"trained regression: y = {weight}x + {bias}")

# 学習過程をグラフ化
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(history.history["loss"])
plt.show()
