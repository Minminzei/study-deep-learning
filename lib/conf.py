# 分類クラス
CLASS_NAMES = [
    "cat",
    "dog",
    "horse",
]

# 画像サイズ
SIZE = 224

# 学習率
LEARNING_RATE = 0.001

# 早期終了
USE_EARLY_STOPPING = False

# エポック数
EPOCHS = 10

# バッチサイズ
BATCH_SIZE = 32

# 検証ステップ
VALIDATION_STEP = 3

# 評価指標
METRICS = "accuracy"  # ["accuracy", "precision", "recall", "f1"]


MODEL_DIR = "models"
SEPARATOR = "."
